//! Approve-time handler for `ReviewKind::ConceptCandidate` review items.
//!
//! bn-lw06: the `missing_concepts` lint (bn-31lt) queues
//! `lint:concept-candidate:<slug>` review items when a term appears in
//! enough source documents but has no concept page. On approve, the CLI
//! calls [`apply_concept_candidate`]:
//!
//! 1. Load snippets from each source's `normalized/<doc>/source.md` that
//!    mentions the candidate term.
//! 2. Call [`kb_llm::LlmAdapter::generate_concept_from_candidate`] to draft
//!    the canonical name, aliases, category, and definition.
//! 3. Optionally run a second pass via
//!    [`kb_llm::LlmAdapter::generate_concept_body`] (bn-1w5 two-step) to
//!    refine a variant-narrowed body. Best-effort: any failure here falls
//!    back to the first-pass definition.
//! 4. Write `wiki/concepts/<slug>.md` with proper frontmatter (frontmatter
//!    shape mirrors what `run_concept_merge_pass` produces so backlink and
//!    index scanners handle it identically — bn-eqx7 category field, bn-i5r
//!    index refresh, bn-2zy inline apply).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_yaml::{Mapping, Value};

use kb_core::fs::atomic_write;
use kb_core::frontmatter::read_frontmatter;
use kb_core::{ReviewItem, ReviewKind, slug_from_title};
use kb_llm::{
    CandidateSourceSnippet, GenerateConceptBodyRequest, GenerateConceptFromCandidateRequest,
    LlmAdapter,
};

use crate::concept_merge::WIKI_CONCEPTS_DIR;

/// Maximum characters per snippet fed into the prompt. Snippets are short
/// windows around each mention — long enough to disambiguate, short enough
/// to keep the prompt compact for batch approves.
const MAX_SNIPPET_CHARS: usize = 280;

/// Number of characters before the mention to include in each snippet.
const SNIPPET_LOOKBEHIND: usize = 80;

/// Maximum number of snippets per candidate. Hard cap so a term that
/// appears in 100 sources doesn't blow up the prompt.
const MAX_SNIPPETS: usize = 12;

/// Result of applying a `concept_candidate` review item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedConceptCandidate {
    /// Path of the concept page that was written (relative to the kb root).
    pub concept_path: PathBuf,
    /// Canonical name chosen by the LLM.
    pub canonical_name: String,
    /// Aliases chosen by the LLM.
    pub aliases: Vec<String>,
    /// Category chosen by the LLM (if any).
    pub category: Option<String>,
    /// Source document ids carried into the concept frontmatter.
    pub source_document_ids: Vec<String>,
}

/// Apply a `concept_candidate` review item by drafting and writing a new
/// concept page.
///
/// The review item is expected to carry:
/// - `citations` = source document ids where the term appears (also
///   mirrored in `metadata.dependencies`).
/// - `target_entity_id` = the review id in the form
///   `lint:concept-candidate:<slug>` (the slug is derived from the original
///   candidate name; the LLM may pick a cleaner canonical name, in which
///   case the file still lands under the slug derived from the LLM's
///   `canonical_name` so the concept id matches the file stem).
/// - `comment` contains the candidate term (but we parse the slug from the
///   id to avoid depending on comment formatting).
///
/// # Errors
///
/// Returns an error when:
/// - the item is not a `concept_candidate`,
/// - no sources could be read (both the LLM call and the snippet gather
///   would yield empty input, which we surface rather than writing a
///   low-quality concept page),
/// - the LLM call fails,
/// - the target concept page already exists (collision — the caller should
///   use `kb review reject` + manual edit instead of silently overwriting),
/// - the page cannot be written atomically.
pub fn apply_concept_candidate<A: LlmAdapter + ?Sized>(
    adapter: &A,
    root: &Path,
    item: &ReviewItem,
) -> Result<ApprovedConceptCandidate> {
    if item.kind != ReviewKind::ConceptCandidate {
        bail!(
            "apply_concept_candidate called on a {:?} review item",
            item.kind
        );
    }

    let candidate_name = extract_candidate_name(item)
        .with_context(|| format!("extract candidate name from review '{}'", item.metadata.id))?;

    // Source document ids in the review item's `citations` (set by the
    // lint writer — see `build_concept_candidate_review_item`). Fall back
    // to `metadata.dependencies` for forward-compat with other producers.
    let source_document_ids: Vec<String> = if item.citations.is_empty() {
        item.metadata.dependencies.clone()
    } else {
        item.citations.clone()
    };

    let snippets = gather_source_snippets(root, &candidate_name, &source_document_ids)
        .with_context(|| format!("gather snippets for candidate '{candidate_name}'"))?;

    if snippets.is_empty() {
        bail!(
            "no source snippets mentioning '{candidate_name}' could be read from normalized/; \
             cannot draft concept page (listed sources: {})",
            source_document_ids.join(", ")
        );
    }

    let existing_categories = collect_existing_categories(root).unwrap_or_default();

    // Primary draft: canonical + aliases + category + definition in one call.
    let request = GenerateConceptFromCandidateRequest {
        candidate_name: candidate_name.clone(),
        source_snippets: snippets.clone(),
        existing_categories,
    };
    let (draft, _provenance) = adapter
        .generate_concept_from_candidate(request)
        .context("LLM generate_concept_from_candidate failed")?;

    let NormalizedDraft {
        canonical_name,
        aliases,
        category,
        first_pass_body,
    } = normalize_draft(draft, candidate_name);

    // bn-1w5 two-step safety net: if the first-pass definition looks
    // variant-narrowed, re-synthesize with the dedicated body prompt. Any
    // failure here falls back to the first-pass definition so approve
    // never fails for a second-call outage.
    let body_text = synthesize_body(
        adapter,
        &canonical_name,
        &aliases,
        &snippets,
        first_pass_body,
    );

    let slug = slug_from_title(&canonical_name);
    if slug.is_empty() {
        bail!(
            "canonical name '{canonical_name}' slugs to empty string; cannot write concept page"
        );
    }
    let concept_rel = PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));
    let concept_path = root.join(&concept_rel);

    // Refuse to overwrite an existing concept page — that collision should
    // be resolved via `kb review reject` + manual edit, not by this path
    // silently clobbering an existing author's work.
    if concept_path.exists() {
        bail!(
            "concept page already exists at {}; reject this review item and edit manually, \
             or run `kb review reject {}` and open a dedicated merge review instead",
            concept_rel.display(),
            item.metadata.id,
        );
    }

    // Deterministic, sorted, de-duplicated sources (mirrors
    // `render_concept_page` output shape — bn-eqx7 category field included).
    let sorted_sources: Vec<String> = source_document_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<String>>()
        .into_iter()
        .collect();

    let content = render_concept_markdown(
        &slug,
        &canonical_name,
        &aliases,
        category.as_deref(),
        &sorted_sources,
        &body_text,
    )?;

    atomic_write(&concept_path, content.as_bytes())
        .with_context(|| format!("write concept page {}", concept_path.display()))?;

    Ok(ApprovedConceptCandidate {
        concept_path: concept_rel,
        canonical_name,
        aliases,
        category,
        source_document_ids: sorted_sources,
    })
}

/// Normalized primary-draft fields ready to feed into page rendering.
struct NormalizedDraft {
    canonical_name: String,
    aliases: Vec<String>,
    category: Option<String>,
    first_pass_body: String,
}

/// Literal fallback category used when the LLM returns no category (or an
/// empty/whitespace-only string). Matches the existing "Uncategorized"
/// bucket rendered by bn-eqx7's `wiki/concepts/index.md` grouper — the
/// index normalizes category values case-insensitively, so the literal
/// lowercase string lands alongside other pages missing a real category.
const UNCATEGORIZED_FALLBACK: &str = "uncategorized";

/// Trim + sanitize the raw adapter response: pick a non-empty canonical
/// name (fallback to the candidate term), de-dupe aliases, clamp alias
/// count at 5, drop aliases that equal the canonical, normalize the
/// category (empty → literal "uncategorized" fallback so the page still
/// lands in the concept index's Uncategorized bucket rather than being
/// absent from the index), and trim the first-pass body.
fn normalize_draft(
    draft: kb_llm::GenerateConceptFromCandidateResponse,
    candidate_name: String,
) -> NormalizedDraft {
    let canonical_name = {
        let trimmed = draft.canonical_name.trim();
        if trimmed.is_empty() {
            candidate_name
        } else {
            trimmed.to_string()
        }
    };
    let canonical_lower = canonical_name.to_ascii_lowercase();
    let mut alias_set: BTreeSet<String> = BTreeSet::new();
    for a in draft.aliases {
        let trimmed = a.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.to_ascii_lowercase() == canonical_lower {
            continue;
        }
        alias_set.insert(trimmed.to_string());
        if alias_set.len() >= 5 {
            break;
        }
    }
    let aliases: Vec<String> = alias_set.into_iter().collect();

    // bn-39fw: fall back to a literal "uncategorized" when the LLM returns
    // no category. The frontmatter field is always written so bn-eqx7's
    // index-page renderer groups the concept into its existing
    // Uncategorized bucket instead of dropping it from the index.
    let category = Some(
        draft
            .category
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map_or_else(|| UNCATEGORIZED_FALLBACK.to_string(), ToString::to_string),
    );

    let first_pass_body = draft.definition.trim().to_string();

    NormalizedDraft {
        canonical_name,
        aliases,
        category,
        first_pass_body,
    }
}

/// bn-1w5 two-step body synthesis. Returns the refined body when the
/// heuristic flags the first-pass one as narrowed AND the dedicated call
/// succeeds with a non-empty result; otherwise returns the first-pass body.
/// Errors from the body adapter are logged (warn-level) and fall through
/// to the first-pass body so approve never hard-fails on a transient
/// refinement error.
fn synthesize_body<A: LlmAdapter + ?Sized>(
    adapter: &A,
    canonical_name: &str,
    aliases: &[String],
    snippets: &[CandidateSourceSnippet],
    first_pass_body: String,
) -> String {
    if !should_refine_body(&first_pass_body, canonical_name, aliases) {
        return first_pass_body;
    }
    let candidate_quotes: Vec<String> = snippets
        .iter()
        .map(|s| s.snippet.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    match adapter.generate_concept_body(GenerateConceptBodyRequest {
        canonical_name: canonical_name.to_string(),
        aliases: aliases.to_vec(),
        candidate_quotes,
    }) {
        Ok((resp, _prov)) => {
            let b = resp.body.trim().to_string();
            if b.is_empty() { first_pass_body } else { b }
        }
        Err(err) => {
            tracing::warn!(
                concept = %canonical_name,
                error = %err,
                "generate_concept_body refinement failed; keeping first-pass body"
            );
            first_pass_body
        }
    }
}

/// Recover the candidate term from the review item.
///
/// Lint-emitted items have id `lint:concept-candidate:<slug>`; the slug is
/// `slug_from_title(name)` so we can't perfectly recover casing from the id
/// alone. Prefer parsing the candidate term out of the comment (which
/// stores the original surface form literally); fall back to the slug with
/// hyphens replaced by spaces when the comment is unparseable.
fn extract_candidate_name(item: &ReviewItem) -> Result<String> {
    // The comment format from `build_concept_candidate_review_item` starts
    // with "Term '<name>' is mentioned in ..." — pull out the quoted term.
    if let Some(rest) = item.comment.split_once("Term '") {
        if let Some((name, _)) = rest.1.split_once('\'') {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }
    }

    // Fallback: derive from the review id slug. This loses casing
    // information, which is why the comment path is tried first.
    let id = item.target_entity_id.as_str();
    let slug = id.rsplit_once(':').map_or(id, |(_, s)| s).trim();
    if slug.is_empty() {
        bail!("review item '{}' has no parseable candidate name", item.metadata.id);
    }
    Ok(slug.replace('-', " "))
}

/// Read each source's `normalized/<id>/source.md` and pull a short snippet
/// around every mention of the candidate term.
///
/// Case-insensitive matching; leaves the original casing in the snippet so
/// the LLM sees how the term is actually used. Snippets are capped at
/// `MAX_SNIPPET_CHARS` and the total number of snippets is clamped to
/// `MAX_SNIPPETS`.
fn gather_source_snippets(
    root: &Path,
    candidate_name: &str,
    source_document_ids: &[String],
) -> Result<Vec<CandidateSourceSnippet>> {
    let needle_lower = candidate_name.to_ascii_lowercase();
    let mut snippets = Vec::new();

    for src_id in source_document_ids {
        if snippets.len() >= MAX_SNIPPETS {
            break;
        }
        let source_path = root.join("normalized").join(src_id).join("source.md");
        let body = match fs::read_to_string(&source_path) {
            Ok(b) => b,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                tracing::debug!(
                    src_id,
                    path = %source_path.display(),
                    "source.md missing while gathering concept-candidate snippets; skipping"
                );
                continue;
            }
            Err(err) => {
                return Err(anyhow::Error::new(err).context(format!(
                    "read normalized source {}",
                    source_path.display()
                )));
            }
        };

        if let Some(snippet) = extract_snippet_around(&body, &needle_lower) {
            snippets.push(CandidateSourceSnippet {
                source_document_id: src_id.clone(),
                snippet,
            });
        }
    }

    Ok(snippets)
}

/// Find the first case-insensitive occurrence of `needle_lower` in `body`
/// and return a window of roughly `MAX_SNIPPET_CHARS` characters around it
/// with the match in the middle. Returns `None` when the needle doesn't
/// appear.
fn extract_snippet_around(body: &str, needle_lower: &str) -> Option<String> {
    let lower = body.to_ascii_lowercase();
    let match_byte = lower.find(needle_lower)?;

    // Map byte offsets back to char boundaries. ASCII-only docs are fine,
    // multibyte docs need a safe char-aligned start.
    let start_byte = match_byte.saturating_sub(SNIPPET_LOOKBEHIND);
    // Walk back to a char boundary.
    let start = nearest_char_boundary(body, start_byte);
    let end_target = match_byte + needle_lower.len() + (MAX_SNIPPET_CHARS - SNIPPET_LOOKBEHIND);
    let end_byte = end_target.min(body.len());
    let end = nearest_char_boundary(body, end_byte);

    let mut snippet = body[start..end].to_string();
    // Collapse runs of whitespace/newlines so the prompt stays compact.
    snippet = snippet
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if snippet.is_empty() {
        return None;
    }
    // Add ellipses if we trimmed either edge.
    let leading = if start > 0 { "…" } else { "" };
    let trailing = if end < body.len() { "…" } else { "" };
    Some(format!("{leading}{snippet}{trailing}"))
}

/// Round `byte` down to the nearest char boundary in `s`.
fn nearest_char_boundary(s: &str, byte: usize) -> usize {
    if byte >= s.len() {
        return s.len();
    }
    let mut b = byte;
    while b > 0 && !s.is_char_boundary(b) {
        b -= 1;
    }
    b
}

/// Walk `wiki/concepts/` and return every non-empty `category:` value seen
/// in existing concept frontmatter. Used to hint the prompt at the project's
/// emergent category vocabulary so the LLM reuses tags instead of spawning
/// a new one per concept.
fn collect_existing_categories(root: &Path) -> Result<Vec<String>> {
    let concepts_dir = root.join(WIKI_CONCEPTS_DIR);
    if !concepts_dir.exists() {
        return Ok(Vec::new());
    }
    let mut set = BTreeSet::new();
    for entry in fs::read_dir(&concepts_dir)
        .with_context(|| format!("scan concepts dir {}", concepts_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if entry.file_name().to_string_lossy() == "index.md" {
            continue;
        }
        let Ok((fm, _)) = read_frontmatter(&path) else {
            continue;
        };
        if let Some(Value::String(cat)) = fm.get(Value::String("category".into())) {
            let trimmed = cat.trim();
            if !trimmed.is_empty() {
                set.insert(trimmed.to_string());
            }
        }
    }
    Ok(set.into_iter().collect())
}

/// Heuristic: is the first-pass body obviously narrowed or empty enough
/// that the bn-1w5 two-step pass is worth paying for? Triggers on:
/// - empty body,
/// - body that does not start with the canonical name,
/// - body shorter than 40 chars (too brief to be a general definition),
/// - multi-alias concepts whose body mentions fewer than 1 alias (hint of
///   variant-narrowing).
fn should_refine_body(body: &str, canonical: &str, aliases: &[String]) -> bool {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.len() < 40 {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    let canon_lower = canonical.to_ascii_lowercase();
    // Accept a leading article.
    let stripped = ["the ", "a ", "an "]
        .iter()
        .find_map(|a| lower.strip_prefix(a))
        .unwrap_or(&lower);
    if !stripped.starts_with(&canon_lower) {
        return true;
    }
    if aliases.len() >= 3 {
        let mentioned = aliases
            .iter()
            .filter(|a| lower.contains(&a.to_ascii_lowercase()))
            .count();
        if mentioned == 0 {
            return true;
        }
    }
    false
}

/// Render the concept-page markdown. Frontmatter shape matches
/// `render_concept_page` in `concept_merge.rs` so backlink/index scanners
/// handle candidate-approved pages identically to merge-pass pages.
fn render_concept_markdown(
    slug: &str,
    canonical_name: &str,
    aliases: &[String],
    category: Option<&str>,
    source_document_ids: &[String],
    body: &str,
) -> Result<String> {
    let mut fm = Mapping::new();
    fm.insert(
        Value::String("id".into()),
        Value::String(format!("concept:{slug}")),
    );
    fm.insert(
        Value::String("name".into()),
        Value::String(canonical_name.to_string()),
    );
    if !aliases.is_empty() {
        let vals: Vec<Value> = aliases
            .iter()
            .map(|a| Value::String(a.clone()))
            .collect();
        fm.insert(Value::String("aliases".into()), Value::Sequence(vals));
    }
    if let Some(cat) = category {
        fm.insert(
            Value::String("category".into()),
            Value::String(cat.to_string()),
        );
    }
    if !source_document_ids.is_empty() {
        let vals: Vec<Value> = source_document_ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect();
        fm.insert(
            Value::String("source_document_ids".into()),
            Value::Sequence(vals),
        );
    }

    let frontmatter_yaml = serde_yaml::to_string(&fm)
        .context("serialize concept frontmatter")?;

    let mut content = format!("---\n{frontmatter_yaml}---\n\n# {canonical_name}\n");
    let body_trimmed = body.trim();
    if !body_trimmed.is_empty() {
        content.push('\n');
        content.push_str(body_trimmed);
        content.push('\n');
    }
    Ok(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kb_core::{EntityMetadata, ReviewStatus, Status};
    use kb_llm::{GenerateConceptBodyResponse, GenerateConceptFromCandidateResponse, LlmAdapterError, ProvenanceRecord};
    use std::sync::Mutex;
    use tempfile::TempDir;

    struct StubAdapter {
        candidate_response: Mutex<Option<GenerateConceptFromCandidateResponse>>,
        body_response: Mutex<Option<GenerateConceptBodyResponse>>,
        last_request: Mutex<Option<GenerateConceptFromCandidateRequest>>,
        last_body_request: Mutex<Option<GenerateConceptBodyRequest>>,
    }

    impl StubAdapter {
        fn new(response: GenerateConceptFromCandidateResponse) -> Self {
            Self {
                candidate_response: Mutex::new(Some(response)),
                body_response: Mutex::new(None),
                last_request: Mutex::new(None),
                last_body_request: Mutex::new(None),
            }
        }
    }

    fn prov() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "stub".to_string(),
            harness_version: None,
            model: "stub".to_string(),
            prompt_template_name: "stub".to_string(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::Hash::from([0u8; 32]),
            started_at: 0,
            ended_at: 0,
            latency_ms: 0,
            retries: 0,
            tokens: None,
            cost_estimate: None,
        }
    }

    impl LlmAdapter for StubAdapter {
        fn summarize_document(
            &self,
            _request: kb_llm::SummarizeDocumentRequest,
        ) -> Result<(kb_llm::SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("unused".to_string()))
        }
        fn extract_concepts(
            &self,
            _request: kb_llm::ExtractConceptsRequest,
        ) -> Result<(kb_llm::ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("unused".to_string()))
        }
        fn merge_concept_candidates(
            &self,
            _request: kb_llm::MergeConceptCandidatesRequest,
        ) -> Result<(kb_llm::MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError>
        {
            Err(LlmAdapterError::Other("unused".to_string()))
        }
        fn generate_concept_body(
            &self,
            request: GenerateConceptBodyRequest,
        ) -> Result<(GenerateConceptBodyResponse, ProvenanceRecord), LlmAdapterError> {
            self.last_body_request
                .lock()
                .expect("stub mutex poisoned")
                .replace(request);
            let taken = self.body_response.lock().expect("stub mutex poisoned").take();
            taken.map_or_else(
                || {
                    Err(LlmAdapterError::Other(
                        "stub body response not configured".to_string(),
                    ))
                },
                |resp| Ok((resp, prov())),
            )
        }
        fn generate_concept_from_candidate(
            &self,
            request: GenerateConceptFromCandidateRequest,
        ) -> Result<(GenerateConceptFromCandidateResponse, ProvenanceRecord), LlmAdapterError>
        {
            self.last_request
                .lock()
                .expect("stub mutex poisoned")
                .replace(request);
            let taken = self
                .candidate_response
                .lock()
                .expect("stub mutex poisoned")
                .take();
            taken.map_or_else(
                || {
                    Err(LlmAdapterError::Other(
                        "stub candidate response not configured".to_string(),
                    ))
                },
                |resp| Ok((resp, prov())),
            )
        }
        fn answer_question(
            &self,
            _request: kb_llm::AnswerQuestionRequest,
        ) -> Result<(kb_llm::AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("unused".to_string()))
        }
        fn generate_slides(
            &self,
            _request: kb_llm::GenerateSlidesRequest,
        ) -> Result<(kb_llm::GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("unused".to_string()))
        }
        fn run_health_check(
            &self,
            _request: kb_llm::RunHealthCheckRequest,
        ) -> Result<(kb_llm::RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("unused".to_string()))
        }
    }

    fn seed_source(root: &Path, id: &str, body: &str) {
        let dir = root.join("normalized").join(id);
        std::fs::create_dir_all(&dir).expect("mkdir normalized");
        std::fs::write(dir.join("source.md"), body).expect("write source.md");
    }

    fn review_item(name: &str, slug: &str, sources: &[&str]) -> ReviewItem {
        let id = format!("lint:concept-candidate:{slug}");
        let comment = format!(
            "Term '{name}' is mentioned in {} source(s) (10 total mention(s)) but has no concept page. \
             Sources: {}. Approve to generate wiki/concepts/{slug}.md from the mentions.",
            sources.len(),
            sources.join(", "),
        );
        let dest = PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));
        ReviewItem {
            metadata: EntityMetadata {
                id: id.clone(),
                created_at_millis: 0,
                updated_at_millis: 0,
                source_hashes: vec![],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: sources.iter().map(|s| (*s).to_string()).collect(),
                output_paths: vec![dest.clone()],
                status: Status::NeedsReview,
            },
            kind: ReviewKind::ConceptCandidate,
            target_entity_id: id,
            proposed_destination: Some(dest.clone()),
            citations: sources.iter().map(|s| (*s).to_string()).collect(),
            affected_pages: vec![dest],
            created_at_millis: 0,
            status: ReviewStatus::Pending,
            comment,
        }
    }

    #[test]
    fn apply_writes_concept_page_with_frontmatter_and_sources() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        seed_source(
            root,
            "doc-a",
            "# A\n\nThe FooBar System is interesting. FooBar System has many parts.\n",
        );
        seed_source(
            root,
            "doc-b",
            "# B\n\nFooBar System unique aspects. The FooBar System model.\n",
        );

        let stub = StubAdapter::new(GenerateConceptFromCandidateResponse {
            canonical_name: "FooBar System".to_string(),
            aliases: vec!["FooBar".to_string()],
            category: Some("storage".to_string()),
            definition:
                "FooBar System is a distributed architecture described across multiple sources covering its partitioning, consistency, and recovery semantics."
                    .to_string(),
        });

        let item = review_item("FooBar System", "foobar-system", &["doc-a", "doc-b"]);
        let applied = apply_concept_candidate(&stub, root, &item).expect("apply");

        assert_eq!(applied.canonical_name, "FooBar System");
        assert_eq!(applied.aliases, vec!["FooBar".to_string()]);
        assert_eq!(applied.category.as_deref(), Some("storage"));
        assert_eq!(
            applied.source_document_ids,
            vec!["doc-a".to_string(), "doc-b".to_string()]
        );

        let page_path = root.join("wiki/concepts/foobar-system.md");
        assert!(page_path.is_file(), "concept page must exist");
        let content = std::fs::read_to_string(&page_path).expect("read page");
        assert!(content.contains("id: concept:foobar-system"));
        assert!(content.contains("name: FooBar System"));
        assert!(content.contains("- FooBar"));
        assert!(content.contains("category: storage"));
        assert!(content.contains("- doc-a"));
        assert!(content.contains("- doc-b"));
        assert!(content.contains("# FooBar System"));
        assert!(content.contains("FooBar System is a distributed architecture"));

        // The stub received the prompt with the actual candidate term.
        let req = stub
            .last_request
            .lock()
            .expect("mutex poisoned")
            .clone()
            .expect("request");
        assert_eq!(req.candidate_name, "FooBar System");
        assert_eq!(req.source_snippets.len(), 2);
        assert!(
            req.source_snippets
                .iter()
                .all(|s| s.snippet.to_lowercase().contains("foobar system")),
            "each snippet must include the candidate term: {:?}",
            req.source_snippets
        );
    }

    #[test]
    fn apply_falls_back_to_uncategorized_when_llm_returns_no_category() {
        // bn-39fw: the LLM may legitimately return category: None when the
        // snippets are uninformative. The writer must still emit a
        // `category:` field so bn-eqx7's index groups the page into the
        // existing Uncategorized bucket rather than dropping it.
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        seed_source(
            root,
            "doc-a",
            "# A\n\nThe FooBar System is interesting. FooBar System has many parts.\n",
        );

        let stub = StubAdapter::new(GenerateConceptFromCandidateResponse {
            canonical_name: "FooBar System".to_string(),
            aliases: vec![],
            category: None,
            definition:
                "FooBar System is a distributed architecture described across multiple sources covering its partitioning, consistency, and recovery semantics."
                    .to_string(),
        });

        let item = review_item("FooBar System", "foobar-system", &["doc-a"]);
        let applied = apply_concept_candidate(&stub, root, &item).expect("apply");

        assert_eq!(
            applied.category.as_deref(),
            Some("uncategorized"),
            "writer must fall back to literal 'uncategorized' when LLM returns None"
        );

        let page_path = root.join("wiki/concepts/foobar-system.md");
        let content = std::fs::read_to_string(&page_path).expect("read page");
        assert!(
            content.contains("category: uncategorized"),
            "written page must emit `category: uncategorized` frontmatter field:\n{content}"
        );
    }

    #[test]
    fn apply_falls_back_to_uncategorized_when_llm_returns_whitespace_category() {
        // bn-39fw: whitespace-only categories are treated the same as None
        // — the normalizer must not emit them verbatim.
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        seed_source(
            root,
            "doc-a",
            "# A\n\nFooBar System is a thing people talk about. FooBar System!\n",
        );

        let stub = StubAdapter::new(GenerateConceptFromCandidateResponse {
            canonical_name: "FooBar System".to_string(),
            aliases: vec![],
            category: Some("   ".to_string()),
            definition:
                "FooBar System is a distributed architecture described across multiple sources covering its partitioning, consistency, and recovery semantics."
                    .to_string(),
        });

        let item = review_item("FooBar System", "foobar-system", &["doc-a"]);
        let applied = apply_concept_candidate(&stub, root, &item).expect("apply");

        assert_eq!(applied.category.as_deref(), Some("uncategorized"));
        let page_path = root.join("wiki/concepts/foobar-system.md");
        let content = std::fs::read_to_string(&page_path).expect("read page");
        assert!(content.contains("category: uncategorized"));
    }

    #[test]
    fn apply_preserves_llm_assigned_category_verbatim() {
        // bn-39fw: regression guard — the fallback must not clobber a
        // real category returned by the LLM.
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        seed_source(
            root,
            "doc-a",
            "# A\n\nThe FooBar System is interesting. FooBar System has many parts.\n",
        );

        let stub = StubAdapter::new(GenerateConceptFromCandidateResponse {
            canonical_name: "FooBar System".to_string(),
            aliases: vec![],
            category: Some("storage".to_string()),
            definition:
                "FooBar System is a distributed architecture described across multiple sources covering its partitioning, consistency, and recovery semantics."
                    .to_string(),
        });

        let item = review_item("FooBar System", "foobar-system", &["doc-a"]);
        let applied = apply_concept_candidate(&stub, root, &item).expect("apply");

        assert_eq!(applied.category.as_deref(), Some("storage"));
        let page_path = root.join("wiki/concepts/foobar-system.md");
        let content = std::fs::read_to_string(&page_path).expect("read page");
        assert!(content.contains("category: storage"));
        assert!(
            !content.contains("category: uncategorized"),
            "LLM-assigned category must not be clobbered by fallback:\n{content}"
        );
    }

    #[test]
    fn apply_refuses_to_overwrite_existing_concept_page() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        seed_source(root, "doc-a", "FooBar System is interesting");
        let existing = root.join("wiki/concepts/foobar-system.md");
        std::fs::create_dir_all(existing.parent().expect("has parent")).expect("mkdir");
        std::fs::write(&existing, "---\nid: concept:foobar-system\n---\n# existing").expect("write");

        let stub = StubAdapter::new(GenerateConceptFromCandidateResponse {
            canonical_name: "FooBar System".to_string(),
            aliases: vec![],
            category: None,
            definition:
                "FooBar System is a placeholder used to guard against overwriting an author's page."
                    .to_string(),
        });
        let item = review_item("FooBar System", "foobar-system", &["doc-a"]);

        let err = apply_concept_candidate(&stub, root, &item).expect_err("should refuse overwrite");
        assert!(
            err.to_string().contains("already exists"),
            "error must mention collision: {err:#}"
        );
    }

    #[test]
    fn apply_errors_when_no_source_mentions_found() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        // Seed a source that does NOT mention the candidate term.
        seed_source(root, "doc-a", "unrelated content that does not mention it");

        let stub = StubAdapter::new(GenerateConceptFromCandidateResponse {
            canonical_name: "Never Called".to_string(),
            aliases: vec![],
            category: None,
            definition: "should not be used".to_string(),
        });
        let item = review_item("NeverMentioned", "nevermentioned", &["doc-a"]);

        let err = apply_concept_candidate(&stub, root, &item).expect_err("should error");
        assert!(
            err.to_string().contains("no source snippets"),
            "error must describe missing snippets: {err:#}"
        );
    }

    #[test]
    fn should_refine_body_triggers_on_empty() {
        assert!(should_refine_body("", "Foo", &[]));
    }

    #[test]
    fn should_refine_body_triggers_on_missing_canonical_prefix() {
        assert!(should_refine_body(
            "Bar is a thing unrelated to the subject matter.",
            "Foo",
            &[]
        ));
    }

    #[test]
    fn should_refine_body_passes_reasonable_body() {
        assert!(!should_refine_body(
            "Foo is a distributed mechanism that coordinates workers across shards and tolerates failures.",
            "Foo",
            &[]
        ));
    }

    #[test]
    fn extract_candidate_name_reads_comment_quoted_term() {
        let item = review_item("Multi-Word Name", "multi-word-name", &["doc-a"]);
        assert_eq!(
            extract_candidate_name(&item).expect("candidate name"),
            "Multi-Word Name"
        );
    }

    #[test]
    fn collect_existing_categories_dedupes_and_sorts() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let concepts = root.join(WIKI_CONCEPTS_DIR);
        std::fs::create_dir_all(&concepts).expect("mkdir");
        std::fs::write(
            concepts.join("a.md"),
            "---\nid: concept:a\nname: A\ncategory: storage\n---\n# A\n",
        )
        .expect("write a.md");
        std::fs::write(
            concepts.join("b.md"),
            "---\nid: concept:b\nname: B\ncategory: async\n---\n# B\n",
        )
        .expect("write b.md");
        std::fs::write(
            concepts.join("c.md"),
            "---\nid: concept:c\nname: C\ncategory: storage\n---\n# C\n",
        )
        .expect("write c.md");
        // Index files are skipped.
        std::fs::write(concepts.join("index.md"), "---\n---\n# index\n")
            .expect("write index.md");

        let cats = collect_existing_categories(root).expect("collect categories");
        assert_eq!(cats, vec!["async".to_string(), "storage".to_string()]);
    }
}
