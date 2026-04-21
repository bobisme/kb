//! Cross-source contradiction detection (bn-3axp).
//!
//! For each concept page, gather `sources[].quote` entries, bucket them by
//! source document, and — when quotes span ≥ `min_sources` distinct sources
//! — hand them to the LLM with a strict JSON prompt asking whether the
//! claims contradict each other. Flagged concepts are emitted as
//! `ReviewKind::Contradiction` items under `reviews/contradictions/`.
//!
//! This pass is expensive (one LLM call per multi-source concept), so it
//! is OFF by default and must be invoked explicitly via
//! `kb lint --check contradictions`.
//!
//! Acknowledgement persistence: each review item's `source_hashes` contains
//! a fingerprint of (`concept_id`, sorted-quote-texts). Before making any LLM
//! call, the pass loads any existing contradiction review items — if the
//! same fingerprint is already **Approved** (acknowledged) or **Rejected**
//! (declared intended nuance), the concept is skipped. Unchanged inputs
//! never re-fire the LLM or the review item.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kb_core::frontmatter::read_frontmatter;
use kb_core::{
    EntityMetadata, ReviewItem, ReviewKind, ReviewStatus, Status, hash_many, list_review_items,
    normalized_dir, slug_from_title,
};
use kb_llm::{ContradictionQuote, DetectContradictionsRequest, LlmAdapter};
use serde_yaml::Value;

const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";
/// Concepts with more than this many cited quotes are truncated to the first
/// N quotes per source before the LLM call. The cost model is one LLM call
/// per concept; a single huge concept won't dominate, but the prompt itself
/// must still fit in context.
const MAX_QUOTES_PER_CONCEPT: usize = 20;
/// Max length of any individual quote passed to the LLM. Quotes are trimmed
/// (not hard-cut) so sentence fragments stay readable. Keeps prompt size
/// bounded on pathological pages that drop whole paragraphs into `quote:`.
const MAX_QUOTE_CHARS: usize = 600;

/// Configuration for the `contradictions` lint check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContradictionsConfig {
    /// Whether the check runs at all. **OFF by default** — this is an
    /// O(concepts) LLM pass. Toggle to `true` in `kb.toml` only if you
    /// want `--check contradictions` to honor the default when callers
    /// don't pass a value.
    pub enabled: bool,
    /// A concept is checked only when its quotes span at least this many
    /// distinct source documents. Default: `2`.
    pub min_sources: usize,
}

impl Default for ContradictionsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_sources: 2,
        }
    }
}

/// Scan concept pages, group cited quotes by source document, and ask the
/// LLM whether the quotes make contradictory claims. Returns one
/// `ReviewItem` per detected contradiction.
///
/// Concepts with an already-approved or already-rejected review item that
/// fingerprints the same concept+quote-set are skipped entirely — no LLM
/// call is issued. This is the "acknowledgement persists across runs"
/// guarantee from the bone spec.
///
/// # Errors
///
/// Returns an error when concept pages cannot be scanned, the normalized
/// directory's heading metadata cannot be parsed, or the system clock
/// cannot be queried. Per-concept LLM failures are logged and the concept
/// is skipped — one flaky call must not poison the whole pass.
pub fn check_contradictions<A: LlmAdapter + ?Sized>(
    root: &Path,
    adapter: &A,
    config: &ContradictionsConfig,
) -> Result<Vec<ReviewItem>> {
    if config.min_sources < 2 {
        tracing::warn!(
            "contradictions check requires min_sources >= 2; got {} — forcing 2",
            config.min_sources
        );
    }
    let effective_min_sources = config.min_sources.max(2);

    let concepts_dir = root.join(WIKI_CONCEPTS_DIR);
    if !concepts_dir.exists() {
        return Ok(Vec::new());
    }

    let anchor_to_source_docs = build_anchor_to_source_docs(root)?;
    let acked_fingerprints = load_acked_or_rejected_fingerprints(root)?;

    let mut items = Vec::new();
    for entry in fs::read_dir(&concepts_dir)
        .with_context(|| format!("scan concept pages dir {}", concepts_dir.display()))?
    {
        let path = entry
            .with_context(|| format!("read entry in {}", concepts_dir.display()))?
            .path();
        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        if path.file_name().is_some_and(|n| n == "index.md") {
            continue;
        }

        let Ok(concept) = parse_concept_for_contradictions(&path, &anchor_to_source_docs) else {
            continue;
        };
        let Some(concept) = concept else { continue };

        if distinct_sources(&concept.quotes) < effective_min_sources {
            continue;
        }

        let fingerprint = fingerprint(&concept.concept_id, &concept.quotes);
        if acked_fingerprints.contains(&fingerprint) {
            tracing::debug!(
                "skipping concept '{}': same quote set already acknowledged",
                concept.concept_id
            );
            continue;
        }

        let request = DetectContradictionsRequest {
            concept_name: concept.name.clone(),
            quotes: concept.quotes.clone(),
        };
        let response = match adapter.detect_contradictions(request) {
            Ok((response, _provenance)) => response,
            Err(err) => {
                tracing::warn!(
                    "contradictions: LLM call failed for concept '{}': {err}",
                    concept.concept_id
                );
                continue;
            }
        };

        if !response.contradiction {
            continue;
        }

        let Some(item) = build_contradiction_review_item(&concept, &response, fingerprint) else {
            continue;
        };
        items.push(item);
    }

    Ok(items)
}

/// Surface the contradictions pass as `LintIssue`s for `kb lint` output.
///
/// Runs the same underlying LLM pass as [`check_contradictions`], then maps
/// each detected contradiction to a single `Warning` issue anchored at the
/// concept page. Callers typically persist the review items in addition to
/// printing the warnings — the review queue is the durable surface, the
/// lint output is the operator hint.
///
/// # Errors
///
/// Propagates errors from [`check_contradictions`].
pub fn detect_contradictions_issues<A: LlmAdapter + ?Sized>(
    root: &Path,
    adapter: &A,
    config: &ContradictionsConfig,
) -> Result<Vec<crate::LintIssue>> {
    let items = check_contradictions(root, adapter, config)?;
    Ok(items
        .into_iter()
        .map(|item| crate::LintIssue {
            severity: crate::IssueSeverity::Warning,
            kind: crate::IssueKind::Contradiction,
            referring_page: item
                .affected_pages
                .first()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            line: 0,
            target: item.target_entity_id.clone(),
            message: item.comment.clone(),
            suggested_fix: Some(format!(
                "run 'kb review show {}' to inspect, then approve (acknowledge) or reject (mark intended nuance)",
                item.metadata.id
            )),
        })
        .collect())
}

/// One concept's worth of extracted input for the contradictions pass.
struct ConceptWithQuotes {
    concept_id: String,
    name: String,
    /// Relative path (from `root`) to the concept page — used for the
    /// review item's `affected_pages`.
    page_path: PathBuf,
    /// Source-attributed quotes in a stable order (grouped by source, then
    /// by original position). Only the first `MAX_QUOTES_PER_CONCEPT` are
    /// retained to keep prompt size bounded.
    quotes: Vec<ContradictionQuote>,
}

fn parse_concept_for_contradictions(
    path: &Path,
    anchor_to_source_docs: &BTreeMap<String, Vec<String>>,
) -> Result<Option<ConceptWithQuotes>> {
    let (fm, _body) =
        read_frontmatter(path).with_context(|| format!("read frontmatter {}", path.display()))?;

    let concept_id = fm
        .get(Value::String("id".into()))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if concept_id.is_empty() {
        return Ok(None);
    }

    let name = fm
        .get(Value::String("name".into()))
        .and_then(Value::as_str)
        .unwrap_or_else(|| concept_id.trim_start_matches("concept:"))
        .to_string();

    // Authoritative list of contributing sources. When a quote has no
    // `heading_anchor` — or the anchor maps to nothing — we fall back to
    // the single entry here (if there is exactly one).
    let concept_source_docs: Vec<String> = fm
        .get(Value::String("source_document_ids".into()))
        .and_then(Value::as_sequence)
        .map(|seq| {
            seq.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let sources_seq = fm
        .get(Value::String("sources".into()))
        .and_then(Value::as_sequence);
    let Some(sources) = sources_seq else {
        return Ok(None);
    };

    // Bucket by source doc id so `min_sources` is a bucket count, not a
    // quote count. Within each source we preserve original order.
    let mut by_source: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in sources {
        let Some(map) = entry.as_mapping() else {
            continue;
        };
        let quote_text = map
            .get(Value::String("quote".into()))
            .and_then(Value::as_str)
            .map(trim_quote_for_prompt)
            .unwrap_or_default();
        if quote_text.is_empty() {
            continue;
        }

        let anchor = map
            .get(Value::String("heading_anchor".into()))
            .and_then(Value::as_str);

        let source_id = resolve_source_for_quote(anchor, anchor_to_source_docs, &concept_source_docs);
        let Some(source_id) = source_id else {
            // Unattributable quote — skip rather than lump it under a
            // synthetic bucket, which would inflate `distinct_sources`.
            continue;
        };

        by_source
            .entry(source_id)
            .or_default()
            .push(quote_text);
    }

    // Flatten to a single Vec<ContradictionQuote> with deterministic order:
    // sources alphabetical, quotes in original order. Cap total at
    // MAX_QUOTES_PER_CONCEPT by round-robining across sources so every
    // source gets representation even on pathological pages.
    let quotes = flatten_with_cap(&by_source, MAX_QUOTES_PER_CONCEPT);

    let rel_path = path
        .strip_prefix(
            path.ancestors()
                .find(|p| p.file_name().is_some_and(|n| n == "concepts"))
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .unwrap_or(path),
        )
        .unwrap_or(path)
        .to_path_buf();

    Ok(Some(ConceptWithQuotes {
        concept_id,
        name,
        page_path: rel_path,
        quotes,
    }))
}

/// Pick the `source_document_id` that a given quote came from.
///
/// Priority:
/// 1. If the `heading_anchor` resolves to exactly one source in
///    `anchor_to_source_docs` and that source is in the concept's
///    `source_document_ids`, use it.
/// 2. Else if the concept has exactly one contributing source document,
///    attribute to that.
/// 3. Else bail — an ambiguous anchor on a multi-source concept is worse
///    than silently dropping the quote: we'd rather check fewer
///    contradictions than invent source attribution.
fn resolve_source_for_quote(
    anchor: Option<&str>,
    anchor_to_source_docs: &BTreeMap<String, Vec<String>>,
    concept_source_docs: &[String],
) -> Option<String> {
    if let Some(anchor) = anchor {
        if let Some(candidates) = anchor_to_source_docs.get(anchor) {
            // Intersect with the concept's declared sources to avoid a
            // shared anchor name (e.g. `## Overview`) on an unrelated
            // source hijacking the attribution.
            let concept_set: BTreeSet<&str> =
                concept_source_docs.iter().map(String::as_str).collect();
            let mut filtered: Vec<&String> = if concept_set.is_empty() {
                candidates.iter().collect()
            } else {
                candidates
                    .iter()
                    .filter(|c| concept_set.contains(c.as_str()))
                    .collect()
            };
            filtered.sort();
            filtered.dedup();
            if filtered.len() == 1 {
                return Some(filtered[0].clone());
            }
        }
    }
    // Fallback: concept-unique source. Explicitly requires exactly one so
    // we don't misattribute.
    if concept_source_docs.len() == 1 {
        return Some(concept_source_docs[0].clone());
    }
    None
}

/// Flatten the source -> quotes map into a flat list, capping the total at
/// `cap`. We round-robin across sources so a single source with many
/// quotes doesn't crowd out representation from the others.
fn flatten_with_cap(
    by_source: &BTreeMap<String, Vec<String>>,
    cap: usize,
) -> Vec<ContradictionQuote> {
    let mut cursors: Vec<(String, &Vec<String>, usize)> = by_source
        .iter()
        .map(|(k, v)| (k.clone(), v, 0usize))
        .collect();

    let mut out = Vec::new();
    while out.len() < cap {
        let mut made_progress = false;
        for (label, quotes, cursor) in &mut cursors {
            if *cursor < quotes.len() {
                out.push(ContradictionQuote {
                    source_label: label.clone(),
                    text: quotes[*cursor].clone(),
                });
                *cursor += 1;
                made_progress = true;
                if out.len() >= cap {
                    break;
                }
            }
        }
        if !made_progress {
            break;
        }
    }
    out
}

fn distinct_sources(quotes: &[ContradictionQuote]) -> usize {
    quotes
        .iter()
        .map(|q| q.source_label.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

fn trim_quote_for_prompt(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.chars().count() <= MAX_QUOTE_CHARS {
        return trimmed.to_string();
    }
    // char-boundary-safe truncation + ellipsis marker.
    let cut: String = trimmed.chars().take(MAX_QUOTE_CHARS).collect();
    format!("{cut}…")
}

/// Compute a stable fingerprint of (`concept_id`, sorted-quote-texts). This
/// is the acknowledgement key: the same concept + same quote set across
/// multiple runs yields the same fingerprint, so a previously approved or
/// rejected review item suppresses re-firing the LLM call.
///
/// Source labels are intentionally excluded so cosmetic source-id changes
/// (e.g. renaming a source document's slug) don't silently re-fire
/// contradictions the user already triaged. The quote *texts* are what
/// decide whether the substantive claim content is unchanged.
fn fingerprint(concept_id: &str, quotes: &[ContradictionQuote]) -> String {
    let mut sorted_texts: Vec<&[u8]> = quotes.iter().map(|q| q.text.as_bytes()).collect();
    sorted_texts.sort_unstable();
    sorted_texts.dedup();

    let mut parts: Vec<&[u8]> = Vec::with_capacity(sorted_texts.len() + 2);
    parts.push(concept_id.as_bytes());
    parts.push(b"\x00"); // separator so "concept:x" + "y" != "concept:" + "xy"
    parts.extend(sorted_texts);

    hash_many(&parts).to_hex()
}

/// Load fingerprints from every existing contradiction review item whose
/// status is Approved or Rejected. Used to skip re-checking concepts the
/// user has already triaged with unchanged inputs.
fn load_acked_or_rejected_fingerprints(root: &Path) -> Result<BTreeSet<String>> {
    let mut set = BTreeSet::new();
    let items = list_review_items(root)?;
    for item in items {
        if item.kind != ReviewKind::Contradiction {
            continue;
        }
        if item.status == ReviewStatus::Pending {
            continue;
        }
        for hash in &item.metadata.source_hashes {
            set.insert(hash.clone());
        }
    }
    Ok(set)
}

fn build_contradiction_review_item(
    concept: &ConceptWithQuotes,
    response: &kb_llm::DetectContradictionsResponse,
    fingerprint: String,
) -> Option<ReviewItem> {
    if response.conflicting_quotes.is_empty() {
        // The LLM said `contradiction: true` but named zero indices. That's
        // a malformed response per the prompt contract; skip rather than
        // emit a useless item.
        tracing::warn!(
            "contradictions: LLM returned contradiction=true with empty conflicting_quotes for '{}'; skipping",
            concept.concept_id
        );
        return None;
    }

    let now = unix_time_ms().ok()?;
    let slug = slug_from_title(
        concept
            .concept_id
            .strip_prefix("concept:")
            .unwrap_or(&concept.concept_id),
    );
    let id = format!("lint:contradiction:{slug}");

    let citations = response
        .conflicting_quotes
        .iter()
        .filter_map(|&idx| concept.quotes.get(idx))
        .map(|q| format!("{}: {}", q.source_label, truncate_for_citation(&q.text)))
        .collect::<Vec<_>>();

    let source_ids: Vec<String> = concept
        .quotes
        .iter()
        .map(|q| q.source_label.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let comment = format!(
        "Contradiction detected across {} source(s) for '{}': {}",
        source_ids.len(),
        concept.name,
        response.explanation.trim(),
    );

    Some(ReviewItem {
        metadata: EntityMetadata {
            id,
            created_at_millis: now,
            updated_at_millis: now,
            // The fingerprint is the whole `source_hashes` vec (not
            // prepended to anything else) so
            // `load_acked_or_rejected_fingerprints` can match on set
            // containment directly.
            source_hashes: vec![fingerprint],
            model_version: None,
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: None,
            dependencies: source_ids,
            output_paths: vec![concept.page_path.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::Contradiction,
        target_entity_id: concept.concept_id.clone(),
        proposed_destination: Some(concept.page_path.clone()),
        citations,
        affected_pages: vec![concept.page_path.clone()],
        created_at_millis: now,
        status: ReviewStatus::Pending,
        comment,
    })
}

fn truncate_for_citation(text: &str) -> String {
    const MAX: usize = 140;
    if text.chars().count() <= MAX {
        return text.to_string();
    }
    let cut: String = text.chars().take(MAX).collect();
    format!("{cut}…")
}

fn unix_time_ms() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before Unix epoch")?;
    duration
        .as_millis()
        .try_into()
        .context("system time exceeds u64 millisecond range")
}

/// Build `heading_anchor -> [source_document_id, ...]` by reading each
/// `normalized/<id>/metadata.json`'s `heading_ids` array. Mirrors the
/// helper in `kb-compile::backlinks`, kept here because the dependency
/// direction is `kb-compile -> kb-lint` would cycle.
fn build_anchor_to_source_docs(root: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let normalized_root = normalized_dir(root);
    if !normalized_root.exists() {
        return Ok(map);
    }

    for entry in fs::read_dir(&normalized_root)
        .with_context(|| format!("scan normalized dir {}", normalized_root.display()))?
    {
        let entry = entry.with_context(|| format!("walk {}", normalized_root.display()))?;
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let source_doc_id = entry.file_name().to_string_lossy().into_owned();
        let metadata_path = entry.path().join("metadata.json");
        if !metadata_path.is_file() {
            continue;
        }
        let raw = fs::read_to_string(&metadata_path)
            .with_context(|| format!("read {}", metadata_path.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", metadata_path.display()))?;
        let Some(headings) = parsed.get("heading_ids").and_then(|v| v.as_array()) else {
            continue;
        };
        for heading in headings {
            if let Some(h) = heading.as_str() {
                map.entry(h.to_string())
                    .or_default()
                    .push(source_doc_id.clone());
            }
        }
    }

    for docs in map.values_mut() {
        docs.sort();
        docs.dedup();
    }
    Ok(map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use kb_llm::{
        AnswerQuestionRequest, AnswerQuestionResponse, DetectContradictionsResponse,
        ExtractConceptsRequest, ExtractConceptsResponse, GenerateSlidesRequest,
        GenerateSlidesResponse, LlmAdapterError, MergeConceptCandidatesRequest,
        MergeConceptCandidatesResponse, ProvenanceRecord, RunHealthCheckRequest,
        RunHealthCheckResponse, SummarizeDocumentRequest, SummarizeDocumentResponse,
    };
    use std::sync::Mutex;
    use tempfile::TempDir;

    struct StubAdapter {
        detect_response: Mutex<Option<DetectContradictionsResponse>>,
        calls: Mutex<Vec<DetectContradictionsRequest>>,
        fail: bool,
    }

    impl StubAdapter {
        fn ok(response: DetectContradictionsResponse) -> Self {
            Self {
                detect_response: Mutex::new(Some(response)),
                calls: Mutex::new(Vec::new()),
                fail: false,
            }
        }

        fn erroring() -> Self {
            Self {
                detect_response: Mutex::new(None),
                calls: Mutex::new(Vec::new()),
                fail: true,
            }
        }
    }

    fn fresh_provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "stub".to_string(),
            harness_version: None,
            model: "stub".to_string(),
            prompt_template_name: "detect_contradictions.md".to_string(),
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
            _r: SummarizeDocumentRequest,
        ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("not used".into()))
        }

        fn extract_concepts(
            &self,
            _r: ExtractConceptsRequest,
        ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("not used".into()))
        }

        fn merge_concept_candidates(
            &self,
            _r: MergeConceptCandidatesRequest,
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError>
        {
            Err(LlmAdapterError::Other("not used".into()))
        }

        fn answer_question(
            &self,
            _r: AnswerQuestionRequest,
        ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("not used".into()))
        }

        fn generate_slides(
            &self,
            _r: GenerateSlidesRequest,
        ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("not used".into()))
        }

        fn run_health_check(
            &self,
            _r: RunHealthCheckRequest,
        ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("not used".into()))
        }

        fn detect_contradictions(
            &self,
            request: DetectContradictionsRequest,
        ) -> Result<(DetectContradictionsResponse, ProvenanceRecord), LlmAdapterError> {
            self.calls.lock().unwrap().push(request);
            if self.fail {
                return Err(LlmAdapterError::Other("simulated failure".into()));
            }
            let response = self
                .detect_response
                .lock()
                .unwrap()
                .clone()
                .expect("response configured");
            Ok((response, fresh_provenance()))
        }
    }

    fn write_concept_page(
        root: &Path,
        slug: &str,
        name: &str,
        source_document_ids: &[&str],
        quotes: &[(&str, &str)], // (heading_anchor, quote)
    ) {
        use std::fmt::Write as _;

        let dir = root.join(WIKI_CONCEPTS_DIR);
        fs::create_dir_all(&dir).unwrap();

        let mut fm = String::new();
        writeln!(fm, "id: concept:{slug}").unwrap();
        writeln!(fm, "name: {name}").unwrap();

        if !source_document_ids.is_empty() {
            fm.push_str("source_document_ids:\n");
            for id in source_document_ids {
                writeln!(fm, "- {id}").unwrap();
            }
        }

        fm.push_str("sources:\n");
        for (anchor, quote) in quotes {
            if anchor.is_empty() {
                writeln!(fm, "- quote: \"{}\"", quote.replace('"', "\\\"")).unwrap();
            } else {
                writeln!(
                    fm,
                    "- heading_anchor: {anchor}\n  quote: \"{}\"",
                    quote.replace('"', "\\\"")
                )
                .unwrap();
            }
        }

        let content = format!("---\n{fm}---\n\n# {name}\n");
        fs::write(dir.join(format!("{slug}.md")), content).unwrap();
    }

    fn write_normalized_metadata(root: &Path, src_id: &str, heading_ids: &[&str]) {
        let dir = normalized_dir(root).join(src_id);
        fs::create_dir_all(&dir).unwrap();
        let value = serde_json::json!({ "heading_ids": heading_ids });
        fs::write(dir.join("metadata.json"), value.to_string()).unwrap();
    }

    #[test]
    fn emits_review_item_when_llm_reports_contradiction() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["claim-a"]);
        write_normalized_metadata(root, "src-b", &["claim-b"]);
        write_concept_page(
            root,
            "widget",
            "Widget",
            &["src-a", "src-b"],
            &[
                ("claim-a", "Widgets are blue."),
                ("claim-b", "Widgets are always red."),
            ],
        );

        let adapter = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "Quotes disagree on widget color.".to_string(),
            conflicting_quotes: vec![0, 1],
        });

        let items = check_contradictions(root, &adapter, &ContradictionsConfig::default()).unwrap();
        assert_eq!(items.len(), 1);
        let item = &items[0];
        assert_eq!(item.kind, ReviewKind::Contradiction);
        assert_eq!(item.target_entity_id, "concept:widget");
        assert!(item.comment.contains("Widget"));
        assert!(item.comment.contains("widget color"));
        assert_eq!(item.citations.len(), 2);
        assert!(item.citations.iter().any(|c| c.contains("src-a")));
        assert!(item.citations.iter().any(|c| c.contains("src-b")));
        assert_eq!(adapter.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn skips_concepts_with_only_one_source() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["claim-a", "claim-b"]);
        write_concept_page(
            root,
            "mono",
            "Mono",
            &["src-a"],
            &[
                ("claim-a", "Mono is fast."),
                ("claim-b", "Mono is slow."),
            ],
        );

        let adapter = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "should not fire".into(),
            conflicting_quotes: vec![0, 1],
        });

        let items = check_contradictions(root, &adapter, &ContradictionsConfig::default()).unwrap();
        assert!(items.is_empty());
        // LLM must not have been called for a single-source concept.
        assert_eq!(adapter.calls.lock().unwrap().len(), 0);
    }

    #[test]
    fn skips_when_llm_says_no_contradiction() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["a"]);
        write_normalized_metadata(root, "src-b", &["b"]);
        write_concept_page(
            root,
            "clean",
            "Clean",
            &["src-a", "src-b"],
            &[("a", "Claim one"), ("b", "Claim two")],
        );

        let adapter = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: false,
            explanation: "no disagreement".into(),
            conflicting_quotes: vec![],
        });

        let items = check_contradictions(root, &adapter, &ContradictionsConfig::default()).unwrap();
        assert!(items.is_empty());
        assert_eq!(adapter.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn acknowledgement_suppresses_re_firing() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["a"]);
        write_normalized_metadata(root, "src-b", &["b"]);
        write_concept_page(
            root,
            "acked",
            "Acked",
            &["src-a", "src-b"],
            &[("a", "X is big."), ("b", "X is small.")],
        );

        // First run — flag it.
        let adapter_first = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "sizes".into(),
            conflicting_quotes: vec![0, 1],
        });
        let items =
            check_contradictions(root, &adapter_first, &ContradictionsConfig::default()).unwrap();
        assert_eq!(items.len(), 1);
        // Persist the item as approved (simulating `kb review approve`).
        let mut approved = items[0].clone();
        approved.status = ReviewStatus::Approved;
        kb_core::save_review_item(root, &approved).unwrap();

        // Second run — same concept + quotes — must skip.
        let adapter_second = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "should not fire".into(),
            conflicting_quotes: vec![0, 1],
        });
        let items2 =
            check_contradictions(root, &adapter_second, &ContradictionsConfig::default()).unwrap();
        assert!(items2.is_empty());
        assert_eq!(
            adapter_second.calls.lock().unwrap().len(),
            0,
            "LLM should not be called for already-acknowledged concepts"
        );
    }

    #[test]
    fn rejection_also_suppresses_re_firing() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["a"]);
        write_normalized_metadata(root, "src-b", &["b"]);
        write_concept_page(
            root,
            "nuance",
            "Nuance",
            &["src-a", "src-b"],
            &[("a", "Big."), ("b", "Small.")],
        );

        let adapter_first = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "disagreement".into(),
            conflicting_quotes: vec![0, 1],
        });
        let items =
            check_contradictions(root, &adapter_first, &ContradictionsConfig::default()).unwrap();
        assert_eq!(items.len(), 1);
        let mut rejected = items[0].clone();
        rejected.status = ReviewStatus::Rejected;
        kb_core::save_review_item(root, &rejected).unwrap();

        let adapter_second = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "should not fire".into(),
            conflicting_quotes: vec![0, 1],
        });
        let items2 =
            check_contradictions(root, &adapter_second, &ContradictionsConfig::default()).unwrap();
        assert!(items2.is_empty());
        assert_eq!(adapter_second.calls.lock().unwrap().len(), 0);
    }

    #[test]
    fn llm_failure_is_logged_and_skipped() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["a"]);
        write_normalized_metadata(root, "src-b", &["b"]);
        write_concept_page(
            root,
            "flaky",
            "Flaky",
            &["src-a", "src-b"],
            &[("a", "one"), ("b", "two")],
        );

        let adapter = StubAdapter::erroring();
        let items = check_contradictions(root, &adapter, &ContradictionsConfig::default()).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn unattributable_quotes_are_dropped_from_bucket_count() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        // Two sources but only one has a resolvable anchor; the other
        // quote has an unknown anchor and the concept lists both sources
        // → we cannot disambiguate, so that quote is dropped. Result:
        // distinct sources drops to 1 and the check is skipped.
        write_normalized_metadata(root, "src-a", &["known-anchor"]);
        write_normalized_metadata(root, "src-b", &["unrelated"]);
        write_concept_page(
            root,
            "amb",
            "Amb",
            &["src-a", "src-b"],
            &[
                ("known-anchor", "Attributed to src-a."),
                ("mystery-anchor", "Unattributable."),
            ],
        );

        let adapter = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "should not fire".into(),
            conflicting_quotes: vec![0, 1],
        });
        let items = check_contradictions(root, &adapter, &ContradictionsConfig::default()).unwrap();
        assert!(items.is_empty());
        assert_eq!(adapter.calls.lock().unwrap().len(), 0);
    }

    #[test]
    fn empty_conflicting_quotes_response_is_rejected() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        write_normalized_metadata(root, "src-a", &["a"]);
        write_normalized_metadata(root, "src-b", &["b"]);
        write_concept_page(
            root,
            "bad",
            "Bad",
            &["src-a", "src-b"],
            &[("a", "one"), ("b", "two")],
        );

        let adapter = StubAdapter::ok(DetectContradictionsResponse {
            contradiction: true,
            explanation: "malformed".into(),
            conflicting_quotes: vec![],
        });
        let items = check_contradictions(root, &adapter, &ContradictionsConfig::default()).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn fingerprint_is_stable_for_reordered_quotes() {
        let fp1 = fingerprint(
            "concept:x",
            &[
                ContradictionQuote {
                    source_label: "src-a".into(),
                    text: "A".into(),
                },
                ContradictionQuote {
                    source_label: "src-b".into(),
                    text: "B".into(),
                },
            ],
        );
        let fp2 = fingerprint(
            "concept:x",
            &[
                ContradictionQuote {
                    source_label: "src-b".into(),
                    text: "B".into(),
                },
                ContradictionQuote {
                    source_label: "src-a".into(),
                    text: "A".into(),
                },
            ],
        );
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_changes_when_quote_text_changes() {
        let fp1 = fingerprint(
            "concept:x",
            &[ContradictionQuote {
                source_label: "src-a".into(),
                text: "old text".into(),
            }],
        );
        let fp2 = fingerprint(
            "concept:x",
            &[ContradictionQuote {
                source_label: "src-a".into(),
                text: "new text".into(),
            }],
        );
        assert_ne!(fp1, fp2);
    }
}
