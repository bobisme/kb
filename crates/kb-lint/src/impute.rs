//! Web-search-backed gap backfill (bn-xt4o).
//!
//! Called when the user runs `kb lint --impute`. Walks a set of lint
//! findings, calls a web-search-capable LLM agent for each one, and
//! persists the drafts as `ReviewKind::ImputedFix` review items.
//!
//! Supported gap kinds in v1:
//!
//! - **`missing_concept`** — term flagged by `missing_concepts` lint
//!   (`IssueKind::ConceptCandidate`) that has no concept page. Impute
//!   drafts a new page from web sources; approve writes
//!   `wiki/concepts/<slug>.md`.
//!
//! - **`thin_concept_body`** — existing concept page whose body has fewer
//!   than N words (see [`crate::ThinConceptBodyConfig`]). Impute drafts a
//!   replacement body; approve rewrites the page body in place.
//!
//! The user must explicitly opt in via `--impute`. The pass never runs as
//! part of the default lint sweep — it's gated on network access,
//! expensive, and the user needs to review every draft before it lands.
//!
//! Contradictions and broken-links are intentionally **not** imputed in
//! v1: the former is already a human-judgement pass (acknowledge vs.
//! nuance), and the latter is a refactor task that wouldn't benefit from
//! web search.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kb_core::{
    EntityMetadata, ReviewItem, ReviewKind, ReviewStatus, Status, hash_many, list_review_items,
    normalized_dir, slug_from_title,
};
use kb_llm::{
    CandidateSourceSnippet, ImputeGapKind, ImputeGapRequest, ImputeGapResponse, ImputedWebSource,
    LlmAdapter,
};
use serde::{Deserialize, Serialize};

use crate::{
    ConceptCandidateHit, IssueKind, IssueSeverity, LintIssue, MissingConceptsConfig, ThinConceptHit,
    detect_thin_concept_bodies,
};

const MAX_SNIPPET_CHARS: usize = 280;
const SNIPPET_LOOKBEHIND: usize = 80;
const MAX_SNIPPETS_PER_REQUEST: usize = 6;

/// Configuration for the impute pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImputeConfig {
    /// Cap the total number of LLM calls per `kb lint --impute` invocation.
    /// Each call hits web search + the model and is billable; the cap keeps
    /// an over-eager run from burning the user's quota on the first pass.
    /// Default: `10`.
    pub max_imputations: usize,
    /// Which gap kinds to impute. Default: both missing concepts and thin
    /// concept bodies.
    pub kinds: ImputeKindSelector,
    /// Settings for the missing-concepts filter (min sources / mentions).
    /// Reused from the regular lint pass so both checks agree on what
    /// counts as a candidate.
    pub missing_concepts: MissingConceptsConfig,
    /// Settings for the thin-concept-body discovery pass.
    pub thin_concept_body: crate::ThinConceptBodyConfig,
}

impl Default for ImputeConfig {
    fn default() -> Self {
        Self {
            max_imputations: 10,
            kinds: ImputeKindSelector::default(),
            missing_concepts: MissingConceptsConfig::default(),
            thin_concept_body: crate::ThinConceptBodyConfig::default(),
        }
    }
}

/// Flags for which gap kinds the impute pass should backfill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImputeKindSelector {
    /// Impute missing concepts (no page yet). Default: `true`.
    pub missing_concepts: bool,
    /// Impute thin concept bodies (page exists but body is too short).
    /// Default: `true`.
    pub thin_concept_body: bool,
}

impl Default for ImputeKindSelector {
    fn default() -> Self {
        Self {
            missing_concepts: true,
            thin_concept_body: true,
        }
    }
}

/// A single imputation outcome.
///
/// Either a new review item + payload to persist, or a skip reason.
/// Callers typically collect all outcomes, persist the items that
/// succeeded (item + sidecar), and report the skips to the user.
///
/// The `Item` variant is boxed so the two variants have similar sizes —
/// clippy flags the plain shape as a large-variant mismatch because
/// `ReviewItem` + `ImputedFixPayload` together dwarf the `Skipped` record.
#[derive(Debug)]
pub enum ImputeOutcome {
    /// LLM drafted a fix; caller should save both the review item and the
    /// sidecar payload (via [`save_imputed_fix_payload`]).
    Item(Box<ImputedItem>),
    /// Gap was found but the LLM call failed or returned unusable content.
    Skipped {
        /// Concept name the impute attempted.
        concept_name: String,
        /// Kind of gap that was skipped.
        gap_kind: ImputeGapKind,
        /// Short human-readable reason (shown in CLI output).
        reason: String,
    },
}

/// A ready-to-persist imputed fix: the review item + its sidecar payload.
#[derive(Debug, Clone)]
pub struct ImputedItem {
    /// The review item to persist to `reviews/imputed_fixes/<id>.json`.
    pub item: ReviewItem,
    /// The structured payload to persist to
    /// `reviews/imputed_fixes/<id>.payload.json`. Carries the fields the
    /// approve handler needs (page path, definition, sources) in a shape
    /// that doesn't depend on comment parsing.
    pub payload: ImputedFixPayload,
}

/// JSON payload written into the review item's `output_paths[0]` so the
/// approve handler has structured data without parsing the comment.
///
/// The filename is `reviews/imputed_fixes/<id>.payload.json`; the approve
/// handler reads it to get the definition + sources for the apply step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImputedFixPayload {
    /// Kind of gap this fix addresses.
    pub gap_kind: String,
    /// Concept name the fix targets. For missing concepts this is the
    /// LLM-approved canonical form; for thin bodies it's the existing
    /// concept name from frontmatter.
    pub concept_name: String,
    /// For `thin_concept_body`: the existing `concept:<slug>` id to locate
    /// the page. For `missing_concept`: derived from the canonical name via
    /// `slug_from_title`.
    pub concept_id: String,
    /// For `thin_concept_body`: existing page path (relative to kb root).
    /// For `missing_concept`: the proposed new path.
    pub page_path: PathBuf,
    /// Body the LLM drafted, trimmed.
    pub definition: String,
    /// Web sources the LLM cited.
    pub sources: Vec<ImputedWebSource>,
    /// Model's confidence self-assessment.
    pub confidence: String,
    /// Rationale note the model produced.
    pub rationale: String,
    /// Local source documents the impute pass seeded the prompt with. Used
    /// for `missing_concept` applies to populate the new page's
    /// `source_document_ids` frontmatter.
    #[serde(default)]
    pub local_source_document_ids: Vec<String>,
}

/// Run the impute pass.
///
/// For each gap identified by the lint library, call the adapter's
/// `impute_gap`; collect either a ready-to-save review item or a skip
/// record per gap. The caller persists review items via
/// [`kb_core::save_review_item`] and prints skips.
///
/// # Errors
///
/// Returns an error when the KB directory cannot be scanned or review-item
/// fingerprints cannot be loaded. Per-gap LLM failures are **not** errors
/// — they surface as `ImputeOutcome::Skipped` entries so one flaky call
/// doesn't poison the whole pass.
pub fn run_impute_pass<A: LlmAdapter + ?Sized>(
    root: &Path,
    adapter: &A,
    config: &ImputeConfig,
) -> Result<Vec<ImputeOutcome>> {
    let mut outcomes = Vec::new();
    let ack = load_imputed_fingerprints(root)?;
    let mut imputed = 0usize;

    if config.kinds.missing_concepts && imputed < config.max_imputations {
        let hits = crate::check_missing_concepts_hits(root, &config.missing_concepts)?;
        for hit in hits {
            if imputed >= config.max_imputations {
                break;
            }
            let fingerprint = missing_concept_fingerprint(&hit);
            if ack.contains(&fingerprint) {
                tracing::debug!(
                    "impute: skipping already-imputed missing-concept '{}'",
                    hit.name
                );
                continue;
            }
            let outcome = impute_missing_concept(adapter, root, &hit, fingerprint);
            imputed += 1;
            outcomes.push(outcome);
        }
    }

    if config.kinds.thin_concept_body && imputed < config.max_imputations {
        let hits = detect_thin_concept_bodies(root, &config.thin_concept_body)?;
        for hit in hits {
            if imputed >= config.max_imputations {
                break;
            }
            let fingerprint = thin_body_fingerprint(&hit);
            if ack.contains(&fingerprint) {
                tracing::debug!(
                    "impute: skipping already-imputed thin-body concept '{}'",
                    hit.name
                );
                continue;
            }
            let outcome = impute_thin_concept_body(adapter, root, &hit, fingerprint);
            imputed += 1;
            outcomes.push(outcome);
        }
    }

    Ok(outcomes)
}

fn impute_missing_concept<A: LlmAdapter + ?Sized>(
    adapter: &A,
    root: &Path,
    hit: &ConceptCandidateHit,
    fingerprint: String,
) -> ImputeOutcome {
    let snippets = gather_source_snippets(root, &hit.name, &hit.source_ids);
    let request = ImputeGapRequest {
        gap_kind: ImputeGapKind::MissingConcept,
        concept_name: hit.name.clone(),
        existing_body: String::new(),
        local_snippets: snippets,
    };
    let response = match adapter.impute_gap(request) {
        Ok((response, _)) => response,
        Err(err) => {
            return ImputeOutcome::Skipped {
                concept_name: hit.name.clone(),
                gap_kind: ImputeGapKind::MissingConcept,
                reason: format!("impute LLM call failed: {err}"),
            };
        }
    };

    let slug = slug_from_title(&hit.name);
    if slug.is_empty() {
        return ImputeOutcome::Skipped {
            concept_name: hit.name.clone(),
            gap_kind: ImputeGapKind::MissingConcept,
            reason: "candidate name slugs to empty string".to_string(),
        };
    }
    let page_path = PathBuf::from("wiki/concepts").join(format!("{slug}.md"));
    let concept_id = format!("concept:{slug}");

    let trimmed_def = response.definition.trim();
    if trimmed_def.is_empty() {
        return ImputeOutcome::Skipped {
            concept_name: hit.name.clone(),
            gap_kind: ImputeGapKind::MissingConcept,
            reason: "LLM returned empty definition".to_string(),
        };
    }

    let Ok(now) = unix_time_ms() else {
        return ImputeOutcome::Skipped {
            concept_name: hit.name.clone(),
            gap_kind: ImputeGapKind::MissingConcept,
            reason: "system clock failure".to_string(),
        };
    };

    let payload = ImputedFixPayload {
        gap_kind: ImputeGapKind::MissingConcept.as_str().to_string(),
        concept_name: hit.name.clone(),
        concept_id,
        page_path: page_path.clone(),
        definition: trimmed_def.to_string(),
        sources: response.sources.clone(),
        confidence: normalize_confidence(&response.confidence),
        rationale: response.rationale.trim().to_string(),
        local_source_document_ids: hit.source_ids.clone(),
    };

    let item = build_imputed_review_item(
        ImputeGapKind::MissingConcept,
        &slug,
        &hit.name,
        &page_path,
        &payload,
        &response,
        fingerprint,
        now,
    );

    ImputeOutcome::Item(Box::new(ImputedItem { item, payload }))
}

fn impute_thin_concept_body<A: LlmAdapter + ?Sized>(
    adapter: &A,
    _root: &Path,
    hit: &ThinConceptHit,
    fingerprint: String,
) -> ImputeOutcome {
    // No source-snippet gathering for thin bodies — the page already holds
    // what the KB knows. The model gets only the existing body + name.
    let request = ImputeGapRequest {
        gap_kind: ImputeGapKind::ThinConceptBody,
        concept_name: hit.name.clone(),
        existing_body: hit.body.clone(),
        local_snippets: Vec::new(),
    };
    let response = match adapter.impute_gap(request) {
        Ok((response, _)) => response,
        Err(err) => {
            return ImputeOutcome::Skipped {
                concept_name: hit.name.clone(),
                gap_kind: ImputeGapKind::ThinConceptBody,
                reason: format!("impute LLM call failed: {err}"),
            };
        }
    };

    let trimmed_def = response.definition.trim();
    if trimmed_def.is_empty() {
        return ImputeOutcome::Skipped {
            concept_name: hit.name.clone(),
            gap_kind: ImputeGapKind::ThinConceptBody,
            reason: "LLM returned empty definition".to_string(),
        };
    }

    let Ok(now) = unix_time_ms() else {
        return ImputeOutcome::Skipped {
            concept_name: hit.name.clone(),
            gap_kind: ImputeGapKind::ThinConceptBody,
            reason: "system clock failure".to_string(),
        };
    };

    let slug = hit.concept_id.strip_prefix("concept:").map_or_else(
        || {
            hit.page_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string()
        },
        str::to_string,
    );

    let payload = ImputedFixPayload {
        gap_kind: ImputeGapKind::ThinConceptBody.as_str().to_string(),
        concept_name: hit.name.clone(),
        concept_id: if hit.concept_id.is_empty() {
            format!("concept:{slug}")
        } else {
            hit.concept_id.clone()
        },
        page_path: hit.page_path.clone(),
        definition: trimmed_def.to_string(),
        sources: response.sources.clone(),
        confidence: normalize_confidence(&response.confidence),
        rationale: response.rationale.trim().to_string(),
        local_source_document_ids: Vec::new(),
    };

    let item = build_imputed_review_item(
        ImputeGapKind::ThinConceptBody,
        &slug,
        &hit.name,
        &hit.page_path,
        &payload,
        &response,
        fingerprint,
        now,
    );

    ImputeOutcome::Item(Box::new(ImputedItem { item, payload }))
}

#[allow(clippy::too_many_arguments)]
fn build_imputed_review_item(
    gap_kind: ImputeGapKind,
    slug: &str,
    concept_name: &str,
    page_path: &Path,
    payload: &ImputedFixPayload,
    response: &ImputeGapResponse,
    fingerprint: String,
    now: u64,
) -> ReviewItem {
    let id = format!("lint:imputed-fix:{}:{}", gap_kind.as_str(), slug);
    let payload_path = PathBuf::from("reviews/imputed_fixes").join(format!("{id}.payload.json"));

    let source_bullets = if response.sources.is_empty() {
        "(no web sources cited)".to_string()
    } else {
        response
            .sources
            .iter()
            .map(|s| format!("- {} — {} ({})", s.title.trim(), s.url.trim(), s.note.trim()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let rationale_line = if response.rationale.trim().is_empty() {
        String::new()
    } else {
        format!("\nRationale: {}", response.rationale.trim())
    };
    let comment = format!(
        "Imputed fix for '{}' ({}).\n\nConfidence: {}{}\n\nProposed {}:\n{}\n\nWeb sources:\n{}",
        concept_name,
        gap_kind.as_str(),
        normalize_confidence(&response.confidence),
        rationale_line,
        match gap_kind {
            ImputeGapKind::MissingConcept => "new concept page body",
            ImputeGapKind::ThinConceptBody => "replacement body",
        },
        response.definition.trim(),
        source_bullets,
    );

    // Target id is the concept id regardless of gap kind — missing concepts
    // and thin-body gaps both resolve to the same canonical concept slug.
    let target_id = payload.concept_id.clone();

    // Citations carry the web URLs + any local source ids so they show up
    // in `kb review show`. Prefix each citation so the reviewer can tell
    // web vs. local at a glance.
    let mut citations = Vec::new();
    for src in &response.sources {
        citations.push(format!("web: {}", src.url.trim()));
    }
    for local in &payload.local_source_document_ids {
        citations.push(format!("local: {local}"));
    }

    ReviewItem {
        metadata: EntityMetadata {
            id,
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![fingerprint],
            model_version: None,
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: None,
            dependencies: payload.local_source_document_ids.clone(),
            // `output_paths[0]` is the payload sidecar; apply reads it.
            // `output_paths[1]` is the target page path so the review
            // lister's "Affected pages" column surfaces the right file.
            output_paths: vec![payload_path, page_path.to_path_buf()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ImputedFix,
        target_entity_id: target_id,
        proposed_destination: Some(page_path.to_path_buf()),
        citations,
        affected_pages: vec![page_path.to_path_buf()],
        created_at_millis: now,
        status: ReviewStatus::Pending,
        comment,
    }
}

/// Compute a stable fingerprint for a missing-concept gap. Sorted source
/// ids + the candidate name — so adding/removing a single source re-fires
/// the impute, but a re-run on identical input is suppressed.
fn missing_concept_fingerprint(hit: &ConceptCandidateHit) -> String {
    let mut sorted: Vec<String> = hit.source_ids.clone();
    sorted.sort();
    sorted.dedup();
    let mut parts: Vec<&[u8]> = Vec::with_capacity(sorted.len() + 3);
    parts.push(b"missing:");
    parts.push(hit.name.as_bytes());
    parts.push(b"\x00");
    let src_strs: Vec<String> = sorted;
    for s in &src_strs {
        parts.push(s.as_bytes());
    }
    hash_many(&parts).to_hex()
}

/// Compute a stable fingerprint for a thin-concept-body gap. Keyed on the
/// concept id + a hash of the existing body, so editing the body to add
/// substantive content re-fires the impute (if the body still looks thin
/// by the word threshold), while repeated runs against unchanged input
/// dedup.
fn thin_body_fingerprint(hit: &ThinConceptHit) -> String {
    let parts: Vec<&[u8]> = vec![
        b"thin:",
        hit.concept_id.as_bytes(),
        b"\x00",
        hit.body.as_bytes(),
    ];
    hash_many(&parts).to_hex()
}

/// Load fingerprints from every existing `ImputedFix` review item whose
/// status is Approved or Rejected. Pending items are not in this set so a
/// pending item is re-computed on the next run — the user may want the
/// newer draft.
fn load_imputed_fingerprints(root: &Path) -> Result<BTreeSet<String>> {
    let mut set = BTreeSet::new();
    let items = list_review_items(root)?;
    for item in items {
        if item.kind != ReviewKind::ImputedFix {
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

/// Pull one snippet per source document mentioning the term. Case-insensitive
/// search; preserves original casing in the snippet. Capped at
/// `MAX_SNIPPETS_PER_REQUEST` so the prompt stays compact.
fn gather_source_snippets(
    root: &Path,
    concept_name: &str,
    source_ids: &[String],
) -> Vec<CandidateSourceSnippet> {
    let needle_lower = concept_name.to_ascii_lowercase();
    let mut snippets = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    for src_id in source_ids {
        if snippets.len() >= MAX_SNIPPETS_PER_REQUEST {
            break;
        }
        if seen.contains(src_id) {
            continue;
        }
        let path = normalized_dir(root).join(src_id).join("source.md");
        let Ok(body) = fs::read_to_string(&path) else {
            continue;
        };
        if let Some(snippet) = extract_snippet_around(&body, &needle_lower) {
            snippets.push(CandidateSourceSnippet {
                source_document_id: src_id.clone(),
                snippet,
            });
            seen.insert(src_id.clone());
        }
    }
    snippets
}

fn extract_snippet_around(body: &str, needle_lower: &str) -> Option<String> {
    let lower = body.to_ascii_lowercase();
    let match_byte = lower.find(needle_lower)?;

    let start_byte = match_byte.saturating_sub(SNIPPET_LOOKBEHIND);
    let start = nearest_char_boundary(body, start_byte);
    let end_target = match_byte + needle_lower.len() + (MAX_SNIPPET_CHARS - SNIPPET_LOOKBEHIND);
    let end_byte = end_target.min(body.len());
    let end = nearest_char_boundary(body, end_byte);

    let mut snippet = body[start..end].to_string();
    snippet = snippet.split_whitespace().collect::<Vec<_>>().join(" ");
    if snippet.is_empty() {
        return None;
    }
    let leading = if start > 0 { "…" } else { "" };
    let trailing = if end < body.len() { "…" } else { "" };
    Some(format!("{leading}{snippet}{trailing}"))
}

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

fn normalize_confidence(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase();
    match lower.as_str() {
        "high" | "medium" | "low" => lower,
        // Everything else round-trips as-is so the user sees what the model
        // returned; the CLI treats anything non-standard as an opaque label.
        _ => {
            if lower.is_empty() {
                "medium".to_string()
            } else {
                lower
            }
        }
    }
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

/// Persist the payload sidecar for an imputed-fix review item.
///
/// Writes to `reviews/imputed_fixes/<id>.payload.json`. Called by the CLI
/// after a successful [`run_impute_pass`] alongside
/// [`kb_core::save_review_item`].
///
/// # Errors
///
/// Returns an error when the directory cannot be created or the file
/// cannot be written.
pub fn save_imputed_fix_payload(
    root: &Path,
    item: &ReviewItem,
    payload: &ImputedFixPayload,
) -> Result<()> {
    let sidecar_rel = item
        .metadata
        .output_paths
        .first()
        .cloned()
        .unwrap_or_else(|| {
            PathBuf::from("reviews/imputed_fixes")
                .join(format!("{}.payload.json", item.metadata.id))
        });
    let sidecar = root.join(&sidecar_rel);
    if let Some(parent) = sidecar.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create imputed-fix payload dir {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(payload)
        .with_context(|| format!("serialize imputed-fix payload {}", item.metadata.id))?;
    fs::write(&sidecar, json)
        .with_context(|| format!("write imputed-fix payload {}", sidecar.display()))?;
    Ok(())
}

/// Load the payload sidecar for an imputed-fix review item.
///
/// # Errors
///
/// Returns an error when the sidecar is missing or can't be parsed.
pub fn load_imputed_fix_payload(root: &Path, item: &ReviewItem) -> Result<ImputedFixPayload> {
    let sidecar_rel = item
        .metadata
        .output_paths
        .first()
        .cloned()
        .unwrap_or_else(|| {
            PathBuf::from("reviews/imputed_fixes")
                .join(format!("{}.payload.json", item.metadata.id))
        });
    let sidecar = root.join(&sidecar_rel);
    let raw = fs::read_to_string(&sidecar)
        .with_context(|| format!("read imputed-fix payload {}", sidecar.display()))?;
    let payload: ImputedFixPayload = serde_json::from_str(&raw)
        .with_context(|| format!("parse imputed-fix payload {}", sidecar.display()))?;
    Ok(payload)
}

/// Surface impute outcomes as lint issues for the existing printer.
///
/// One warning per successful imputation; skip entries are rendered as
/// info-level warnings too so the user knows the pass ran even when
/// every call failed.
#[must_use]
pub fn outcomes_to_lint_issues(outcomes: &[ImputeOutcome]) -> Vec<LintIssue> {
    outcomes
        .iter()
        .map(|outcome| match outcome {
            ImputeOutcome::Item(payload) => {
                let item = &payload.item;
                LintIssue {
                    severity: IssueSeverity::Warning,
                    kind: IssueKind::ConceptCandidate,
                    referring_page: item
                        .affected_pages
                        .first()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    line: 0,
                    target: item.target_entity_id.clone(),
                    message: format!(
                        "imputed-fix review item queued: {} (run 'kb review show {}' to inspect, approve to apply)",
                        item.target_entity_id, item.metadata.id
                    ),
                    suggested_fix: Some(format!(
                        "kb review show {} | kb review approve {} | kb review reject {}",
                        item.metadata.id, item.metadata.id, item.metadata.id
                    )),
                }
            }
            ImputeOutcome::Skipped {
                concept_name,
                gap_kind,
                reason,
            } => LintIssue {
                severity: IssueSeverity::Warning,
                kind: IssueKind::ConceptCandidate,
                referring_page: String::new(),
                line: 0,
                target: concept_name.clone(),
                message: format!(
                    "imputed-fix skipped for '{}' ({}): {}",
                    concept_name,
                    gap_kind.as_str(),
                    reason
                ),
                suggested_fix: None,
            },
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use kb_llm::{
        AnswerQuestionRequest, AnswerQuestionResponse, DetectContradictionsRequest,
        DetectContradictionsResponse, ExtractConceptsRequest, ExtractConceptsResponse,
        GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapterError,
        MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, ProvenanceRecord,
        RunHealthCheckRequest, RunHealthCheckResponse, SummarizeDocumentRequest,
        SummarizeDocumentResponse,
    };
    use std::sync::Mutex;
    use tempfile::TempDir;

    struct StubAdapter {
        impute_response: Mutex<Option<ImputeGapResponse>>,
        calls: Mutex<Vec<ImputeGapRequest>>,
    }

    impl StubAdapter {
        fn new(response: ImputeGapResponse) -> Self {
            Self {
                impute_response: Mutex::new(Some(response)),
                calls: Mutex::new(Vec::new()),
            }
        }
    }

    fn prov() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "stub".to_string(),
            harness_version: None,
            model: "stub".to_string(),
            prompt_template_name: "impute_gap.md".to_string(),
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
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
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
            _r: DetectContradictionsRequest,
        ) -> Result<(DetectContradictionsResponse, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other("not used".into()))
        }
        fn impute_gap(
            &self,
            r: ImputeGapRequest,
        ) -> Result<(ImputeGapResponse, ProvenanceRecord), LlmAdapterError> {
            self.calls.lock().unwrap().push(r);
            let taken = self.impute_response.lock().unwrap().take();
            taken.map_or_else(
                || Err(LlmAdapterError::Other("no stub response".to_string())),
                |resp| Ok((resp, prov())),
            )
        }
    }

    fn write_concept(root: &Path, slug: &str, name: &str, body: &str) {
        let dir = root.join("wiki").join("concepts");
        fs::create_dir_all(&dir).unwrap();
        let content = format!(
            "---\nid: concept:{slug}\nname: {name}\n---\n\n# {name}\n\n{body}\n"
        );
        fs::write(dir.join(format!("{slug}.md")), content).unwrap();
    }

    #[test]
    fn thin_body_detection_flags_short_bodies() {
        let tmp = TempDir::new().unwrap();
        write_concept(tmp.path(), "short", "Short Page", "tiny.");
        write_concept(
            tmp.path(),
            "long",
            "Long Page",
            "This concept page has plenty of body text that clearly exceeds the minimum word threshold set by the thin-body detector.",
        );
        let hits = detect_thin_concept_bodies(
            tmp.path(),
            &crate::ThinConceptBodyConfig::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].concept_id, "concept:short");
    }

    #[test]
    fn impute_thin_body_produces_review_item_and_payload() {
        let tmp = TempDir::new().unwrap();
        write_concept(tmp.path(), "widget", "Widget", "thin.");

        let stub = StubAdapter::new(ImputeGapResponse {
            definition: "Widget is a test concept used for impute integration tests.".to_string(),
            sources: vec![ImputedWebSource {
                url: "https://example.com/widget".to_string(),
                title: "Widget — Example".to_string(),
                note: "Overview used to seed the definition.".to_string(),
            }],
            confidence: "medium".to_string(),
            rationale: "Stub response for tests.".to_string(),
        });

        let config = ImputeConfig {
            kinds: ImputeKindSelector {
                missing_concepts: false,
                thin_concept_body: true,
            },
            ..Default::default()
        };

        let outcomes = run_impute_pass(tmp.path(), &stub, &config).unwrap();
        assert_eq!(outcomes.len(), 1, "one thin-body hit should be imputed");
        let (item, payload) = match &outcomes[0] {
            ImputeOutcome::Item(i) => (&i.item, &i.payload),
            ImputeOutcome::Skipped { .. } => panic!("expected item, got skipped"),
        };
        assert_eq!(item.kind, ReviewKind::ImputedFix);
        assert!(item.metadata.id.starts_with("lint:imputed-fix:thin_concept_body:widget"));
        assert_eq!(item.target_entity_id, "concept:widget");
        assert!(item.citations.iter().any(|c| c.starts_with("web: ")));
        assert_eq!(payload.concept_id, "concept:widget");
        assert_eq!(payload.gap_kind, "thin_concept_body");

        // Round-trip through the save/load helpers.
        kb_core::save_review_item(tmp.path(), item).unwrap();
        save_imputed_fix_payload(tmp.path(), item, payload).unwrap();
        let loaded = load_imputed_fix_payload(tmp.path(), item).unwrap();
        assert_eq!(loaded.concept_id, "concept:widget");
        assert_eq!(loaded.definition, payload.definition);
    }

    #[test]
    fn impute_skips_already_processed_fingerprints() {
        let tmp = TempDir::new().unwrap();
        write_concept(tmp.path(), "widget", "Widget", "thin.");
        let stub = StubAdapter::new(ImputeGapResponse {
            definition: "Widget is a thing.".to_string(),
            sources: vec![],
            confidence: "low".to_string(),
            rationale: String::new(),
        });
        let config = ImputeConfig {
            kinds: ImputeKindSelector {
                missing_concepts: false,
                thin_concept_body: true,
            },
            ..Default::default()
        };

        // First run: emits an item.
        let first = run_impute_pass(tmp.path(), &stub, &config).unwrap();
        let first_item = match &first[0] {
            ImputeOutcome::Item(i) => i.item.clone(),
            ImputeOutcome::Skipped { .. } => panic!("expected item"),
        };

        // Mark the item approved so its fingerprint is in the ack set.
        let mut approved = first_item;
        approved.status = ReviewStatus::Approved;
        kb_core::save_review_item(tmp.path(), &approved).unwrap();

        // Second run: no impute call because fingerprint is acknowledged.
        let second = run_impute_pass(tmp.path(), &stub, &config).unwrap();
        assert!(second.is_empty(), "expected no new outcomes, got {second:?}");
    }
}
