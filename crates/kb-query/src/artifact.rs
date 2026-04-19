use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::LazyLock;

use regex::Regex;

use crate::lexical::{AssembledContext, ContextManifestEntry};

pub struct CitationManifest {
    pub entries: BTreeMap<u32, ManifestEntry>,
}

pub struct ManifestEntry {
    pub source_id: String,
    pub anchor: Option<String>,
    pub label: String,
}

pub struct ArtifactResult {
    pub body: String,
    pub valid_citations: Vec<u32>,
    pub invalid_citations: Vec<u32>,
    pub has_uncertainty_banner: bool,
}

#[must_use]
pub fn build_citation_manifest(ctx: &AssembledContext) -> CitationManifest {
    let mut entries = BTreeMap::new();
    for (i, entry) in ctx.manifest.iter().enumerate() {
        let key = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let label = format_manifest_label(entry);
        entries.insert(
            key,
            ManifestEntry {
                source_id: entry.source_id.clone(),
                anchor: entry.anchor.clone(),
                label,
            },
        );
    }
    CitationManifest { entries }
}

fn format_manifest_label(entry: &ContextManifestEntry) -> String {
    let kind = format!("{:?}", entry.chunk_kind).to_lowercase();
    entry.anchor.as_ref().map_or_else(
        || format!("{} ({})", entry.source_id, kind),
        |anchor| format!("{}#{} ({})", entry.source_id, anchor, kind),
    )
}

#[must_use]
pub fn render_manifest_for_prompt(manifest: &CitationManifest) -> String {
    let mut out = String::new();
    for (key, entry) in &manifest.entries {
        let _ = writeln!(out, "[{key}] {}", entry.label);
    }
    out
}

static CITATION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(\d+)\]").expect("valid citation regex")
});

#[must_use]
pub fn postprocess_answer(
    raw_answer: &str,
    manifest: &CitationManifest,
    _ctx: &AssembledContext,
) -> ArtifactResult {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();

    for cap in CITATION_RE.captures_iter(raw_answer) {
        let n: u32 = cap[1].parse().unwrap_or(0);
        if n == 0 {
            continue;
        }
        if manifest.entries.contains_key(&n) {
            if !valid.contains(&n) {
                valid.push(n);
            }
        } else if !invalid.contains(&n) {
            invalid.push(n);
        }
    }

    let needs_banner = should_show_uncertainty_banner(&valid, &invalid);

    let mut body = String::new();

    if needs_banner {
        body.push_str("> **Note:** The available sources provide limited coverage for this question. ");
        body.push_str("The answer below may be incomplete. Consider ingesting additional sources.\n\n");
    }

    body.push_str(raw_answer);

    if !invalid.is_empty() {
        body.push_str("\n\n---\n\n");
        body.push_str("**Unresolved citations:** ");
        let labels: Vec<String> = invalid.iter().map(|n| format!("[{n}]")).collect();
        body.push_str(&labels.join(", "));
        body.push_str(" — these citation keys do not match any source in the manifest.\n");
    }

    ArtifactResult {
        body,
        valid_citations: valid,
        invalid_citations: invalid,
        has_uncertainty_banner: needs_banner,
    }
}

/// Decide whether the final answer needs an uncertainty banner.
///
/// The banner should only fire when the answer is truly weakly grounded:
///
/// - No valid citations at all (the model didn't cite any real source), or
/// - More hallucinated citation keys than valid ones (answer references
///   sources that do not exist in the manifest more often than real ones).
///
/// Heuristics based purely on candidate count or token-budget utilization
/// are intentionally absent: a short, well-cited answer with a small
/// context is not uncertain.
const fn should_show_uncertainty_banner(
    valid_citations: &[u32],
    invalid_citations: &[u32],
) -> bool {
    if valid_citations.is_empty() {
        return true;
    }
    invalid_citations.len() > valid_citations.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::ContextChunkKind;

    fn sample_context() -> AssembledContext {
        AssembledContext {
            text: "Some context text here.".to_string(),
            token_budget: 1000,
            estimated_tokens: 500,
            manifest: vec![
                ContextManifestEntry {
                    start_offset: 0,
                    end_offset: 10,
                    source_id: "wiki/sources/doc-a.md".to_string(),
                    anchor: None,
                    chunk_kind: ContextChunkKind::FullDocument,
                },
                ContextManifestEntry {
                    start_offset: 10,
                    end_offset: 22,
                    source_id: "wiki/sources/doc-b.md".to_string(),
                    anchor: Some("intro".to_string()),
                    chunk_kind: ContextChunkKind::Section,
                },
            ],
        }
    }

    #[test]
    fn build_manifest_assigns_sequential_keys() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        assert_eq!(manifest.entries.len(), 2);
        assert!(manifest.entries.contains_key(&1));
        assert!(manifest.entries.contains_key(&2));
        assert_eq!(manifest.entries[&1].source_id, "wiki/sources/doc-a.md");
        assert_eq!(manifest.entries[&2].source_id, "wiki/sources/doc-b.md");
        assert_eq!(manifest.entries[&2].anchor.as_deref(), Some("intro"));
    }

    #[test]
    fn render_manifest_produces_numbered_lines() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let rendered = render_manifest_for_prompt(&manifest);
        assert!(rendered.contains("[1]"));
        assert!(rendered.contains("[2]"));
        assert!(rendered.contains("wiki/sources/doc-a.md"));
        assert!(rendered.contains("doc-b.md#intro"));
    }

    #[test]
    fn postprocess_validates_citations() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "According to the source [1], this is true. Also see [2]. But [99] is wrong.";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert_eq!(result.valid_citations, vec![1, 2]);
        assert_eq!(result.invalid_citations, vec![99]);
        assert!(result.body.contains("[99]"));
        assert!(result.body.contains("Unresolved citations"));
    }

    #[test]
    fn postprocess_no_invalid_citations_skips_warning() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "This is grounded [1] and also [2].";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert_eq!(result.valid_citations, vec![1, 2]);
        assert!(result.invalid_citations.is_empty());
        assert!(!result.body.contains("Unresolved citations"));
    }

    #[test]
    fn uncertainty_banner_when_low_coverage() {
        // Low coverage now means no valid citations at all — a tiny context
        // with a manifest entry but a raw answer that cites nothing real
        // should still fire the banner.
        let ctx = AssembledContext {
            text: "tiny".to_string(),
            token_budget: 10000,
            estimated_tokens: 50,
            manifest: vec![ContextManifestEntry {
                start_offset: 0,
                end_offset: 4,
                source_id: "wiki/sources/tiny.md".to_string(),
                anchor: None,
                chunk_kind: ContextChunkKind::Summary,
            }],
        };
        let manifest = build_citation_manifest(&ctx);
        let raw = "Some answer with no citations at all.";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert!(result.has_uncertainty_banner);
        assert!(result.body.contains("limited coverage"));
    }

    #[test]
    fn uncertainty_banner_when_empty_context() {
        let ctx = AssembledContext {
            text: String::new(),
            token_budget: 1000,
            estimated_tokens: 0,
            manifest: vec![],
        };
        let manifest = build_citation_manifest(&ctx);
        let raw = "I have no sources to draw from.";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert!(result.has_uncertainty_banner);
    }

    #[test]
    fn no_uncertainty_banner_when_good_coverage() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "Well-grounded answer [1] with evidence [2].";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert!(!result.has_uncertainty_banner);
    }

    #[test]
    fn no_banner_with_five_valid_zero_invalid() {
        // Regression: 5 valid citations, 0 invalid should never show the
        // banner regardless of context size vs token budget.
        let ctx = AssembledContext {
            text: "context".to_string(),
            // A big budget with relatively few estimated tokens used to
            // trip the old "low coverage" heuristic — verify it no longer does.
            token_budget: 100_000,
            estimated_tokens: 500,
            manifest: (0..5)
                .map(|i| ContextManifestEntry {
                    start_offset: i * 10,
                    end_offset: i * 10 + 10,
                    source_id: format!("wiki/sources/doc-{i}.md"),
                    anchor: None,
                    chunk_kind: ContextChunkKind::Section,
                })
                .collect(),
        };
        let manifest = build_citation_manifest(&ctx);
        let raw = "Rust has three ownership rules [1][2][3][4][5].";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert_eq!(result.valid_citations.len(), 5);
        assert!(result.invalid_citations.is_empty());
        assert!(!result.has_uncertainty_banner);
        assert!(!result.body.contains("limited coverage"));
    }

    #[test]
    fn banner_with_zero_valid_citations() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "Ungrounded claim with no citations.";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert!(result.valid_citations.is_empty());
        assert!(result.has_uncertainty_banner);
    }

    #[test]
    fn banner_when_invalid_exceeds_valid() {
        // 1 valid, 3 invalid — more hallucinated than grounded.
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "Claim [1] plus fabrications [42], [77], and [99].";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert_eq!(result.valid_citations, vec![1]);
        assert_eq!(result.invalid_citations, vec![42, 77, 99]);
        assert!(result.has_uncertainty_banner);
    }

    #[test]
    fn no_banner_when_valid_equals_or_exceeds_invalid() {
        // 2 valid, 1 invalid — majority grounded, no banner.
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "Grounded [1] and [2] with one stray [99].";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert_eq!(result.valid_citations, vec![1, 2]);
        assert_eq!(result.invalid_citations, vec![99]);
        assert!(!result.has_uncertainty_banner);
    }

    #[test]
    fn deduplicates_citation_keys() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "Claim [1] and again [1] and [1].";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert_eq!(result.valid_citations, vec![1]);
    }
}
