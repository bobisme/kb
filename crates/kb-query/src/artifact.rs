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

    // bn-1319: rewrite valid `[N]` markers to Obsidian wikilinks so they're
    // clickable inside the vault. Invalid `[N]` are left as plain text so the
    // unresolved-citations footer below still labels them.
    let rewritten = CITATION_RE.replace_all(raw_answer, |caps: &regex::Captures<'_>| {
        let raw = caps.get(0).map_or("", |m| m.as_str()).to_string();
        let n: u32 = caps[1].parse().unwrap_or(0);
        manifest
            .entries
            .get(&n)
            .map_or(raw, |entry| citation_wikilink(entry, n))
    });

    let mut body = String::new();

    if needs_banner {
        body.push_str("> **Note:** The available sources provide limited coverage for this question. ");
        body.push_str("The answer below may be incomplete. Consider ingesting additional sources.\n\n");
    }

    body.push_str(&rewritten);

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

/// Render an Obsidian wikilink for a citation manifest entry, displayed as the
/// numeric key the model emitted (`[[wiki/path|N]]`). Real heading anchors are
/// preserved as `#anchor`; the assembler's pseudo-anchors (`summary`,
/// `section`) are skipped because they don't correspond to anything in the
/// underlying page.
fn citation_wikilink(entry: &ManifestEntry, n: u32) -> String {
    let target = entry
        .source_id
        .strip_suffix(".md")
        .unwrap_or(&entry.source_id);
    match entry.anchor.as_deref() {
        Some(anchor) if !anchor.is_empty() && anchor != "summary" && anchor != "section" => {
            format!("[[{target}#{anchor}|{n}]]")
        }
        _ => format!("[[{target}|{n}]]"),
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

/// Strip the LLM's inlined tool-call narration from the head of a chart-format
/// answer body.
///
/// bn-31uk: when `kb ask --format=chart` runs, opencode drives the model
/// through a bash/write tool cycle to produce the PNG, and streams the
/// model's per-tool-call "what I'm about to do" commentary into stdout
/// alongside the real final caption. The narration leaks into `answer.md`:
///
/// ```text
/// Checking the output location, then I'll write and run a minimal script…
/// Writing the chart script now, then I'll execute it and verify the PNG…
///
/// # Chart caption
///
/// The chart shows …
/// ```
///
/// Strategy (deliberately conservative — no phrase-level regex):
///
/// 1. If the body contains a line-start markdown heading (`# ` through
///    `######`), return the slice starting at the first such heading. The
///    `ask_chart.md` prompt doesn't mandate a heading, but models often emit
///    one, and when they do it is unambiguously the start of the caption.
/// 2. Else, if the body has at least one blank-line-separated paragraph
///    break, return the slice from the *last* paragraph. The narration
///    opencode streams is organized into its own paragraphs separate from
///    the final assistant message, so the final paragraph is the caption.
/// 3. Else, return the full trimmed input unchanged. When the model emits
///    one run-together block with no paragraph breaks at all, we have no
///    robust structural cue to split on — dropping known "I'll"/"Checking"
///    phrases by regex is too fragile (bone bn-31uk explicitly forbids it),
///    so we keep the body intact rather than risk stripping real content.
///
/// Applied only on the chart / figure code path — markdown and JSON formats
/// pass through unchanged.
#[must_use]
pub fn strip_tool_narration(text: &str) -> &str {
    let trimmed = text.trim_start_matches(['\n', '\r']);

    // Rule 1: first line-start markdown heading anchors the caption.
    if let Some(idx) = find_first_heading(trimmed) {
        return trimmed[idx..].trim_end();
    }

    // Rule 2: last blank-line-separated paragraph.
    if let Some(idx) = find_last_paragraph_start(trimmed) {
        return trimmed[idx..].trim_end();
    }

    // Rule 3: no structural signal — return input unchanged (still trimmed).
    trimmed.trim_end()
}

/// Return the byte offset of the first line-start markdown heading
/// (1–6 `#` chars followed by a space), or `None` if there is no heading.
/// Only considers headings at the true start of a line (after a `\n` or at
/// the beginning of the text), so inline `#` characters inside prose are
/// ignored.
fn find_first_heading(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let at_line_start = i == 0 || bytes[i - 1] == b'\n';
        if at_line_start && bytes[i] == b'#' {
            let mut j = i;
            while j < bytes.len() && j - i < 6 && bytes[j] == b'#' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b' ' {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Return the byte offset of the start of the last blank-line-separated
/// paragraph. A "paragraph break" is two or more consecutive newlines
/// (optionally with spaces/tabs on the blank lines). Returns `None` when
/// the text has no paragraph break at all.
fn find_last_paragraph_start(text: &str) -> Option<usize> {
    // Walk the text to locate all paragraph breaks, then return the offset
    // of the character just after the *last* one. We consider a run of
    // "\n[ \t]*\n" (one or more times) a break.
    let bytes = text.as_bytes();
    let mut last_break_end: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            // Scan forward over any blank lines.
            let mut j = i + 1;
            let mut saw_blank_line = false;
            loop {
                // Consume leading spaces/tabs on this line.
                let line_start = j;
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'\n' {
                    // Entire line was whitespace → counts as blank.
                    saw_blank_line = true;
                    j += 1;
                } else {
                    // Non-blank line reached. If we saw any blank line, this
                    // is a paragraph break and the break ends at `line_start`.
                    if saw_blank_line {
                        last_break_end = Some(line_start);
                    }
                    break;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    last_break_end
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
        // Valid citations are rewritten to clickable Obsidian wikilinks; invalid
        // ones stay as plain text so the unresolved-citations footer reads cleanly.
        assert!(result.body.contains("[[wiki/sources/doc-a|1]]"));
        assert!(result.body.contains("[[wiki/sources/doc-b#intro|2]]"));
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

    /// bn-1319 regression: valid `[N]` markers must turn into Obsidian
    /// wikilinks so they're clickable in the vault, with `.md` stripped from
    /// the target and pseudo-anchors (`summary`, `section`) omitted because
    /// they don't correspond to a real heading on the destination page.
    #[test]
    fn postprocess_rewrites_valid_citations_to_obsidian_wikilinks() {
        let ctx = AssembledContext {
            text: String::new(),
            token_budget: 1000,
            estimated_tokens: 0,
            manifest: vec![
                ContextManifestEntry {
                    start_offset: 0,
                    end_offset: 1,
                    source_id: "wiki/sources/full-doc.md".to_string(),
                    anchor: None,
                    chunk_kind: ContextChunkKind::FullDocument,
                },
                ContextManifestEntry {
                    start_offset: 1,
                    end_offset: 2,
                    source_id: "wiki/sources/with-summary.md".to_string(),
                    anchor: Some("summary".to_string()),
                    chunk_kind: ContextChunkKind::Summary,
                },
                ContextManifestEntry {
                    start_offset: 2,
                    end_offset: 3,
                    source_id: "wiki/sources/with-section.md".to_string(),
                    anchor: Some("section".to_string()),
                    chunk_kind: ContextChunkKind::Section,
                },
                ContextManifestEntry {
                    start_offset: 3,
                    end_offset: 4,
                    source_id: "wiki/concepts/topic.md".to_string(),
                    anchor: Some("ownership-rules".to_string()),
                    chunk_kind: ContextChunkKind::Section,
                },
            ],
        };
        let manifest = build_citation_manifest(&ctx);
        let raw = "Claim a [1]; claim b [2]; claim c [3]; claim d [4].";
        let result = postprocess_answer(raw, &manifest, &ctx);

        // No-anchor source: link to the page itself.
        assert!(
            result.body.contains("[[wiki/sources/full-doc|1]]"),
            "got body: {}",
            result.body
        );
        // Pseudo "summary" anchor must be stripped.
        assert!(result.body.contains("[[wiki/sources/with-summary|2]]"));
        assert!(!result.body.contains("with-summary#summary"));
        // Pseudo "section" anchor must be stripped.
        assert!(result.body.contains("[[wiki/sources/with-section|3]]"));
        assert!(!result.body.contains("with-section#section"));
        // Real heading anchors survive on concept pages too.
        assert!(result.body.contains("[[wiki/concepts/topic#ownership-rules|4]]"));

        // No bare `[N]` markers should remain in the body for valid citations.
        for n in 1..=4 {
            let bare = format!("[{n}]");
            assert!(
                !result.body.contains(&bare),
                "bare {bare} should have been rewritten to a wikilink, body: {}",
                result.body
            );
        }
    }

    /// bn-1319 regression: adjacent citations like `[1][2]` must each be
    /// rewritten independently and produce well-formed back-to-back wikilinks
    /// without corrupting the surrounding text.
    #[test]
    fn postprocess_rewrites_adjacent_citations() {
        let ctx = sample_context();
        let manifest = build_citation_manifest(&ctx);
        let raw = "Stacked claim [1][2] still resolves.";
        let result = postprocess_answer(raw, &manifest, &ctx);
        assert!(
            result
                .body
                .contains("[[wiki/sources/doc-a|1]][[wiki/sources/doc-b#intro|2]]"),
            "got: {}",
            result.body
        );
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

    // bn-31uk: strip_tool_narration tests. The helper is applied only on the
    // chart code path; it must be surgical — no phrase-level regex — and
    // preserve real caption content intact.

    #[test]
    fn strip_tool_narration_keeps_from_first_heading() {
        let raw = "Checking the output location, then I'll write the script.\n\
                   Writing the chart script now.\n\n\
                   # Chart caption\n\n\
                   Body text of the caption.\n";
        assert_eq!(
            strip_tool_narration(raw),
            "# Chart caption\n\n\
             Body text of the caption."
        );
    }

    #[test]
    fn strip_tool_narration_prefers_first_heading_over_last_paragraph() {
        // When both rules could match, the heading wins — it's the stronger
        // structural signal.
        let raw = "narration line\n\n\
                   # Caption header\n\n\
                   Caption body paragraph.\n\n\
                   Trailing paragraph.\n";
        let stripped = strip_tool_narration(raw);
        assert!(stripped.starts_with("# Caption header"), "got: {stripped:?}");
        assert!(stripped.contains("Caption body paragraph."));
        assert!(stripped.contains("Trailing paragraph."));
    }

    #[test]
    fn strip_tool_narration_falls_back_to_last_paragraph() {
        // No heading but narration and caption are separated by a blank line.
        let raw = "Checking the output location.\n\
                   Writing the chart script now.\n\n\
                   The chart shows a horizontal comparison of two categories.\n";
        assert_eq!(
            strip_tool_narration(raw),
            "The chart shows a horizontal comparison of two categories."
        );
    }

    #[test]
    fn strip_tool_narration_returns_input_when_no_paragraph_break_and_no_heading() {
        // Pathological real-world case (bn-31uk example): narration and
        // caption are all in one run-together block with no blank lines and
        // no heading. The helper must NOT lossily drop real content by
        // guessing; it returns the (trimmed) input so the caller still sees
        // the full answer and downstream behavior is unchanged from pre-fix.
        let raw = "Checking the output location.\n\
                   Writing the chart script now.\n\
                   The chart shows a horizontal comparison of two categories.\n";
        assert_eq!(
            strip_tool_narration(raw),
            "Checking the output location.\n\
             Writing the chart script now.\n\
             The chart shows a horizontal comparison of two categories."
        );
    }

    #[test]
    fn strip_tool_narration_ignores_inline_hash_characters() {
        // A `#` that is NOT at the start of a line is not a markdown heading.
        // The helper must not mistake inline `#` or `#tag` for a section
        // break.
        let raw = "narration with #hashtag inline.\n\n\
                   actual caption paragraph.\n";
        // No line-start heading, so rule 2 (last paragraph) wins.
        assert_eq!(
            strip_tool_narration(raw),
            "actual caption paragraph."
        );
    }

    #[test]
    fn strip_tool_narration_handles_heading_at_start_of_input() {
        // If the heading is already at position 0, the helper returns the
        // whole thing unchanged modulo trailing whitespace.
        let raw = "# Caption\n\nBody.\n";
        assert_eq!(strip_tool_narration(raw), "# Caption\n\nBody.");
    }

    #[test]
    fn strip_tool_narration_preserves_trailing_paragraphs_after_heading() {
        // After the heading the entire remainder is the caption content,
        // including any subsequent paragraphs.
        let raw = "intro narration.\n\n\
                   # Title\n\n\
                   Paragraph one.\n\n\
                   Paragraph two with citation [1].\n";
        let stripped = strip_tool_narration(raw);
        assert!(stripped.starts_with("# Title"));
        assert!(stripped.contains("Paragraph one."));
        assert!(stripped.contains("Paragraph two with citation [1]."));
    }

    #[test]
    fn strip_tool_narration_handles_blank_lines_with_whitespace() {
        // Real LLM output sometimes emits paragraph breaks with a space or
        // tab on the "blank" line. Treat those as paragraph breaks too.
        let raw = "narration paragraph.\n   \n\
                   caption paragraph.\n";
        assert_eq!(
            strip_tool_narration(raw),
            "caption paragraph."
        );
    }

    #[test]
    fn strip_tool_narration_empty_input_returns_empty() {
        assert_eq!(strip_tool_narration(""), "");
        assert_eq!(strip_tool_narration("\n\n\n"), "");
    }

    #[test]
    fn strip_tool_narration_single_line_no_heading_returns_input() {
        // No narration at all — a one-line caption passes through.
        assert_eq!(
            strip_tool_narration("Just a single caption line."),
            "Just a single caption line."
        );
    }

    #[test]
    fn strip_tool_narration_preserves_citation_keys_in_caption() {
        // Integration with postprocess_answer's citation-key extraction: the
        // helper must never eat `[N]` markers from the caption.
        let raw = "narration.\n\n\
                   The chart shows X versus Y [1][2].\n";
        assert_eq!(
            strip_tool_narration(raw),
            "The chart shows X versus Y [1][2]."
        );
    }
}
