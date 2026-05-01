//! Markdown chunking for per-section embeddings (bn-3rzz).
//!
//! Long sources used to embed as a single vector with a hard char cap;
//! MiniLM-L6-v2's 256-token training window meant a 5000-token source
//! about many topics averaged out and matched nothing well. This module
//! splits a source body into semantically-coherent chunks so each chunk
//! can be embedded independently and aggregated to source-level by
//! max-chunk-score at query time.
//!
//! Chunking algorithm:
//!
//! 1. Strip managed-region fences (the markers themselves, not contents).
//! 2. Split by H2 (`^## `). Pre-first-H2 prose becomes a chunk with no heading.
//! 3. For each section, if `estimated_tokens > 800`, split by H3 (`^### `).
//!    If still > 800, split by blank-line paragraphs.
//! 4. Token estimate = `chars / 4` (soft heuristic — the 100/800 thresholds
//!    are not load-bearing; they bias toward chunks small enough to fit a
//!    256-token window without splitting mid-thought too aggressively).
//! 5. Merge chunks under 100 `estimated_tokens` into the previous chunk
//!    (or the next one when there's no previous, e.g. a tiny pre-H2 prose
//!    chunk before a long H2 section).
//!
//! The chunker is pure — no I/O, easy to unit-test.

use kb_core::extract_managed_regions;

/// Soft upper bound on a chunk's `estimated_tokens` before further splitting.
///
/// When a chunk's tokens exceed this, the chunker tries H3 splitting,
/// then paragraph fallback. Not load-bearing — picked to fit MiniLM-L6-v2's
/// 256-token training window with headroom for header/title prefixes.
pub const TARGET_MAX_TOKENS: usize = 800;

/// Soft lower bound on a chunk's `estimated_tokens` before we merge it
/// upward into the previous chunk. Avoids one-line stub chunks dominating
/// retrieval through the chunk-id tail.
pub const TARGET_MIN_TOKENS: usize = 100;

/// Approximate char-to-token ratio. `MiniLM` tokenizes about 4 chars per
/// token on English prose; the actual ratio drifts on code-heavy sources
/// but the thresholds (100/800) tolerate a wide band, so the simple
/// heuristic is fine.
const CHARS_PER_TOKEN: usize = 4;

/// One slice of a source's body, ready to be embedded independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The H2/H3 heading text this chunk belongs to. `None` for pre-first-heading
    /// prose or when a chunk is the body of a top-level document with no headings.
    pub heading: Option<String>,
    /// The chunk's body text (heading line itself NOT included; the `heading`
    /// field carries that). Trailing/leading blank lines are trimmed.
    pub body: String,
    /// Soft `chars / 4` token estimate. Used by the chunker to decide
    /// further splitting / merging; consumers can ignore it.
    pub estimated_tokens: usize,
    /// PDF page span this chunk covers, when the source body carries
    /// `<!-- kb:page N -->` markers (bn-3ij3). `None` for sources without
    /// page metadata (existing markdown, transcripts, web articles, …).
    /// Inclusive on both ends: `(7, 7)` means "lives entirely on page 7";
    /// `(7, 9)` means "spans pages 7 through 9 inclusive".
    pub page_range: Option<(u32, u32)>,
}

impl Chunk {
    fn new(heading: Option<String>, body: &str) -> Self {
        let trimmed = body.trim().to_string();
        let estimated_tokens = estimate_tokens(&trimmed);
        Self {
            heading,
            body: trimmed,
            estimated_tokens,
            page_range: None,
        }
    }

}

#[inline]
fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / CHARS_PER_TOKEN
}

/// Chunk a markdown body into per-section pieces (see module docs).
///
/// Empty bodies, single-section bodies, and concept-style pages with no
/// H2/H3 all produce sensible output:
///
/// - Empty input → empty `Vec`.
/// - No H2 at all → single chunk with `heading: None` containing the
///   whole body (subject to the > 800 token paragraph fallback).
/// - Mixed H2/H3 → an entry per H2 section; an oversized section gets
///   re-split by its H3 children, then by paragraph if any H3 chunk is
///   still too big.
///
/// bn-3ij3: when the body carries `<!-- kb:page N -->` page markers
/// (emitted by page-aware PDF ingest), each chunk's `page_range` field is
/// populated and chunks never span pages — the page boundary is treated
/// as a hard boundary, with the merge pass collapsing small pages into
/// their neighbor only if doing so keeps a single chunk on a single page
/// or a contiguous page range.
#[must_use]
pub fn chunk_markdown(body: &str) -> Vec<Chunk> {
    let cleaned = strip_managed_region_fences(body);

    // Page-aware path: if the body has `<!-- kb:page N -->` markers, slice
    // by page first and chunk each page independently. The merge pass
    // collapses tiny adjacent pages into a contiguous range so a one-line
    // page (a chapter heading on its own page, say) doesn't end up as a
    // standalone chunk.
    let pages = split_by_page_markers(&cleaned);
    if pages.iter().any(|p| p.page_range.is_some()) {
        let mut chunks: Vec<Chunk> = Vec::new();
        for page in pages {
            for chunk in chunk_section(&page.body, None) {
                let mut c = chunk;
                c.page_range = page.page_range;
                chunks.push(c);
            }
        }
        return merge_short_chunks(chunks);
    }

    // Non-paged path: legacy H2/H3/paragraph chunking.
    let chunks = chunk_section(&cleaned, None);
    merge_short_chunks(chunks)
}

/// Chunk a single contiguous markdown body (already stripped of managed-
/// region fences and page markers). The optional `outer_heading` is used
/// when the caller wants to label every chunk with the same H2 heading
/// (e.g. when a single page's body is being chunked but the page lives
/// inside an outer section). Today every call site passes `None` and lets
/// the inner heading detection do the labeling.
fn chunk_section(body: &str, outer_heading: Option<&str>) -> Vec<Chunk> {
    let sections = split_by_heading(body, 2);

    let mut chunks: Vec<Chunk> = Vec::new();
    for section in sections {
        let section_label = section
            .heading
            .clone()
            .or_else(|| outer_heading.map(str::to_string));
        let section_chunk = Chunk::new(section_label.clone(), &section.body);
        if section_chunk.body.is_empty() {
            // A bare H2 with no body under it. Drop — embedding noise.
            continue;
        }
        if section_chunk.estimated_tokens <= TARGET_MAX_TOKENS {
            chunks.push(section_chunk);
            continue;
        }

        // Section too big — split by H3 first.
        let h3_sections = split_by_heading(&section.body, 3);

        // If splitting by H3 didn't yield any sub-headings (nothing matched
        // `^### `), fall straight through to paragraph splitting on the
        // entire section body.
        let h3_has_real_split = h3_sections.iter().any(|s| s.heading.is_some());
        if !h3_has_real_split {
            for piece in split_by_paragraph(&section.body) {
                let c = Chunk::new(section_label.clone(), &piece);
                if !c.body.is_empty() {
                    chunks.push(c);
                }
            }
            continue;
        }

        for sub in h3_sections {
            // The pre-first-H3 prose under the H2 keeps the H2 heading;
            // each H3's chunks adopt the H3 heading.
            let label = sub.heading.clone().or_else(|| section_label.clone());
            let sub_chunk = Chunk::new(label.clone(), &sub.body);
            if sub_chunk.body.is_empty() {
                continue;
            }
            if sub_chunk.estimated_tokens <= TARGET_MAX_TOKENS {
                chunks.push(sub_chunk);
                continue;
            }
            // Still too big — split by paragraph.
            for piece in split_by_paragraph(&sub.body) {
                let c = Chunk::new(label.clone(), &piece);
                if !c.body.is_empty() {
                    chunks.push(c);
                }
            }
        }
    }
    chunks
}

/// One page's slice of the body, produced by [`split_by_page_markers`].
#[derive(Debug, Clone)]
struct PageSection {
    /// `Some((page, page))` when the slice was preceded by a
    /// `<!-- kb:page N -->` marker; `None` for pre-first-marker prose
    /// (which in practice should be empty for ingest-emitted PDFs but
    /// could carry a YAML frontmatter or stray prose for hand-edited
    /// pages).
    page_range: Option<(u32, u32)>,
    body: String,
}

/// Split `body` by `<!-- kb:page N -->` markers. Each marker opens a new
/// `PageSection` whose `page_range` is `(N, N)`; the body of the section
/// is everything between the marker and the next marker (or EOF). Bodies
/// without any marker return a single section with `page_range: None` and
/// the entire body — the caller can then fall through to legacy chunking.
fn split_by_page_markers(body: &str) -> Vec<PageSection> {
    let mut sections: Vec<PageSection> = Vec::new();
    let mut current = PageSection {
        page_range: None,
        body: String::new(),
    };
    for line in body.lines() {
        if let Some(page) = parse_page_marker(line) {
            // Push the in-progress section before starting a new one. We
            // tolerate empty pre-marker prose by dropping the leading
            // empty section in the post-pass below.
            sections.push(std::mem::replace(
                &mut current,
                PageSection {
                    page_range: Some((page, page)),
                    body: String::new(),
                },
            ));
        } else {
            current.body.push_str(line);
            current.body.push('\n');
        }
    }
    sections.push(current);

    // Drop the leading no-page section when its body is empty (the common
    // case for PDFs ingested via the page-aware path). We keep it when
    // non-empty so frontmatter-style preamble doesn't silently disappear.
    if let Some(first) = sections.first()
        && first.page_range.is_none()
        && first.body.trim().is_empty()
    {
        sections.remove(0);
    }
    sections
}

/// Recognize `<!-- kb:page N -->` (with arbitrary surrounding ASCII
/// whitespace inside the comment). Returns `Some(N)` on match.
fn parse_page_marker(line: &str) -> Option<u32> {
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix("<!--")?.strip_suffix("-->")?;
    let inner = inner.trim();
    let rest = inner.strip_prefix("kb:page")?;
    rest.trim().parse::<u32>().ok()
}

/// Strip managed-region fences (`<!-- kb:begin id=... -->` / `<!-- kb:end id=... -->`)
/// from the body but keep the inner content. Mirrors `embed.rs::strip_managed_regions`
/// — kept independent here so the chunker can be unit-tested without owning a
/// full embedding pipeline.
fn strip_managed_region_fences(body: &str) -> String {
    let regions = extract_managed_regions(body);
    if regions.is_empty() {
        return body.to_string();
    }

    let mut out = String::with_capacity(body.len());
    let mut cursor = 0_usize;
    for region in &regions {
        if region.full_start >= cursor {
            out.push_str(&body[cursor..region.full_start]);
        }
        if region.content_start <= region.content_end {
            out.push_str(&body[region.content_start..region.content_end]);
        }
        cursor = region.full_end;
    }
    if cursor < body.len() {
        out.push_str(&body[cursor..]);
    }
    out
}

#[derive(Debug, Clone)]
struct HeadingSection {
    /// `Some(heading text)` for sections with a heading at the requested level;
    /// `None` for the pre-first-heading prose at the top of the input.
    heading: Option<String>,
    /// Body text of this section, NOT including the heading line itself.
    body: String,
}

/// Split `text` by ATX-style `#`-prefixed headings of exactly `level`.
///
/// Lines inside fenced code blocks (```` ``` ````/`~~~`) are not treated
/// as headings even when they begin with `#`.
///
/// The first section's `heading` is `None` when the input starts before the
/// first matching heading (i.e. there is pre-heading prose). Otherwise every
/// returned section carries a `Some(heading_text)`.
fn split_by_heading(text: &str, level: usize) -> Vec<HeadingSection> {
    let prefix = "#".repeat(level);
    let mut sections: Vec<HeadingSection> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_body = String::new();
    let mut in_fence = false;

    for line in text.lines() {
        let trimmed = line.trim_start();

        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            current_body.push_str(line);
            current_body.push('\n');
            continue;
        }

        let is_heading = !in_fence && is_heading_at_level(trimmed, &prefix);
        if is_heading {
            if !current_body.is_empty() || current_heading.is_some() {
                sections.push(HeadingSection {
                    heading: current_heading.take(),
                    body: std::mem::take(&mut current_body),
                });
            }
            // `prefix` matches `#` * level; capture the heading text.
            let after_hashes = &trimmed[prefix.len()..];
            let heading_text = after_hashes.trim().trim_end_matches('#').trim().to_string();
            current_heading = Some(heading_text);
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    sections.push(HeadingSection {
        heading: current_heading,
        body: current_body,
    });

    // Drop a leading prose-only section that's empty (the input started with
    // a heading at the requested level — there is no pre-heading prose).
    if let Some(first) = sections.first()
        && first.heading.is_none()
        && first.body.trim().is_empty()
    {
        sections.remove(0);
    }
    sections
}

/// Return true when `line` is an ATX heading at exactly the given prefix
/// level (`prefix` is `"##"` or `"###"`). Heading detection requires the
/// hashes to be followed by whitespace; this rejects e.g. `###tag` and
/// reflects the markdown spec.
fn is_heading_at_level(line: &str, prefix: &str) -> bool {
    if !line.starts_with(prefix) {
        return false;
    }
    let after = &line[prefix.len()..];
    // Reject deeper headings (e.g. `###` when looking for `##`). We only
    // match exactly this level.
    if after.starts_with('#') {
        return false;
    }
    // Heading text must be separated from the hashes by whitespace.
    after.starts_with(' ') || after.starts_with('\t')
}

/// Split `text` by blank-line-separated paragraphs. Each returned `String`
/// is one paragraph's text with no leading/trailing whitespace. Empty
/// paragraphs are dropped.
fn split_by_paragraph(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !buf.trim().is_empty() {
                out.push(buf.trim().to_string());
            }
            buf.clear();
        } else {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
        }
    }
    if !buf.trim().is_empty() {
        out.push(buf.trim().to_string());
    }
    out
}

/// Walk `chunks` and merge any chunk under [`TARGET_MIN_TOKENS`] into the
/// previous chunk. When the first chunk is short, it's merged into the
/// next chunk instead. Invariant: never returns a chunk with
/// `estimated_tokens < TARGET_MIN_TOKENS` unless that's the only chunk
/// left (a tiny single-chunk page is the user's natural state).
fn merge_short_chunks(chunks: Vec<Chunk>) -> Vec<Chunk> {
    if chunks.len() <= 1 {
        return chunks;
    }

    let mut out: Vec<Chunk> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        if chunk.estimated_tokens < TARGET_MIN_TOKENS
            && let Some(prev) = out.last_mut()
        {
            merge_into(prev, &chunk);
            continue;
        }
        // Either above the threshold, or below it but no previous chunk
        // exists yet (the first short pre-H2 prose). Keep it for now —
        // the post-pass below folds a leading short chunk forward.
        out.push(chunk);
    }

    // Post-pass: if the FIRST chunk is short and there's a next, fold it
    // forward. (`merge_short_chunks` runs once; if multiple consecutive
    // shorts pile up at the head this still handles them by chained
    // forward-merge.)
    while out.len() >= 2 && out[0].estimated_tokens < TARGET_MIN_TOKENS {
        let head = out.remove(0);
        let next = &mut out[0];
        // Prepend `head.body` so the surviving chunk reads in source order.
        let mut merged_body = head.body;
        if !merged_body.is_empty() && !next.body.is_empty() {
            merged_body.push_str("\n\n");
        }
        merged_body.push_str(&next.body);
        // Keep `next.heading` — it's the more specific one (or the only one);
        // `head.heading` was usually `None` anyway (pre-first-H2 prose).
        next.body = merged_body;
        next.estimated_tokens = estimate_tokens(&next.body);
        // Forward-merge widens the page range too: the surviving chunk now
        // covers both pages.
        next.page_range = merge_page_ranges(head.page_range, next.page_range);
    }

    out
}

fn merge_into(target: &mut Chunk, src: &Chunk) {
    if !target.body.is_empty() && !src.body.is_empty() {
        target.body.push_str("\n\n");
    }
    target.body.push_str(&src.body);
    target.estimated_tokens = estimate_tokens(&target.body);
    // Preserve `target.heading`; merging short tail chunks shouldn't
    // promote the merger's heading into the merged-into chunk.
    // Extend the page range to cover both contributors. A merge across a
    // page boundary turns a single-page chunk into a multi-page chunk;
    // citation rendering then emits `pp.M-N` instead of `p.M`.
    target.page_range = merge_page_ranges(target.page_range, src.page_range);
}

/// Merge two optional page ranges. `(a, b) ⊕ (c, d) = (min(a,c), max(b,d))`.
/// One-sided `None` returns the other side unchanged; both-`None` is
/// itself `None`.
fn merge_page_ranges(
    a: Option<(u32, u32)>,
    b: Option<(u32, u32)>,
) -> Option<(u32, u32)> {
    match (a, b) {
        (Some((a0, a1)), Some((b0, b1))) => Some((a0.min(b0), a1.max(b1))),
        (Some(r), None) | (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::unnecessary_wraps)]
    fn h(text: &str) -> Option<String> {
        Some(text.to_owned())
    }

    #[test]
    fn empty_body_produces_no_chunks() {
        assert!(chunk_markdown("").is_empty());
        assert!(chunk_markdown("\n\n  \n").is_empty());
    }

    #[test]
    fn body_without_h2_is_a_single_chunk_no_heading() {
        let body = "Just some prose.\n\nA second paragraph.\n";
        let chunks = chunk_markdown(body);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, None);
        assert!(chunks[0].body.contains("Just some prose."));
        assert!(chunks[0].body.contains("second paragraph"));
    }

    #[test]
    fn body_with_mixed_h2_and_h3_splits_per_h2() {
        // Three H2 sections, each comfortably above TARGET_MIN_TOKENS so
        // the merger doesn't collapse them. The middle one carries an H3
        // — without forcing the H3-split path (we'd need the H2 body
        // itself > TARGET_MAX_TOKENS for that), the H3 just rides along
        // inside Beta's body, which is the right semantics for a healthy
        // section that already fits the embedding window.
        let bulk = "lorem ipsum dolor sit amet ".repeat(20); // ~540 chars per section
        let body = format!(
            "Some lead-in prose here.\n\n\
             ## Alpha section\n\n{bulk}\n\n\
             ## Beta section\n\n{bulk}\n\n\
             ### Beta sub\n\n{bulk}\n\n\
             ## Gamma section\n\n{bulk}\n",
        );
        let chunks = chunk_markdown(&body);
        let headings: Vec<Option<String>> = chunks.iter().map(|c| c.heading.clone()).collect();
        assert_eq!(
            headings,
            vec![h("Alpha section"), h("Beta section"), h("Gamma section")],
            "expected one chunk per H2 section"
        );
        // Lead-in prose was tiny → merged forward into the first H2 chunk.
        assert!(chunks[0].body.contains("lead-in"));
        // Beta's body and its inline H3 sub-body both ride along on the
        // Beta chunk (the H2 section as a whole fits the embedding window
        // so H3 splitting doesn't fire).
        assert!(chunks[1].body.contains("### Beta sub"));
    }

    #[test]
    fn long_section_without_h3_falls_back_to_paragraph_splits() {
        // One H2 section over 800 tokens (~3200 chars) with no H3 — must
        // split by paragraph instead of returning one giant chunk.
        let para = "x ".repeat(700); // ~1400 chars per paragraph
        let body = format!(
            "## Big section\n\n{para}\n\n{para}\n\n{para}\n\n{para}\n",
        );
        let chunks = chunk_markdown(&body);
        assert!(
            chunks.len() >= 2,
            "expected paragraph fallback to produce >1 chunk, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert_eq!(c.heading, h("Big section"));
        }
    }

    #[test]
    fn long_section_with_h3_splits_by_h3() {
        // One huge H2 with two H3s under it. The chunker should produce a
        // chunk per H3 branch — and the chunks should carry the H3
        // heading text, not the H2's.
        let big_para = "y ".repeat(1800); // ~3600 chars, one chunk on its own
        let body = format!(
            "## Top heading\n\n\
             ### First sub\n\n\
             {big_para}\n\n\
             ### Second sub\n\n\
             {big_para}\n",
        );
        let chunks = chunk_markdown(&body);
        let labels: Vec<Option<String>> = chunks.iter().map(|c| c.heading.clone()).collect();
        // We expect at least one chunk per H3, possibly more if a single
        // H3's body is itself > TARGET_MAX_TOKENS (it is in this fixture).
        assert!(labels.iter().any(|l| l.as_deref() == Some("First sub")));
        assert!(labels.iter().any(|l| l.as_deref() == Some("Second sub")));
    }

    #[test]
    fn short_trailing_chunk_is_merged_upward() {
        // Two H2s; the second is tiny. The merger folds the tail into the
        // previous chunk so we don't keep a 5-token chunk that pollutes
        // retrieval.
        let big_para = "z ".repeat(500); // ~1000 chars
        let body = format!(
            "## Main\n\n{big_para}\n\n## Tiny\n\nstub.\n",
        );
        let chunks = chunk_markdown(&body);
        // After merging the tail, only one chunk survives — and it's the
        // Main chunk with the tail's body appended.
        assert_eq!(chunks.len(), 1, "got {chunks:#?}");
        assert_eq!(chunks[0].heading, h("Main"));
        assert!(chunks[0].body.contains("stub."));
    }

    #[test]
    fn short_leading_pre_h2_prose_is_merged_into_first_section() {
        // A 2-line lead-in followed by a healthy H2 — the lead-in alone
        // is well under 100 tokens, so the merger should fold it into
        // the first H2 chunk.
        let big_para = "w ".repeat(500);
        let body = format!(
            "Lead-in line.\n\n## First\n\n{big_para}\n",
        );
        let chunks = chunk_markdown(&body);
        assert_eq!(chunks.len(), 1);
        // The merged chunk keeps the H2's heading.
        assert_eq!(chunks[0].heading, h("First"));
        assert!(chunks[0].body.contains("Lead-in line."));
    }

    #[test]
    fn managed_region_fences_are_stripped_but_content_kept() {
        let body = "## Section\n\n\
                    <!-- kb:begin id=summary -->managed body<!-- kb:end id=summary -->\n\
                    \n\
                    More prose.\n";
        let chunks = chunk_markdown(body);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].body.contains("managed body"));
        assert!(!chunks[0].body.contains("kb:begin"));
        assert!(!chunks[0].body.contains("kb:end"));
    }

    #[test]
    fn fenced_code_with_hashes_is_not_treated_as_heading() {
        // A `## comment` inside a fenced code block must not split the chunk.
        let body = "## Real heading\n\n\
                    ```\n\
                    ## not a heading inside code\n\
                    ```\n\n\
                    Body text.\n";
        let chunks = chunk_markdown(body);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, h("Real heading"));
        assert!(chunks[0].body.contains("not a heading inside code"));
    }

    #[test]
    fn heading_with_trailing_hashes_is_normalized() {
        let body = "## Title ##\n\nbody.\n";
        let chunks = chunk_markdown(body);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, h("Title"));
    }

    #[test]
    fn empty_section_under_a_heading_is_dropped() {
        // `## A` with no body, then `## B` with content. The empty section
        // shouldn't show up as a zero-body chunk in the output.
        let body = "## A\n\n## B\n\nB body content.\n";
        let chunks = chunk_markdown(body);
        let headings: Vec<Option<String>> = chunks.iter().map(|c| c.heading.clone()).collect();
        assert!(!headings.iter().any(|l| l.as_deref() == Some("A")));
        assert!(headings.iter().any(|l| l.as_deref() == Some("B")));
    }

    /// bn-3ij3: when the body has no page markers, every chunk has
    /// `page_range: None` and the legacy chunking behavior is preserved.
    #[test]
    fn chunks_without_page_markers_have_no_page_range() {
        let body = "## Section\n\nbody text\n";
        let chunks = chunk_markdown(body);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page_range, None);
    }

    /// bn-3ij3: page markers split the body so each chunk's `page_range`
    /// matches the page it lives on. Healthy multi-page sources produce
    /// one chunk per page when each page is comfortably above MIN tokens.
    #[test]
    fn chunks_with_page_markers_carry_per_page_range() {
        let big_para = "alpha ".repeat(120); // ~720 chars per page
        let body = format!(
            "<!-- kb:page 1 -->\n{big_para}\n\n<!-- kb:page 2 -->\n{big_para}\n",
        );
        let chunks = chunk_markdown(&body);
        assert_eq!(chunks.len(), 2, "two healthy pages should be two chunks");
        assert_eq!(chunks[0].page_range, Some((1, 1)));
        assert_eq!(chunks[1].page_range, Some((2, 2)));
    }

    /// bn-3ij3: a chunk never spans pages unless a small page is merged
    /// upward into the next chunk by [`merge_short_chunks`]. When a page's
    /// body is too small to stand alone, the merger folds it into a
    /// neighbor and the surviving chunk's range covers both pages.
    #[test]
    fn small_page_merges_upward_and_widens_range() {
        let big_para = "z ".repeat(500); // ~1000 chars
        // Page 1 is a tiny chapter heading; page 2 is a fat body. The
        // merger folds page 1 into page 2 so the surviving chunk's range
        // is (1, 2).
        let body = format!(
            "<!-- kb:page 1 -->\nshort intro line.\n\n<!-- kb:page 2 -->\n{big_para}\n",
        );
        let chunks = chunk_markdown(&body);
        assert_eq!(chunks.len(), 1, "tiny page-1 should fold into page-2");
        assert_eq!(chunks[0].page_range, Some((1, 2)));
        assert!(chunks[0].body.contains("short intro line"));
    }

    /// bn-3ij3: a page boundary is a HARD boundary for healthy chunks.
    /// Two adjacent healthy pages must NOT be merged by the H2-style
    /// section logic — the page split happens before H2 chunking, so
    /// page 1 and page 2 produce independent chunks even when they share
    /// the same H2.
    #[test]
    fn page_boundary_is_a_hard_boundary_for_healthy_chunks() {
        let big_para = "alpha ".repeat(120);
        // Same H2 on both pages (a multi-page section); chunker still
        // emits two chunks, one per page, both labelled with the H2.
        let body = format!(
            "<!-- kb:page 1 -->\n## Topic\n\n{big_para}\n\n<!-- kb:page 2 -->\n## Topic\n\n{big_para}\n",
        );
        let chunks = chunk_markdown(&body);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_range, Some((1, 1)));
        assert_eq!(chunks[1].page_range, Some((2, 2)));
        assert_eq!(chunks[0].heading.as_deref(), Some("Topic"));
        assert_eq!(chunks[1].heading.as_deref(), Some("Topic"));
    }

    /// bn-3ij3: a page that's larger than `TARGET_MAX_TOKENS` still gets
    /// split internally (by H2/H3/paragraph), and every produced chunk
    /// keeps the same `page_range` — splitting WITHIN a page doesn't widen
    /// the range.
    #[test]
    fn oversized_page_splits_internally_keeping_one_page() {
        let para = "x ".repeat(700); // ~1400 chars per paragraph
        let body = format!(
            "<!-- kb:page 1 -->\n## Big section\n\n{para}\n\n{para}\n\n{para}\n\n{para}\n",
        );
        let chunks = chunk_markdown(&body);
        assert!(
            chunks.len() >= 2,
            "expected paragraph-fallback splits, got {}",
            chunks.len()
        );
        for chunk in &chunks {
            assert_eq!(
                chunk.page_range,
                Some((1, 1)),
                "internal splits must keep the page range",
            );
        }
    }

    #[test]
    fn parse_page_marker_recognizes_inline_form() {
        assert_eq!(parse_page_marker("<!-- kb:page 7 -->"), Some(7));
        assert_eq!(parse_page_marker("  <!-- kb:page 12 -->  "), Some(12));
        // Non-marker comments are not pages.
        assert_eq!(parse_page_marker("<!-- kb:begin id=summary -->"), None);
        assert_eq!(parse_page_marker("regular line"), None);
    }

    #[test]
    fn merge_page_ranges_combines_endpoints() {
        assert_eq!(merge_page_ranges(None, None), None);
        assert_eq!(merge_page_ranges(Some((3, 5)), None), Some((3, 5)));
        assert_eq!(merge_page_ranges(None, Some((3, 5))), Some((3, 5)));
        assert_eq!(
            merge_page_ranges(Some((3, 5)), Some((4, 7))),
            Some((3, 7)),
        );
        assert_eq!(
            merge_page_ranges(Some((9, 9)), Some((1, 2))),
            Some((1, 9)),
        );
    }

    #[test]
    fn chunk_estimated_tokens_is_chars_over_four() {
        let body = "## Heading\n\n".to_string() + &"x".repeat(400);
        let chunks = chunk_markdown(&body);
        assert_eq!(chunks.len(), 1);
        // 400 chars / 4 = 100 tokens. Allow ±a few for the heading-bearing
        // body's exact char count after trim.
        assert!(
            chunks[0].estimated_tokens >= 95 && chunks[0].estimated_tokens <= 105,
            "got {}",
            chunks[0].estimated_tokens
        );
    }
}
