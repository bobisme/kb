//! Image asset resolution for retrieval candidates.
//!
//! bn-3dkw: when a retrieval candidate's body references a local image
//! (e.g. `![diagram](assets/foo.png)` in a normalized source, or
//! `![fig](../../.kb/normalized/src-xyz/assets/bar.png)` in a compiled wiki
//! page), we want to hand those image files to multimodal LLMs alongside
//! the prompt — not just leave the text reference inert.
//!
//! This module walks the candidate list, scans each candidate's file on
//! disk for markdown image references, resolves them to absolute paths
//! (relative to the right anchor — normalized source vs. wiki page), and
//! returns the deduplicated list, capped to avoid blowing the context.
//!
//! # Scope guardrails
//! - Only bodies of selected candidates are scanned; the full corpus is never
//!   walked.
//! - URL-scheme refs (http, https, data) are ignored.
//! - Refs that don't resolve to an existing file on disk are silently
//!   dropped — the markdown text still reaches the LLM verbatim, the image
//!   just isn't attached.
//! - The first [`MAX_IMAGES_PER_QUERY`] resolvable images are returned; the
//!   rest are dropped in order so the context doesn't blow up.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use kb_core::normalized_dir;
use regex::Regex;

use crate::lexical::RetrievalPlan;

/// Hard cap on images per `kb ask` invocation. Keeps the attachment payload
/// bounded even for plans that sweep up many image-heavy pages.
pub const MAX_IMAGES_PER_QUERY: usize = 5;

/// Matches `![alt](path)` image references in markdown. Mirrors the scanners
/// in `kb_ingest::image_refs` and `kb_compile::source_page` — kept local here
/// so this crate doesn't have to depend on either.
///
/// Capture groups:
/// - group 1: angle-bracket-wrapped path (`<path with spaces.png>`) when that
///   form is used — spaces OK, no unescaped `>` inside.
/// - group 2: plain path (`path.png`) — no whitespace, no angle brackets.
///
/// An optional `CommonMark` title (`"caption"`) after the path is tolerated
/// and discarded. See bn-18qs: Obsidian emits the angle-bracket form when a
/// filename contains spaces, and the old `[^)\s]+` pattern missed them
/// entirely — so multimodal retrieval never saw those screenshots.
static IMAGE_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^)\s"]+))(?:\s+"[^"]*")?\s*\)"#)
        .expect("hard-coded image regex is valid")
});

/// Extract the raw path from an [`IMAGE_REF_RE`] capture, trimming any
/// surrounding whitespace. Returns `None` if neither capture group is
/// present.
fn captured_raw_path<'a>(capture: &regex::Captures<'a>) -> Option<&'a str> {
    capture
        .get(1)
        .or_else(|| capture.get(2))
        .map(|m| m.as_str().trim())
}

/// Resolve image attachments referenced by the pages in `plan`, relative to
/// the KB `root`.
///
/// Returns absolute paths to on-disk image files, in first-seen order,
/// deduplicated, and capped at [`MAX_IMAGES_PER_QUERY`].
///
/// Callers that know no candidate can have images (empty plan, or a
/// pre-check via [`plan_mentions_images`]) should skip this call entirely —
/// it's a cheap walk but still touches disk.
///
/// For `wiki/sources/src-<id>.md` candidates, the wiki page is an LLM-
/// generated summary that frequently drops the original `![](assets/…)`
/// references from the raw source. So we additionally scan
/// `normalized/<id>/source.md` (resolving refs relative to
/// `normalized/<id>/`) — that's where the ingest-time image refs actually
/// live. Concept pages (`wiki/concepts/*.md`) get no such fallback:
/// concepts don't own sources, and any image a concept cites will be
/// picked up when the source itself is also a candidate.
#[must_use]
pub fn resolve_candidate_image_paths(root: &Path, plan: &RetrievalPlan) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for candidate in &plan.candidates {
        if out.len() >= MAX_IMAGES_PER_QUERY {
            break;
        }

        // Primary scan: the candidate's wiki page itself.
        let page_path = root.join(&candidate.id);
        scan_file_for_images(&page_path, None, &mut out, &mut seen);
        if out.len() >= MAX_IMAGES_PER_QUERY {
            break;
        }

        // Fallback scan: for `wiki/sources/src-<id>.md` candidates, also
        // scan the normalized source file. The wiki page is a summary that
        // usually drops the original image refs.
        if let Some(source_id) = wiki_source_id(&candidate.id) {
            let source_root = normalized_dir(root).join(source_id);
            let source_path = source_root.join("source.md");
            scan_file_for_images(&source_path, Some(&source_root), &mut out, &mut seen);
        }
    }

    out
}

/// True when any candidate's on-disk body mentions at least one image ref.
///
/// Used as the zero-cost fast-path guard: if no candidate mentions any
/// image, the ask pipeline can skip image resolution and adapter-side
/// attachment plumbing entirely.
///
/// Matches [`resolve_candidate_image_paths`]'s scope: for
/// `wiki/sources/src-<id>.md` candidates we also check
/// `normalized/<id>/source.md`, since the wiki summary frequently drops
/// the original refs.
#[must_use]
pub fn plan_mentions_images(root: &Path, plan: &RetrievalPlan) -> bool {
    for candidate in &plan.candidates {
        let page_path = root.join(&candidate.id);
        if file_contains_image_ref(&page_path) {
            return true;
        }
        if let Some(source_id) = wiki_source_id(&candidate.id) {
            let source_path = normalized_dir(root)
                .join(source_id)
                .join("source.md");
            if file_contains_image_ref(&source_path) {
                return true;
            }
        }
    }
    false
}

/// If `candidate_id` looks like `wiki/sources/src-<id>.md`, return the
/// `<id>` portion (e.g. `src-xyz`). Otherwise `None`.
///
/// The candidate id uses forward slashes regardless of OS, since it comes
/// from the KB's compiled index, not local filesystem traversal.
fn wiki_source_id(candidate_id: &str) -> Option<&str> {
    let rest = candidate_id.strip_prefix("wiki/sources/")?;
    let stem = rest.strip_suffix(".md")?;
    // Only `src-*` files are sources we ingest; ignore other wiki/sources
    // files (e.g. an index) that don't have a matching normalized dir.
    if stem.starts_with("src-") { Some(stem) } else { None }
}

/// Read `path` and append resolved image refs into `out`, respecting the
/// global [`MAX_IMAGES_PER_QUERY`] cap and deduping against `seen`.
///
/// `anchor_override`, when set, is used as the base dir for relative refs;
/// otherwise the file's parent is used. This lets the normalized-source
/// fallback resolve refs against `normalized/<id>/` rather than the
/// source.md file itself (which is equivalent in this case, but keeps the
/// intent explicit).
fn scan_file_for_images(
    path: &Path,
    anchor_override: Option<&Path>,
    out: &mut Vec<PathBuf>,
    seen: &mut HashSet<PathBuf>,
) {
    let Ok(body) = std::fs::read_to_string(path) else {
        return;
    };
    let anchor = anchor_override
        .or_else(|| path.parent())
        .unwrap_or_else(|| Path::new("."));

    for capture in IMAGE_REF_RE.captures_iter(&body) {
        if out.len() >= MAX_IMAGES_PER_QUERY {
            break;
        }
        let Some(raw_path) = captured_raw_path(&capture) else {
            continue;
        };
        let Some(resolved) = resolve_image_ref(raw_path, anchor) else {
            continue;
        };
        if seen.insert(resolved.clone()) {
            out.push(resolved);
        }
    }
}

/// True when the file at `path` exists and contains at least one markdown
/// image ref. Missing/unreadable files are treated as empty.
fn file_contains_image_ref(path: &Path) -> bool {
    std::fs::read_to_string(path).is_ok_and(|body| IMAGE_REF_RE.is_match(&body))
}

/// Resolve a raw image reference to an absolute on-disk path, or return
/// `None` if it's a URL / data URI / broken reference.
fn resolve_image_ref(raw_path: &str, page_parent: &Path) -> Option<PathBuf> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return None;
    }

    // External / non-local schemes.
    let head_len = trimmed.len().min(8);
    let head = trimmed.get(..head_len).unwrap_or("").to_ascii_lowercase();
    if head.starts_with("http://")
        || head.starts_with("https://")
        || head.starts_with("data:")
        || trimmed.starts_with('#')
    {
        return None;
    }

    let candidate = Path::new(trimmed);
    let absolute = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        page_parent.join(candidate)
    };

    // Canonicalize when the target exists; fall back to the un-canonicalized
    // form otherwise (so callers can diagnose missing refs if they want to).
    match std::fs::canonicalize(&absolute) {
        Ok(abs) if abs.is_file() => Some(abs),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::RetrievalCandidate;
    use std::fmt::Write as _;
    use std::fs;
    use tempfile::TempDir;

    fn make_plan(candidates: Vec<RetrievalCandidate>) -> RetrievalPlan {
        RetrievalPlan {
            query: "test".to_string(),
            token_budget: 5000,
            estimated_tokens: 0,
            candidates,
            fallback_reason: None,
        }
    }

    #[test]
    fn resolves_wiki_source_image_to_normalized_assets() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        // Lay down a fake KB with one normalized source + its wiki page.
        let normalized_dir = normalized_dir(root).join("src-xyz/assets");
        fs::create_dir_all(&normalized_dir).expect("normalized assets");
        fs::write(normalized_dir.join("foo.png"), b"\x89PNG").expect("write png");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-xyz.md"),
            "---\ntype: source\n---\n![d](../../.kb/normalized/src-xyz/assets/foo.png)\n",
        )
        .expect("write wiki page");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/sources/src-xyz.md".to_string(),
            title: "src-xyz".to_string(),
            score: 10,
            estimated_tokens: 100,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), 1, "one image resolved: {images:?}");
        assert!(images[0].ends_with("foo.png"));
        assert!(images[0].is_absolute(), "absolute path: {:?}", images[0]);
    }

    #[test]
    fn skips_external_urls() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("c.md"),
            "![](https://example.com/x.png)\n![](data:image/png;base64,AAAA)\n",
        )
        .expect("write");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/concepts/c.md".to_string(),
            title: "c".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert!(images.is_empty(), "no images from URLs: {images:?}");
    }

    #[test]
    fn silently_drops_broken_refs() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(wiki_dir.join("c.md"), "![nope](does-not-exist.png)\n")
            .expect("write");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/concepts/c.md".to_string(),
            title: "c".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert!(images.is_empty());
    }

    #[test]
    fn dedupes_same_image_across_candidates() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let normalized_dir = normalized_dir(root).join("src-a/assets");
        fs::create_dir_all(&normalized_dir).expect("normalized dir");
        fs::write(normalized_dir.join("shared.png"), b"PNG").expect("png");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-a.md"),
            "![x](../../.kb/normalized/src-a/assets/shared.png)\n",
        )
        .expect("src-a");
        fs::write(
            wiki_dir.join("src-b.md"),
            "![y](../../.kb/normalized/src-a/assets/shared.png)\n",
        )
        .expect("src-b");

        let plan = make_plan(vec![
            RetrievalCandidate {
                id: "wiki/sources/src-a.md".to_string(),
                title: "a".to_string(),
                score: 3,
                estimated_tokens: 10,
                reasons: Vec::new(),
            },
            RetrievalCandidate {
                id: "wiki/sources/src-b.md".to_string(),
                title: "b".to_string(),
                score: 2,
                estimated_tokens: 10,
                reasons: Vec::new(),
            },
        ]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), 1, "deduplicated: {images:?}");
    }

    #[test]
    fn caps_at_max_images() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let normalized_dir = normalized_dir(root).join("src/assets");
        fs::create_dir_all(&normalized_dir).expect("normalized dir");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");

        // 10 images > MAX_IMAGES_PER_QUERY (5).
        let mut body = String::new();
        for i in 0..10 {
            let name = format!("i{i}.png");
            fs::write(normalized_dir.join(&name), b"PNG").expect("png");
            writeln!(body, "![](../../.kb/normalized/src/assets/{name})")
                .expect("write to string never fails");
        }
        fs::write(wiki_dir.join("src.md"), body).expect("src page");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/sources/src.md".to_string(),
            title: "s".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), MAX_IMAGES_PER_QUERY);
    }

    #[test]
    fn plan_mentions_images_returns_true_when_any_candidate_has_image_ref() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(wiki_dir.join("a.md"), "plain text, no image\n").expect("a");
        fs::write(wiki_dir.join("b.md"), "![x](assets/x.png)\n").expect("b");

        let plan = make_plan(vec![
            RetrievalCandidate {
                id: "wiki/concepts/a.md".to_string(),
                title: "a".to_string(),
                score: 1,
                estimated_tokens: 10,
                reasons: Vec::new(),
            },
            RetrievalCandidate {
                id: "wiki/concepts/b.md".to_string(),
                title: "b".to_string(),
                score: 1,
                estimated_tokens: 10,
                reasons: Vec::new(),
            },
        ]);

        assert!(plan_mentions_images(root, &plan));
    }

    #[test]
    fn plan_mentions_images_returns_false_when_no_candidate_has_image_ref() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(wiki_dir.join("a.md"), "no images here\n").expect("a");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/concepts/a.md".to_string(),
            title: "a".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        assert!(!plan_mentions_images(root, &plan));
    }

    // bn-3nvm: wiki source summaries drop image refs. For `wiki/sources/src-X.md`
    // candidates we must also scan `normalized/<X>/source.md`.

    #[test]
    fn resolves_image_from_normalized_source_when_wiki_summary_dropped_ref() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        // Normalized source has the image ref; asset lives beside it.
        let normalized_dir = normalized_dir(root).join("src-xyz");
        fs::create_dir_all(normalized_dir.join("assets")).expect("assets");
        fs::write(normalized_dir.join("assets/foo.png"), b"\x89PNG").expect("png");
        fs::write(
            normalized_dir.join("source.md"),
            "# source\n\n![diagram](assets/foo.png)\n",
        )
        .expect("source.md");

        // Wiki summary has NO image ref (this is the real-world bug).
        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-xyz.md"),
            "---\ntype: source\n---\n\nSummary text without image refs.\n",
        )
        .expect("wiki page");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/sources/src-xyz.md".to_string(),
            title: "src-xyz".to_string(),
            score: 10,
            estimated_tokens: 100,
            reasons: Vec::new(),
        }]);

        assert!(
            plan_mentions_images(root, &plan),
            "plan_mentions_images must detect ref in normalized source.md",
        );

        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), 1, "one image resolved: {images:?}");
        assert!(images[0].ends_with("foo.png"));
        assert!(images[0].is_absolute());
    }

    #[test]
    fn normalized_source_and_wiki_page_both_scanned_dedup_and_cap() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        // Enough images across both files to exceed the cap (5).
        // 3 in the wiki page (unique), 3 in the normalized source where
        // one duplicates a wiki-page ref (so 5 distinct images total; cap
        // should still allow all 5 but would cut off any 6th).
        let normalized_dir = normalized_dir(root).join("src-many");
        let normalized_assets = normalized_dir.join("assets");
        fs::create_dir_all(&normalized_assets).expect("normalized assets");
        for i in 0..6 {
            fs::write(normalized_assets.join(format!("i{i}.png")), b"PNG").expect("png");
        }

        // Wiki page refs: i0, i1, i2 via `../../.kb/normalized/src-many/assets/iN.png`.
        let mut wiki_body = String::from("---\ntype: source\n---\n");
        for i in 0..3 {
            writeln!(
                wiki_body,
                "![w{i}](../../.kb/normalized/src-many/assets/i{i}.png)"
            )
            .expect("write");
        }
        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(wiki_dir.join("src-many.md"), wiki_body).expect("wiki page");

        // Normalized source refs: i2 (duplicate), i3, i4, i5 — four refs,
        // three unique (i3, i4, i5) after dedup. Total distinct: 6 (i0..i5)
        // but cap is 5, so we get exactly MAX_IMAGES_PER_QUERY.
        let mut source_body = String::from("# src\n");
        for i in 2..6 {
            writeln!(source_body, "![s{i}](assets/i{i}.png)").expect("write");
        }
        fs::write(normalized_dir.join("source.md"), source_body).expect("source.md");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/sources/src-many.md".to_string(),
            title: "src-many".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(
            images.len(),
            MAX_IMAGES_PER_QUERY,
            "cap honored across both scan paths: {images:?}"
        );
        // First three must be from the wiki page (scanned first).
        assert!(images[0].ends_with("i0.png"));
        assert!(images[1].ends_with("i1.png"));
        assert!(images[2].ends_with("i2.png"));
        // i2 must not be duplicated.
        let i2_count = images.iter().filter(|p| p.ends_with("i2.png")).count();
        assert_eq!(i2_count, 1, "i2 deduplicated");
    }

    #[test]
    fn concept_candidate_does_not_trigger_normalized_fallback() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        // Lay down a normalized source dir that would match if the fallback
        // were mistakenly applied to a concept candidate — verify it doesn't.
        let normalized_dir = normalized_dir(root).join("src-hex");
        fs::create_dir_all(normalized_dir.join("assets")).expect("assets");
        fs::write(normalized_dir.join("assets/bar.png"), b"PNG").expect("png");
        fs::write(
            normalized_dir.join("source.md"),
            "![x](assets/bar.png)\n",
        )
        .expect("source.md");

        // Concept page with no image refs of its own.
        let concept_dir = root.join("wiki/concepts");
        fs::create_dir_all(&concept_dir).expect("concepts");
        fs::write(concept_dir.join("src-hex.md"), "Just a concept.\n")
            .expect("concept page");

        let plan = make_plan(vec![RetrievalCandidate {
            // Even if a concept's stem collides with a source id, the
            // `wiki/concepts/` path must NOT trigger the normalized fallback.
            id: "wiki/concepts/src-hex.md".to_string(),
            title: "hex".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        assert!(
            !plan_mentions_images(root, &plan),
            "concept candidate must not fall through to normalized sources",
        );
        assert!(resolve_candidate_image_paths(root, &plan).is_empty());
    }

    #[test]
    fn wiki_page_image_ref_still_resolves_without_normalized_source() {
        // Regression guard: the existing primary-scan path keeps working
        // even when there is no matching normalized/<id>/source.md on disk.
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        let normalized_assets = normalized_dir(root).join("src-only-wiki/assets");
        fs::create_dir_all(&normalized_assets).expect("assets");
        fs::write(normalized_assets.join("baz.png"), b"PNG").expect("png");
        // Note: no source.md file — the only ref lives in the wiki page.

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-only-wiki.md"),
            "![d](../../.kb/normalized/src-only-wiki/assets/baz.png)\n",
        )
        .expect("wiki page");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/sources/src-only-wiki.md".to_string(),
            title: "s".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        assert!(plan_mentions_images(root, &plan));
        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), 1);
        assert!(images[0].ends_with("baz.png"));
    }

    #[test]
    fn wiki_source_id_parses_expected_forms() {
        assert_eq!(
            wiki_source_id("wiki/sources/src-xyz.md"),
            Some("src-xyz"),
        );
        assert_eq!(
            wiki_source_id("wiki/sources/src-1oz.md"),
            Some("src-1oz"),
        );
        // Non-src files in wiki/sources (e.g. an index) should not map.
        assert_eq!(wiki_source_id("wiki/sources/index.md"), None);
        assert_eq!(wiki_source_id("wiki/concepts/c.md"), None);
        assert_eq!(wiki_source_id("wiki/sources/src-xyz.txt"), None);
    }

    // ---- bn-18qs: angle-bracket form + paths with spaces + titles ----

    /// Obsidian emits `![](<./Screenshot with spaces.png>)`; the ingest-time
    /// rewrite preserves that form, so the normalized source.md we scan here
    /// can contain `![](<assets/Screenshot with spaces.png>)`. We must
    /// detect, extract, and resolve those refs.
    #[test]
    fn angle_bracket_ref_in_normalized_source_resolves() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        let normalized = normalized_dir(root).join("src-ab");
        fs::create_dir_all(normalized.join("assets")).expect("assets");
        fs::write(
            normalized.join("assets/screenshot with spaces.png"),
            b"\x89PNG",
        )
        .expect("png");
        fs::write(
            normalized.join("source.md"),
            "# src\n\n![](<assets/screenshot with spaces.png>)\n",
        )
        .expect("source.md");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-ab.md"),
            "---\ntype: source\n---\nSummary text only.\n",
        )
        .expect("wiki page");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/sources/src-ab.md".to_string(),
            title: "src-ab".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        assert!(
            plan_mentions_images(root, &plan),
            "angle-bracket ref must count as an image mention",
        );
        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), 1, "one image resolved: {images:?}");
        assert!(images[0].ends_with("screenshot with spaces.png"));
    }

    /// `CommonMark` title attributes are tolerated — the title is discarded,
    /// the path still resolves.
    #[test]
    fn title_attribute_tolerated_in_ref() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();

        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        let img = wiki_dir.join("pic.png");
        fs::write(&img, b"PNG").expect("png");
        fs::write(
            wiki_dir.join("c.md"),
            r#"![x](pic.png "caption")"#,
        )
        .expect("c.md");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/concepts/c.md".to_string(),
            title: "c".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert_eq!(images.len(), 1);
        assert!(images[0].ends_with("pic.png"));
    }

    /// Angle-bracket URL refs stay skipped.
    #[test]
    fn angle_bracket_url_ref_still_skipped() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("c.md"),
            "![](<https://example.com/foo bar.png>)\n",
        )
        .expect("c.md");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/concepts/c.md".to_string(),
            title: "c".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        let images = resolve_candidate_image_paths(root, &plan);
        assert!(images.is_empty(), "URL in angle-brackets still skipped: {images:?}");
    }

    /// Angle-bracket ref to a missing file is silently dropped, matching
    /// the plain-form behavior.
    #[test]
    fn angle_bracket_missing_ref_dropped() {
        let tmp = TempDir::new().expect("tempdir");
        let root = tmp.path();
        let wiki_dir = root.join("wiki/concepts");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("c.md"),
            "![](<does not exist.png>)\n",
        )
        .expect("c.md");

        let plan = make_plan(vec![RetrievalCandidate {
            id: "wiki/concepts/c.md".to_string(),
            title: "c".to_string(),
            score: 1,
            estimated_tokens: 10,
            reasons: Vec::new(),
        }]);

        assert!(
            plan_mentions_images(root, &plan),
            "regex must still match, even if the file is missing",
        );
        let images = resolve_candidate_image_paths(root, &plan);
        assert!(images.is_empty());
    }
}
