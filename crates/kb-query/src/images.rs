//! Image asset resolution for retrieval candidates.
//!
//! bn-3dkw: when a retrieval candidate's body references a local image
//! (e.g. `![diagram](assets/foo.png)` in a normalized source, or
//! `![fig](../../normalized/src-xyz/assets/bar.png)` in a compiled wiki
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

use regex::Regex;

use crate::lexical::RetrievalPlan;

/// Hard cap on images per `kb ask` invocation. Keeps the attachment payload
/// bounded even for plans that sweep up many image-heavy pages.
pub const MAX_IMAGES_PER_QUERY: usize = 5;

/// Matches `![alt](path)` image references in markdown. Mirrors the scanners
/// in `kb_ingest::image_refs` and `kb_compile::source_page` — kept local here
/// so this crate doesn't have to depend on either.
static IMAGE_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"!\[[^\]]*\]\(([^)\s]+)\)").expect("hard-coded image regex is valid")
});

/// Resolve image attachments referenced by the pages in `plan`, relative to
/// the KB `root`.
///
/// Returns absolute paths to on-disk image files, in first-seen order,
/// deduplicated, and capped at [`MAX_IMAGES_PER_QUERY`].
///
/// Callers that know no candidate can have images (empty plan, or a
/// pre-check via [`plan_mentions_images`]) should skip this call entirely —
/// it's a cheap walk but still touches disk.
#[must_use]
pub fn resolve_candidate_image_paths(root: &Path, plan: &RetrievalPlan) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for candidate in &plan.candidates {
        if out.len() >= MAX_IMAGES_PER_QUERY {
            break;
        }

        let page_path = root.join(&candidate.id);
        let Ok(body) = std::fs::read_to_string(&page_path) else {
            continue;
        };

        // The wiki page lives at e.g. `wiki/sources/src-xxx.md`; a relative
        // path inside it resolves against its parent dir.
        let page_parent = page_path.parent().unwrap_or(root);

        for capture in IMAGE_REF_RE.captures_iter(&body) {
            if out.len() >= MAX_IMAGES_PER_QUERY {
                break;
            }
            let Some(raw_path) = capture.get(1).map(|m| m.as_str()) else {
                continue;
            };
            let Some(resolved) = resolve_image_ref(raw_path, page_parent) else {
                continue;
            };
            if seen.insert(resolved.clone()) {
                out.push(resolved);
            }
        }
    }

    out
}

/// True when any candidate's on-disk body mentions at least one image ref.
///
/// Used as the zero-cost fast-path guard: if no candidate mentions any
/// image, the ask pipeline can skip image resolution and adapter-side
/// attachment plumbing entirely.
#[must_use]
pub fn plan_mentions_images(root: &Path, plan: &RetrievalPlan) -> bool {
    for candidate in &plan.candidates {
        let page_path = root.join(&candidate.id);
        let Ok(body) = std::fs::read_to_string(&page_path) else {
            continue;
        };
        if IMAGE_REF_RE.is_match(&body) {
            return true;
        }
    }
    false
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
        let normalized_dir = root.join("normalized/src-xyz/assets");
        fs::create_dir_all(&normalized_dir).expect("normalized assets");
        fs::write(normalized_dir.join("foo.png"), b"\x89PNG").expect("write png");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-xyz.md"),
            "---\ntype: source\n---\n![d](../../normalized/src-xyz/assets/foo.png)\n",
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
        let normalized_dir = root.join("normalized/src-a/assets");
        fs::create_dir_all(&normalized_dir).expect("normalized dir");
        fs::write(normalized_dir.join("shared.png"), b"PNG").expect("png");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");
        fs::write(
            wiki_dir.join("src-a.md"),
            "![x](../../normalized/src-a/assets/shared.png)\n",
        )
        .expect("src-a");
        fs::write(
            wiki_dir.join("src-b.md"),
            "![y](../../normalized/src-a/assets/shared.png)\n",
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
        let normalized_dir = root.join("normalized/src/assets");
        fs::create_dir_all(&normalized_dir).expect("normalized dir");

        let wiki_dir = root.join("wiki/sources");
        fs::create_dir_all(&wiki_dir).expect("wiki dir");

        // 10 images > MAX_IMAGES_PER_QUERY (5).
        let mut body = String::new();
        for i in 0..10 {
            let name = format!("i{i}.png");
            fs::write(normalized_dir.join(&name), b"PNG").expect("png");
            writeln!(body, "![](../../normalized/src/assets/{name})")
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
}
