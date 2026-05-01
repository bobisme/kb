//! bn-2qda: vision-LLM auto-captions for undescribed images.
//!
//! Walks each compiled `wiki/sources/*.md` page for `![alt](path)` image
//! references. For each ref whose alt text is empty, generic, or equal to the
//! filename stem, computes a content hash of the image bytes, looks the
//! caption up in `<root>/.kb/cache/captions/<hash>.txt`, and on miss invokes
//! a vision-capable [`LlmAdapter`] to generate a 2-3 sentence description.
//!
//! Captions are injected next to the image as a managed region so the lexical
//! and embedding indexes pick them up automatically on the same compile pass:
//!
//! ```markdown
//! ![](diagram.png)
//! <!-- kb:begin id=caption-<hash> -->
//! <caption text>
//! <!-- kb:end id=caption-<hash> -->
//! ```
//!
//! ## Behavior
//!
//! - **alt-text gate**: caption only when alt is empty, equals "image", or
//!   matches the filename stem (e.g. alt="diagram" for `diagram.png`).
//! - **`allow_paths` privacy guard**: skip images outside the configured prefix
//!   list so private files (e.g. `~/.ssh/id_rsa.png`) are never sent to a
//!   remote model.
//! - **content-hash cache**: identical bytes produce identical caption text;
//!   pay the LLM cost once per unique image.
//! - **graceful degradation**: a "vision unsupported" error from the adapter
//!   logs a warn and skips captioning rather than failing the compile.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use kb_core::fs::atomic_write;
use kb_core::{cache_dir, hash_bytes, rewrite_managed_region};
use kb_llm::{LlmAdapter, LlmAdapterError};
use regex::Regex;

/// Default prompt sent to the vision-capable adapter.
///
/// Matches the bone description: 2-3 sentences, focused on what is depicted
/// (text / diagrams / charts / relationships), plain prose without markdown so
/// the injected caption doesn't fight with the surrounding markdown context.
pub const DEFAULT_PROMPT: &str = "Describe this image in 2-3 sentences for a knowledge base. Focus on what is depicted, any text, diagrams, charts, or relationships shown. Plain prose, no markdown.";

/// Generic alt-text values that should be treated as missing (i.e. trigger
/// captioning even though the alt slot is technically non-empty).
const GENERIC_ALT_TEXT: &[&str] = &["image", "img", "picture", "screenshot"];

/// Subdirectory under `.kb/cache/` where caption text is stored.
///
/// Keyed by blake3 content-hash of the image bytes. Each entry is a plain
/// UTF-8 file containing only the caption — no JSON envelope so cache
/// files are trivially diffable.
pub const CAPTIONS_CACHE_SUBDIR: &str = "captions";

/// Same image-ref regex used by [`crate::source_page`]:
/// - capture group 1: alt text
/// - capture group 2: angle-bracket-wrapped path (`<path with spaces.png>`)
/// - capture group 3: plain path (no whitespace)
///
/// An optional `CommonMark` title after the path is tolerated and discarded.
static IMAGE_REF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"!\[([^\]]*)\]\(\s*(?:<([^>]+)>|([^)\s"]+))(?:\s+"[^"]*")?\s*\)"#)
        .expect("hard-coded image regex is valid")
});

/// Configuration for the captions pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptionsConfig {
    /// Master switch. When `false` the pass is a no-op.
    pub enabled: bool,
    /// Ordered list of path prefixes (relative to the KB root) the pass is
    /// allowed to read images from. An image path that doesn't start with any
    /// of these prefixes is skipped silently — privacy guard. Empty list
    /// means "no images allowed", not "all images allowed", to fail-closed
    /// against a misconfigured TOML.
    pub allow_paths: Vec<String>,
    /// Caption prompt sent to the adapter. Stored on the config so callers
    /// can override per-deployment without recompiling.
    pub prompt: String,
}

impl Default for CaptionsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_paths: vec![
                "wiki/".to_string(),
                "sources/".to_string(),
                "raw/".to_string(),
            ],
            prompt: DEFAULT_PROMPT.to_string(),
        }
    }
}

/// One image reference parsed out of a markdown page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRef {
    /// Alt text between `[` and `]`.
    pub alt: String,
    /// Path between `(` and `)` after stripping any optional title.
    pub path: String,
    /// Absolute byte offset of the start of the `![` token in the page body.
    pub start: usize,
    /// Absolute byte offset of the end of the closing `)`.
    pub end: usize,
}

/// Result of the captions pass for a single page.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CaptionsPageReport {
    /// Number of image refs that were found, regardless of whether they got
    /// captioned.
    pub images_seen: usize,
    /// Number of refs that triggered caption generation (cache hit OR adapter
    /// call).
    pub images_captioned: usize,
    /// Number of refs that hit the disk cache (no adapter call).
    pub cache_hits: usize,
    /// Number of refs where the adapter actually ran (cache miss).
    pub adapter_calls: usize,
    /// Number of refs that were skipped because the alt text was already
    /// descriptive.
    pub skipped_alt_present: usize,
    /// Number of refs that were skipped because the path is outside
    /// `allow_paths`.
    pub skipped_blocked_path: usize,
    /// Number of refs whose image file could not be read.
    pub skipped_missing_file: usize,
    /// Number of refs where the adapter returned an "unsupported" error.
    pub skipped_unsupported: usize,
    /// `true` when the page body changed and should be written back.
    pub body_modified: bool,
}

/// Walk `body` for image refs and inject `kb:caption` managed regions next to
/// every undescribed image. Returns the (possibly-modified) body and a
/// per-page report.
///
/// `page_path` is the absolute path of the page being scanned and is used for
/// resolving relative image paths. `root` is the KB root so the privacy
/// allow-list can be evaluated against root-relative prefixes.
///
/// # Errors
///
/// Returns an error when the cache directory cannot be created or a cache
/// file cannot be written. Adapter errors and unreadable image files are
/// surfaced as `skipped_*` counters in the report — they never fail the pass.
// Captions need a single linear walk over image refs that branches into
// alt-gate / allow_paths / cache-hit / cache-miss / inject; splitting it
// across helpers obscures the per-ref decision tree more than it helps.
#[allow(clippy::too_many_lines)]
pub fn caption_page<A: LlmAdapter + ?Sized>(
    adapter: &A,
    root: &Path,
    page_path: &Path,
    body: &str,
    config: &CaptionsConfig,
) -> Result<(String, CaptionsPageReport)> {
    let mut report = CaptionsPageReport::default();

    if !config.enabled {
        return Ok((body.to_string(), report));
    }

    let refs = parse_image_refs(body);
    report.images_seen = refs.len();

    if refs.is_empty() {
        return Ok((body.to_string(), report));
    }

    let cache_root = captions_cache_dir(root);

    // Track caption-region IDs we've already inserted on this pass so two
    // refs to the same image (same content hash) on one page don't both try
    // to inject a caption block — the second one would collide with the
    // first's managed-region id and rewrite_managed_region would touch only
    // the first occurrence.
    let mut injected_ids: HashSet<String> = HashSet::new();

    let mut body_out = body.to_string();
    for image_ref in refs {
        if !needs_captioning(&image_ref) {
            report.skipped_alt_present += 1;
            continue;
        }

        let Some(resolved) = resolve_image_path(root, page_path, &image_ref.path) else {
            report.skipped_missing_file += 1;
            continue;
        };

        if !path_is_allowed(root, &resolved, &config.allow_paths) {
            report.skipped_blocked_path += 1;
            continue;
        }

        let bytes = match std::fs::read(&resolved) {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(
                    "captions: skip {}: {err}",
                    resolved.display()
                );
                report.skipped_missing_file += 1;
                continue;
            }
        };

        let hash = hash_bytes(&bytes).to_hex();
        let region_id = format!("caption-{hash}");

        // If we already injected this caption region above, skip the second
        // ref entirely. The page already has the caption near the first ref.
        if injected_ids.contains(&region_id) {
            continue;
        }

        // If the page already carries this exact caption region, that's a
        // cache-on-page hit: increment images_captioned but do not re-inject.
        if body_out.contains(&format!("<!-- kb:begin id={region_id} -->")) {
            report.images_captioned += 1;
            report.cache_hits += 1;
            injected_ids.insert(region_id);
            continue;
        }

        let cache_path = cache_root.join(format!("{hash}.txt"));
        let caption = if cache_path.exists() {
            match std::fs::read_to_string(&cache_path) {
                Ok(text) => {
                    report.cache_hits += 1;
                    text.trim().to_string()
                }
                Err(err) => {
                    tracing::warn!(
                        "captions: failed to read cache {}: {err}",
                        cache_path.display()
                    );
                    continue;
                }
            }
        } else {
            match adapter.caption_image(&resolved, &config.prompt) {
                Ok((text, _provenance)) => {
                    let trimmed = text.trim().to_string();
                    if let Err(err) = atomic_write(&cache_path, trimmed.as_bytes()) {
                        tracing::warn!(
                            "captions: failed to persist cache {}: {err}",
                            cache_path.display()
                        );
                    }
                    report.adapter_calls += 1;
                    trimmed
                }
                Err(LlmAdapterError::Other(msg))
                    if msg.contains("vision unsupported")
                        || msg.contains("not implemented") =>
                {
                    tracing::warn!(
                        "captions: adapter does not support vision; skipping {}",
                        resolved.display()
                    );
                    report.skipped_unsupported += 1;
                    continue;
                }
                Err(err) => {
                    tracing::warn!(
                        "captions: adapter call failed for {}: {err}",
                        resolved.display()
                    );
                    report.skipped_unsupported += 1;
                    continue;
                }
            }
        };

        // Inject (or rewrite) the caption managed region next to the image.
        body_out = inject_caption_region(&body_out, &image_ref, &region_id, &caption);
        injected_ids.insert(region_id);
        report.images_captioned += 1;
        report.body_modified = true;
    }

    Ok((body_out, report))
}

/// Parse all `![alt](path)` refs out of `body` in document order.
///
/// # Panics
///
/// Panics if the hard-coded image regex unexpectedly returns a capture
/// without group 0 — by construction this is unreachable.
#[must_use]
pub fn parse_image_refs(body: &str) -> Vec<ImageRef> {
    let mut refs = Vec::new();
    for capture in IMAGE_REF_RE.captures_iter(body) {
        let whole = capture.get(0).expect("regex always has group 0");
        let alt = capture.get(1).map_or("", |m| m.as_str()).to_string();
        let path = capture
            .get(2)
            .or_else(|| capture.get(3))
            .map_or("", |m| m.as_str().trim())
            .to_string();

        // Skip data: URIs and external URLs — those are never on-disk files
        // we'd hash and caption.
        if path.starts_with("data:") || looks_like_url(&path) {
            continue;
        }

        refs.push(ImageRef {
            alt,
            path,
            start: whole.start(),
            end: whole.end(),
        });
    }
    refs
}

/// Decide whether this image ref should trigger captioning based on its alt
/// text alone.
///
/// Returns `true` when alt is empty, matches a generic placeholder from
/// [`GENERIC_ALT_TEXT`], or equals the filename stem of the path
/// (case-insensitive).
#[must_use]
pub fn needs_captioning(image_ref: &ImageRef) -> bool {
    let alt = image_ref.alt.trim();
    if alt.is_empty() {
        return true;
    }
    let alt_lower = alt.to_ascii_lowercase();
    if GENERIC_ALT_TEXT.iter().any(|g| *g == alt_lower) {
        return true;
    }

    if let Some(stem) = filename_stem(&image_ref.path) {
        if stem.eq_ignore_ascii_case(alt) {
            return true;
        }
    }

    false
}

/// Extract the filename stem (no extension) from a path string. Returns
/// `None` when the path has no final segment or its stem is empty.
fn filename_stem(path: &str) -> Option<String> {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .filter(|s| !s.is_empty())
}

/// Resolve a markdown image path against the source page directory. Returns
/// the canonicalized absolute path on disk, or `None` when the file does not
/// exist.
///
/// Absolute paths are returned verbatim (after canonicalization) — the
/// privacy `allow_paths` check downstream gates them.
fn resolve_image_path(root: &Path, page_path: &Path, raw: &str) -> Option<PathBuf> {
    let candidate = if raw.starts_with('/') {
        PathBuf::from(raw)
    } else {
        page_path.parent().map_or_else(
            || root.to_path_buf(),
            std::path::Path::to_path_buf,
        )
        .join(raw)
    };

    // Normalize the path so the prefix check works regardless of `..`/`.` segments.
    // canonicalize() requires the file to exist; we rely on that as the
    // existence test too.
    candidate.canonicalize().ok()
}

/// `true` when `resolved` lives under any of the allow-list prefixes
/// (interpreted relative to `root`). Empty `allow_paths` returns `false`
/// (fail-closed).
fn path_is_allowed(root: &Path, resolved: &Path, allow_paths: &[String]) -> bool {
    if allow_paths.is_empty() {
        return false;
    }
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let Ok(rel) = resolved.strip_prefix(&canonical_root) else {
        return false;
    };
    let rel_str = rel.to_string_lossy();
    allow_paths.iter().any(|prefix| {
        let prefix = prefix.trim_start_matches("./");
        rel_str.starts_with(prefix)
    })
}

/// Cheaply detect whether the path looks like a URL (`http://`, `https://`,
/// `ftp://`, etc.) so we don't treat it as a local file.
fn looks_like_url(path: &str) -> bool {
    path.find("://").is_some_and(|idx| {
        let scheme = &path[..idx];
        // Any non-empty alphanumeric scheme counts as a URL.
        !scheme.is_empty()
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-')
    })
}

/// Absolute path to `<root>/.kb/cache/captions/`.
#[must_use]
pub fn captions_cache_dir(root: &Path) -> PathBuf {
    cache_dir(root).join(CAPTIONS_CACHE_SUBDIR)
}

/// Inject a `kb:begin id=caption-<hash>` managed region into `body` directly
/// after the closing `)` of `image_ref`, or rewrite an existing region with
/// the same id when one is already present.
fn inject_caption_region(
    body: &str,
    image_ref: &ImageRef,
    region_id: &str,
    caption: &str,
) -> String {
    let region_text = render_caption_region(region_id, caption);

    // If the page already has a region with this id, rewrite its content in
    // place — the caller already checked for the begin marker, but
    // rewrite_managed_region is idempotent so calling it again is harmless.
    let region_inner = format!("\n{}\n", caption.trim());
    if let Some(updated) = rewrite_managed_region(body, region_id, &region_inner) {
        return updated;
    }

    // Otherwise, insert the region right after the image. We also leave a
    // single newline between the image and the region so block-level parsers
    // (Obsidian, kb's own renderer) treat them as adjacent paragraphs rather
    // than a single inline run.
    let mut out = String::with_capacity(body.len() + region_text.len() + 1);
    out.push_str(&body[..image_ref.end]);
    if !body[image_ref.end..].starts_with('\n') {
        out.push('\n');
    }
    out.push_str(&region_text);
    out.push_str(&body[image_ref.end..]);
    out
}

/// Render the full begin/content/end fence triple for a caption region.
fn render_caption_region(region_id: &str, caption: &str) -> String {
    format!(
        "<!-- kb:begin id={region_id} -->\n{caption}\n<!-- kb:end id={region_id} -->\n",
        region_id = region_id,
        caption = caption.trim(),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use kb_llm::{
        AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest,
        ExtractConceptsResponse, GenerateSlidesRequest, GenerateSlidesResponse,
        MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, ProvenanceRecord,
        RunHealthCheckRequest, RunHealthCheckResponse, SummarizeDocumentRequest,
        SummarizeDocumentResponse, TokenUsage,
    };
    use std::fs;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// Stub adapter that records the number of `caption_image` calls made and
    /// returns a fixed caption. Uses `Mutex<usize>` for the call counter so
    /// the type is `Sync` without an unsafe impl (the crate forbids unsafe).
    struct StubAdapter {
        caption: String,
        calls: Mutex<usize>,
    }

    impl StubAdapter {
        fn new(caption: &str) -> Self {
            Self {
                caption: caption.to_string(),
                calls: Mutex::new(0),
            }
        }

        fn call_count(&self) -> usize {
            *self.calls.lock().expect("poisoned counter")
        }
    }

    impl LlmAdapter for StubAdapter {
        fn caption_image(
            &self,
            _path: &Path,
            _prompt: &str,
        ) -> Result<(String, ProvenanceRecord), LlmAdapterError> {
            *self.calls.lock().expect("poisoned counter") += 1;
            Ok((self.caption.clone(), test_provenance()))
        }

        fn summarize_document(
            &self,
            _: SummarizeDocumentRequest,
        ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("captions test does not call summarize_document")
        }

        fn extract_concepts(
            &self,
            _: ExtractConceptsRequest,
        ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("captions test does not call extract_concepts")
        }

        fn merge_concept_candidates(
            &self,
            _: MergeConceptCandidatesRequest,
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError>
        {
            unreachable!("captions test does not call merge_concept_candidates")
        }

        fn answer_question(
            &self,
            _: AnswerQuestionRequest,
        ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("captions test does not call answer_question")
        }

        fn generate_slides(
            &self,
            _: GenerateSlidesRequest,
        ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("captions test does not call generate_slides")
        }

        fn run_health_check(
            &self,
            _: RunHealthCheckRequest,
        ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("captions test does not call run_health_check")
        }
    }

    /// Adapter that always returns "vision unsupported" — used to verify the
    /// graceful-skip path.
    struct UnsupportedAdapter;

    impl LlmAdapter for UnsupportedAdapter {
        fn caption_image(
            &self,
            _path: &Path,
            _prompt: &str,
        ) -> Result<(String, ProvenanceRecord), LlmAdapterError> {
            Err(LlmAdapterError::Other(
                "vision unsupported by stub adapter".to_string(),
            ))
        }

        fn summarize_document(
            &self,
            _: SummarizeDocumentRequest,
        ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!()
        }

        fn extract_concepts(
            &self,
            _: ExtractConceptsRequest,
        ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!()
        }

        fn merge_concept_candidates(
            &self,
            _: MergeConceptCandidatesRequest,
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError>
        {
            unreachable!()
        }

        fn answer_question(
            &self,
            _: AnswerQuestionRequest,
        ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!()
        }

        fn generate_slides(
            &self,
            _: GenerateSlidesRequest,
        ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!()
        }

        fn run_health_check(
            &self,
            _: RunHealthCheckRequest,
        ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!()
        }
    }

    fn test_provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "test".to_string(),
            harness_version: None,
            model: "test-vision".to_string(),
            prompt_template_name: "caption_image".to_string(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::Hash::from([0u8; 32]),
            started_at: 1,
            ended_at: 2,
            latency_ms: 1,
            retries: 0,
            tokens: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
            cost_estimate: None,
        }
    }

    fn write_image(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, bytes).expect("write fixture image");
        p
    }

    fn fixture_root() -> TempDir {
        let dir = TempDir::new().expect("tempdir");
        // Mimic kb layout: wiki/, .kb/cache/captions/.
        fs::create_dir_all(dir.path().join("wiki/sources")).expect("mkdir wiki/sources");
        fs::create_dir_all(dir.path().join(".kb/cache/captions")).expect("mkdir cache");
        dir
    }

    // --- alt-text classifier tests ---

    #[test]
    fn needs_captioning_triggers_on_empty_alt() {
        let r = ImageRef {
            alt: String::new(),
            path: "diagram.png".to_string(),
            start: 0,
            end: 0,
        };
        assert!(needs_captioning(&r));
    }

    #[test]
    fn needs_captioning_triggers_on_generic_alt_image() {
        let r = ImageRef {
            alt: "image".to_string(),
            path: "diagram.png".to_string(),
            start: 0,
            end: 0,
        };
        assert!(needs_captioning(&r));

        let r2 = ImageRef {
            alt: "Image".to_string(),
            path: "diagram.png".to_string(),
            start: 0,
            end: 0,
        };
        assert!(needs_captioning(&r2));
    }

    #[test]
    fn needs_captioning_triggers_on_filename_stem_alt() {
        // alt="diagram" for `diagram.png` is treated as a placeholder — most
        // markdown editors auto-fill this.
        let r = ImageRef {
            alt: "diagram".to_string(),
            path: "diagram.png".to_string(),
            start: 0,
            end: 0,
        };
        assert!(needs_captioning(&r));

        // Case-insensitive.
        let r2 = ImageRef {
            alt: "Diagram".to_string(),
            path: "DIAGRAM.png".to_string(),
            start: 0,
            end: 0,
        };
        assert!(needs_captioning(&r2));
    }

    #[test]
    fn needs_captioning_skips_descriptive_alt() {
        let r = ImageRef {
            alt: "a high-level architecture overview of the auth pipeline".to_string(),
            path: "diagram.png".to_string(),
            start: 0,
            end: 0,
        };
        assert!(!needs_captioning(&r));
    }

    // --- allow_paths gating ---

    #[test]
    fn allow_paths_blocks_path_outside_list() {
        let dir = fixture_root();
        let root = dir.path();
        // An "outside" image lives at a tempdir path that is NOT under any
        // configured prefix.
        let outside = TempDir::new().expect("tempdir");
        let img_path = write_image(outside.path(), "secret.png", b"\x89PNGbytes");

        let allowed = path_is_allowed(
            root,
            &img_path.canonicalize().unwrap(),
            &["wiki/".to_string()],
        );
        assert!(
            !allowed,
            "image outside KB root must not match any allow-list prefix"
        );
    }

    #[test]
    fn allow_paths_admits_path_under_listed_prefix() {
        let dir = fixture_root();
        let root = dir.path();
        let img = write_image(&root.join("wiki/sources"), "diagram.png", b"\x89PNGbytes");
        let allowed = path_is_allowed(
            root,
            &img.canonicalize().unwrap(),
            &["wiki/".to_string()],
        );
        assert!(allowed);
    }

    #[test]
    fn allow_paths_empty_list_blocks_everything() {
        let dir = fixture_root();
        let root = dir.path();
        let img = write_image(&root.join("wiki/sources"), "diagram.png", b"\x89PNGbytes");
        let allowed = path_is_allowed(root, &img.canonicalize().unwrap(), &[]);
        assert!(!allowed, "empty allow_paths fails closed");
    }

    // --- pass behavior tests ---

    #[test]
    fn caption_pass_calls_adapter_on_cache_miss_and_persists() {
        let dir = fixture_root();
        let root = dir.path();
        let page_dir = root.join("wiki/sources");
        let img = write_image(&page_dir, "diagram.png", b"\x89PNGfakebytes");

        let page_path = page_dir.join("page.md");
        let body = "# Page\n\n![](diagram.png)\n";

        let adapter = StubAdapter::new("A diagram showing service A talking to service B.");
        let cfg = CaptionsConfig::default();

        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(adapter.call_count(), 1);
        assert_eq!(report.images_seen, 1);
        assert_eq!(report.adapter_calls, 1);
        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.images_captioned, 1);
        assert!(report.body_modified);

        // The page now contains the caption block.
        let hash = hash_bytes(&fs::read(&img).unwrap()).to_hex();
        assert!(
            out.contains(&format!("<!-- kb:begin id=caption-{hash} -->")),
            "expected kb:caption block; got body:\n{out}"
        );
        assert!(
            out.contains("A diagram showing service A talking to service B."),
            "expected caption text inline; got body:\n{out}"
        );

        // Cache file persisted.
        let cache_file = captions_cache_dir(root).join(format!("{hash}.txt"));
        assert!(cache_file.exists(), "cache file should be written on miss");
        let cached = fs::read_to_string(&cache_file).unwrap();
        assert_eq!(
            cached.trim(),
            "A diagram showing service A talking to service B."
        );
    }

    #[test]
    fn caption_pass_uses_cache_when_present() {
        let dir = fixture_root();
        let root = dir.path();
        let page_dir = root.join("wiki/sources");
        let img = write_image(&page_dir, "diagram.png", b"\x89PNGfakebytes");

        // Pre-populate the cache.
        let hash = hash_bytes(&fs::read(&img).unwrap()).to_hex();
        let cache_file = captions_cache_dir(root).join(format!("{hash}.txt"));
        fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        fs::write(&cache_file, "Cached caption text here.").unwrap();

        let page_path = page_dir.join("page.md");
        let body = "# Page\n\n![](diagram.png)\n";

        let adapter = StubAdapter::new("WRONG — should not be called");
        let cfg = CaptionsConfig::default();

        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(
            adapter.call_count(),
            0,
            "adapter MUST NOT be called when cache file exists"
        );
        assert_eq!(report.cache_hits, 1);
        assert_eq!(report.adapter_calls, 0);
        assert_eq!(report.images_captioned, 1);
        assert!(out.contains("Cached caption text here."));
    }

    #[test]
    fn caption_pass_skips_when_alt_is_descriptive() {
        let dir = fixture_root();
        let root = dir.path();
        let page_dir = root.join("wiki/sources");
        write_image(&page_dir, "diagram.png", b"\x89PNGfakebytes");
        let page_path = page_dir.join("page.md");
        let body = "# Page\n\n![architecture overview of the auth flow](diagram.png)\n";

        let adapter = StubAdapter::new("never called");
        let cfg = CaptionsConfig::default();

        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(adapter.call_count(), 0);
        assert_eq!(report.images_seen, 1);
        assert_eq!(report.skipped_alt_present, 1);
        assert_eq!(report.images_captioned, 0);
        assert_eq!(out, body, "page body must be unchanged");
    }

    #[test]
    fn caption_pass_skips_when_path_is_outside_allow_paths() {
        let dir = fixture_root();
        let root = dir.path();
        let outside = TempDir::new().expect("tempdir for outside file");
        let img = write_image(outside.path(), "secret.png", b"\x89PNGsecret");

        let page_path = root.join("wiki/sources/page.md");
        // Use absolute path to the outside image.
        let body = format!("# Page\n\n![]({})\n", img.display());

        let adapter = StubAdapter::new("never called");
        let cfg = CaptionsConfig::default();

        let (out, report) = caption_page(&adapter, root, &page_path, &body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(adapter.call_count(), 0);
        assert_eq!(report.skipped_blocked_path, 1);
        assert_eq!(report.images_captioned, 0);
        assert_eq!(out, body, "page body must be unchanged");
    }

    #[test]
    fn caption_pass_unsupported_adapter_logs_warn_and_skips() {
        let dir = fixture_root();
        let root = dir.path();
        let page_dir = root.join("wiki/sources");
        write_image(&page_dir, "diagram.png", b"\x89PNGfakebytes");
        let page_path = page_dir.join("page.md");
        let body = "# Page\n\n![](diagram.png)\n";

        let adapter = UnsupportedAdapter;
        let cfg = CaptionsConfig::default();

        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds even with unsupported adapter");

        assert_eq!(report.skipped_unsupported, 1);
        assert_eq!(report.images_captioned, 0);
        assert!(!report.body_modified);
        assert_eq!(out, body, "body unchanged when adapter can't caption");
    }

    #[test]
    fn caption_pass_disabled_is_noop() {
        let dir = fixture_root();
        let root = dir.path();
        let page_dir = root.join("wiki/sources");
        write_image(&page_dir, "diagram.png", b"\x89PNGfakebytes");
        let page_path = page_dir.join("page.md");
        let body = "![](diagram.png)";

        let adapter = StubAdapter::new("never called");
        let cfg = CaptionsConfig {
            enabled: false,
            ..Default::default()
        };

        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(adapter.call_count(), 0);
        assert_eq!(report, CaptionsPageReport::default());
        assert_eq!(out, body);
    }

    #[test]
    fn caption_pass_idempotent_on_re_run() {
        // Re-running compile must NOT call the adapter again, even though
        // the kb:caption block is already inline. The cache file still
        // exists from the first run; the page already has the caption block.
        let dir = fixture_root();
        let root = dir.path();
        let page_dir = root.join("wiki/sources");
        write_image(&page_dir, "diagram.png", b"\x89PNGfakebytes");
        let page_path = page_dir.join("page.md");
        let body = "# Page\n\n![](diagram.png)\n";

        let adapter = StubAdapter::new("First caption.");
        let cfg = CaptionsConfig::default();

        let (after_first, _) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("first pass");
        let calls_after_first = adapter.call_count();
        assert_eq!(calls_after_first, 1);

        // Second pass on the already-captioned body.
        let (after_second, report2) = caption_page(&adapter, root, &page_path, &after_first, &cfg)
            .expect("second pass");

        assert_eq!(
            adapter.call_count(),
            calls_after_first,
            "adapter must not be called on re-run"
        );
        assert!(!report2.body_modified || after_second == after_first,
            "re-run must not reshape the body");
    }

    #[test]
    fn caption_pass_skips_data_uri_image_refs() {
        // Data URIs are not on-disk files and never need captioning.
        let dir = fixture_root();
        let root = dir.path();
        let page_path = root.join("wiki/sources/page.md");
        let body = "![](data:image/png;base64,iVBORw0KGgo=)";

        let adapter = StubAdapter::new("never called");
        let cfg = CaptionsConfig::default();
        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(adapter.call_count(), 0);
        assert_eq!(report.images_seen, 0, "data URI must not register as an image ref");
        assert_eq!(out, body);
    }

    #[test]
    fn caption_pass_skips_external_url_image_refs() {
        let dir = fixture_root();
        let root = dir.path();
        let page_path = root.join("wiki/sources/page.md");
        let body = "![](https://example.com/diagram.png)";

        let adapter = StubAdapter::new("never called");
        let cfg = CaptionsConfig::default();
        let (out, report) = caption_page(&adapter, root, &page_path, body, &cfg)
            .expect("captions pass succeeds");

        assert_eq!(adapter.call_count(), 0);
        assert_eq!(report.images_seen, 0);
        assert_eq!(out, body);
    }

    // --- managed-region injection tests ---

    #[test]
    fn render_caption_region_uses_canonical_fence_format() {
        let rendered = render_caption_region("caption-abc", "An image of a cat.");
        assert_eq!(
            rendered,
            "<!-- kb:begin id=caption-abc -->\nAn image of a cat.\n<!-- kb:end id=caption-abc -->\n"
        );
    }

    #[test]
    fn parse_image_refs_handles_angle_bracket_paths() {
        let body = "![my image](<path with spaces.png>)";
        let refs = parse_image_refs(body);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].alt, "my image");
        assert_eq!(refs[0].path, "path with spaces.png");
    }
}
