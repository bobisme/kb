//! OCR fallback for scan-only PDFs.
//!
//! A scan-only PDF (photographed pages, OCR'd legal docs, image-only scientific
//! papers) extracts to empty or near-empty text via the markitdown
//! preprocessor: there's nothing to extract because the content is pixels, not
//! text. Without a fallback those PDFs become silent black holes — present in
//! the corpus but invisible to retrieval.
//!
//! This module wraps the classic `pdftoppm` (rasterize) + `tesseract` (OCR)
//! pipeline and emits page-tagged markdown that downstream consumers can split
//! by page. It is intentionally NOT a Rust-side image library: shelling out to
//! the system tools is the same posture the codebase already uses for
//! markitdown, keeps build time low, and avoids the pile of native deps that
//! image/text-rec crates pull in.
//!
//! ## Failure modes — none are fatal
//!
//! - `enabled = false`: returns `Skipped` immediately. Caller proceeds with the
//!   original (possibly empty) text.
//! - `tesseract` or `pdftoppm` not on PATH: warn once per binary, return
//!   `Skipped`. Ingest must continue on machines without the binaries
//!   installed (CI, dev boxes).
//! - Any per-page subprocess failure: warn, drop that page, continue with the
//!   rest.
//!
//! ## Caching
//!
//! Each rasterized page image is hashed (BLAKE3, via `kb_core::hash_bytes`,
//! matching the rest of the codebase's hashing) and the OCR output is parked
//! at `<root>/.kb/cache/ocr/<hex>.txt`. Re-running OCR on the same PDF is a
//! no-op once the cache is warm — `pdftoppm` always reproduces the same PNGs
//! for the same input, so the page hashes are stable across runs.

use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use kb_core::{cache_dir, hash_bytes};

/// Default minimum character count below which a markitdown extraction is
/// considered "essentially empty" and we attempt the OCR fallback.
pub const DEFAULT_MIN_CHARS_THRESHOLD: usize = 100;

/// Default rasterization DPI for `pdftoppm`. Tesseract recommends 300 DPI for
/// general-purpose printed text — lower hurts accuracy, higher mostly burns CPU.
const RASTER_DPI: u32 = 300;

/// User-visible settings for the OCR fallback step, parsed from
/// `[ingest.ocr]` in `kb.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrOptions {
    /// Master switch. When false, OCR fallback never triggers regardless of
    /// extracted text length.
    pub enabled: bool,
    /// Tesseract command name (or absolute path). Whitespace-split into
    /// program + leading argv, mirroring `MarkitdownOptions.command`.
    pub command: String,
    /// Pdftoppm command name (or absolute path). Whitespace-split into
    /// program + leading argv. Defaults to `"pdftoppm"` (PATH lookup).
    /// Configurable so tests can swap in a stub without process-wide PATH
    /// mutation.
    pub pdftoppm_command: String,
    /// Tesseract `-l` argument. Pass-through; tesseract accepts `+`-joined
    /// language codes (`eng+fra`) so we don't validate here.
    pub language: String,
    /// Threshold (in chars, after trimming) below which we trigger OCR.
    /// `0` disables the threshold (only empty text triggers OCR).
    pub min_chars_threshold: usize,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "tesseract".to_string(),
            pdftoppm_command: "pdftoppm".to_string(),
            language: "eng".to_string(),
            min_chars_threshold: DEFAULT_MIN_CHARS_THRESHOLD,
        }
    }
}

impl OcrOptions {
    /// A disabled options bundle for callers that want to bypass OCR
    /// (legacy `IngestOptions::new`, tests).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            command: "tesseract".to_string(),
            pdftoppm_command: "pdftoppm".to_string(),
            language: "eng".to_string(),
            min_chars_threshold: DEFAULT_MIN_CHARS_THRESHOLD,
        }
    }
}

/// Decision returned by [`should_ocr_fallback`].
///
/// We expose a tri-state instead of `bool` so the caller can log the reason
/// when a fallback fires (helpful when debugging "why did this PDF take 30s
/// to ingest").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// Extracted text is empty (after trim) — definitely worth OCR'ing.
    Empty,
    /// Extracted text is non-empty but shorter than the configured threshold.
    BelowThreshold,
}

/// Returns `Some(reason)` if the extraction is empty or short enough that the
/// caller should attempt an OCR fallback. `None` means "extraction looks
/// fine, no fallback needed".
#[must_use]
pub fn should_ocr_fallback(extracted: &str, min_chars_threshold: usize) -> Option<FallbackReason> {
    let trimmed = extracted.trim();
    if trimmed.is_empty() {
        return Some(FallbackReason::Empty);
    }
    // `chars().count()` rather than `len()` so multibyte text isn't
    // double-counted toward the threshold.
    if trimmed.chars().count() < min_chars_threshold {
        return Some(FallbackReason::BelowThreshold);
    }
    None
}

/// Outcome of an OCR attempt for one PDF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrOutcome {
    /// OCR ran (possibly partly from cache) and produced markdown with at
    /// least one page block.
    Recovered { markdown: String, pages: usize },
    /// OCR was attempted but produced no usable text (e.g. all pages OCR'd
    /// to empty strings, or every per-page subprocess failed).
    Empty,
    /// OCR was not attempted: either disabled in config, or a required
    /// binary is missing on PATH. The caller should treat this as "OCR
    /// fallback opted out" and surface the original empty/short extraction.
    Skipped,
}

/// Warn at most once per process per missing binary, mirroring
/// markitdown's `WARNED_MISSING` pattern. Avoids spamming stderr when a
/// directory walk hits N PDFs and tesseract isn't installed.
static WARNED_MISSING_TESSERACT: AtomicBool = AtomicBool::new(false);
static WARNED_MISSING_PDFTOPPM: AtomicBool = AtomicBool::new(false);

/// Run the OCR fallback pipeline for a PDF. Returns the OCR'd markdown (with
/// page markers) or an [`OcrOutcome::Skipped`] if the path was opted out.
///
/// `kb_root` is the KB root that owns the cache directory (`.kb/cache/ocr/`).
///
/// # Errors
///
/// Currently never returns an error: per-binary problems degrade to
/// `Skipped`, per-page problems degrade to a missing page in the output.
/// The `Result` shape is preserved so a future strict mode can opt into
/// fatal OCR failures without churning every caller.
#[allow(clippy::too_many_lines)]
pub fn ocr_pdf(pdf_path: &Path, kb_root: &Path, options: &OcrOptions) -> Result<OcrOutcome> {
    if !options.enabled {
        return Ok(OcrOutcome::Skipped);
    }

    let mut pdftoppm_parts = options.pdftoppm_command.split_ascii_whitespace();
    let Some(pdftoppm_program) = pdftoppm_parts.next() else {
        eprintln!("warning: ingest.ocr.pdftoppm_command is empty; skipping OCR fallback");
        return Ok(OcrOutcome::Skipped);
    };
    let pdftoppm_extra: Vec<String> = pdftoppm_parts.map(str::to_owned).collect();
    if !binary_on_path(pdftoppm_program) {
        warn_missing(&WARNED_MISSING_PDFTOPPM, pdftoppm_program);
        return Ok(OcrOutcome::Skipped);
    }

    let mut tesseract_parts = options.command.split_ascii_whitespace();
    let Some(tesseract_program) = tesseract_parts.next() else {
        eprintln!("warning: ingest.ocr.command is empty; skipping OCR fallback");
        return Ok(OcrOutcome::Skipped);
    };
    let tesseract_extra: Vec<String> = tesseract_parts.map(str::to_owned).collect();
    if !binary_on_path(tesseract_program) {
        warn_missing(&WARNED_MISSING_TESSERACT, tesseract_program);
        return Ok(OcrOutcome::Skipped);
    }

    // Rasterize into a per-PDF tempdir so concurrent OCR invocations don't
    // clobber each other's PNGs.
    let raster_dir = tempfile::TempDir::new()
        .context("failed to create tempdir for pdftoppm output")?;
    let prefix = raster_dir.path().join("page");
    let raster = Command::new(pdftoppm_program)
        .args(&pdftoppm_extra)
        .arg("-r")
        .arg(RASTER_DPI.to_string())
        .arg("-png")
        .arg(pdf_path)
        .arg(&prefix)
        .stdin(Stdio::null())
        .output();
    let raster = match raster {
        Ok(out) => out,
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                warn_missing(&WARNED_MISSING_PDFTOPPM, pdftoppm_program);
                return Ok(OcrOutcome::Skipped);
            }
            eprintln!(
                "warning: failed to run `{pdftoppm_program}` for {}: {err}",
                pdf_path.display()
            );
            return Ok(OcrOutcome::Skipped);
        }
    };
    if !raster.status.success() {
        let stderr = String::from_utf8_lossy(&raster.stderr);
        eprintln!(
            "warning: pdftoppm failed for {} (exit {:?}): {}",
            pdf_path.display(),
            raster.status.code(),
            stderr.trim()
        );
        return Ok(OcrOutcome::Skipped);
    }

    let mut pages: Vec<PathBuf> = fs::read_dir(raster_dir.path())
        .with_context(|| {
            format!(
                "failed to read pdftoppm output dir {}",
                raster_dir.path().display()
            )
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    // pdftoppm emits e.g. page-1.png, page-2.png, ... page-10.png. Sorting
    // by file name gives lexicographic order, which puts page-10 before
    // page-2; sort by the trailing numeric suffix instead so page markers
    // come out in document order.
    pages.sort_by_key(|p| extract_page_index(p).unwrap_or(usize::MAX));

    if pages.is_empty() {
        eprintln!(
            "warning: pdftoppm produced no pages for {}; skipping OCR",
            pdf_path.display()
        );
        return Ok(OcrOutcome::Skipped);
    }

    let cache = OcrCache::new(cache_dir(kb_root).join("ocr"));

    let mut out = String::new();
    let mut emitted_pages = 0usize;
    let mut any_text = false;
    for (idx, page) in pages.iter().enumerate() {
        let page_no = idx + 1;
        let page_text = match ocr_one_page(
            page,
            &cache,
            tesseract_program,
            &tesseract_extra,
            &options.language,
        ) {
            Ok(text) => text,
            Err(err) => {
                eprintln!(
                    "warning: OCR failed for page {page_no} of {}: {err}",
                    pdf_path.display()
                );
                String::new()
            }
        };
        if !page_text.trim().is_empty() {
            any_text = true;
        }
        // Always emit the page marker — downstream tools may want to know a
        // page existed even if OCR returned nothing for it (e.g. a blank
        // scanned page). bn-3ij3 will own the citation-format integration.
        writeln!(out, "<!-- kb:page {page_no} -->").expect("writeln to String");
        out.push_str(page_text.trim_end_matches('\n'));
        out.push('\n');
        if page_no < pages.len() {
            out.push('\n');
        }
        emitted_pages += 1;
    }

    if !any_text {
        return Ok(OcrOutcome::Empty);
    }

    Ok(OcrOutcome::Recovered {
        markdown: out,
        pages: emitted_pages,
    })
}

/// Try to extract `N` from a `pdftoppm` output filename like `page-12.png`.
fn extract_page_index(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_str()?;
    let dash = stem.rfind('-')?;
    stem[dash + 1..].parse::<usize>().ok()
}

fn binary_on_path(name: &str) -> bool {
    // An absolute or `./relative` path is checked literally; bare names are
    // resolved against $PATH. `which`-style logic without the dependency.
    let p = Path::new(name);
    if p.is_absolute() || name.contains('/') {
        return p.is_file();
    }
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path_var).any(|dir| dir.join(name).is_file())
}

fn warn_missing(flag: &AtomicBool, command: &str) {
    if flag.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "warning: PDF OCR fallback skipped: `{command}` not found on PATH. \
         Install poppler-utils (pdftoppm) and tesseract, or disable with \
         `ingest.ocr.enabled = false` in kb.toml."
    );
}

/// Filesystem-backed cache mapping a page-image hash to its OCR text.
pub struct OcrCache {
    root: PathBuf,
}

impl OcrCache {
    /// Construct a cache rooted at `root`. The directory is created lazily on
    /// first write so a never-used cache leaves no on-disk trace.
    #[must_use]
    pub const fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Path where cache entry for the given hash hex would live (or already
    /// does).
    #[must_use]
    pub fn entry_path(&self, hash_hex: &str) -> PathBuf {
        self.root.join(format!("{hash_hex}.txt"))
    }

    /// Look up `hash_hex` in the cache. Returns `None` on miss; surfaces I/O
    /// errors only for genuinely unexpected failures (a hit that we can't
    /// read).
    ///
    /// # Errors
    ///
    /// Returns an error if the cache entry exists but cannot be read (e.g.
    /// permission denied, or the entry happens to be a directory rather
    /// than a file). A clean miss (`NotFound`) is surfaced as `Ok(None)`.
    pub fn get(&self, hash_hex: &str) -> Result<Option<String>> {
        let path = self.entry_path(hash_hex);
        match fs::read_to_string(&path) {
            Ok(text) => Ok(Some(text)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(anyhow::Error::from(err)
                .context(format!("failed to read OCR cache entry {}", path.display()))),
        }
    }

    /// Write `text` to the cache slot for `hash_hex`. Creates the cache dir
    /// on demand. Failures are propagated; the caller should warn-and-skip
    /// since cache writes are an optimization, not load-bearing for
    /// correctness.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be created or the
    /// entry file cannot be written.
    pub fn put(&self, hash_hex: &str, text: &str) -> Result<()> {
        fs::create_dir_all(&self.root).with_context(|| {
            format!("failed to create OCR cache dir {}", self.root.display())
        })?;
        let path = self.entry_path(hash_hex);
        let mut f = fs::File::create(&path)
            .with_context(|| format!("failed to write OCR cache entry {}", path.display()))?;
        f.write_all(text.as_bytes()).with_context(|| {
            format!("failed to write OCR cache entry {}", path.display())
        })?;
        Ok(())
    }
}

/// OCR a single page image. Hits the cache first; on a miss, spawns
/// tesseract and writes the result back into the cache.
fn ocr_one_page(
    page_png: &Path,
    cache: &OcrCache,
    tesseract: &str,
    tesseract_extra: &[String],
    language: &str,
) -> Result<String> {
    let bytes = fs::read(page_png)
        .with_context(|| format!("failed to read page image {}", page_png.display()))?;
    let hash_hex = hash_bytes(&bytes).to_hex();

    if let Some(cached) = cache.get(&hash_hex)? {
        return Ok(cached);
    }

    // tesseract <png> - -l <lang>  → writes recognized text to stdout.
    let output = Command::new(tesseract)
        .args(tesseract_extra)
        .arg(page_png)
        .arg("-")
        .arg("-l")
        .arg(language)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("failed to run `{tesseract}` on {}", page_png.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "tesseract failed (exit {:?}): {}",
            output.status.code(),
            stderr.trim()
        );
    }

    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    // Cache misses for empty results too — re-running tesseract on a blank
    // page is wasted work even though we have no text to surface.
    if let Err(err) = cache.put(&hash_hex, &text) {
        eprintln!("warning: failed to populate OCR cache: {err:#}");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn detector_flags_empty() {
        assert_eq!(should_ocr_fallback("", 100), Some(FallbackReason::Empty));
        assert_eq!(
            should_ocr_fallback("   \n\t  ", 100),
            Some(FallbackReason::Empty)
        );
    }

    #[test]
    fn detector_flags_short_under_threshold() {
        assert_eq!(
            should_ocr_fallback("only a few words here", 100),
            Some(FallbackReason::BelowThreshold)
        );
    }

    #[test]
    fn detector_passes_long_extraction() {
        let long = "x".repeat(500);
        assert_eq!(should_ocr_fallback(&long, 100), None);
    }

    #[test]
    fn detector_uses_chars_not_bytes() {
        // 50 multibyte chars: well under 100 chars but ~150 bytes.
        let s = "✓".repeat(50);
        assert_eq!(
            should_ocr_fallback(&s, 100),
            Some(FallbackReason::BelowThreshold)
        );
    }

    #[test]
    fn detector_threshold_zero_only_flags_empty() {
        assert_eq!(should_ocr_fallback("a", 0), None);
        assert_eq!(should_ocr_fallback("", 0), Some(FallbackReason::Empty));
    }

    #[test]
    fn cache_round_trips() {
        let tmp = TempDir::new().expect("mktemp");
        let cache = OcrCache::new(tmp.path().join("ocr"));
        assert!(cache.get("abc123").expect("get miss").is_none());
        cache.put("abc123", "hello world").expect("put");
        let got = cache.get("abc123").expect("get hit").expect("hit text");
        assert_eq!(got, "hello world");
    }

    #[test]
    fn cache_entry_path_is_deterministic() {
        let cache = OcrCache::new(PathBuf::from("/tmp/whatever"));
        assert_eq!(
            cache.entry_path("deadbeef"),
            PathBuf::from("/tmp/whatever/deadbeef.txt")
        );
    }

    #[test]
    fn cache_get_propagates_unexpected_io() {
        // Cache root that exists as a *file* — read_to_string on a
        // non-existent child would still NotFound, so simulate corruption
        // by creating the entry as a directory.
        let tmp = TempDir::new().expect("mktemp");
        let root = tmp.path().join("ocr");
        fs::create_dir_all(&root).expect("mkdir cache root");
        fs::create_dir_all(root.join("ff.txt")).expect("mkdir corrupt entry");
        let cache = OcrCache::new(root);
        let err = cache.get("ff").expect_err("expected I/O error");
        assert!(format!("{err:#}").contains("OCR cache entry"));
    }

    #[test]
    fn ocr_disabled_returns_skipped() {
        let tmp = TempDir::new().expect("mktemp");
        let pdf = tmp.path().join("doc.pdf");
        fs::write(&pdf, b"%PDF-1.4 fake").expect("write pdf");
        let outcome =
            ocr_pdf(&pdf, tmp.path(), &OcrOptions::disabled()).expect("never errors");
        assert_eq!(outcome, OcrOutcome::Skipped);
    }

    #[test]
    fn binary_on_path_resolves_relative_against_path() {
        let tmp = TempDir::new().expect("mktemp");
        // Bare name that obviously doesn't exist anywhere.
        assert!(!binary_on_path("definitely-not-a-real-binary-bn2hyr"));
        // Absolute path — checked directly, no $PATH walk.
        let made_up = tmp.path().join("made-up");
        assert!(!binary_on_path(made_up.to_str().expect("utf-8 path")));
        // Absolute path that DOES exist.
        let real = tmp.path().join("real");
        fs::write(&real, b"#!/bin/sh\n").expect("write real");
        assert!(binary_on_path(real.to_str().expect("utf-8 path")));
    }

    #[test]
    fn page_index_extraction() {
        assert_eq!(extract_page_index(Path::new("page-1.png")), Some(1));
        assert_eq!(extract_page_index(Path::new("page-12.png")), Some(12));
        assert_eq!(extract_page_index(Path::new("page-1.PNG")), Some(1));
        assert_eq!(extract_page_index(Path::new("/tmp/x/page-7.png")), Some(7));
        // No trailing number → None (caller falls back to usize::MAX).
        assert_eq!(extract_page_index(Path::new("blank.png")), None);
    }
}
