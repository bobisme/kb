//! Integration tests for the OCR fallback (bn-2hyr).
//!
//! These exercise the full ingest pipeline end-to-end with shell-script
//! stubs standing in for `pdftoppm` and `tesseract`. The stubs run via
//! `sh` (no executable bit needed), matching the markitdown shim pattern
//! used in `crates/kb-ingest/src/lib.rs#tests::markitdown_ingest`. The
//! tests deliberately avoid PATH manipulation — `OcrOptions.command` and
//! `OcrOptions.pdftoppm_command` are configurable so we can supply
//! absolute paths to the stubs directly. That keeps the tests safe to run
//! in parallel and works on machines that don't have the real binaries
//! installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};

use kb_ingest::{
    IngestOptions, MarkitdownOptions, OcrCache, OcrOptions, PdfPageExtractOptions,
    ingest_paths_with_config,
};
use tempfile::TempDir;

fn init_kb_root() -> TempDir {
    let temp = TempDir::new().expect("create tempdir");
    fs::create_dir_all(temp.path().join("raw/inbox")).expect("create raw inbox");
    temp
}

/// Write a shell script (no executable bit) that the test will invoke
/// via `sh <script>`. Side-steps Linux's ETXTBSY race when parallel test
/// workers exec a freshly-written binary.
fn write_shim(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, body).expect("write shim");
    path
}

/// pdftoppm stub: writes one fake-PNG per "page" to `<prefix>-N.png` for
/// N in 1..=PAGES. The bytes are deterministic so the OCR cache hashes
/// stably across runs (the second `ingest_paths_with_config` is meant to
/// hit the cache and skip re-running tesseract).
const PDFTOPPM_STUB: &str = r#"#!/bin/sh
# pdftoppm stub: emit two PNGs at the given prefix. Argv layout matches
# the real tool when called from ocr.rs:
#     pdftoppm -r 300 -png <pdf> <prefix>
# We just grab the last positional arg as the prefix.
set -e
prefix="$(eval echo \$$#)"
# Distinct per-page bytes so the cache keys are different.
printf 'PAGE1-IMAGE-BYTES' > "${prefix}-1.png"
printf 'PAGE2-IMAGE-BYTES' > "${prefix}-2.png"
"#;

/// tesseract stub: reads the input PNG path (first positional arg) and
/// emits canned recognition text on stdout. We key on the page-number
/// suffix so each page produces different markdown — that's how the test
/// verifies the page markers match the page contents.
const TESSERACT_STUB: &str = r#"#!/bin/sh
# tesseract stub: tesseract <png> - -l <lang>
# We only care about the first arg (the PNG path).
set -e
png="$1"
case "$png" in
    *page-1.png)
        printf 'Recognized text from page one.\nA second line of OCR output.\n'
        ;;
    *page-2.png)
        printf 'Recognized text from page two.\n'
        ;;
    *)
        printf 'unknown page'
        ;;
esac
"#;

fn ocr_test_options(shim_dir: &Path) -> IngestOptions {
    let pdftoppm = write_shim(shim_dir, "pdftoppm.sh", PDFTOPPM_STUB);
    let tesseract = write_shim(shim_dir, "tesseract.sh", TESSERACT_STUB);
    // Markitdown shim: emits empty output to force the OCR fallback.
    let markitdown = write_shim(shim_dir, "markitdown.sh", "#!/bin/sh\nprintf ''\n");

    IngestOptions {
        dry_run: false,
        allow_empty: false,
        markitdown: MarkitdownOptions {
            enabled: true,
            command: format!("sh {}", markitdown.display()),
            extensions: vec!["pdf".to_string()],
            optional_extensions: Vec::new(),
        },
        ocr: OcrOptions {
            enabled: true,
            command: format!("sh {}", tesseract.display()),
            pdftoppm_command: format!("sh {}", pdftoppm.display()),
            language: "eng".to_string(),
            min_chars_threshold: 100,
        },
        // Page-aware extraction off so the OCR fallback path is exercised
        // exactly as before bn-3ij3 — these tests assert end-to-end on the
        // markitdown→OCR pipeline, not on the new page-aware path.
        pdf_page_extract: PdfPageExtractOptions::disabled(),
    }
}

#[test]
fn ocr_recovers_text_when_markitdown_returns_empty() {
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");

    let pdf = source_root.path().join("scan.pdf");
    fs::write(&pdf, b"%PDF-1.4 \x00 fake binary").expect("write pdf");

    let options = ocr_test_options(shim_dir.path());
    let reports = ingest_paths_with_config(kb_root.path(), std::slice::from_ref(&pdf), &options)
        .expect("ingest succeeds");
    assert_eq!(reports.len(), 1, "scan-only PDF should ingest after OCR");

    // The normalized markdown contains the OCR'd text + page markers.
    let src_id = &reports[0].ingested.document.metadata.id;
    let normalized = fs::read_to_string(
        kb_root.path().join(".kb/normalized").join(src_id).join("source.md"),
    )
    .expect("read normalized markdown");
    assert!(
        normalized.contains("<!-- kb:page 1 -->"),
        "must include page-1 marker, got:\n{normalized}"
    );
    assert!(
        normalized.contains("<!-- kb:page 2 -->"),
        "must include page-2 marker, got:\n{normalized}"
    );
    assert!(
        normalized.contains("Recognized text from page one"),
        "must include page-1 OCR text, got:\n{normalized}"
    );
    assert!(
        normalized.contains("Recognized text from page two"),
        "must include page-2 OCR text, got:\n{normalized}"
    );
}

#[test]
fn ocr_re_run_is_a_cache_hit_no_op() {
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");

    let pdf = source_root.path().join("scan.pdf");
    fs::write(&pdf, b"%PDF-1.4 \x00 fake binary").expect("write pdf");

    let options = ocr_test_options(shim_dir.path());
    let _first = ingest_paths_with_config(kb_root.path(), std::slice::from_ref(&pdf), &options)
        .expect("first ingest");

    // The cache directory must contain one entry per distinct page hash
    // (two pages → two cache files). Re-running ingest must NOT add new
    // entries; the second pass should hit the cache for both pages.
    let cache_dir = kb_root.path().join(".kb/cache/ocr");
    let entries_before: Vec<_> = fs::read_dir(&cache_dir)
        .expect("cache dir exists after first ingest")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .collect();
    assert_eq!(
        entries_before.len(),
        2,
        "expected 2 OCR cache entries (one per page), got: {entries_before:?}"
    );

    // Second pass: same bytes → same revision → ingest reports skipped.
    // The cache should not gain new entries either.
    let second = ingest_paths_with_config(kb_root.path(), &[pdf], &options)
        .expect("second ingest");
    assert_eq!(second.len(), 1);

    let entries_after: Vec<_> = fs::read_dir(&cache_dir)
        .expect("cache dir still exists")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .collect();
    assert_eq!(
        entries_after.len(),
        2,
        "second ingest must NOT add new cache entries (got {entries_after:?})"
    );
}

#[test]
fn ocr_disabled_skips_fallback_and_pdf_is_dropped() {
    // When [ingest.ocr] is disabled, an empty markitdown extraction for a
    // scan-only PDF surfaces as "skipped — empty source", matching the
    // pre-bn-2hyr behavior. No ingest record, no panic.
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");

    let pdf = source_root.path().join("scan.pdf");
    fs::write(&pdf, b"%PDF-1.4 \x00 fake binary").expect("write pdf");

    let mut options = ocr_test_options(shim_dir.path());
    options.ocr.enabled = false;
    // markitdown shim still emits empty output → empty-source skip path.
    let reports = ingest_paths_with_config(kb_root.path(), &[pdf], &options)
        .expect("ingest succeeds");
    assert!(
        reports.is_empty(),
        "with OCR off, an empty extraction must skip the file"
    );
    assert!(
        !kb_root.path().join(".kb/cache/ocr").exists()
            || fs::read_dir(kb_root.path().join(".kb/cache/ocr"))
                .map_or(0, std::iter::Iterator::count)
                == 0,
        "OCR cache must not be populated when OCR is disabled"
    );
}

#[test]
fn ocr_missing_binaries_are_warned_not_fatal() {
    // Both pdftoppm and tesseract point at paths that don't exist. Ingest
    // must succeed (no panic, no error); the file is simply skipped at
    // the empty-source gate, exactly as if OCR were disabled. This is
    // the dev/CI workflow on machines without poppler/tesseract.
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");

    let pdf = source_root.path().join("scan.pdf");
    fs::write(&pdf, b"%PDF-1.4 \x00 fake binary").expect("write pdf");

    // Markitdown is still wired up (and emits empty output) so the
    // fallback condition fires, but the OCR binaries are missing.
    let markitdown = write_shim(shim_dir.path(), "markitdown.sh", "#!/bin/sh\nprintf ''\n");
    let options = IngestOptions {
        dry_run: false,
        allow_empty: false,
        markitdown: MarkitdownOptions {
            enabled: true,
            command: format!("sh {}", markitdown.display()),
            extensions: vec!["pdf".to_string()],
            optional_extensions: Vec::new(),
        },
        ocr: OcrOptions {
            enabled: true,
            command: "/no/such/path/tesseract-bn2hyr".to_string(),
            pdftoppm_command: "/no/such/path/pdftoppm-bn2hyr".to_string(),
            language: "eng".to_string(),
            min_chars_threshold: 100,
        },
        pdf_page_extract: PdfPageExtractOptions::disabled(),
    };

    let reports = ingest_paths_with_config(kb_root.path(), &[pdf], &options)
        .expect("ingest must succeed even without OCR binaries");
    // With both OCR binaries missing, the fallback is skipped and the
    // empty markitdown extraction falls through to the empty-source skip.
    assert!(
        reports.is_empty(),
        "missing binaries → empty extraction → skip; no records expected"
    );
}

#[test]
fn ocr_cache_path_layout_is_under_dot_kb_cache() {
    // Anchors the bone's "aggressive caching: per-page-image hash → text in
    // <kb-root>/.kb/cache/ocr/<hash>.txt" requirement.
    let kb_root = init_kb_root();
    let cache = OcrCache::new(kb_root.path().join(".kb/cache/ocr"));
    cache.put("deadbeef", "hello").expect("put");
    assert!(kb_root.path().join(".kb/cache/ocr/deadbeef.txt").is_file());
    assert_eq!(
        cache.get("deadbeef").expect("get"),
        Some("hello".to_string())
    );
}
