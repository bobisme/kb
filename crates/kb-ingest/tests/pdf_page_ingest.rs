//! Integration tests for page-aware PDF ingest (bn-3ij3).
//!
//! These exercise the full ingest pipeline end-to-end with a shell-script
//! stub standing in for `pdftotext`. The stub runs via `sh` (no executable
//! bit needed), matching the markitdown / OCR shim pattern. The tests
//! deliberately avoid PATH manipulation — `PdfPageExtractOptions.pdftotext_command`
//! is configurable so we can supply absolute paths to the stub directly.
//! That keeps the tests safe to run in parallel and works on machines that
//! don't have the real binary installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};

use kb_ingest::{
    IngestOptions, MarkitdownOptions, OcrOptions, PdfPageExtractOptions,
    ingest_paths_with_config,
};
use tempfile::TempDir;

fn init_kb_root() -> TempDir {
    let temp = TempDir::new().expect("create tempdir");
    fs::create_dir_all(temp.path().join("raw/inbox")).expect("create raw inbox");
    temp
}

/// Write a shell script (no executable bit) that the test invokes via
/// `sh <script>`. Side-steps Linux's ETXTBSY race when parallel test
/// workers exec a freshly-written binary.
fn write_shim(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, body).expect("write shim");
    path
}

/// `pdftotext` stub: takes `-layout -f N -l N <pdf> -` and emits canned
/// per-page text. Page 1, 2 produce real text; page 3 is empty (signaling
/// EOF to the caller via the consecutive-empty heuristic).
const PDFTOTEXT_STUB: &str = r#"#!/bin/sh
# pdftotext stub: pdftotext -layout -f N -l N <pdf> -
# We grab the page number from the -f arg.
set -e
page=""
while [ $# -gt 0 ]; do
    case "$1" in
        -f)
            shift
            page="$1"
            ;;
    esac
    shift
done
case "$page" in
    1)
        printf 'First-page-text from a fake PDF.\nA second line on page one.\n'
        ;;
    2)
        printf 'Page two carries different text entirely.\n'
        ;;
    *)
        # Past EOF: emit empty stdout so the caller's two-consecutive-
        # empty heuristic stops the page loop.
        printf ''
        ;;
esac
"#;

fn pdf_test_options(shim_dir: &Path) -> IngestOptions {
    let pdftotext = write_shim(shim_dir, "pdftotext.sh", PDFTOTEXT_STUB);
    IngestOptions {
        dry_run: false,
        allow_empty: false,
        // Markitdown is on but its disabled-output extension list excludes
        // PDFs in this test — we want the page-aware path to handle PDFs.
        // We instead leave markitdown configured for non-PDF formats only,
        // so a PDF that fails the page-aware path would still be caught.
        markitdown: MarkitdownOptions {
            enabled: true,
            command: "markitdown".to_string(),
            extensions: vec!["docx".to_string()], // exclude pdf
            optional_extensions: Vec::new(),
        },
        // OCR off so we measure only the page-aware path.
        ocr: OcrOptions::disabled(),
        pdf_page_extract: PdfPageExtractOptions {
            enabled: true,
            pdftotext_command: format!("sh {}", pdftotext.display()),
        },
    }
}

#[test]
fn page_aware_extraction_emits_per_page_markers() {
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");

    let pdf = source_root.path().join("paper.pdf");
    fs::write(&pdf, b"%PDF-1.4 fake bytes").expect("write pdf");

    let options = pdf_test_options(shim_dir.path());
    let reports = ingest_paths_with_config(kb_root.path(), std::slice::from_ref(&pdf), &options)
        .expect("ingest succeeds");
    assert_eq!(reports.len(), 1, "page-aware path should ingest the PDF");

    // The normalized markdown contains per-page kb:page markers and content.
    let src_id = &reports[0].ingested.document.metadata.id;
    let normalized = fs::read_to_string(
        kb_root
            .path()
            .join(".kb/normalized")
            .join(src_id)
            .join("source.md"),
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
        normalized.contains("First-page-text"),
        "must include page-1 content, got:\n{normalized}"
    );
    assert!(
        normalized.contains("Page two"),
        "must include page-2 content, got:\n{normalized}"
    );
    // Past-EOF page should be trimmed off by the EOF heuristic.
    assert!(
        !normalized.contains("<!-- kb:page 3 -->"),
        "past-EOF page should be trimmed:\n{normalized}"
    );

    // Original PDF bytes must still be archived under raw/inbox.
    let rev_dir = kb_root
        .path()
        .join("raw/inbox")
        .join(src_id)
        .join(&reports[0].ingested.revision.metadata.id);
    let original = rev_dir.join("original.pdf");
    assert!(original.is_file(), "original.pdf should be archived");
    assert_eq!(
        fs::read(&original).expect("read original"),
        b"%PDF-1.4 fake bytes",
        "original bytes must round-trip verbatim",
    );
}

#[test]
fn page_aware_disabled_falls_through_to_markitdown() {
    // With page-aware extraction off, the existing markitdown / OCR
    // pipeline must still handle the PDF. We supply a markitdown shim
    // configured for PDFs to confirm the fallback is wired up.
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");
    let markitdown = write_shim(
        shim_dir.path(),
        "markitdown.sh",
        "#!/bin/sh\nprintf '# Markitdown produced this\\n\\nbody\\n'\n",
    );

    let pdf = source_root.path().join("paper.pdf");
    fs::write(&pdf, b"%PDF-1.4 fake bytes").expect("write pdf");

    let options = IngestOptions {
        dry_run: false,
        allow_empty: false,
        markitdown: MarkitdownOptions {
            enabled: true,
            command: format!("sh {}", markitdown.display()),
            extensions: vec!["pdf".to_string()],
            optional_extensions: Vec::new(),
        },
        ocr: OcrOptions::disabled(),
        // Disabled.
        pdf_page_extract: PdfPageExtractOptions::disabled(),
    };
    let reports = ingest_paths_with_config(kb_root.path(), &[pdf], &options)
        .expect("ingest succeeds");
    assert_eq!(reports.len(), 1);

    let src_id = &reports[0].ingested.document.metadata.id;
    let normalized = fs::read_to_string(
        kb_root
            .path()
            .join(".kb/normalized")
            .join(src_id)
            .join("source.md"),
    )
    .expect("read normalized");
    assert!(
        normalized.contains("Markitdown produced this"),
        "fallback must use markitdown, got:\n{normalized}"
    );
    // No page markers emitted — this path doesn't know about pages.
    assert!(
        !normalized.contains("<!-- kb:page"),
        "fallback must not invent page markers:\n{normalized}"
    );
}

#[test]
fn pdftotext_missing_falls_through_to_existing_pipeline() {
    // pdftotext command points at a path that doesn't exist. The bn-3ij3
    // path skips, the markitdown shim handles the conversion, and the
    // PDF still gets ingested.
    let kb_root = init_kb_root();
    let shim_dir = TempDir::new().expect("mktemp shim");
    let source_root = TempDir::new().expect("mktemp source");

    let markitdown = write_shim(
        shim_dir.path(),
        "markitdown.sh",
        "#!/bin/sh\nprintf '# fallback\\nbody\\n'\n",
    );

    let pdf = source_root.path().join("paper.pdf");
    fs::write(&pdf, b"%PDF-1.4 fake bytes").expect("write pdf");

    let options = IngestOptions {
        dry_run: false,
        allow_empty: false,
        markitdown: MarkitdownOptions {
            enabled: true,
            command: format!("sh {}", markitdown.display()),
            extensions: vec!["pdf".to_string()],
            optional_extensions: Vec::new(),
        },
        ocr: OcrOptions::disabled(),
        pdf_page_extract: PdfPageExtractOptions {
            enabled: true,
            pdftotext_command: "/no/such/path/pdftotext-bn3ij3".to_string(),
        },
    };
    let reports = ingest_paths_with_config(kb_root.path(), &[pdf], &options)
        .expect("missing binary must not fail ingest");
    assert_eq!(reports.len(), 1);
    let src_id = &reports[0].ingested.document.metadata.id;
    let normalized = fs::read_to_string(
        kb_root
            .path()
            .join(".kb/normalized")
            .join(src_id)
            .join("source.md"),
    )
    .expect("read normalized");
    assert!(
        normalized.contains("fallback"),
        "missing pdftotext should fall through to markitdown",
    );
}
