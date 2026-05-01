//! Page-aware PDF text extraction (bn-3ij3).
//!
//! `pdftotext -layout -f N -l N` extracts a single page's text and writes it
//! to stdout. Running it once per page (we discover the page count via
//! `pdfinfo`, falling back to a single full-document extraction when
//! `pdfinfo` is unavailable) lets us emit `<!-- kb:page N -->` markers
//! between pages. Downstream consumers (the chunker, citation parser, and
//! source-page renderer) lift those markers into per-chunk `page_range`
//! metadata so a `[src-id p.7]` citation can deep-link to the right page.
//!
//! # Failure posture
//!
//! - `enabled = false`: returns `Skipped`. Caller routes through markitdown
//!   as before.
//! - `pdftotext` not on PATH: warn once, return `Skipped`. Same fallback.
//! - `pdftotext` succeeds but produces empty text on every page: returns
//!   `Empty` so the OCR fallback can still rescue scan-only PDFs. The
//!   upstream caller treats this identically to "markitdown emitted empty"
//!   — the page-aware path is opportunistic, not load-bearing.
//! - Per-page subprocess failure: warn, drop that page (the marker is still
//!   emitted), continue with the rest.
//!
//! All shell-out is via `Command::new(...)` with explicit `Stdio::null()`
//! on stdin to avoid `pdftotext` blocking on a closed pipe in degenerate
//! configurations. We mirror the OCR module's posture (binary-on-PATH check
//! before spawning, warn-once-per-process for a missing binary) so a folder
//! of mixed PDFs + markdown ingests cleanly even when poppler isn't
//! installed.

use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

/// Hard upper bound on extracted page count. Picked to keep a malformed
/// `pdfinfo` output (e.g. a multi-GB scan misreporting its page count)
/// from spawning thousands of subprocesses. Real PDFs in kb's corpus are
/// 1–500 pages; 4096 leaves headroom for academic tomes.
const MAX_PAGES: u32 = 4096;

/// User-visible settings for page-aware PDF extraction, parsed from
/// `[ingest.pdf]` in `kb.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfPageExtractOptions {
    /// Master switch. When false, PDFs route directly through markitdown
    /// (no per-page markers). Defaults on; the extraction is a no-op (warn
    /// + skip) if `pdftotext` is missing.
    pub enabled: bool,
    /// `pdftotext` command. Whitespace-split into program + leading argv,
    /// mirroring the OCR options shape so wrappers (`docker run ...`) can
    /// be configured without code changes.
    pub pdftotext_command: String,
}

impl Default for PdfPageExtractOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            pdftotext_command: "pdftotext".to_string(),
        }
    }
}

impl PdfPageExtractOptions {
    /// A disabled-by-default options bundle for callers (legacy
    /// `IngestOptions::new`, tests) that want to bypass page-aware
    /// extraction entirely.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            pdftotext_command: "pdftotext".to_string(),
        }
    }
}

/// Outcome of a page-aware PDF extraction attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdfPageExtractOutcome {
    /// Per-page extraction ran and produced markdown with `<!-- kb:page N -->`
    /// markers between pages.
    Extracted { markdown: String, pages: u32 },
    /// `pdftotext` ran but every page produced empty text. Caller should
    /// fall through to markitdown / OCR as if the path had been opted out.
    Empty,
    /// Page-aware extraction was opted out (disabled in config or
    /// `pdftotext` is missing on PATH). Caller routes through the existing
    /// preprocess / OCR stack.
    Skipped,
}

static WARNED_MISSING: AtomicBool = AtomicBool::new(false);

/// Run page-aware extraction for a PDF. Returns the assembled markdown
/// (with `<!-- kb:page N -->` markers) or [`PdfPageExtractOutcome::Skipped`]
/// when the path was opted out.
///
/// # Errors
///
/// Currently never returns an error: per-page failures degrade to a missing
/// page in the output; missing-binary / disabled cases degrade to
/// `Skipped`. The `Result` shape is preserved so a future strict mode can
/// opt into fatal failures without churning every caller.
pub fn extract_pdf_pages(
    pdf_path: &Path,
    options: &PdfPageExtractOptions,
) -> Result<PdfPageExtractOutcome> {
    if !options.enabled {
        return Ok(PdfPageExtractOutcome::Skipped);
    }

    let mut parts = options.pdftotext_command.split_ascii_whitespace();
    let Some(program) = parts.next() else {
        eprintln!("warning: ingest.pdf.pdftotext_command is empty; skipping page-aware extraction");
        return Ok(PdfPageExtractOutcome::Skipped);
    };
    let extra: Vec<String> = parts.map(str::to_owned).collect();

    if !binary_on_path(program) {
        warn_missing(program);
        return Ok(PdfPageExtractOutcome::Skipped);
    }

    // We can't trust `pdfinfo` to be installed (it ships in poppler-utils
    // alongside pdftotext, but a stripped image might omit it). Plan A:
    // ask pdftotext to report the page count via a single full-document
    // extraction with `-l 1`, increment until it stops emitting text.
    // Plan B: run the page loop and stop when a page extracts to empty AND
    // the previous page also extracted to empty (two empties in a row =
    // EOF). Plan B is robust enough on its own and avoids a second binary
    // dep, so we use it.
    let mut markdown = String::new();
    let mut emitted_pages: u32 = 0;
    let mut consecutive_empty: u32 = 0;
    let mut any_text = false;
    let mut page: u32 = 1;
    while page <= MAX_PAGES {
        match run_pdftotext_for_page(program, &extra, pdf_path, page) {
            Ok(Some(text)) => {
                consecutive_empty = 0;
                if !text.trim().is_empty() {
                    any_text = true;
                }
                append_page(&mut markdown, page, &text);
                emitted_pages = page;
            }
            Ok(None) => {
                // Empty page. Could be a blank page mid-document or past
                // EOF. Two empties in a row is our EOF heuristic.
                append_page(&mut markdown, page, "");
                emitted_pages = page;
                consecutive_empty += 1;
                if consecutive_empty >= 2 {
                    // Roll back the trailing empty pages — they're past
                    // EOF, not real document pages.
                    trim_trailing_empty_pages(&mut markdown, &mut emitted_pages);
                    break;
                }
            }
            Err(err) => {
                eprintln!(
                    "warning: pdftotext failed for page {page} of {}: {err}",
                    pdf_path.display()
                );
                // Drop the page; emit a marker but no text so the page
                // numbering stays stable for downstream chunk metadata.
                append_page(&mut markdown, page, "");
                emitted_pages = page;
                consecutive_empty += 1;
                if consecutive_empty >= 2 {
                    trim_trailing_empty_pages(&mut markdown, &mut emitted_pages);
                    break;
                }
            }
        }
        page = page.saturating_add(1);
    }

    if emitted_pages == 0 || !any_text {
        return Ok(PdfPageExtractOutcome::Empty);
    }

    Ok(PdfPageExtractOutcome::Extracted {
        markdown,
        pages: emitted_pages,
    })
}

/// Append one page's text under a `<!-- kb:page N -->` marker. The marker
/// is always emitted, even when the page text is empty, so consumers can
/// see a blank page existed.
fn append_page(out: &mut String, page: u32, text: &str) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    if !out.is_empty() {
        out.push('\n');
    }
    writeln!(out, "<!-- kb:page {page} -->").expect("writeln to String");
    let trimmed = text.trim_end_matches('\n');
    out.push_str(trimmed);
    if !trimmed.is_empty() {
        out.push('\n');
    }
}

/// Drop trailing empty pages from `out` (and decrement `emitted_pages`).
/// "Empty" means: a `<!-- kb:page N -->` line followed only by whitespace
/// up to either the next marker or EOF. Used after our two-consecutive-
/// empty heuristic decides we ran past the last real page.
fn trim_trailing_empty_pages(out: &mut String, emitted_pages: &mut u32) {
    while *emitted_pages > 0 {
        let marker = format!("<!-- kb:page {} -->", *emitted_pages);
        let Some(idx) = out.rfind(&marker) else {
            break;
        };
        let tail = &out[idx + marker.len()..];
        if tail.chars().all(char::is_whitespace) {
            out.truncate(idx);
            // Drop the surrounding blank line we put in front of the marker.
            while out.ends_with('\n') || out.ends_with(' ') {
                out.pop();
            }
            *emitted_pages -= 1;
        } else {
            break;
        }
    }
}

fn run_pdftotext_for_page(
    program: &str,
    extra: &[String],
    pdf: &Path,
    page: u32,
) -> Result<Option<String>> {
    // `pdftotext -layout -f N -l N <pdf> -` writes the page text to
    // stdout. -layout preserves column structure, which matters for
    // tabular PDFs (paper figures, financial reports, ...).
    let output = Command::new(program)
        .args(extra)
        .arg("-layout")
        .arg("-f")
        .arg(page.to_string())
        .arg("-l")
        .arg(page.to_string())
        .arg(pdf)
        .arg("-")
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("spawn `{program}` for page {page}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trimmed = stderr.trim();
        // `pdftotext` exits 99 with "Wrong page range" when the requested
        // page is past the document's last page. That's the normal
        // termination signal for our page-by-page loop — surface it as
        // an empty page so the EOF heuristic in the caller can advance
        // without spamming warnings.
        if trimmed.is_empty() || is_past_eof_error(trimmed) {
            return Ok(None);
        }
        // Other failures are surfaced to the caller as warnings.
        anyhow::bail!("pdftotext exit {:?}: {trimmed}", output.status.code());
    }

    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    if text.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(text))
}

/// Recognize the stderr output `pdftotext` emits when asked for a page
/// past the document's last page. The exact wording differs across
/// poppler versions but always contains "page range" so we substring-
/// match conservatively.
fn is_past_eof_error(stderr: &str) -> bool {
    let lower = stderr.to_ascii_lowercase();
    lower.contains("wrong page range")
        || lower.contains("page range given")
        || (lower.contains("first page") && lower.contains("can not be after"))
}

fn binary_on_path(name: &str) -> bool {
    let p = Path::new(name);
    if p.is_absolute() || name.contains('/') {
        return p.is_file();
    }
    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path_var).any(|dir| dir.join(name).is_file())
}

fn warn_missing(command: &str) {
    if WARNED_MISSING.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "warning: page-aware PDF extraction skipped: `{command}` not found on PATH. \
         Install poppler-utils, or disable with `ingest.pdf.enabled = false` in kb.toml \
         (citations will fall back to whole-document references)."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Returns true if a real `pdftotext` is on the test machine's PATH.
    /// Several tests below need a working binary; on CI without poppler
    /// they self-skip rather than fail.
    fn pdftotext_available() -> bool {
        binary_on_path("pdftotext")
    }

    /// Builds a minimal two-page PDF using a hand-rolled PDF body. The
    /// content stream uses `Tj` with literal strings so `pdftotext` can
    /// recover the text. Pages are sized 612×792 (US Letter) and share
    /// one Helvetica font resource. Generated inline so the test doesn't
    /// pull in a dependency just to make a fixture.
    ///
    /// This is intentionally minimal — kb's PDF parsing is delegated to
    /// poppler/pdftotext, so the test only needs a valid-enough PDF for
    /// poppler to recognize.
    fn write_two_page_pdf(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("two-pages.pdf");
        // Build the body first so we can compute the xref offsets.
        let stream1 = "BT /F1 12 Tf 72 720 Td (Page one says hello.) Tj ET";
        let stream2 = "BT /F1 12 Tf 72 720 Td (Page two has different text.) Tj ET";

        let mut objects: Vec<String> = Vec::new();
        objects.push("<< /Type /Catalog /Pages 2 0 R >>".to_string());
        objects.push("<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_string());
        objects.push(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R \
             /Resources << /Font << /F1 7 0 R >> >> >>"
                .to_string(),
        );
        objects.push(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R \
             /Resources << /Font << /F1 7 0 R >> >> >>"
                .to_string(),
        );
        objects.push(format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            stream1.len(),
            stream1
        ));
        objects.push(format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            stream2.len(),
            stream2
        ));
        objects.push(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        );

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n");
        let mut offsets: Vec<usize> = Vec::with_capacity(objects.len());
        for (idx, body) in objects.iter().enumerate() {
            offsets.push(bytes.len());
            let header = format!("{} 0 obj\n", idx + 1);
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(body.as_bytes());
            bytes.extend_from_slice(b"\nendobj\n");
        }
        let xref_offset = bytes.len();
        let xref_count = objects.len() + 1; // +1 for object 0 (free).
        let mut xref = format!("xref\n0 {xref_count}\n0000000000 65535 f \n");
        for off in &offsets {
            use std::fmt::Write as _;
            writeln!(xref, "{off:010} 00000 n ").expect("write to String");
        }
        bytes.extend_from_slice(xref.as_bytes());
        let trailer = format!(
            "trailer\n<< /Size {xref_count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        );
        bytes.extend_from_slice(trailer.as_bytes());

        fs::write(&path, &bytes).expect("write pdf fixture");
        path
    }

    #[test]
    fn disabled_returns_skipped() {
        let tmp = TempDir::new().expect("tempdir");
        let pdf = tmp.path().join("doc.pdf");
        fs::write(&pdf, b"%PDF-1.4 fake").expect("write pdf");
        let outcome =
            extract_pdf_pages(&pdf, &PdfPageExtractOptions::disabled()).expect("never errors");
        assert_eq!(outcome, PdfPageExtractOutcome::Skipped);
    }

    #[test]
    fn missing_binary_returns_skipped() {
        let tmp = TempDir::new().expect("tempdir");
        let pdf = tmp.path().join("doc.pdf");
        fs::write(&pdf, b"%PDF-1.4 fake").expect("write pdf");
        let opts = PdfPageExtractOptions {
            enabled: true,
            pdftotext_command: "/no/such/path/pdftotext-bn3ij3".to_string(),
        };
        let outcome = extract_pdf_pages(&pdf, &opts).expect("never errors");
        assert_eq!(outcome, PdfPageExtractOutcome::Skipped);
    }

    #[test]
    fn binary_on_path_resolves_relative_against_path() {
        let tmp = TempDir::new().expect("mktemp");
        assert!(!binary_on_path("definitely-not-a-real-binary-bn3ij3"));
        let real = tmp.path().join("real");
        fs::write(&real, b"#!/bin/sh\n").expect("write real");
        assert!(binary_on_path(real.to_str().expect("utf-8 path")));
    }

    #[test]
    fn append_page_inserts_marker_before_content() {
        let mut buf = String::new();
        append_page(&mut buf, 1, "first page text");
        assert!(buf.starts_with("<!-- kb:page 1 -->"));
        assert!(buf.contains("first page text"));

        append_page(&mut buf, 2, "second page text");
        assert!(buf.contains("<!-- kb:page 2 -->"));
        // Pages are separated by a blank line so the markers are visible
        // when reading the source file directly.
        let between = buf
            .split("<!-- kb:page 2 -->")
            .next()
            .expect("split before page-2");
        assert!(
            between.ends_with("\n\n"),
            "expected blank line between pages, got: {between:?}"
        );
    }

    #[test]
    fn trim_trailing_empty_pages_drops_eof_pages() {
        // Three pages: 1 with text, 2 empty, 3 empty (past EOF). After
        // trim, only page 1 should remain.
        let mut buf = String::new();
        append_page(&mut buf, 1, "real content");
        append_page(&mut buf, 2, "");
        append_page(&mut buf, 3, "");
        let mut emitted: u32 = 3;
        trim_trailing_empty_pages(&mut buf, &mut emitted);
        assert_eq!(emitted, 1);
        assert!(buf.contains("<!-- kb:page 1 -->"));
        assert!(!buf.contains("<!-- kb:page 2 -->"));
        assert!(!buf.contains("<!-- kb:page 3 -->"));
    }

    #[test]
    fn extract_pdf_pages_recovers_two_page_text() {
        if !pdftotext_available() {
            eprintln!("skipping: pdftotext not on PATH");
            return;
        }
        let tmp = TempDir::new().expect("tempdir");
        let pdf = write_two_page_pdf(tmp.path());

        let outcome =
            extract_pdf_pages(&pdf, &PdfPageExtractOptions::default()).expect("extract");
        let PdfPageExtractOutcome::Extracted { markdown, pages } = outcome else {
            panic!("expected Extracted, got {outcome:?}");
        };
        assert_eq!(pages, 2, "two-page fixture should report 2 pages");
        assert!(
            markdown.contains("<!-- kb:page 1 -->"),
            "markdown missing page-1 marker:\n{markdown}"
        );
        assert!(
            markdown.contains("<!-- kb:page 2 -->"),
            "markdown missing page-2 marker:\n{markdown}"
        );
        assert!(
            markdown.contains("Page one"),
            "markdown missing page-1 content:\n{markdown}"
        );
        assert!(
            markdown.contains("Page two"),
            "markdown missing page-2 content:\n{markdown}"
        );
    }
}
