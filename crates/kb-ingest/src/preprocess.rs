//! Preprocess non-markdown source files through Microsoft's `markitdown`.
//!
//! `kb ingest` natively handles text/markdown. Binary formats (PDF, Word,
//! Excel, …) would be rejected by the binary-detection gate in `ingest_file`.
//! To support them we shell out to `markitdown <path>` before that gate runs
//! and swap the file's bytes for the subprocess's stdout, which is
//! UTF-8 markdown. The original bytes are still archived alongside the
//! converted markdown under `raw/inbox/<src>/<rev>/original.<ext>`.
//!
//! The set of extensions routed through markitdown is configurable — see
//! [`MarkitdownOptions`]. By default the "safe" set runs automatically; the
//! "optional" set (OCR on images, audio transcription) is opt-in because
//! those conversions can be slow or require cloud APIs.
//!
//! Failure modes are all non-fatal: if `markitdown` is missing we warn once
//! per process and skip; if a conversion fails or produces empty output we
//! warn and skip the file. The caller then proceeds as if the original file
//! were simply unrecognized, which matches the existing behavior for
//! unsupported binaries.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

/// Warn at most once per process that `markitdown` isn't on PATH. Ingesting a
/// folder with a mix of markdown + PDFs shouldn't spam the log with one
/// identical warning per PDF.
static WARNED_MISSING: AtomicBool = AtomicBool::new(false);

/// User-visible settings for the markitdown preprocessing step, parsed from
/// `[ingest.markitdown]` in `kb.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkitdownOptions {
    /// Master switch. When false, no file is ever routed through markitdown
    /// regardless of its extension.
    pub enabled: bool,
    /// Shell-style command to invoke. Split on ASCII whitespace to produce
    /// the program + leading argv. The input file path is appended as the
    /// final argument. Usually just `"markitdown"`, but users can point at
    /// a wrapper script (e.g. `"uvx markitdown"`) or a custom container
    /// invocation without needing a separate exec layer.
    pub command: String,
    /// Extensions (without the leading dot, lowercased) that are always
    /// converted when `enabled == true`.
    pub extensions: Vec<String>,
    /// Extensions that require an explicit opt-in (slower or cloud-backed
    /// conversions — OCR on .png, audio transcription on .mp3, etc.). Files
    /// matching these are only converted if the user has added them to
    /// `extensions` in their config.
    ///
    /// We keep this field so round-tripping `kb.toml` via serde preserves the
    /// default advisory set, but runtime dispatch only consults `extensions`.
    pub optional_extensions: Vec<String>,
}

impl Default for MarkitdownOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "markitdown".to_string(),
            extensions: default_extensions(),
            optional_extensions: default_optional_extensions(),
        }
    }
}

impl MarkitdownOptions {
    /// Returns `true` if this file should be run through markitdown based on
    /// its extension alone. The caller must still check `self.enabled`.
    #[must_use]
    pub fn matches(&self, path: &Path) -> bool {
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            return false;
        };
        let ext_lower = ext.to_ascii_lowercase();
        self.extensions.iter().any(|e| e == &ext_lower)
    }

    /// A disabled-by-default options bundle, useful for call sites (tests,
    /// dry-run analyses) that want to bypass preprocessing entirely.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            command: "markitdown".to_string(),
            extensions: Vec::new(),
            optional_extensions: Vec::new(),
        }
    }
}

/// Default extensions converted when `ingest.markitdown` is enabled. Mirrors
/// the bone's spec: formats that are effectively always worth converting
/// (fast, local, no external API cost).
#[must_use]
pub fn default_extensions() -> Vec<String> {
    [
        "pdf", "docx", "pptx", "xlsx", "xls", "doc", "ppt", "html", "htm", "xml", "csv", "epub",
        "msg",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

/// Opt-in extensions: OCR on images and transcription on audio. Users must
/// list them in `extensions` explicitly to enable.
#[must_use]
pub fn default_optional_extensions() -> Vec<String> {
    ["png", "jpg", "jpeg", "mp3", "wav", "m4a"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Output of a successful markitdown conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    /// The markdown bytes captured from markitdown's stdout.
    pub markdown: Vec<u8>,
}

/// Result of considering a file for markitdown preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Preprocessed {
    /// Preprocessing didn't apply to this file (disabled, extension
    /// doesn't match, or the binary is missing on PATH). The caller should
    /// fall through to the normal ingest path as if preprocessing had
    /// never been attempted.
    NotApplicable,
    /// The file was routed through markitdown successfully. Its stdout
    /// (markdown bytes) is available for downstream stages.
    Converted(Converted),
    /// Preprocessing applied but failed (subprocess error, nonzero exit,
    /// empty stdout). A warning has already been emitted; the caller
    /// should SKIP the file entirely — falling through to the raw bytes
    /// would re-feed an unsupported format into the text/binary gate.
    SkipFile,
}

/// Runs `markitdown <path>` if and only if the options enable it for this
/// file's extension.
///
/// See [`Preprocessed`] for the return variants. Subprocess problems never
/// bubble up as `Err`; we warn and surface them as `SkipFile` so a batch
/// ingest doesn't abort on one broken PDF.
///
/// # Errors
/// Currently never returns an error. The `Result` shape is preserved for
/// future strict modes where the caller may want to opt into fatal
/// preprocessing failures.
pub fn maybe_preprocess(path: &Path, options: &MarkitdownOptions) -> Result<Preprocessed> {
    if !options.enabled {
        return Ok(Preprocessed::NotApplicable);
    }
    if !options.matches(path) {
        return Ok(Preprocessed::NotApplicable);
    }

    let mut parts = options.command.split_ascii_whitespace();
    let Some(program) = parts.next() else {
        eprintln!(
            "warning: ingest.markitdown.command is empty; skipping preprocessing"
        );
        return Ok(Preprocessed::NotApplicable);
    };
    let extra_args: Vec<&str> = parts.collect();

    let output = match Command::new(program)
        .args(&extra_args)
        .arg(path)
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                warn_missing_binary(program);
                // Missing binary is configuration-level, not per-file —
                // let the caller see the raw bytes and decide (explicit
                // PDFs will still error, directory walks will skip binary
                // content with a warning).
                return Ok(Preprocessed::NotApplicable);
            }
            eprintln!(
                "warning: failed to spawn `{} {}`: {err}",
                options.command,
                path.display()
            );
            return Ok(Preprocessed::SkipFile);
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trimmed = stderr.trim();
        if trimmed.is_empty() {
            eprintln!(
                "warning: markitdown conversion failed for {} (exit code {:?})",
                path.display(),
                output.status.code()
            );
        } else {
            eprintln!(
                "warning: markitdown conversion failed for {}: {trimmed}",
                path.display()
            );
        }
        return Ok(Preprocessed::SkipFile);
    }

    let markdown = output.stdout;
    // Mirror the bn-40r empty-warning pattern: producing nothing useful is
    // treated as a skip, not a hard error.
    if is_effectively_empty(&markdown) {
        eprintln!(
            "warning: markitdown produced empty output for {}; skipping",
            path.display()
        );
        return Ok(Preprocessed::SkipFile);
    }

    Ok(Preprocessed::Converted(Converted { markdown }))
}

fn warn_missing_binary(command: &str) {
    if WARNED_MISSING.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "warning: `{command}` not found on PATH; skipping non-markdown sources. \
         Install from https://github.com/microsoft/markitdown or disable with \
         `ingest.markitdown.enabled = false` in kb.toml."
    );
}

/// Check whether markitdown's stdout is worth keeping. All-whitespace /
/// zero-length output is treated as a conversion failure so we don't pollute
/// the KB with empty source documents. Invalid UTF-8 is surprising from
/// markitdown but we'd rather surface it to the normal pipeline (where the
/// binary check will warn) than silently drop it here.
fn is_effectively_empty(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    std::str::from_utf8(bytes).is_ok_and(|text| text.chars().all(char::is_whitespace))
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Writes a shell-script shim into `dir`. We deliberately DO NOT mark
    /// it executable: the tests invoke it via `sh <script>` so we avoid
    /// Linux's ETXTBSY race (parallel workers exec'ing a just-written
    /// binary while another thread still holds its write handle open).
    fn write_shim(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        fs::write(&path, body).expect("write shim");
        path
    }

    fn make_pdf(dir: &Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        fs::write(&path, b"%PDF-1.4 fake").expect("write pdf");
        path
    }

    /// Options pointing at a shim script run via `sh`. Whitespace splitting
    /// in `maybe_preprocess` produces argv = [`sh`, `<shim>`, `<file>`].
    fn pdf_opts(shim_path: &Path) -> MarkitdownOptions {
        MarkitdownOptions {
            enabled: true,
            command: format!("sh {}", shim_path.display()),
            extensions: vec!["pdf".to_string()],
            optional_extensions: Vec::new(),
        }
    }

    #[test]
    fn matches_respects_configured_extensions() {
        let opts = MarkitdownOptions::default();
        assert!(opts.matches(Path::new("/tmp/x.pdf")));
        assert!(opts.matches(Path::new("/tmp/x.PDF"))); // case-insensitive
        assert!(opts.matches(Path::new("/tmp/x.docx")));
        assert!(!opts.matches(Path::new("/tmp/notes.md")));
        assert!(!opts.matches(Path::new("/tmp/notes.txt")));
        // Optional extensions are not auto-matched.
        assert!(!opts.matches(Path::new("/tmp/pic.png")));
        assert!(!opts.matches(Path::new("/tmp/rec.mp3")));
    }

    #[test]
    fn disabled_skips_everything() {
        let opts = MarkitdownOptions::disabled();
        let result =
            maybe_preprocess(Path::new("/tmp/whatever.pdf"), &opts).expect("preprocess ok");
        assert!(matches!(result, Preprocessed::NotApplicable));
    }

    #[test]
    fn unmatched_extension_skips() {
        let opts = MarkitdownOptions::default();
        let result =
            maybe_preprocess(Path::new("/tmp/notes.md"), &opts).expect("preprocess ok");
        assert!(matches!(result, Preprocessed::NotApplicable));
    }

    #[test]
    fn successful_shim_returns_stdout() {
        let tmp = TempDir::new().expect("mktemp");
        let shim = write_shim(
            tmp.path(),
            "markitdown",
            "#!/bin/sh\nprintf '# Converted\\n\\nbody\\n'\n",
        );
        let pdf = make_pdf(tmp.path(), "doc.pdf");

        let opts = pdf_opts(&shim);
        let result = maybe_preprocess(&pdf, &opts).expect("preprocess ok");
        let Preprocessed::Converted(c) = result else {
            panic!("expected Converted, got {result:?}");
        };
        assert_eq!(c.markdown, b"# Converted\n\nbody\n");
    }

    #[test]
    fn nonzero_exit_yields_skip_with_warning() {
        let tmp = TempDir::new().expect("mktemp");
        let shim = write_shim(
            tmp.path(),
            "markitdown",
            "#!/bin/sh\necho boom >&2\nexit 7\n",
        );
        let pdf = make_pdf(tmp.path(), "bad.pdf");

        let opts = pdf_opts(&shim);
        let result = maybe_preprocess(&pdf, &opts).expect("preprocess ok");
        assert!(matches!(result, Preprocessed::SkipFile));
    }

    #[test]
    fn empty_stdout_yields_skip() {
        let tmp = TempDir::new().expect("mktemp");
        let shim = write_shim(tmp.path(), "markitdown", "#!/bin/sh\nprintf ''\n");
        let pdf = make_pdf(tmp.path(), "blank.pdf");

        let opts = pdf_opts(&shim);
        let result = maybe_preprocess(&pdf, &opts).expect("preprocess ok");
        assert!(matches!(result, Preprocessed::SkipFile));
    }

    #[test]
    fn missing_binary_is_not_applicable() {
        // Config-level failure: the user has `enabled = true` but no
        // binary. Rather than skip every matched file, we surface this as
        // NotApplicable so the caller's existing binary/unsupported
        // handling takes over (and emits its own warnings).
        let opts = MarkitdownOptions {
            enabled: true,
            command: "/nonexistent/definitely/not/markitdown-xyz".to_string(),
            extensions: vec!["pdf".to_string()],
            optional_extensions: Vec::new(),
        };
        let tmp = TempDir::new().expect("mktemp");
        let pdf = make_pdf(tmp.path(), "doc.pdf");
        let result = maybe_preprocess(&pdf, &opts).expect("preprocess ok");
        assert!(matches!(result, Preprocessed::NotApplicable));
    }

    #[test]
    fn whitespace_only_stdout_is_empty() {
        let tmp = TempDir::new().expect("mktemp");
        let shim = write_shim(
            tmp.path(),
            "markitdown",
            "#!/bin/sh\nprintf '   \\n\\t\\n'\n",
        );
        let pdf = make_pdf(tmp.path(), "ws.pdf");

        let opts = pdf_opts(&shim);
        let result = maybe_preprocess(&pdf, &opts).expect("preprocess ok");
        assert!(matches!(result, Preprocessed::SkipFile));
    }
}
