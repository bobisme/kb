//! Shared ONNX-model file caching and download helpers.
//!
//! Both [`super::minilm::MiniLmBackend`] (bn-1rww) and
//! [`super::rerank::CrossEncoder`] (bn-1cp2) cache their ONNX weights and
//! tokenizer JSON under the same `~/.cache/kb/models/` directory and use the
//! same `KB_SEMANTIC_AUTO_DOWNLOAD` opt-out. This module factors out the
//! download/cache plumbing so both backends agree on:
//!
//! - Cache root location ([`cache_root`]).
//! - Auto-download policy ([`auto_download_enabled`]).
//! - Atomic temp-file-then-rename download semantics ([`ensure_file_cached`]).
//!
//! The two backends still own their own URLs, filenames, and once-per-process
//! "we already tried" guards — those bits are model-specific.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};

/// Env var that disables auto-download when set to a falsy value
/// (`0`, `false`, `no`, `off`).
pub const AUTO_DOWNLOAD_ENV: &str = "KB_SEMANTIC_AUTO_DOWNLOAD";

const DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 5;
const DOWNLOAD_READ_TIMEOUT_SECS: u64 = 60;

/// Cache directory used for ONNX models and tokenizers.
///
/// `~/.cache/kb/models/` on Linux, `~/Library/Caches/kb/models/` on macOS,
/// `%LOCALAPPDATA%\kb\models\` on Windows.
///
/// # Errors
///
/// Returns an error when the OS cache directory cannot be determined.
pub fn cache_root() -> Result<PathBuf> {
    let mut path = dirs::cache_dir().context("unable to determine OS cache directory")?;
    path.push("kb");
    path.push("models");
    Ok(path)
}

/// Returns `true` when auto-download is allowed (the default).
///
/// Opting out via `KB_SEMANTIC_AUTO_DOWNLOAD=0` (or any falsy value)
/// makes the loader fail with an actionable error instead of trying to
/// fetch from Hugging Face — useful in air-gapped or pinned-mirror
/// deployments.
#[must_use]
pub fn auto_download_enabled() -> bool {
    std::env::var(AUTO_DOWNLOAD_ENV).ok().is_none_or(|raw| {
        !matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

/// Resolve a download URL via env override (when set and non-empty), else the
/// caller-provided default.
#[must_use]
pub fn url_from_env_or_default(env_var: &str, default: &str) -> String {
    std::env::var(env_var)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Ensure an ONNX-model artifact exists at `path`, downloading when missing.
///
/// Downloads from `url` when the path is not already a regular file.
/// Caller passes a per-process `once_flag` `AtomicBool` so a failed
/// download is not retried in a tight loop within the same process.
///
/// Returns `Ok(())` when the file is on disk after the call.
///
/// # Errors
///
/// Returns an error when auto-download is disabled and the file is missing,
/// when the download fails (HTTP error, transport error), or when the
/// resulting file is not present after the download completes.
pub fn ensure_file_cached(
    path: &Path,
    url: &str,
    artifact_label: &str,
    once_flag: &AtomicBool,
) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }

    if !auto_download_enabled() {
        bail!(
            "{artifact_label} not found at {}. Automatic download is disabled via {AUTO_DOWNLOAD_ENV}=0",
            path.display()
        );
    }

    if once_flag.swap(true, Ordering::SeqCst) {
        bail!(
            "{artifact_label} not found at {} and auto-download was already attempted in this process",
            path.display()
        );
    }

    download_to_path(url, path, artifact_label)
        .with_context(|| format!("failed to fetch {artifact_label} to {}", path.display()))?;

    if !path.is_file() {
        bail!("{artifact_label} download completed but file was not created");
    }

    Ok(())
}

fn download_to_path(url: &str, path: &Path, artifact_label: &str) -> Result<()> {
    let parent = path.parent().with_context(|| {
        format!(
            "{artifact_label} cache path '{}' has no parent directory",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create {} cache directory {}",
            artifact_label,
            parent.display()
        )
    })?;

    let temp_path = parent.join(format!(
        "{}.download",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DOWNLOAD_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DOWNLOAD_READ_TIMEOUT_SECS))
        .build();

    let response = match agent
        .get(url)
        .set("User-Agent", "kb-query/semantic-downloader")
        .call()
    {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) => {
            bail!("{artifact_label} download failed: HTTP {code} from {url}")
        }
        Err(ureq::Error::Transport(err)) => {
            bail!("{artifact_label} download failed from {url}: {err}")
        }
    };

    {
        let mut reader = response.into_reader();
        let mut out = fs::File::create(&temp_path)
            .with_context(|| format!("failed to create temporary file {}", temp_path.display()))?;
        std::io::copy(&mut reader, &mut out)
            .with_context(|| format!("failed to write {artifact_label} download"))?;
        out.flush()
            .with_context(|| format!("failed to flush {artifact_label} download"))?;
    }

    if path.exists() {
        fs::remove_file(path).with_context(|| {
            format!(
                "failed to replace existing {} at {}",
                artifact_label,
                path.display()
            )
        })?;
    }

    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "failed to move downloaded {} from {} to {}",
            artifact_label,
            temp_path.display(),
            path.display()
        )
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_root_uses_kb_subdir() {
        let path = cache_root().expect("cache root resolves");
        let expected_tail = Path::new("kb").join("models");
        assert!(path.ends_with(expected_tail), "got {}", path.display());
    }
}
