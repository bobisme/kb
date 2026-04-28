//! Model auto-download with content-hash verification.
//!
//! Models live at `~/.kb/models/<sha256>.<ext>`. On first call, missing
//! models are downloaded from MIT-licensed mirrors (no HuggingFace
//! authentication required). The sha256 is verified after download; a
//! mismatch deletes the file and returns an error.
//!
//! Override the cache root with `TranscribeConfig::model_cache_root` or
//! the `KB_MODELS_DIR` env var.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("model download failed for {url}: {source}")]
    Download {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("hash mismatch for {name}: expected {expected}, got {actual}")]
    HashMismatch {
        name: String,
        expected: String,
        actual: String,
    },
    #[error("could not determine cache root (HOME unset)")]
    NoCacheRoot,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Human-readable name for logs and errors.
    pub name: &'static str,
    /// Local filename (extension matters: .onnx for ONNX, .bin for whisper GGML).
    pub filename: &'static str,
    /// Direct download URL — must be MIT/Apache-licensed and unauthenticated.
    pub url: &'static str,
    /// Algorithm-prefixed content hash. Format: `<algo>$<lowercase-hex>`.
    /// Currently supported algos: `blake3`, `sha256`. Set to
    /// `"unverified$"` to skip verification (development only).
    pub hash: &'static str,
    pub size_bytes: u64,
}

/// MIT-licensed pyannote-segmentation-3.0 ONNX (~5.7 MB).
///
/// We use the thewh1teagle release mirror (the pyannote-rs author's own
/// export) because the onnx-community mirror at HuggingFace renames its
/// output tensor to `logits` while pyannote-rs's `get_segments`
/// hardcodes a lookup for the `output` tensor. Both are MIT-licensed
/// conversions of the same upstream pyannote weights.
pub const SEGMENTATION_3_0: ModelSpec = ModelSpec {
    name: "pyannote-segmentation-3.0",
    filename: "segmentation-3.0.onnx",
    url: "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx",
    hash: "blake3$6eaeab89d667c33640531d9ec036649287e6349954d50efdd257c5c0203df65f",
    size_bytes: 5_983_836,
};

/// wespeaker-voxceleb-resnet34-LM ONNX, Apache-2.0. ~25 MB.
/// Mirror: k2-fsa/sherpa-onnx GitHub releases.
pub const WESPEAKER_RESNET34_LM: ModelSpec = ModelSpec {
    name: "wespeaker-voxceleb-resnet34-LM",
    filename: "wespeaker-voxceleb-resnet34-LM.onnx",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
    hash: "blake3$0efe839fb7f400913d0be2869158687c402921ed8cac492f18d60f0d1bbd9b65",
    size_bytes: 26_530_000,
};

/// whisper-large-v3 GGML q5_0, MIT (whisper.cpp / OpenAI weights). ~1.1 GB.
/// Hosted on HuggingFace (ungated), via the ggerganov whisper.cpp repo.
pub const WHISPER_LARGE_V3: ModelSpec = ModelSpec {
    name: "whisper-large-v3-q5_0",
    filename: "whisper-large-v3-q5_0.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
    hash: "blake3$2324b561c8e0fcf4cd2c0fde6b19a24a11ac27a4b629fe6687c794a309587840",
    size_bytes: 1_100_000_000,
};

#[derive(Debug, Clone)]
pub struct ModelHandles {
    pub segmentation: PathBuf,
    pub embedding: PathBuf,
    pub whisper: PathBuf,
}

pub struct ModelStore {
    root: PathBuf,
}

impl ModelStore {
    /// Create a `ModelStore` rooted at `override_root`, `$KB_MODELS_DIR`,
    /// or `~/.kb/models` (in that order). Creates the directory if missing.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::NoCacheRoot` if no root can be determined and
    /// `ModelError::Io` if the directory cannot be created.
    pub fn new(override_root: Option<PathBuf>) -> Result<Self, ModelError> {
        let root = match override_root {
            Some(p) => p,
            None => match std::env::var("KB_MODELS_DIR") {
                Ok(p) => PathBuf::from(p),
                Err(_) => default_root().ok_or(ModelError::NoCacheRoot)?,
            },
        };
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Ensure all required models are downloaded; return resolved paths.
    ///
    /// # Errors
    ///
    /// Surfaces I/O, network, or hash-mismatch errors from the underlying
    /// `ensure` calls.
    pub fn ensure_all(&self) -> Result<ModelHandles, ModelError> {
        Ok(ModelHandles {
            segmentation: self.ensure(&SEGMENTATION_3_0)?,
            embedding: self.ensure(&WESPEAKER_RESNET34_LM)?,
            whisper: self.ensure(&WHISPER_LARGE_V3)?,
        })
    }

    /// Ensure a single model spec is present and verified. Downloads if
    /// missing or hash-mismatched, then verifies the new file.
    ///
    /// # Errors
    ///
    /// I/O failures, network failures, or hash mismatches between the
    /// downloaded bytes and `spec.hash`.
    pub fn ensure(&self, spec: &ModelSpec) -> Result<PathBuf, ModelError> {
        let target = self.root.join(spec.filename);
        if target.exists() && verify_hash(&target, spec.hash).unwrap_or(false) {
            return Ok(target);
        }
        download_to(&target, spec)?;
        Ok(target)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgo {
    Blake3,
    Sha256,
    Unverified,
}

fn parse_hash_spec(spec: &str) -> Result<(HashAlgo, &str), ModelError> {
    let (algo, value) = spec.split_once('$').ok_or_else(|| ModelError::HashMismatch {
        name: "<spec>".into(),
        expected: spec.into(),
        actual: "missing algo prefix; expected `algo$value`".into(),
    })?;
    let algo = match algo {
        "blake3" => HashAlgo::Blake3,
        "sha256" => HashAlgo::Sha256,
        "unverified" => HashAlgo::Unverified,
        other => {
            return Err(ModelError::HashMismatch {
                name: "<algo>".into(),
                expected: format!("blake3|sha256|unverified, got {other}"),
                actual: spec.into(),
            });
        }
    };
    Ok((algo, value))
}

fn default_root() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| Path::new(&home).join(".kb").join("models"))
}

fn download_to(target: &Path, spec: &ModelSpec) -> Result<(), ModelError> {
    tracing::info!(
        name = spec.name,
        url = spec.url,
        size_bytes = spec.size_bytes,
        "downloading model"
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|source| ModelError::Download {
            url: spec.url.to_string(),
            source,
        })?;
    let mut resp = client.get(spec.url).send().map_err(|source| ModelError::Download {
        url: spec.url.to_string(),
        source,
    })?;

    let tmp = target.with_extension("partial");
    let mut file = File::create(&tmp)?;
    resp.copy_to(&mut file).map_err(|source| ModelError::Download {
        url: spec.url.to_string(),
        source,
    })?;
    file.flush()?;

    let (algo, expected) = parse_hash_spec(spec.hash)?;
    let actual = hash_file(&tmp, algo)?;
    match algo {
        HashAlgo::Unverified => {
            tracing::warn!(
                name = spec.name,
                actual = %format!("blake3${actual}"),
                "ModelSpec.hash is `unverified$` — replace with the actual hash above to lock the model version"
            );
        }
        _ if actual != expected => {
            let _ = fs::remove_file(&tmp);
            return Err(ModelError::HashMismatch {
                name: spec.name.to_string(),
                expected: spec.hash.to_string(),
                actual: format!("{}${}", algo_name(algo), actual),
            });
        }
        _ => {}
    }
    fs::rename(&tmp, target)?;
    Ok(())
}

fn verify_hash(path: &Path, spec: &str) -> std::io::Result<bool> {
    let Ok((algo, expected)) = parse_hash_spec(spec) else {
        return Ok(false);
    };
    if matches!(algo, HashAlgo::Unverified) {
        // Accept any existing file; we still log the actual hash on download.
        return Ok(true);
    }
    let actual = hash_file(path, algo)?;
    Ok(actual == expected)
}

fn hash_file(path: &Path, algo: HashAlgo) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    match algo {
        HashAlgo::Blake3 | HashAlgo::Unverified => {
            let mut hasher = blake3::Hasher::new();
            loop {
                let n = file.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            Ok(hasher.finalize().to_hex().to_string())
        }
        HashAlgo::Sha256 => {
            // sha256 path uses blake3-style streaming via the `sha2` crate when
            // we add it. For now error so callers aren't silently misled.
            let _ = file;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "sha256 verification not yet implemented — use blake3 or unverified",
            ))
        }
    }
}

const fn algo_name(algo: HashAlgo) -> &'static str {
    match algo {
        HashAlgo::Blake3 => "blake3",
        HashAlgo::Sha256 => "sha256",
        HashAlgo::Unverified => "unverified",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_algo_prefixed_hash() {
        let (a, v) = parse_hash_spec("blake3$abc123").unwrap();
        assert_eq!(a, HashAlgo::Blake3);
        assert_eq!(v, "abc123");
    }

    #[test]
    fn parses_unverified_with_empty_value() {
        let (a, v) = parse_hash_spec("unverified$").unwrap();
        assert_eq!(a, HashAlgo::Unverified);
        assert_eq!(v, "");
    }

    #[test]
    fn rejects_missing_prefix() {
        assert!(parse_hash_spec("abc123").is_err());
    }

    #[test]
    fn rejects_unknown_algo() {
        assert!(parse_hash_spec("md5$abc").is_err());
    }

    #[test]
    fn blake3_round_trip_matches_known_value() {
        // Empty input has a known blake3 hash.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty");
        std::fs::File::create(&path).unwrap();
        let h = hash_file(&path, HashAlgo::Blake3).unwrap();
        assert_eq!(
            h,
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    #[test]
    fn verify_hash_accepts_unverified() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("any");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"anything").unwrap();
        assert!(verify_hash(&path, "unverified$").unwrap());
    }

    #[test]
    fn verify_hash_rejects_wrong_blake3() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("any");
        std::fs::File::create(&path).unwrap();
        let wrong =
            "blake3$0000000000000000000000000000000000000000000000000000000000000000";
        assert!(!verify_hash(&path, wrong).unwrap());
    }
}
