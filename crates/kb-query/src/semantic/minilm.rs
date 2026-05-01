//! ONNX `MiniLM-L6-v2` embedding backend (`semantic-ort` feature).
//!
//! Wraps the Sentence-Transformers `all-MiniLM-L6-v2` model exported to ONNX
//! (Xenova quantized weights). 384-dim, mean-pool over token embeddings,
//! L2-normalize.
//!
//! Pattern lifted from `bones-search`'s `semantic/model.rs`. Cache layout:
//!
//! ```text
//! ${XDG_CACHE_HOME:-~/.cache}/kb/models/
//!   minilm-l6-v2-int8.onnx
//!   minilm-l6-v2-tokenizer.json
//! ```
//!
//! Model + tokenizer are auto-downloaded from Hugging Face on first use
//! unless `KB_SEMANTIC_AUTO_DOWNLOAD=0`. The download URLs can be overridden
//! with `KB_SEMANTIC_MODEL_URL` and `KB_SEMANTIC_TOKENIZER_URL` for
//! air-gapped or mirrored installs.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, anyhow, bail};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

use super::model::{EmbeddingBackend, MINILM_BACKEND_ID, MINILM_DIM};
use super::onnx_model::{cache_root, ensure_file_cached, url_from_env_or_default};

const MODEL_FILENAME: &str = "minilm-l6-v2-int8.onnx";
const TOKENIZER_FILENAME: &str = "minilm-l6-v2-tokenizer.json";

/// Tokens beyond this index are dropped before inference. `MiniLM-L6-v2`
/// was trained with a 256-token window; longer inputs are clipped to that
/// window to match the training distribution.
const MAX_TOKENS: usize = 256;

const MODEL_DOWNLOAD_URL_ENV: &str = "KB_SEMANTIC_MODEL_URL";
const TOKENIZER_DOWNLOAD_URL_ENV: &str = "KB_SEMANTIC_TOKENIZER_URL";
const MODEL_DOWNLOAD_URL_DEFAULT: &str =
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx";
const TOKENIZER_DOWNLOAD_URL_DEFAULT: &str =
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

static MODEL_DOWNLOAD_ATTEMPTED: AtomicBool = AtomicBool::new(false);
static TOKENIZER_DOWNLOAD_ATTEMPTED: AtomicBool = AtomicBool::new(false);

/// ONNX-backed `MiniLM-L6-v2` embedding backend.
pub struct MiniLmBackend {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl MiniLmBackend {
    /// Load the backend.
    ///
    /// `model_path` and `tokenizer_path` override the cache convention when
    /// `Some` — useful for air-gapped installs that pre-stage the files. When
    /// `None`, `~/.cache/kb/models/...` is used and missing files trigger a
    /// download from Hugging Face.
    ///
    /// # Errors
    ///
    /// Returns an error when the model or tokenizer files are unavailable
    /// (and auto-download is disabled), when the ONNX session fails to build,
    /// or when the tokenizer cannot be parsed.
    pub fn load(model_path: Option<&Path>, tokenizer_path: Option<&Path>) -> Result<Self> {
        let model_path = match model_path {
            Some(p) => p.to_path_buf(),
            None => default_model_path()?,
        };
        let tokenizer_path = match tokenizer_path {
            Some(p) => p.to_path_buf(),
            None => default_tokenizer_path()?,
        };

        ensure_file_cached(
            &model_path,
            &model_download_url(),
            "semantic model",
            &MODEL_DOWNLOAD_ATTEMPTED,
        )
        .with_context(|| format!("ensure semantic model at {}", model_path.display()))?;

        ensure_file_cached(
            &tokenizer_path,
            &tokenizer_download_url(),
            "semantic tokenizer",
            &TOKENIZER_DOWNLOAD_ATTEMPTED,
        )
        .with_context(|| format!("ensure semantic tokenizer at {}", tokenizer_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow!(
                "failed to load semantic tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        let session = Session::builder()
            .context("failed to create ONNX Runtime session builder")?
            .commit_from_file(&model_path)
            .with_context(|| {
                format!("failed to load semantic model from {}", model_path.display())
            })?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Cache directory used for the ONNX model and tokenizer.
    ///
    /// `~/.cache/kb/models/` on Linux, `~/Library/Caches/kb/models/` on macOS,
    /// `%LOCALAPPDATA%\kb\models\` on Windows. Delegates to the shared
    /// [`super::onnx_model::cache_root`] helper so the cross-encoder
    /// reranker (bn-1cp2) and any future ONNX backend pick up the same root.
    ///
    /// # Errors
    ///
    /// Returns an error when the OS cache directory cannot be determined.
    pub fn cache_root() -> Result<PathBuf> {
        cache_root()
    }
}

impl EmbeddingBackend for MiniLmBackend {
    fn backend_id(&self) -> &'static str {
        MINILM_BACKEND_ID
    }

    fn dimensions(&self) -> usize {
        MINILM_DIM
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoded = self.encode_text(text)?;
        let mut out = self.run_model_batch(&[encoded])?;
        out.pop()
            .ok_or_else(|| anyhow!("semantic model returned no embedding"))
    }
}

struct EncodedText {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
}

enum InputSource {
    InputIds,
    AttentionMask,
    TokenTypeIds,
}

impl MiniLmBackend {
    fn encode_text(&self, text: &str) -> Result<EncodedText> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("failed to tokenize semantic input: {e}"))?;

        let ids = encoding.get_ids();
        if ids.is_empty() {
            // Empty/whitespace text: return a single CLS-equivalent placeholder
            // so the downstream pooling produces a valid (zero) vector.
            return Ok(EncodedText {
                input_ids: vec![0],
                attention_mask: vec![1],
            });
        }

        let attention = encoding.get_attention_mask();
        let keep = ids.len().min(MAX_TOKENS);

        let mut input_ids = Vec::with_capacity(keep);
        let mut attention_mask = Vec::with_capacity(keep);
        for (idx, id) in ids.iter().enumerate().take(keep) {
            input_ids.push(i64::from(*id));
            attention_mask.push(i64::from(*attention.get(idx).unwrap_or(&1_u32)));
        }
        if attention_mask.iter().all(|v| *v == 0) {
            attention_mask.fill(1);
        }

        Ok(EncodedText {
            input_ids,
            attention_mask,
        })
    }

    #[allow(clippy::significant_drop_tightening, clippy::cast_precision_loss)]
    fn run_model_batch(&self, encoded: &[EncodedText]) -> Result<Vec<Vec<f32>>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        let batch = encoded.len();
        let seq_len = encoded.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
        if seq_len == 0 {
            bail!("semantic batch has no tokens");
        }

        let mut flat_ids = vec![0_i64; batch * seq_len];
        let mut flat_attention = vec![0_i64; batch * seq_len];
        for (row_idx, row) in encoded.iter().enumerate() {
            let row_base = row_idx * seq_len;
            flat_ids[row_base..(row.input_ids.len() + row_base)].copy_from_slice(&row.input_ids);
            flat_attention[row_base..(row.attention_mask.len() + row_base)]
                .copy_from_slice(&row.attention_mask);
        }
        let flat_token_types = vec![0_i64; batch * seq_len];

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("semantic model session mutex poisoned"))?;

        let model_inputs = &session.inputs;
        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(model_inputs.len());
        for (index, input) in model_inputs.iter().enumerate() {
            let input_name = input.name.as_str();
            let source = input_source(index, input_name);
            let data = match source {
                InputSource::InputIds => flat_ids.clone(),
                InputSource::AttentionMask => flat_attention.clone(),
                InputSource::TokenTypeIds => flat_token_types.clone(),
            };
            let tensor = Tensor::<i64>::from_array(([batch, seq_len], data.into_boxed_slice()))
                .with_context(|| format!("failed to build ONNX input tensor '{input_name}'"))?;
            inputs.push((input_name.to_string(), tensor));
        }

        let outputs = session
            .run(inputs)
            .context("failed to run ONNX semantic inference")?;

        if outputs.len() == 0 {
            bail!("semantic model returned no outputs");
        }

        let output = outputs
            .get("sentence_embedding")
            .or_else(|| outputs.get("last_hidden_state"))
            .or_else(|| outputs.get("token_embeddings"))
            .unwrap_or(&outputs[0]);

        let (shape, data) = output.try_extract_tensor::<f32>().with_context(|| {
            "semantic model output tensor is not f32; expected sentence embedding tensor"
        })?;

        decode_embeddings(shape, data, &flat_attention, batch, seq_len)
    }
}

fn input_source(index: usize, input_name: &str) -> InputSource {
    let name = input_name.to_ascii_lowercase();
    if name.contains("attention") {
        return InputSource::AttentionMask;
    }
    if name.contains("token_type") || name.contains("segment") {
        return InputSource::TokenTypeIds;
    }
    if name.contains("input_ids") || (name.contains("input") && name.contains("id")) {
        return InputSource::InputIds;
    }

    match index {
        0 => InputSource::InputIds,
        1 => InputSource::AttentionMask,
        _ => InputSource::TokenTypeIds,
    }
}

#[allow(clippy::cast_precision_loss)]
fn decode_embeddings(
    shape: &[i64],
    data: &[f32],
    flat_attention: &[i64],
    batch: usize,
    seq_len: usize,
) -> Result<Vec<Vec<f32>>> {
    match shape.len() {
        // [batch, hidden]: already pooled.
        2 => {
            let out_batch = usize::try_from(shape[0]).unwrap_or(0);
            let hidden = usize::try_from(shape[1]).unwrap_or(0);
            if out_batch == 0 || hidden == 0 {
                bail!("invalid sentence embedding output shape {shape:?}");
            }
            if out_batch != batch {
                bail!("semantic output batch mismatch: expected {batch}, got {out_batch}");
            }

            let mut out = Vec::with_capacity(out_batch);
            for row in 0..out_batch {
                let start = row * hidden;
                let end = start + hidden;
                let mut emb = data[start..end].to_vec();
                normalize_l2(&mut emb);
                out.push(emb);
            }
            Ok(out)
        }

        // [batch, tokens, hidden]: mean-pool with attention mask.
        3 => {
            let out_batch = usize::try_from(shape[0]).unwrap_or(0);
            let out_tokens = usize::try_from(shape[1]).unwrap_or(0);
            let hidden = usize::try_from(shape[2]).unwrap_or(0);
            if out_batch == 0 || out_tokens == 0 || hidden == 0 {
                bail!("invalid token embedding output shape {shape:?}");
            }
            if out_batch != batch {
                bail!("semantic output batch mismatch: expected {batch}, got {out_batch}");
            }

            let mut out = Vec::with_capacity(out_batch);
            for b in 0..out_batch {
                let mut emb = vec![0.0_f32; hidden];
                let mut weight_sum = 0.0_f32;

                for t in 0..out_tokens {
                    let mask_weight = if t < seq_len {
                        flat_attention[b * seq_len + t] as f32
                    } else {
                        0.0
                    };
                    if mask_weight <= 0.0 {
                        continue;
                    }

                    let token_base = (b * out_tokens + t) * hidden;
                    for h in 0..hidden {
                        emb[h] += data[token_base + h] * mask_weight;
                    }
                    weight_sum += mask_weight;
                }

                if weight_sum > 0.0 {
                    for value in &mut emb {
                        *value /= weight_sum;
                    }
                }
                normalize_l2(&mut emb);
                out.push(emb);
            }
            Ok(out)
        }

        // [hidden]: single-row fallback.
        1 => {
            if batch != 1 {
                bail!("rank-1 semantic output only supported for single-row batch");
            }
            let hidden = usize::try_from(shape[0]).unwrap_or(0);
            if hidden == 0 {
                bail!("invalid rank-1 semantic output shape {shape:?}");
            }
            let mut emb = data[0..hidden].to_vec();
            normalize_l2(&mut emb);
            Ok(vec![emb])
        }

        rank => bail!("unsupported semantic output rank {rank}: shape {shape:?}"),
    }
}

fn normalize_l2(values: &mut [f32]) {
    let mut norm_sq = 0.0_f32;
    for value in values.iter() {
        norm_sq += value * value;
    }

    if norm_sq == 0.0 {
        return;
    }

    let norm = norm_sq.sqrt();
    for value in values {
        *value /= norm;
    }
}

fn default_model_path() -> Result<PathBuf> {
    Ok(cache_root()?.join(MODEL_FILENAME))
}

fn default_tokenizer_path() -> Result<PathBuf> {
    Ok(cache_root()?.join(TOKENIZER_FILENAME))
}

fn model_download_url() -> String {
    url_from_env_or_default(MODEL_DOWNLOAD_URL_ENV, MODEL_DOWNLOAD_URL_DEFAULT)
}

fn tokenizer_download_url() -> String {
    url_from_env_or_default(TOKENIZER_DOWNLOAD_URL_ENV, TOKENIZER_DOWNLOAD_URL_DEFAULT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_root_uses_kb_subdir() {
        // Delegates to `super::onnx_model::cache_root`; assert the wrapper
        // doesn't hide that the path lives under `kb/models`.
        let path = MiniLmBackend::cache_root().expect("cache root resolves");
        let expected_tail = Path::new("kb").join("models");
        assert!(path.ends_with(expected_tail), "got {}", path.display());
    }

    #[test]
    fn input_source_prefers_named_fields() {
        assert!(matches!(
            input_source(5, "attention_mask"),
            InputSource::AttentionMask
        ));
        assert!(matches!(
            input_source(5, "token_type_ids"),
            InputSource::TokenTypeIds
        ));
        assert!(matches!(
            input_source(5, "input_ids"),
            InputSource::InputIds
        ));
        // Index-based fallback for opaque input names.
        assert!(matches!(input_source(0, "x"), InputSource::InputIds));
        assert!(matches!(input_source(1, "y"), InputSource::AttentionMask));
        assert!(matches!(input_source(2, "z"), InputSource::TokenTypeIds));
    }

    #[test]
    fn normalize_l2_produces_unit_norm() {
        let mut emb = vec![3.0_f32, 4.0_f32, 0.0_f32];
        normalize_l2(&mut emb);
        let norm = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_zero_vector_is_unchanged() {
        let mut emb = vec![0.0_f32; 4];
        normalize_l2(&mut emb);
        assert!(emb.iter().all(|v| *v == 0.0));
    }

    // No env-var test for `auto_download_enabled` — the only reasonable way
    // to exercise it would mutate a process-global env var, which is unsafe
    // under modern Rust and races with concurrent tests in `cargo test`.
    // The function is a thin string-match around a single env lookup.
}
