//! Cross-encoder reranker for the second stage of hybrid retrieval (bn-1cp2).
//!
//! Bi-encoder retrieval (`HashEmbedBackend`, `MiniLmBackend`) is *recall*-
//! focused: it embeds query and document into the same space independently
//! and ranks by cosine. That's fast (the corpus vectors are precomputed) but
//! it cannot read the query and the candidate jointly — so it confuses
//! topically related text with answers.
//!
//! A cross-encoder *reads `(query, candidate)` together*, with full
//! attention from query tokens to candidate tokens. It cannot precompute
//! anything for the corpus, so it's slow — but it is dramatically more
//! precise. The standard 2-stage IR pattern is:
//!
//! 1. Bi-encoder + lexical RRF → top-K candidates (recall-focused).
//! 2. Cross-encoder → re-rank K → keep top-N (precision-focused).
//!
//! This module ships stage 2. The model is `cross-encoder/ms-marco-MiniLM-L-6-v2`
//! (~80MB ONNX). Inputs are `[CLS] query [SEP] candidate [SEP]` and the
//! single-logit output is treated as a relevance score (higher = more
//! relevant). Scores are *not* normalized to `[0, 1]`; we only sort by them.
//!
//! Cache layout (shared with [`super::onnx_model`]):
//!
//! ```text
//! ${XDG_CACHE_HOME:-~/.cache}/kb/models/
//!   cross-encoder-ms-marco-minilm-l6-int8.onnx
//!   cross-encoder-ms-marco-minilm-l6-tokenizer.json
//! ```
//!
//! ## Feature gating
//!
//! [`RerankSettings`] is always available — it's a tiny Copy struct that
//! lives inside [`crate::HybridOptions`] regardless of build features. The
//! [`CrossEncoder`] (and its [`Reranker`] impl) is gated behind
//! `semantic-ort` because it pulls in `ort`/`tokenizers`. Without the
//! feature, kb-cli either ships a stub reranker or leaves rerank disabled.

use anyhow::Result;

/// Default top-K candidates fed into the cross-encoder.
///
/// The rerank pass reads `top_k` candidates from the bi-encoder/lexical
/// stage and keeps `keep` of them. `K=30` matches the bone spec and is
/// consistent with typical 2-stage IR setups (cross-encoder is O(K) so
/// K=30 is a few hundred milliseconds on CPU).
pub const DEFAULT_TOP_K: usize = 30;
/// Default number of results returned after rerank. Matches the bone spec.
pub const DEFAULT_KEEP: usize = 8;

/// Trait the hybrid layer calls to score `(query, candidate)` pairs in batch.
///
/// [`CrossEncoder`] is the production implementation; tests build
/// in-memory stubs (e.g. `score = -lexical_rank`) to verify the wiring
/// without dragging in an 80MB ONNX model.
pub trait Reranker: Send + Sync {
    /// Score `candidates` against `query` in a single call. Output `Vec`
    /// is in the same order as `candidates`. Higher = more relevant.
    ///
    /// # Errors
    ///
    /// Returns an error when scoring fails. The hybrid caller logs and
    /// degrades to the un-reranked order rather than failing the whole
    /// query.
    fn score_batch(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>>;
}

/// Compact, Copy-able settings for the rerank pass.
///
/// Sits inside [`crate::HybridOptions`] so callers can toggle rerank
/// without changing function signatures. The actual [`Reranker`] (the
/// loaded model) is passed separately because it owns an ONNX session
/// and is not Copy.
///
/// `enabled` doubles as the "did the caller load a model?" check — the
/// hybrid layer skips the rerank pass when either `enabled` is false or
/// the caller didn't pass a [`Reranker`].
#[derive(Debug, Clone, Copy)]
pub struct RerankSettings {
    /// Master switch. When false, the hybrid retriever skips rerank
    /// regardless of whether a [`Reranker`] is provided.
    pub enabled: bool,
    /// Candidates from the fused stage to feed into the cross-encoder.
    /// `0` => use [`DEFAULT_TOP_K`].
    pub top_k: usize,
    /// Results to keep after rerank. `0` => use [`DEFAULT_KEEP`].
    pub keep: usize,
}

impl Default for RerankSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            top_k: DEFAULT_TOP_K,
            keep: DEFAULT_KEEP,
        }
    }
}

impl RerankSettings {
    /// `top_k` clamped to the documented default when the user passed `0`.
    #[must_use]
    pub const fn effective_top_k(&self) -> usize {
        if self.top_k == 0 {
            DEFAULT_TOP_K
        } else {
            self.top_k
        }
    }

    /// `keep` clamped to the documented default when the user passed `0`,
    /// and to `effective_top_k` so we never ask to keep more than we
    /// scored.
    #[must_use]
    pub const fn effective_keep(&self) -> usize {
        let raw = if self.keep == 0 { DEFAULT_KEEP } else { self.keep };
        let cap = self.effective_top_k();
        if raw > cap { cap } else { raw }
    }
}

#[cfg(feature = "semantic-ort")]
pub use ort_impl::{CrossEncoder, CrossEncoderConfig, RerankConfig};

#[cfg(feature = "semantic-ort")]
mod ort_impl {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    use anyhow::{Context, Result, anyhow, bail};
    use ort::{session::Session, value::Tensor};
    use tokenizers::Tokenizer;

    use super::{Reranker, RerankSettings};
    use crate::semantic::onnx_model::{cache_root, ensure_file_cached, url_from_env_or_default};

    const MODEL_FILENAME: &str = "cross-encoder-ms-marco-minilm-l6-int8.onnx";
    const TOKENIZER_FILENAME: &str = "cross-encoder-ms-marco-minilm-l6-tokenizer.json";

    /// Hard token cap matching the cross-encoder's training window. Inputs
    /// that exceed this length are truncated (we do *not* split into
    /// multiple windows — this is a re-rank pass, the candidate's snippet
    /// is already short).
    const MAX_TOKENS: usize = 256;

    const MODEL_DOWNLOAD_URL_ENV: &str = "KB_RERANK_MODEL_URL";
    const TOKENIZER_DOWNLOAD_URL_ENV: &str = "KB_RERANK_TOKENIZER_URL";
    const MODEL_DOWNLOAD_URL_DEFAULT: &str =
        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx";
    const TOKENIZER_DOWNLOAD_URL_DEFAULT: &str =
        "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json";

    static MODEL_DOWNLOAD_ATTEMPTED: AtomicBool = AtomicBool::new(false);
    static TOKENIZER_DOWNLOAD_ATTEMPTED: AtomicBool = AtomicBool::new(false);

    /// Path-bag for [`CrossEncoder::load_with_paths`].
    ///
    /// `None` falls back to the OS cache convention; auto-download from
    /// Hugging Face fires on first use unless
    /// [`crate::semantic::onnx_model::AUTO_DOWNLOAD_ENV`] is set to a
    /// falsy value.
    #[derive(Debug, Clone, Default)]
    pub struct CrossEncoderConfig {
        pub model_path: Option<PathBuf>,
        pub tokenizer_path: Option<PathBuf>,
    }

    /// Tunable rerank parameters. Exposed through `kb.toml [semantic.rerank]`
    /// in `kb-cli`, mirrored here so `kb-query` stays free of toml/serde
    /// plumbing the way [`crate::SemanticBackendConfig`] is.
    ///
    /// `enabled = false` skips the cross-encoder entirely — the hybrid
    /// layer behaves identically to the pre-rerank pipeline. This is the
    /// v1 default because each enable adds a model download (~80MB on
    /// first use) and a few hundred milliseconds of CPU latency per query.
    #[derive(Debug, Clone, Default)]
    pub struct RerankConfig {
        /// Master switch. When false, the hybrid retriever skips the
        /// rerank pass and degrades to the pre-rerank fused order.
        pub enabled: bool,
        /// Candidates from the fused stage to feed into the cross-encoder.
        /// `0` => use [`DEFAULT_TOP_K`].
        pub top_k: usize,
        /// Results to keep after rerank. `0` => use [`DEFAULT_KEEP`].
        pub keep: usize,
        /// Cross-encoder model file paths (overrides cache convention).
        pub paths: CrossEncoderConfig,
    }

    impl RerankConfig {
        /// Distill the `Copy` subset of this config that travels through
        /// [`crate::HybridOptions`].
        #[must_use]
        pub const fn settings(&self) -> RerankSettings {
            RerankSettings {
                enabled: self.enabled,
                top_k: self.top_k,
                keep: self.keep,
            }
        }

        /// `top_k` clamped to the documented default when the user passed
        /// `0`.
        #[must_use]
        pub const fn effective_top_k(&self) -> usize {
            self.settings().effective_top_k()
        }

        /// `keep` clamped to the documented default when the user passed
        /// `0`. Always `<= effective_top_k`.
        #[must_use]
        pub const fn effective_keep(&self) -> usize {
            self.settings().effective_keep()
        }
    }

    /// A loaded ONNX cross-encoder.
    ///
    /// `Mutex<Session>` mirrors [`crate::semantic::minilm::MiniLmBackend`]
    /// — `ort::Session::run` takes `&mut self`, so we serialize calls.
    pub struct CrossEncoder {
        session: Mutex<Session>,
        tokenizer: Tokenizer,
    }

    impl CrossEncoder {
        /// Load the cross-encoder using the OS cache convention.
        ///
        /// `model_path` and `tokenizer_path` override the cache convention
        /// when `Some` — useful for air-gapped installs that pre-stage the
        /// files. When `None`, `~/.cache/kb/models/cross-encoder-ms-marco-minilm-l6-*`
        /// is used and missing files trigger a download from Hugging Face
        /// (override URLs with `KB_RERANK_MODEL_URL` and
        /// `KB_RERANK_TOKENIZER_URL`).
        ///
        /// # Errors
        ///
        /// Returns an error when the model or tokenizer files cannot be
        /// loaded (and auto-download is disabled or fails), the ONNX
        /// session fails to build, or the tokenizer cannot be parsed.
        pub fn load(model_path: Option<&Path>, tokenizer_path: Option<&Path>) -> Result<Self> {
            Self::load_with_paths(&CrossEncoderConfig {
                model_path: model_path.map(Path::to_path_buf),
                tokenizer_path: tokenizer_path.map(Path::to_path_buf),
            })
        }

        /// `CrossEncoderConfig`-flavored variant of [`Self::load`].
        ///
        /// # Errors
        ///
        /// Same as [`Self::load`].
        pub fn load_with_paths(config: &CrossEncoderConfig) -> Result<Self> {
            let model_path = match config.model_path.as_deref() {
                Some(p) => p.to_path_buf(),
                None => default_model_path()?,
            };
            let tokenizer_path = match config.tokenizer_path.as_deref() {
                Some(p) => p.to_path_buf(),
                None => default_tokenizer_path()?,
            };

            ensure_file_cached(
                &model_path,
                &model_download_url(),
                "cross-encoder model",
                &MODEL_DOWNLOAD_ATTEMPTED,
            )
            .with_context(|| {
                format!("ensure cross-encoder model at {}", model_path.display())
            })?;

            ensure_file_cached(
                &tokenizer_path,
                &tokenizer_download_url(),
                "cross-encoder tokenizer",
                &TOKENIZER_DOWNLOAD_ATTEMPTED,
            )
            .with_context(|| {
                format!(
                    "ensure cross-encoder tokenizer at {}",
                    tokenizer_path.display()
                )
            })?;

            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                anyhow!(
                    "failed to load cross-encoder tokenizer from {}: {e}",
                    tokenizer_path.display()
                )
            })?;

            let session = Session::builder()
                .context("failed to create ONNX Runtime session builder")?
                .commit_from_file(&model_path)
                .with_context(|| {
                    format!(
                        "failed to load cross-encoder model from {}",
                        model_path.display()
                    )
                })?;

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
            })
        }

        /// Cache directory for the cross-encoder artifacts. Shares the
        /// `MiniLM` cache root via [`crate::semantic::onnx_model::cache_root`].
        ///
        /// # Errors
        ///
        /// Returns an error when the OS cache directory cannot be
        /// determined.
        pub fn cache_root() -> Result<PathBuf> {
            cache_root()
        }

        /// Score a single `(query, candidate)` pair. Higher = more
        /// relevant. The score is the raw model logit; do not assume any
        /// specific range.
        ///
        /// # Errors
        ///
        /// Returns an error when tokenization or inference fails.
        pub fn score(&self, query: &str, candidate: &str) -> Result<f32> {
            let mut out = self.score_batch_inner(query, &[candidate])?;
            out.pop()
                .ok_or_else(|| anyhow!("cross-encoder returned no score"))
        }

        /// Score a batch of `(query, candidate_i)` pairs in a single
        /// ONNX session run. For K candidates this is dramatically
        /// faster than K independent `score` calls because the ONNX
        /// runtime amortizes per-call overhead. Output `Vec` is in the
        /// same order as `candidates`.
        ///
        /// # Errors
        ///
        /// Returns an error when tokenization or inference fails.
        pub fn score_batch(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>> {
            self.score_batch_inner(query, candidates)
        }

        #[allow(clippy::significant_drop_tightening)]
        fn score_batch_inner(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>> {
            if candidates.is_empty() {
                return Ok(Vec::new());
            }

            let encoded = candidates
                .iter()
                .map(|cand| self.encode_pair(query, cand))
                .collect::<Result<Vec<_>>>()?;

            let batch = encoded.len();
            let seq_len = encoded.iter().map(|e| e.input_ids.len()).max().unwrap_or(0);
            if seq_len == 0 {
                bail!("cross-encoder batch has no tokens");
            }

            let mut flat_ids = vec![0_i64; batch * seq_len];
            let mut flat_attention = vec![0_i64; batch * seq_len];
            let mut flat_token_types = vec![0_i64; batch * seq_len];
            for (row_idx, row) in encoded.iter().enumerate() {
                let row_base = row_idx * seq_len;
                flat_ids[row_base..(row_base + row.input_ids.len())]
                    .copy_from_slice(&row.input_ids);
                flat_attention[row_base..(row_base + row.attention_mask.len())]
                    .copy_from_slice(&row.attention_mask);
                flat_token_types[row_base..(row_base + row.token_type_ids.len())]
                    .copy_from_slice(&row.token_type_ids);
            }

            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow!("cross-encoder session mutex poisoned"))?;

            let model_inputs = &session.inputs;
            let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(model_inputs.len());
            for (index, input) in model_inputs.iter().enumerate() {
                let name = input.name.as_str();
                let source = input_source(index, name);
                let data = match source {
                    CrossInput::InputIds => flat_ids.clone(),
                    CrossInput::AttentionMask => flat_attention.clone(),
                    CrossInput::TokenTypeIds => flat_token_types.clone(),
                };
                let tensor =
                    Tensor::<i64>::from_array(([batch, seq_len], data.into_boxed_slice()))
                        .with_context(|| {
                            format!("failed to build cross-encoder input tensor '{name}'")
                        })?;
                inputs.push((name.to_string(), tensor));
            }

            let outputs = session
                .run(inputs)
                .context("failed to run cross-encoder inference")?;

            if outputs.len() == 0 {
                bail!("cross-encoder returned no outputs");
            }

            // ms-marco-MiniLM-L-6-v2 ONNX export emits one output named
            // "logits" of shape [batch, 1] (single-logit regression head).
            // Some exports use "score" or rank by index — fall back
            // gracefully.
            let output = outputs
                .get("logits")
                .or_else(|| outputs.get("score"))
                .or_else(|| outputs.get("scores"))
                .unwrap_or(&outputs[0]);

            let (shape, data) = output.try_extract_tensor::<f32>().context(
                "cross-encoder output tensor is not f32; expected logit tensor",
            )?;

            decode_scores(shape, data, batch)
        }

        fn encode_pair(&self, query: &str, candidate: &str) -> Result<EncodedPair> {
            // Pair encoding: tokenizer.json's post-processor inserts the
            // special tokens ([CLS] q [SEP] c [SEP]) and emits per-token
            // type-ids (0 for query, 1 for candidate). We hard-cap on our
            // side because tokenizers may not enable truncation by default
            // in older exports.
            let encoding = self
                .tokenizer
                .encode((query.to_string(), candidate.to_string()), true)
                .map_err(|e| anyhow!("failed to tokenize cross-encoder input: {e}"))?;

            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            if ids.is_empty() {
                // Empty input: produce a single placeholder token so
                // downstream tensor construction has at least one column.
                return Ok(EncodedPair {
                    input_ids: vec![0],
                    attention_mask: vec![1],
                    token_type_ids: vec![0],
                });
            }

            let keep = ids.len().min(MAX_TOKENS);
            let mut input_ids = Vec::with_capacity(keep);
            let mut attention_mask = Vec::with_capacity(keep);
            let mut token_type_ids = Vec::with_capacity(keep);
            for (idx, id) in ids.iter().enumerate().take(keep) {
                input_ids.push(i64::from(*id));
                attention_mask.push(i64::from(*attention.get(idx).unwrap_or(&1_u32)));
                token_type_ids.push(i64::from(*type_ids.get(idx).unwrap_or(&0_u32)));
            }
            if attention_mask.iter().all(|v| *v == 0) {
                attention_mask.fill(1);
            }

            Ok(EncodedPair {
                input_ids,
                attention_mask,
                token_type_ids,
            })
        }
    }

    impl Reranker for CrossEncoder {
        fn score_batch(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>> {
            Self::score_batch(self, query, candidates)
        }
    }

    struct EncodedPair {
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        token_type_ids: Vec<i64>,
    }

    #[derive(Clone, Copy)]
    enum CrossInput {
        InputIds,
        AttentionMask,
        TokenTypeIds,
    }

    fn input_source(index: usize, input_name: &str) -> CrossInput {
        let name = input_name.to_ascii_lowercase();
        if name.contains("attention") {
            return CrossInput::AttentionMask;
        }
        if name.contains("token_type") || name.contains("segment") {
            return CrossInput::TokenTypeIds;
        }
        if name.contains("input_ids") || (name.contains("input") && name.contains("id")) {
            return CrossInput::InputIds;
        }
        match index {
            0 => CrossInput::InputIds,
            1 => CrossInput::AttentionMask,
            _ => CrossInput::TokenTypeIds,
        }
    }

    fn decode_scores(shape: &[i64], data: &[f32], batch: usize) -> Result<Vec<f32>> {
        match shape.len() {
            // [batch, 1] or [batch, num_labels=1] — the standard cross-encoder
            // regression head.
            2 => {
                let out_batch = usize::try_from(shape[0]).unwrap_or(0);
                let logits = usize::try_from(shape[1]).unwrap_or(0);
                if out_batch != batch {
                    bail!(
                        "cross-encoder batch mismatch: expected {batch}, got {out_batch}"
                    );
                }
                if logits == 0 {
                    bail!("cross-encoder produced empty per-row logit dimension {shape:?}");
                }
                let mut out = Vec::with_capacity(out_batch);
                for row in 0..out_batch {
                    // Take the first logit. Some exports emit a 2-class
                    // softmax head ([negative, positive]); for those we
                    // take the positive logit (column 1). Single-logit
                    // heads use column 0.
                    let pick = usize::from(logits >= 2);
                    out.push(data[row * logits + pick]);
                }
                Ok(out)
            }
            // [batch] — flat logit vector, single-row shapes.
            1 => {
                let out_batch = usize::try_from(shape[0]).unwrap_or(0);
                if out_batch != batch {
                    bail!(
                        "cross-encoder batch mismatch: expected {batch}, got {out_batch}"
                    );
                }
                Ok(data[..out_batch].to_vec())
            }
            rank => bail!("unsupported cross-encoder output rank {rank}: shape {shape:?}"),
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
        use super::super::{DEFAULT_KEEP, DEFAULT_TOP_K};
        use super::*;

        #[test]
        fn rerank_config_defaults_use_documented_constants() {
            let cfg = RerankConfig::default();
            assert!(!cfg.enabled, "rerank must default to disabled (bone spec)");
            assert_eq!(cfg.effective_top_k(), DEFAULT_TOP_K);
            assert_eq!(cfg.effective_keep(), DEFAULT_KEEP);
        }

        #[test]
        fn rerank_config_explicit_values_are_honored() {
            let cfg = RerankConfig {
                enabled: true,
                top_k: 50,
                keep: 12,
                paths: CrossEncoderConfig::default(),
            };
            assert_eq!(cfg.effective_top_k(), 50);
            assert_eq!(cfg.effective_keep(), 12);
        }

        #[test]
        fn rerank_config_keep_clamped_to_top_k() {
            // keep > top_k is a misconfiguration (no extra candidates
            // exist beyond top_k). Clamp silently rather than erroring.
            let cfg = RerankConfig {
                enabled: true,
                top_k: 5,
                keep: 20,
                paths: CrossEncoderConfig::default(),
            };
            assert_eq!(cfg.effective_top_k(), 5);
            assert_eq!(cfg.effective_keep(), 5);
        }

        #[test]
        fn rerank_config_settings_round_trip() {
            let cfg = RerankConfig {
                enabled: true,
                top_k: 25,
                keep: 7,
                paths: CrossEncoderConfig::default(),
            };
            let s = cfg.settings();
            assert!(s.enabled);
            assert_eq!(s.top_k, 25);
            assert_eq!(s.keep, 7);
        }

        #[test]
        fn default_paths_live_in_kb_models_cache() {
            let model = default_model_path().expect("model path resolves");
            let tokenizer = default_tokenizer_path().expect("tokenizer path resolves");
            assert!(
                model.ends_with(MODEL_FILENAME),
                "got {}",
                model.display()
            );
            assert!(
                tokenizer.ends_with(TOKENIZER_FILENAME),
                "got {}",
                tokenizer.display()
            );
            let expected_dir_tail = std::path::Path::new("kb").join("models");
            assert!(
                model
                    .parent()
                    .is_some_and(|p| p.ends_with(&expected_dir_tail)),
                "model parent should be the shared kb/models cache, got {}",
                model.display()
            );
        }

        #[test]
        fn input_source_routes_named_fields_correctly() {
            assert!(matches!(
                input_source(7, "attention_mask"),
                CrossInput::AttentionMask
            ));
            assert!(matches!(
                input_source(7, "token_type_ids"),
                CrossInput::TokenTypeIds
            ));
            assert!(matches!(
                input_source(7, "input_ids"),
                CrossInput::InputIds
            ));
            // Index fallback for opaque names.
            assert!(matches!(input_source(0, "x"), CrossInput::InputIds));
            assert!(matches!(input_source(1, "y"), CrossInput::AttentionMask));
            assert!(matches!(input_source(2, "z"), CrossInput::TokenTypeIds));
        }

        #[test]
        fn decode_scores_picks_positive_logit_for_two_class_head() {
            // Two-class softmax exports: [negative, positive] per row.
            // The reranker should sort by `positive`, so column 1 wins.
            let shape = [2_i64, 2_i64];
            let data = [0.9_f32, 0.1, 0.2, 0.8];
            let out = decode_scores(&shape, &data, 2).expect("decode");
            assert!((out[0] - 0.1).abs() < 1e-6, "row 0 picks positive logit");
            assert!((out[1] - 0.8).abs() < 1e-6, "row 1 picks positive logit");
        }

        #[test]
        fn decode_scores_handles_single_logit_head() {
            let shape = [3_i64, 1_i64];
            let data = [-0.5_f32, 1.5, 0.0];
            let out = decode_scores(&shape, &data, 3).expect("decode");
            assert_eq!(out, vec![-0.5, 1.5, 0.0]);
        }

        #[test]
        fn decode_scores_rejects_batch_mismatch() {
            let shape = [3_i64, 1_i64];
            let data = [0.0_f32, 0.0, 0.0];
            let err = decode_scores(&shape, &data, 5).expect_err("must reject mismatch");
            assert!(err.to_string().contains("batch mismatch"));
        }

        #[test]
        fn cross_encoder_load_does_not_panic_on_missing_files() {
            // Validate the load path runs to completion (Ok or Err)
            // rather than panicking when the pinned files don't exist
            // on disk. Three plausible outcomes:
            //  1. Auto-download is disabled → clean Err.
            //  2. Auto-download attempts a network fetch and fails →
            //     clean Err with the URL embedded.
            //  3. Auto-download fetches the real model successfully →
            //     Ok (this happens in dev environments where the
            //     network is up).
            // All three are fine; we just don't want a panic.
            let tmp = tempfile::tempdir().expect("tmpdir");
            let model = tmp.path().join("missing-model.onnx");
            let tokenizer = tmp.path().join("missing-tokenizer.json");
            let cfg = CrossEncoderConfig {
                model_path: Some(model),
                tokenizer_path: Some(tokenizer),
            };
            let _ = CrossEncoder::load_with_paths(&cfg);
        }
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn settings_defaults_are_disabled_with_documented_top_k_and_keep() {
        let s = RerankSettings::default();
        assert!(!s.enabled, "rerank ships disabled by default");
        assert_eq!(s.effective_top_k(), DEFAULT_TOP_K);
        assert_eq!(s.effective_keep(), DEFAULT_KEEP);
    }

    #[test]
    fn settings_zero_top_k_falls_back_to_default() {
        let s = RerankSettings {
            enabled: true,
            top_k: 0,
            keep: 0,
        };
        assert_eq!(s.effective_top_k(), DEFAULT_TOP_K);
        assert_eq!(s.effective_keep(), DEFAULT_KEEP);
    }

    #[test]
    fn settings_keep_is_capped_at_top_k() {
        let s = RerankSettings {
            enabled: true,
            top_k: 4,
            keep: 50,
        };
        assert_eq!(s.effective_top_k(), 4);
        assert_eq!(s.effective_keep(), 4);
    }
}
