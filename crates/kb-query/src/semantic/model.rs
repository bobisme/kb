//! Embedding backend trait and the available implementations.
//!
//! kb v1 ships two backends:
//!
//! - [`HashEmbedBackend`] — always available, zero new dependencies.
//!   Hashes 3..=5 character n-grams plus whitespace tokens into a 256-dim
//!   vector via FNV-1a, then L2-normalizes. Catches morphological similarity
//!   (`authentication` / `authn`) and rephrased queries, but not full
//!   conceptual paraphrases.
//!
//! - [`MiniLmBackend`] — feature-gated behind `semantic-ort`. Wraps the
//!   Sentence-Transformers `all-MiniLM-L6-v2` ONNX model (384-dim, mean-pool +
//!   L2 norm). Closes the lexical/semantic gap; bn-1rww. The model file is
//!   cached under [`MiniLmBackend::cache_root`] and auto-downloaded from
//!   Hugging Face on first use unless `KB_SEMANTIC_AUTO_DOWNLOAD=0`.
//!
//! Hash-pattern lifted from `bones-search`'s `hash_embed.rs`; ORT-pattern
//! lifted from `bones-search`'s `semantic/model.rs`.

use std::path::PathBuf;

#[cfg(feature = "semantic-ort")]
use anyhow::Context;
use anyhow::Result;
#[cfg(not(feature = "semantic-ort"))]
use anyhow::bail;
use serde::{Deserialize, Serialize};

#[cfg(feature = "semantic-ort")]
use super::minilm::MiniLmBackend;

/// Default embedding dimension for the always-available hash backend.
pub const HASH_DIM: usize = 256;

/// Stable identifier for the hash backend. Stored in `semantic_meta` so
/// changing backends triggers a re-embed of the corpus on the next compile.
pub const HASH_BACKEND_ID: &str = "hash-embed-256-chunked";

/// Stable identifier for the ORT `MiniLM` backend.
pub const MINILM_BACKEND_ID: &str = "ort-minilm-384-chunked";

/// Output dimensionality of the `MiniLM-L6-v2` backend.
pub const MINILM_DIM: usize = 384;

const NGRAM_MIN: usize = 3;
const NGRAM_MAX: usize = 5;

/// Trait for embedding backends.
pub trait EmbeddingBackend {
    /// Stable identifier persisted in `semantic_meta.backend_id`.
    fn backend_id(&self) -> &'static str;

    /// Output dimensionality.
    fn dimensions(&self) -> usize;

    /// Embed a single text string. Output is L2-normalized.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails. The hash backend never fails
    /// in practice; the trait method allows ML backends to surface runtime
    /// errors uniformly.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Hash-based embedding backend using character n-grams and word hashing.
///
/// Always available, zero new dependencies. Always L2-normalizes output.
#[derive(Debug, Clone, Copy)]
pub struct HashEmbedBackend {
    dimensions: usize,
}

impl HashEmbedBackend {
    /// Construct a backend producing [`HASH_DIM`]-dimensional vectors.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dimensions: HASH_DIM,
        }
    }
}

impl Default for HashEmbedBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingBackend for HashEmbedBackend {
    fn backend_id(&self) -> &'static str {
        HASH_BACKEND_ID
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(hash_embed(text, self.dimensions))
    }
}

/// Backend selector parsed from `kb.toml`.
///
/// `Default` is platform-aware (bn-2xbd, mirrors bones-cli):
/// - non-Windows: [`Self::Minilm`] (assumes the `semantic-ort` feature is
///   compiled in — kb-cli enables it via a target-specific dep).
/// - Windows: [`Self::Hash`] — kb has no model2vec backend yet, and
///   `ort`'s ONNX Runtime can clash with the MSVC CRT in the default
///   distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticBackendKind {
    /// Always-available hash n-gram backend ([`HashEmbedBackend`]).
    #[cfg_attr(target_os = "windows", default)]
    Hash,
    /// ONNX `MiniLM-L6-v2` (`semantic-ort` feature).
    #[cfg_attr(not(target_os = "windows"), default)]
    Minilm,
}

impl SemanticBackendKind {
    /// Backend-tuned default for the [`crate::hybrid::HybridOptions::min_semantic_score`]
    /// floor. Hash and `MiniLM` live in different cosine-score regimes — hash
    /// unrelated text stays in `[0.05, 0.35]`, `MiniLM` in `[0.0, 0.15]` — so
    /// a single global floor either lets noise through (low) or filters
    /// real matches (high). Callers without an explicit user override
    /// should consult this method.
    #[must_use]
    pub const fn default_min_semantic_score(self) -> f32 {
        match self {
            Self::Hash => crate::hybrid::HASH_MIN_SEMANTIC_SCORE,
            Self::Minilm => crate::hybrid::MINILM_MIN_SEMANTIC_SCORE,
        }
    }

    /// Backend-tuned default for the
    /// [`crate::hybrid::HybridOptions`] top-when-no-lexical floor. Same
    /// rationale as [`Self::default_min_semantic_score`].
    #[must_use]
    pub const fn default_min_semantic_top_score_no_lexical(self) -> f32 {
        match self {
            Self::Hash => crate::hybrid::HASH_MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL,
            Self::Minilm => crate::hybrid::MINILM_MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL,
        }
    }
}

/// Configuration bag for [`SemanticBackend::from_config`].
///
/// `model_path` and `tokenizer_path` are only consulted by
/// [`SemanticBackendKind::Minilm`]. When `None`, the OS cache convention
/// applies — the file is fetched from Hugging Face on first run.
#[derive(Debug, Clone, Default)]
pub struct SemanticBackendConfig {
    pub kind: SemanticBackendKind,
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
}

/// Runtime-dispatched embedding backend.
///
/// Produced by [`SemanticBackend::from_config`]. Implements
/// [`EmbeddingBackend`] by delegating to the wrapped concrete backend, so it
/// drops into [`super::embed::sync_embeddings`] and the hybrid query path
/// without further plumbing.
///
/// The `MiniLm` variant is large (it holds an ONNX session and tokenizer)
/// but a process only ever owns one `SemanticBackend` at a time, so the
/// size disparity between variants doesn't materially affect memory use —
/// only one `Box` allocation would be saved by indirection. We follow
/// `bones-search` here and accept the warning rather than add the heap
/// indirection.
#[allow(clippy::large_enum_variant)]
pub enum SemanticBackend {
    Hash(HashEmbedBackend),
    #[cfg(feature = "semantic-ort")]
    MiniLm(MiniLmBackend),
}

impl std::fmt::Debug for SemanticBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hash(b) => f.debug_tuple("SemanticBackend::Hash").field(b).finish(),
            #[cfg(feature = "semantic-ort")]
            Self::MiniLm(_) => f
                .debug_struct("SemanticBackend::MiniLm")
                .field("backend_id", &MINILM_BACKEND_ID)
                .field("dim", &MINILM_DIM)
                .finish(),
        }
    }
}

impl SemanticBackend {
    /// Recover the [`SemanticBackendKind`] this instance was loaded with.
    /// Used by callers (kb-web, kb-cli ask path) that hold a backend and
    /// need to look up backend-tuned threshold defaults without re-reading
    /// the config.
    #[must_use]
    pub const fn kind(&self) -> SemanticBackendKind {
        match self {
            Self::Hash(_) => SemanticBackendKind::Hash,
            #[cfg(feature = "semantic-ort")]
            Self::MiniLm(_) => SemanticBackendKind::Minilm,
        }
    }

    /// Construct a backend from the parsed config.
    ///
    /// # Errors
    ///
    /// Returns an error when `kind = Minilm` is requested but the
    /// `semantic-ort` feature is not compiled in, or when the underlying
    /// model/tokenizer cannot be loaded.
    pub fn from_config(config: &SemanticBackendConfig) -> Result<Self> {
        match config.kind {
            SemanticBackendKind::Hash => Ok(Self::Hash(HashEmbedBackend::new())),
            SemanticBackendKind::Minilm => Self::load_minilm(config),
        }
    }

    #[cfg(feature = "semantic-ort")]
    fn load_minilm(config: &SemanticBackendConfig) -> Result<Self> {
        let backend = MiniLmBackend::load(
            config.model_path.as_deref(),
            config.tokenizer_path.as_deref(),
        )
        .context("load MiniLM-L6-v2 backend")?;
        Ok(Self::MiniLm(backend))
    }

    #[cfg(not(feature = "semantic-ort"))]
    fn load_minilm(_config: &SemanticBackendConfig) -> Result<Self> {
        bail!(
            "kb.toml requests semantic backend `minilm` but this kb binary was built without \
             the `semantic-ort` feature. Rebuild with `cargo build --features semantic-ort` \
             (or `just install --features semantic-ort`) to enable the ONNX `MiniLM` backend."
        )
    }
}

impl EmbeddingBackend for SemanticBackend {
    fn backend_id(&self) -> &'static str {
        match self {
            Self::Hash(b) => b.backend_id(),
            #[cfg(feature = "semantic-ort")]
            Self::MiniLm(b) => b.backend_id(),
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            Self::Hash(b) => b.dimensions(),
            #[cfg(feature = "semantic-ort")]
            Self::MiniLm(b) => b.dimensions(),
        }
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Hash(b) => b.embed(text),
            #[cfg(feature = "semantic-ort")]
            Self::MiniLm(b) => b.embed(text),
        }
    }
}

fn hash_embed(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vec = vec![0.0_f32; dimensions];
    let normalized = text.to_lowercase();
    let chars: Vec<char> = normalized.chars().collect();

    if chars.is_empty() {
        return vec;
    }

    // Word-level signal: hash each whitespace-delimited token.
    for word in normalized.split_whitespace() {
        if !word.is_empty() {
            let idx = fnv1a(word.as_bytes()) % dimensions;
            vec[idx] += 1.0;
        }
    }

    // Character n-grams (3..=5) for morphological similarity.
    for n in NGRAM_MIN..=NGRAM_MAX {
        if chars.len() < n {
            continue;
        }
        for window in chars.windows(n) {
            let s: String = window.iter().collect();
            let idx = fnv1a(s.as_bytes()) % dimensions;
            vec[idx] += 1.0;
        }
    }

    normalize_l2(&mut vec);
    vec
}

/// FNV-1a hash. Fast, simple, good distribution for feature hashing.
fn fnv1a(bytes: &[u8]) -> usize {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    // Truncation to usize is intentional: we only need a well-distributed
    // index into the embedding dimension, not the full 64-bit value.
    #[allow(clippy::cast_possible_truncation)]
    {
        hash as usize
    }
}

fn normalize_l2(values: &mut [f32]) {
    let norm_sq: f32 = values.iter().map(|v| v * v).sum();
    if norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for v in values {
            *v *= inv_norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_produces_correct_dimensions() {
        let backend = HashEmbedBackend::new();
        let embedding = backend.embed("hello world").expect("embed");
        assert_eq!(embedding.len(), HASH_DIM);
    }

    #[test]
    fn embed_is_l2_normalized() {
        let backend = HashEmbedBackend::new();
        let embedding = backend.embed("some text to embed").expect("embed");
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn empty_text_produces_zero_vector() {
        let backend = HashEmbedBackend::new();
        let embedding = backend.embed("").expect("embed");
        assert!(embedding.iter().all(|&x| x == 0.0));
        assert_eq!(embedding.len(), HASH_DIM);
    }

    #[test]
    fn deterministic_output() {
        let backend = HashEmbedBackend::new();
        let a = backend.embed("deterministic test").expect("a");
        let b = backend.embed("deterministic test").expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn similar_texts_have_higher_similarity_than_unrelated_ones() {
        let backend = HashEmbedBackend::new();
        let a = backend.embed("fix login bug in authentication").expect("a");
        let b = backend.embed("fix auth login issue").expect("b");
        let c = backend.embed("add dark mode to settings page").expect("c");

        let related = dot(&a, &b);
        let unrelated = dot(&a, &c);
        assert!(
            related > unrelated,
            "similar texts should score higher: related={related} vs unrelated={unrelated}"
        );
    }

    #[test]
    fn backend_id_is_stable() {
        let backend = HashEmbedBackend::new();
        assert_eq!(backend.backend_id(), "hash-embed-256-chunked");
        assert_eq!(backend.dimensions(), HASH_DIM);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn semantic_backend_default_is_hash_on_windows() {
        let backend = SemanticBackend::from_config(&SemanticBackendConfig::default())
            .expect("hash backend always loads");
        assert_eq!(backend.backend_id(), HASH_BACKEND_ID);
        assert_eq!(backend.dimensions(), HASH_DIM);
        let v = backend.embed("hello").expect("embed");
        assert_eq!(v.len(), HASH_DIM);
    }

    #[test]
    fn semantic_backend_explicit_hash_kind_dispatches_to_hash() {
        // Independent of the platform default — when a user explicitly
        // selects `Hash` (e.g. via `kb.toml backend = "hash"`), the
        // dispatcher must honor it and return a hash-embed backend.
        let backend = SemanticBackend::from_config(&SemanticBackendConfig {
            kind: SemanticBackendKind::Hash,
            ..SemanticBackendConfig::default()
        })
        .expect("hash backend always loads");
        assert_eq!(backend.backend_id(), HASH_BACKEND_ID);
        assert_eq!(backend.dimensions(), HASH_DIM);
    }

    #[cfg(not(feature = "semantic-ort"))]
    #[test]
    fn minilm_kind_errors_without_feature() {
        let cfg = SemanticBackendConfig {
            kind: SemanticBackendKind::Minilm,
            ..SemanticBackendConfig::default()
        };
        let err = SemanticBackend::from_config(&cfg).expect_err("must fail without feature");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("semantic-ort"),
            "error should point at the feature flag, got: {msg}"
        );
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}
