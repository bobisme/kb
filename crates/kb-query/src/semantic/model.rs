//! Embedding backend trait and the always-available hash-embed implementation.
//!
//! kb v1 ships a single backend: [`HashEmbedBackend`]. It hashes 3-to-5
//! character n-grams plus whitespace-delimited word tokens into a 256-dim
//! vector via FNV-1a, then L2-normalizes. No ML model, no extra deps. The
//! quality is intentionally weak — it catches morphological similarity
//! ("authentication" / "authn") and rephrased queries, but not full
//! conceptual paraphrases. That's Phase 2.
//!
//! Pattern lifted from `bones-search`'s `hash_embed.rs`.

use anyhow::Result;

/// Default embedding dimension.
pub const HASH_DIM: usize = 256;

/// Stable identifier for the active backend. Stored in `semantic_meta` so
/// changing backends triggers a re-embed of the corpus on the next compile.
pub const HASH_BACKEND_ID: &str = "hash-embed-256";

const NGRAM_MIN: usize = 3;
const NGRAM_MAX: usize = 5;

/// Trait for embedding backends. v1 has a single impl; Phase 2 may add
/// ONNX-backed and model2vec-backed implementations behind feature flags.
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
    /// in practice; the trait method allows Phase 2 backends to surface
    /// runtime errors uniformly.
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
        assert_eq!(backend.backend_id(), "hash-embed-256");
        assert_eq!(backend.dimensions(), HASH_DIM);
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}
