//! Semantic search support for kb.
//!
//! Phase 1 shipped a zero-dependency hash-embed backend, content-hash-deduped
//! incremental embedding pipeline, and KNN search with a sqlite-vec fast path
//! plus a Rust-side cosine fallback.
//!
//! Phase 2 (bn-1rww) adds an opt-in ONNX MiniLM-L6-v2 backend behind the
//! `semantic-ort` feature. Backend selection lives in `kb.toml [semantic]`
//! and is dispatched at runtime via [`SemanticBackend`].

pub mod chunk;
pub mod embed;
#[cfg(feature = "semantic-ort")]
pub mod minilm;
pub mod model;
pub mod search;

pub use chunk::{Chunk, TARGET_MAX_TOKENS, TARGET_MIN_TOKENS, chunk_markdown};
pub use embed::{
    EMBEDDING_DB_REL, EmbeddingPipeline, SemanticIndexStats, SyncStats, chunk_id_for,
    embedding_db_path, ensure_embedding_schema, item_id_for_relpath, semantic_index_stats,
    sync_embeddings,
};
pub use model::{
    EmbeddingBackend, HASH_BACKEND_ID, HashEmbedBackend, MINILM_BACKEND_ID, SemanticBackend,
    SemanticBackendConfig, SemanticBackendKind,
};
pub use search::{
    SemanticChunkHit, SemanticItemHit, SemanticSearchResult, aggregate_chunks_to_items, knn_search,
};
