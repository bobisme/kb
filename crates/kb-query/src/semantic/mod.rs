//! Semantic search support for kb.
//!
//! Phase 1: zero-dependency hash-embed backend, content-hash-deduped
//! incremental embedding pipeline, and KNN search with a sqlite-vec fast
//! path plus a Rust-side cosine fallback. See `notes/semantic-retrieval.md`
//! in the workspace for the design.

pub mod embed;
pub mod model;
pub mod search;

pub use embed::{
    EMBEDDING_DB_REL, EmbeddingPipeline, SemanticIndexStats, SyncStats, embedding_db_path,
    ensure_embedding_schema, item_id_for_relpath, semantic_index_stats, sync_embeddings,
};
pub use model::{EmbeddingBackend, HashEmbedBackend};
pub use search::{SemanticSearchResult, knn_search};
