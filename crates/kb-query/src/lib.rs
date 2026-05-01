#![forbid(unsafe_code)]

pub mod artifact;
pub mod grounding;
pub mod hybrid;
pub mod images;
pub mod lexical;
pub mod paths;
pub mod semantic;
pub mod writer;

pub use artifact::{
    ArtifactResult, CitationManifest, build_citation_manifest, postprocess_answer,
    render_manifest_for_prompt, strip_tool_narration,
};
pub use hybrid::{
    HybridOptions, HybridResult, MIN_SEMANTIC_SCORE, MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL, RRF_K,
    hybrid_search, hybrid_search_with_backend, hybrid_search_with_index,
    hybrid_search_with_index_and_backend, hybrid_search_with_options, plan_retrieval_hybrid,
    plan_retrieval_hybrid_with_backend,
};
pub use images::{MAX_IMAGES_PER_QUERY, plan_mentions_images, resolve_candidate_image_paths};
pub use lexical::{
    AssembledContext, ContextChunkKind, ContextManifestEntry, LexicalEntry, LexicalIndex,
    RetrievalCandidate, RetrievalPlan, STOPWORDS, SearchResult, assemble_context,
    build_lexical_index, lexical_index_path, query_reduced_to_stopwords, tokenize_query,
};
pub use paths::resolve_question_dir;
pub use semantic::{
    EMBEDDING_DB_REL, EmbeddingBackend, HASH_BACKEND_ID, HashEmbedBackend, MINILM_BACKEND_ID,
    SemanticBackend, SemanticBackendConfig, SemanticBackendKind, SemanticIndexStats, SyncStats,
    embedding_db_path, semantic_index_stats, sync_embeddings,
};
pub use writer::{ArtifactSidecar, WriteArtifactInput, WriteArtifactOutput, write_artifact};
