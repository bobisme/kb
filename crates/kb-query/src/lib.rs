#![forbid(unsafe_code)]

pub mod artifact;
pub mod grounding;
pub mod images;
pub mod lexical;
pub mod writer;

pub use artifact::{
    ArtifactResult, CitationManifest, build_citation_manifest, postprocess_answer,
    render_manifest_for_prompt,
};
pub use images::{MAX_IMAGES_PER_QUERY, plan_mentions_images, resolve_candidate_image_paths};
pub use lexical::{
    AssembledContext, ContextChunkKind, ContextManifestEntry, LexicalEntry, LexicalIndex,
    RetrievalCandidate, RetrievalPlan, STOPWORDS, SearchResult, assemble_context,
    build_lexical_index, lexical_index_path, query_reduced_to_stopwords, tokenize_query,
};
pub use writer::{ArtifactSidecar, WriteArtifactInput, WriteArtifactOutput, write_artifact};
