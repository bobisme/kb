#![forbid(unsafe_code)]

pub mod artifact;
pub mod lexical;
pub mod writer;

pub use artifact::{
    ArtifactResult, CitationManifest, build_citation_manifest, postprocess_answer,
    render_manifest_for_prompt,
};
pub use lexical::{
    AssembledContext, ContextChunkKind, ContextManifestEntry, LexicalEntry, LexicalIndex,
    RetrievalCandidate, RetrievalPlan, SearchResult, assemble_context, build_lexical_index,
};
pub use writer::{ArtifactSidecar, WriteArtifactInput, WriteArtifactOutput, write_artifact};
