#![forbid(unsafe_code)]

pub mod lexical;

pub use lexical::{
    AssembledContext, ContextChunkKind, ContextManifestEntry, LexicalEntry, LexicalIndex,
    RetrievalCandidate, RetrievalPlan, SearchResult, assemble_context, build_lexical_index,
};
