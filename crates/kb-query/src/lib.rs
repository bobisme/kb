#![forbid(unsafe_code)]

pub mod lexical;

pub use lexical::{
    LexicalEntry, LexicalIndex, RetrievalCandidate, RetrievalPlan, SearchResult,
    build_lexical_index,
};
