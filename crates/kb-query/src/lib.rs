#![forbid(unsafe_code)]

pub mod lexical;

pub use lexical::{LexicalEntry, LexicalIndex, SearchResult, build_lexical_index};
