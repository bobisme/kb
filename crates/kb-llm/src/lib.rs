#![forbid(unsafe_code)]

mod provenance;
pub mod adapter;
mod templates;

pub use adapter::{
    LlmAdapter, LlmAdapterError, SummarizeDocumentRequest, SummarizeDocumentResponse,
    ExtractConceptsRequest, ExtractConceptsResponse, MergeConceptCandidatesRequest,
    MergeConceptCandidatesResponse, AnswerQuestionRequest, AnswerQuestionResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, RunHealthCheckRequest, RunHealthCheckResponse,
};
pub use provenance::{BackendResponse, ProvenanceRecord, TokenUsage};
pub use templates::{Template, RenderedTemplate};
