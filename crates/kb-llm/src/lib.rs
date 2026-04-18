#![forbid(unsafe_code)]

mod provenance;
pub mod adapter;

pub use adapter::{
    LlmAdapter, LlmAdapterError, SummarizeDocumentRequest, SummarizeDocumentResponse,
    ExtractConceptsRequest, ExtractConceptsResponse, MergeConceptCandidatesRequest,
    MergeConceptCandidatesResponse, AnswerQuestionRequest, AnswerQuestionResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, RunHealthCheckRequest, RunHealthCheckResponse,
};
pub use provenance::{BackendResponse, ProvenanceRecord, TokenUsage};
