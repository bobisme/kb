#![forbid(unsafe_code)]

pub mod adapter;
mod provenance;
pub mod subprocess;
mod templates;

pub use adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter, LlmAdapterError,
    MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, RunHealthCheckRequest,
    RunHealthCheckResponse, SummarizeDocumentRequest, SummarizeDocumentResponse,
};
pub use provenance::{BackendResponse, ProvenanceRecord, TokenUsage};
pub use subprocess::{run_shell_command, SubprocessError, SubprocessOutput};
pub use templates::{RenderedTemplate, Template};
