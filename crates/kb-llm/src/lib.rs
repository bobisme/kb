#![forbid(unsafe_code)]

pub mod adapter;
pub mod claude;
mod provenance;
pub mod subprocess;
mod templates;

pub use adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter, LlmAdapterError,
    MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, RunHealthCheckRequest,
    RunHealthCheckResponse, SummarizeDocumentRequest, SummarizeDocumentResponse,
};
pub use claude::{ClaudeCliAdapter, ClaudeCliConfig};
pub use provenance::{BackendResponse, ProvenanceRecord, TokenUsage};
pub use subprocess::{SubprocessError, SubprocessOutput, run_shell_command};
pub use templates::{RenderedTemplate, Template};
