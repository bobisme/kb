#![forbid(unsafe_code)]

pub mod adapter;
pub mod claude;
pub mod opencode;
mod provenance;
pub mod router;
pub mod subprocess;
mod templates;

pub use adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, ConceptCandidate, ExtractConceptsRequest,
    ExtractConceptsResponse, GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter,
    LlmAdapterError, MergeConceptCandidatesRequest, MergeConceptCandidatesResponse,
    RunHealthCheckRequest, RunHealthCheckResponse, SourceAnchor, SummarizeDocumentRequest,
    SummarizeDocumentResponse,
};
pub use claude::{ClaudeCliAdapter, ClaudeCliConfig};
pub use opencode::{OpencodeAdapter, OpencodeConfig};
pub use provenance::{BackendResponse, ProvenanceRecord, TokenUsage};
pub use router::{Backend, BackendRouter, Router};
pub use subprocess::{SubprocessError, SubprocessOutput, run_shell_command};
pub use templates::{RenderedTemplate, Template};
