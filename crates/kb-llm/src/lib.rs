#![forbid(unsafe_code)]

pub mod adapter;
pub mod claude;
pub mod opencode;
mod provenance;
pub mod router;
pub mod subprocess;
mod templates;

pub use adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, CandidateSourceSnippet, ConceptCandidate,
    ContradictionQuote, DetectContradictionsRequest, DetectContradictionsResponse,
    ExtractConceptsRequest, ExtractConceptsResponse, GenerateConceptBodyRequest,
    GenerateConceptBodyResponse, GenerateConceptFromCandidateRequest,
    GenerateConceptFromCandidateResponse, GenerateSlidesRequest, GenerateSlidesResponse,
    ImputeGapKind, ImputeGapRequest, ImputeGapResponse, ImputedWebSource, LlmAdapter,
    LlmAdapterError, MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, MergeGroup,
    RunHealthCheckRequest, RunHealthCheckResponse, SourceAnchor, SummarizeDocumentRequest,
    SummarizeDocumentResponse, parse_detect_contradictions_json,
    parse_generate_concept_from_candidate_json, parse_impute_gap_json,
    parse_merge_concept_candidates_json,
};
pub use claude::{ClaudeCliAdapter, ClaudeCliConfig};
pub use opencode::{OpencodeAdapter, OpencodeConfig};
pub use provenance::{BackendResponse, ProvenanceRecord, TokenUsage};
pub use router::{Backend, BackendRouter, Router};
pub use subprocess::{SubprocessError, SubprocessOutput, run_shell_command};
pub use templates::{RenderedTemplate, Template};
