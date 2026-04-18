use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ProvenanceRecord;

/// Error type for LlmAdapter operations with classification by error class.
#[derive(Debug, Error)]
pub enum LlmAdapterError {
    /// Transport-level failures (connection refused, DNS resolution, etc.).
    #[error("Transport error: {0}")]
    Transport(String),

    /// Request or response exceeded time limits.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Failed to parse response from backend.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Authentication or authorization failure.
    #[error("Authentication error: {0}")]
    Auth(String),

    /// Rate limit or quota exceeded.
    #[error("Rate limit: {0}")]
    RateLimit(String),

    /// Other unclassified errors.
    #[error("Other error: {0}")]
    Other(String),
}

/// Request to summarize a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeDocumentRequest {
    /// The document text to summarize.
    pub document_text: String,
    /// Optional instruction for summarization style (e.g. "bullet points", "paragraph").
    pub style: Option<String>,
}

/// Response containing the document summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeDocumentResponse {
    /// The generated summary text.
    pub summary: String,
}

/// Request to extract concepts from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractConceptsRequest {
    /// The document text to analyze.
    pub document_text: String,
    /// Maximum number of concepts to extract (None for no limit).
    pub max_concepts: Option<usize>,
}

/// Response containing extracted concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractConceptsResponse {
    /// List of extracted concept names.
    pub concepts: Vec<String>,
}

/// Request to merge concept candidates into a canonical list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConceptCandidatesRequest {
    /// Multiple lists of candidate concepts to merge.
    pub candidate_lists: Vec<Vec<String>>,
    /// Strategy for merging (e.g. "union", "intersection").
    pub merge_strategy: String,
}

/// Response containing the merged concept list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConceptCandidatesResponse {
    /// The merged and deduplicated list of concepts.
    pub merged_concepts: Vec<String>,
}

/// Request to answer a user question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerQuestionRequest {
    /// The user's question.
    pub question: String,
    /// Context documents or retrieved passages to use in answering.
    pub context: Vec<String>,
    /// Optional format instruction (e.g. "concise", "detailed").
    pub format: Option<String>,
}

/// Response containing the answer to a question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerQuestionResponse {
    /// The generated answer text.
    pub answer: String,
    /// Optional list of relevant concept IDs or references.
    pub references: Option<Vec<String>>,
}

/// Request to generate slides from content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSlidesRequest {
    /// The source content to convert into slides.
    pub content: String,
    /// Number of slides to generate (None for auto-determined).
    pub target_slide_count: Option<usize>,
    /// Optional output format (e.g. "markdown", "json").
    pub output_format: Option<String>,
}

/// Response containing generated slide content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSlidesResponse {
    /// The generated slide deck content.
    pub slides: String,
    /// Actual number of slides in the output.
    pub slide_count: usize,
}

/// Request to run a health check on the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunHealthCheckRequest {
    /// Optional details about which checks to perform.
    pub check_details: Option<String>,
}

/// Response indicating backend health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunHealthCheckResponse {
    /// Overall health status (e.g. "healthy", "degraded", "unhealthy").
    pub status: String,
    /// Optional details about individual component health.
    pub details: Option<String>,
}

/// Adapter trait for interacting with LLM backends.
///
/// All methods are synchronous and return a response payload along with provenance metadata.
/// Each operation is independent and may be retried independently.
pub trait LlmAdapter: Send + Sync {
    /// Summarize a document.
    ///
    /// Takes a document and optional style instruction, returns a summary.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or encounters authentication issues.
    fn summarize_document(
        &self,
        request: SummarizeDocumentRequest,
    ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError>;

    /// Extract named concepts from a document.
    ///
    /// Identifies and extracts key concepts, optionally limiting the result count.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or encounters authentication issues.
    fn extract_concepts(
        &self,
        request: ExtractConceptsRequest,
    ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError>;

    /// Merge multiple lists of concept candidates into a canonical list.
    ///
    /// Applies a merge strategy to deduplicate and consolidate concepts from multiple sources.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or encounters authentication issues.
    fn merge_concept_candidates(
        &self,
        request: MergeConceptCandidatesRequest,
    ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError>;

    /// Answer a user question using context documents.
    ///
    /// Takes a question and context passages, returns a generated answer and optional references.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or encounters authentication issues.
    fn answer_question(
        &self,
        request: AnswerQuestionRequest,
    ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError>;

    /// Generate a slide deck from source content.
    ///
    /// Converts content into a structured slide presentation with optional format and count constraints.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or encounters authentication issues.
    fn generate_slides(
        &self,
        request: GenerateSlidesRequest,
    ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError>;

    /// Perform a health check on the backend.
    ///
    /// Verifies the backend is operational and returns overall and component-level status.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or encounters authentication issues.
    fn run_health_check(
        &self,
        request: RunHealthCheckRequest,
    ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError>;
}
