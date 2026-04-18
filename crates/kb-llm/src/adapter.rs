use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ProvenanceRecord;

/// Error type for `LlmAdapter` operations with classification by error class.
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
    /// Display title for the source document being summarized.
    pub title: String,
    /// Canonical body text to summarize.
    pub body: String,
    /// Maximum target length for the returned summary.
    pub max_words: usize,
}

/// Response containing the document summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeDocumentResponse {
    /// The generated summary text.
    pub summary: String,
}

/// Anchor back into the source that supports an extracted concept candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAnchor {
    /// Optional stable heading/section anchor within the source.
    pub heading_anchor: Option<String>,
    /// Optional exact quote or short supporting snippet from the source.
    pub quote: Option<String>,
}

/// Candidate concept emitted by the extraction pass before canonicalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptCandidate {
    /// Preferred concept label for this candidate.
    pub name: String,
    /// Alternate labels or spellings observed in the source.
    pub aliases: Vec<String>,
    /// Short definition or gloss for the concept.
    pub definition_hint: Option<String>,
    /// Source-backed anchors supporting this extraction.
    pub source_anchors: Vec<SourceAnchor>,
}

/// Request to extract concept candidates from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractConceptsRequest {
    /// Display title for the source document being analyzed.
    pub title: String,
    /// Canonical body text to analyze.
    pub body: String,
    /// Optional precomputed source summary to give the extractor more context.
    pub summary: Option<String>,
    /// Maximum number of concepts to extract (None for no limit).
    pub max_concepts: Option<usize>,
}

/// Response containing extracted concept candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractConceptsResponse {
    /// Candidate concepts found in the source.
    pub concepts: Vec<ConceptCandidate>,
}

/// One cluster of candidates merged into a single canonical concept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeGroup {
    /// Proposed canonical name for this concept.
    pub canonical_name: String,
    /// Alternate names, abbreviations, or spellings collapsed into this concept.
    pub aliases: Vec<String>,
    /// Original candidates clustered into this group.
    pub members: Vec<ConceptCandidate>,
    /// True when the merge is unambiguous; false routes the group to human review.
    pub confident: bool,
    /// Explanation for uncertain merges.
    pub rationale: Option<String>,
}

/// Request to merge concept candidates into a canonical list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConceptCandidatesRequest {
    /// Flat list of all candidates to cluster, sorted deterministically by the caller.
    pub candidates: Vec<ConceptCandidate>,
}

/// Response containing the merged concept list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConceptCandidatesResponse {
    /// Grouped and deduplicated concept clusters.
    pub groups: Vec<MergeGroup>,
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

/// Extract JSON candidate payloads from LLM text output.
///
/// # Errors
///
/// Returns [`LlmAdapterError::Parse`] when the text is empty, not valid JSON, or does not
/// match the expected concept-candidate response schema.
pub fn parse_extract_concepts_json(text: &str) -> Result<ExtractConceptsResponse, LlmAdapterError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmAdapterError::Parse(
            "extract_concepts response was empty".to_string(),
        ));
    }

    let json_text = trimmed
        .strip_prefix("```")
        .and_then(|body| body.split_once('\n').map(|(_, rest)| rest))
        .and_then(|body| body.rsplit_once("```").map(|(json, _)| json.trim()))
        .unwrap_or(trimmed);

    let value: serde_json::Value = serde_json::from_str(json_text).map_err(|err| {
        LlmAdapterError::Parse(format!("extract_concepts response was not valid JSON: {err}"))
    })?;

    if value.is_array() {
        let concepts = serde_json::from_value(value).map_err(|err| {
            LlmAdapterError::Parse(format!("extract_concepts response had invalid candidate list: {err}"))
        })?;
        return Ok(ExtractConceptsResponse { concepts });
    }

    serde_json::from_value(value).map_err(|err| {
        LlmAdapterError::Parse(format!(
            "extract_concepts response had invalid envelope shape: {err}"
        ))
    })
}

/// Parse a `MergeConceptCandidatesResponse` from LLM text output.
///
/// Accepts raw JSON or a fenced code block.
///
/// # Errors
///
/// Returns [`LlmAdapterError::Parse`] when the text is empty, not valid JSON, or does not
/// match the expected merge-response schema.
pub fn parse_merge_concept_candidates_json(
    text: &str,
) -> Result<MergeConceptCandidatesResponse, LlmAdapterError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmAdapterError::Parse(
            "merge_concept_candidates response was empty".to_string(),
        ));
    }

    let json_text = trimmed
        .strip_prefix("```")
        .and_then(|body| body.split_once('\n').map(|(_, rest)| rest))
        .and_then(|body| body.rsplit_once("```").map(|(json, _)| json.trim()))
        .unwrap_or(trimmed);

    serde_json::from_str(json_text).map_err(|err| {
        LlmAdapterError::Parse(format!(
            "merge_concept_candidates response had invalid shape: {err}"
        ))
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_extract_concepts_accepts_enveloped_object() {
        let parsed = parse_extract_concepts_json(
            r#"{
                "concepts": [
                    {
                        "name": "Rust ownership",
                        "aliases": ["ownership"],
                        "definition_hint": "Rules that govern value moves and borrowing.",
                        "source_anchors": [
                            {"heading_anchor": "ownership", "quote": "Ownership is Rust's most unique feature."}
                        ]
                    }
                ]
            }"#,
        )
        .expect("parse candidate envelope");

        assert_eq!(parsed.concepts.len(), 1);
        assert_eq!(parsed.concepts[0].name, "Rust ownership");
        assert_eq!(parsed.concepts[0].aliases, vec!["ownership"]);
        assert_eq!(
            parsed.concepts[0].source_anchors[0].heading_anchor.as_deref(),
            Some("ownership")
        );
    }

    #[test]
    fn parse_extract_concepts_accepts_fenced_array() {
        let parsed = parse_extract_concepts_json(
            "```json\n[{\"name\":\"Borrow checker\",\"aliases\":[],\"definition_hint\":null,\"source_anchors\":[{\"heading_anchor\":null,\"quote\":\"The borrow checker validates references.\"}]}]\n```",
        )
        .expect("parse fenced candidate list");

        assert_eq!(parsed.concepts.len(), 1);
        assert_eq!(parsed.concepts[0].name, "Borrow checker");
        assert_eq!(
            parsed.concepts[0].source_anchors[0].quote.as_deref(),
            Some("The borrow checker validates references.")
        );
    }
}
