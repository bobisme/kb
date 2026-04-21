use std::path::PathBuf;

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
    /// Optional 1-3 word tag grouping related concepts under a common heading
    /// (e.g. "async", "consensus"). Emitted into concept frontmatter as
    /// `category:` and used by `wiki/concepts/index.md` to bucket entries.
    ///
    /// Serialized field is flattened and defaults to `None` so older
    /// adapter responses (and hand-written fixtures) parse without a
    /// `category` key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
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

/// Request to generate a general-scope body for a canonical concept.
///
/// bn-1w5: body synthesis used to happen inside the single merge call, which
/// is where the "LLM latches onto the most-quoted variant" regression lives.
/// Splitting it out lets the prompt be minimal + strict ("describe what the
/// aliases have in COMMON"), which the merge prompt can't enforce because it
/// has to carry all the clustering rules too.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConceptBodyRequest {
    /// Canonical concept name picked by the merge step.
    pub canonical_name: String,
    /// Folded aliases under the canonical name.
    pub aliases: Vec<String>,
    /// Candidate source quotes to show the model as context only
    /// (the prompt instructs the model NOT to copy them).
    pub candidate_quotes: Vec<String>,
}

/// Response containing a plain-text concept body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConceptBodyResponse {
    /// 1-3 sentence general body — already stripped of code fences / markdown.
    pub body: String,
}

/// Request to draft a canonical concept entry from a lint-flagged candidate.
///
/// bn-lw06: the `missing_concepts` lint queues `concept_candidate` review
/// items with a surface term (e.g. `"FooBar System"`) and the source
/// documents that mention it. On approve, the CLI hands a snippet of each
/// source body to the LLM so it can propose the canonical shape: preferred
/// name, aliases, category tag, and a general-scope 2-3 sentence
/// definition. Mirrors the `bn-1w5` two-step pattern — callers may refine
/// the body via a subsequent
/// [`generate_concept_body`](LlmAdapter::generate_concept_body) call if the
/// returned definition looks variant-narrowed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateConceptFromCandidateRequest {
    /// Surface term flagged by the lint (what the user sees in `kb review show`).
    pub candidate_name: String,
    /// Source-document ids with a short snippet each — one bullet per source.
    /// The adapter renders these verbatim into the prompt so the LLM can see
    /// how the term is used in context.
    pub source_snippets: Vec<CandidateSourceSnippet>,
    /// Pre-existing concept category tags in the knowledge base. The prompt
    /// instructs the model to reuse one of these when a reasonable fit exists
    /// rather than spawning a new tag per concept.
    pub existing_categories: Vec<String>,
}

/// One source's contribution to a concept-candidate prompt.
///
/// The `snippet` is already trimmed/truncated by the caller before being
/// handed to the adapter; the adapter treats it as opaque text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSourceSnippet {
    /// The source document id (e.g. `repo/foo` or a hashed content id).
    pub source_document_id: String,
    /// A short excerpt from the source that mentions the candidate term.
    pub snippet: String,
}

/// Response containing a drafted canonical concept entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateConceptFromCandidateResponse {
    /// Preferred canonical display name (may be identical to the candidate).
    pub canonical_name: String,
    /// Alternate labels observed in the sources (0-5).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Optional 1-3 word tag for the concepts index. `None` puts the
    /// concept under "Uncategorized" (matches the merge-pass convention).
    #[serde(default)]
    pub category: Option<String>,
    /// 2-3 sentence general definition. May be refined by a follow-up
    /// `generate_concept_body` call; the prompt already asks for a general
    /// scope, but the two-step pattern gives a safety net.
    pub definition: String,
}

/// Parse a `GenerateConceptFromCandidateResponse` from LLM text output.
///
/// Accepts raw JSON or a fenced code block. Unknown fields are ignored so
/// model drift (e.g. an extra `rationale`) doesn't break parsing.
///
/// # Errors
///
/// Returns [`LlmAdapterError::Parse`] when the text is empty, not valid JSON,
/// or does not match the expected schema.
pub fn parse_generate_concept_from_candidate_json(
    text: &str,
) -> Result<GenerateConceptFromCandidateResponse, LlmAdapterError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmAdapterError::Parse(
            "generate_concept_from_candidate response was empty".to_string(),
        ));
    }

    let json_text = trimmed
        .strip_prefix("```")
        .and_then(|body| body.split_once('\n').map(|(_, rest)| rest))
        .and_then(|body| body.rsplit_once("```").map(|(json, _)| json.trim()))
        .unwrap_or(trimmed);

    serde_json::from_str(json_text).map_err(|err| {
        LlmAdapterError::Parse(format!(
            "generate_concept_from_candidate response had invalid shape: {err}"
        ))
    })
}

/// Request to answer a user question.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnswerQuestionRequest {
    /// The user's question.
    pub question: String,
    /// Context documents or retrieved passages to use in answering.
    pub context: Vec<String>,
    /// Optional format instruction (e.g. "concise", "detailed").
    pub format: Option<String>,
    /// Optional prompt template name to load. Defaults to `ask.md` when `None`.
    /// Used by alternative ask pipelines (e.g. `--format=chart` -> `ask_chart.md`)
    /// to swap the prompt without adding a separate adapter method.
    #[serde(default)]
    pub template_name: Option<String>,
    /// Optional output-path directive (substituted into `{{output_path}}` in
    /// the template). Used by chart/figure pipelines to tell the LLM the exact
    /// file path it must write, so the post-run check can assert the file
    /// landed where kb expects it.
    #[serde(default)]
    pub output_path: Option<String>,
    /// Absolute paths to image assets referenced by the retrieval context.
    ///
    /// Adapters are expected to attach these to the LLM invocation so the
    /// model can see the images instead of just their markdown references
    /// (bn-3dkw — multimodal retrieval). The caller caps the list (typically
    /// at 5) and is responsible for only passing files that actually exist.
    /// Empty vec means "no images" — adapters must treat that as a cheap
    /// no-op and not pay any attachment cost.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_paths: Vec<PathBuf>,
    /// When `true`, adapters that speak a tool-use streaming protocol should
    /// request a structured event stream from the runner and extract only the
    /// final assistant message, skipping any intermediate tool-narration text.
    ///
    /// bn-1ikn: the `--format=chart` path drives the LLM through a
    /// write/bash tool cycle, so opencode streams per-tool "what I'm about to
    /// do" commentary ahead of the real caption. When this flag is set, the
    /// opencode adapter invokes `opencode run --format json` and keeps only
    /// the `text` event carrying `metadata.openai.phase = "final_answer"`.
    /// Adapters that don't expose a structured stream (e.g. `claude` already
    /// parses its JSON output) MAY ignore this flag — it's a hint, not a
    /// contract. Defaults to `false` so existing callers are unaffected.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub structured_output: bool,
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

/// Request to check a set of source quotes about a concept for mutual
/// contradictions. One request == one concept; indices in the response refer
/// back to positions in the supplied `quotes` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectContradictionsRequest {
    /// Display name for the concept the quotes are about (for the prompt).
    pub concept_name: String,
    /// Ordered quotes from different source documents. `source_label` is a
    /// human-readable origin (e.g. source-doc id) used only for the prompt;
    /// the LLM refers to quotes by their numeric index when reporting.
    pub quotes: Vec<ContradictionQuote>,
}

/// One quote fed to the contradiction detector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContradictionQuote {
    /// The source document id (or other stable origin label) this quote came
    /// from. Used both to bucket quotes by source (the check only fires when
    /// quotes span ≥ 2 sources) and to render the prompt.
    pub source_label: String,
    /// The quote text itself.
    pub text: String,
}

/// Response from a contradiction-detection pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectContradictionsResponse {
    /// `true` when the quotes make contradictory claims about the concept.
    pub contradiction: bool,
    /// Short natural-language explanation of the contradiction (or a brief
    /// "no contradiction" note when `contradiction` is `false`).
    pub explanation: String,
    /// Zero-based indices into the request's `quotes` list naming the
    /// specific quotes that disagree. Empty when `contradiction` is `false`.
    #[serde(default)]
    pub conflicting_quotes: Vec<usize>,
}

/// Parse a `DetectContradictionsResponse` from LLM text output. Accepts raw
/// JSON or fenced code blocks (```json ... ```).
///
/// # Errors
///
/// Returns [`LlmAdapterError::Parse`] when the text is empty, not valid JSON,
/// or does not match the expected contradiction-response schema.
pub fn parse_detect_contradictions_json(
    text: &str,
) -> Result<DetectContradictionsResponse, LlmAdapterError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmAdapterError::Parse(
            "detect_contradictions response was empty".to_string(),
        ));
    }

    let json_text = trimmed
        .strip_prefix("```")
        .and_then(|body| body.split_once('\n').map(|(_, rest)| rest))
        .and_then(|body| body.rsplit_once("```").map(|(json, _)| json.trim()))
        .unwrap_or(trimmed);

    serde_json::from_str(json_text).map_err(|err| {
        LlmAdapterError::Parse(format!(
            "detect_contradictions response had invalid shape: {err}"
        ))
    })
}

/// The kind of lint gap being imputed.
///
/// Controls prompt wording (e.g. "the concept does not exist yet" vs.
/// "the concept's body is too thin") and informs the approve handler
/// whether to create a new page or extend an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImputeGapKind {
    /// No concept page exists for a term mentioned across several sources.
    /// Approve drafts a new `wiki/concepts/<slug>.md`.
    MissingConcept,
    /// A concept page exists but its body is empty or too short to be
    /// useful. Approve extends the existing page's body.
    ThinConceptBody,
}

impl ImputeGapKind {
    /// Human-readable name used in prompts and UI. Stable — changes here
    /// change prompt render hashes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MissingConcept => "missing_concept",
            Self::ThinConceptBody => "thin_concept_body",
        }
    }

    /// One-sentence description used to hint the LLM at what the fix should
    /// look like. Visible in the prompt, kept minimal so prompt rendering
    /// stays compact.
    #[must_use]
    pub const fn prompt_description(self) -> &'static str {
        match self {
            Self::MissingConcept => {
                "No concept page exists yet for this term. Draft a from-scratch general-scope definition."
            }
            Self::ThinConceptBody => {
                "A concept page exists but its body is empty or too short. Draft a replacement body (general-scope)."
            }
        }
    }
}

/// Request to impute (web-search-backfill) missing or thin KB content.
///
/// The adapter MUST enable the runner's web-search tool when it exposes
/// one — this is the one call kb makes that expects the model to browse
/// external sources, not just reason over the KB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImputeGapRequest {
    /// What kind of gap this is (missing vs. thin body).
    pub gap_kind: ImputeGapKind,
    /// Display name for the concept being imputed.
    pub concept_name: String,
    /// Existing body text (empty string when `gap_kind == MissingConcept`).
    /// For `ThinConceptBody` this is the current page body so the model can
    /// decide whether to extend or rewrite.
    pub existing_body: String,
    /// Short snippets from local sources that mention the concept — used to
    /// disambiguate which variant of the term the KB cares about. The
    /// adapter treats these as opaque text; the caller truncates before
    /// passing them in.
    pub local_snippets: Vec<CandidateSourceSnippet>,
}

/// One web source cited by an imputed fix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImputedWebSource {
    /// The page URL the model browsed.
    pub url: String,
    /// Page title as displayed by the source.
    pub title: String,
    /// Short (<= 20 words) reason why this source supports the definition.
    pub note: String,
}

/// Response with the drafted definition + cited web sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImputeGapResponse {
    /// The imputed 2-4 sentence definition (general-scope). First sentence
    /// must start with the concept name followed by a linking verb; the
    /// prompt enforces this but callers should still tolerate minor drift.
    pub definition: String,
    /// Web sources the model used (between 1 and 3).
    #[serde(default)]
    pub sources: Vec<ImputedWebSource>,
    /// Model's self-assessed confidence. Surfaced in `kb review show` so
    /// reviewers know how much to trust the draft.
    pub confidence: String,
    /// 1-2 sentence note on which variant of the concept was picked and why.
    #[serde(default)]
    pub rationale: String,
}

/// Parse an [`ImputeGapResponse`] from LLM text output.
///
/// Accepts raw JSON or a fenced code block. Unknown fields are ignored so
/// model drift doesn't break parsing.
///
/// # Errors
///
/// Returns [`LlmAdapterError::Parse`] when the text is empty, not valid JSON,
/// or does not match the expected schema.
pub fn parse_impute_gap_json(text: &str) -> Result<ImputeGapResponse, LlmAdapterError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(LlmAdapterError::Parse(
            "impute_gap response was empty".to_string(),
        ));
    }

    let json_text = trimmed
        .strip_prefix("```")
        .and_then(|body| body.split_once('\n').map(|(_, rest)| rest))
        .and_then(|body| body.rsplit_once("```").map(|(json, _)| json.trim()))
        .unwrap_or(trimmed);

    serde_json::from_str(json_text).map_err(|err| {
        LlmAdapterError::Parse(format!(
            "impute_gap response had invalid shape: {err}"
        ))
    })
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
        LlmAdapterError::Parse(format!(
            "extract_concepts response was not valid JSON: {err}"
        ))
    })?;

    if value.is_array() {
        let concepts = serde_json::from_value(value).map_err(|err| {
            LlmAdapterError::Parse(format!(
                "extract_concepts response had invalid candidate list: {err}"
            ))
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

    /// Generate a 1-3 sentence general-scope body for a canonical concept.
    ///
    /// This is the bn-1w5 two-step body synthesis call: after
    /// [`merge_concept_candidates`](Self::merge_concept_candidates) picks the
    /// canonical name and aliases, the merge pipeline calls this on each
    /// confident group so the body is generated in isolation with a minimal,
    /// strict prompt that forces the "what do the aliases have in common?"
    /// framing.
    ///
    /// The default implementation returns [`LlmAdapterError::Other`] so that
    /// adapters without runner support (e.g. test doubles that only stub
    /// `merge_concept_candidates`) cause the caller to fall back to the
    /// merge-step `definition_hint` instead of hard-failing the pass.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or the adapter does not implement this call.
    fn generate_concept_body(
        &self,
        _request: GenerateConceptBodyRequest,
    ) -> Result<(GenerateConceptBodyResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "generate_concept_body is not implemented by this adapter".to_string(),
        ))
    }

    /// Detect whether a set of source quotes about a single concept make
    /// contradictory claims.
    ///
    /// The default implementation returns [`LlmAdapterError::Other`] so that
    /// adapters that do not yet implement the contradiction prompt cause the
    /// caller to skip the check rather than hard-fail. See
    /// `crates/kb-llm/prompts/detect_contradictions.md` for the prompt
    /// contract.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or the adapter does not implement this call.
    fn detect_contradictions(
        &self,
        _request: DetectContradictionsRequest,
    ) -> Result<(DetectContradictionsResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "detect_contradictions is not implemented by this adapter".to_string(),
        ))
    }

    /// Draft a canonical concept entry from a lint-flagged candidate.
    ///
    /// bn-lw06: called from `kb review approve concept-candidate:<name>`. The
    /// CLI passes the surface term plus a short snippet from each source
    /// document that mentions it; the model returns the canonical name,
    /// aliases, category, and a general-scope 2-3 sentence definition.
    ///
    /// Default implementation returns [`LlmAdapterError::Other`] so that
    /// adapters without runner support (test doubles) cause the caller to
    /// surface a clear error rather than crashing.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or the adapter does not implement this call.
    fn generate_concept_from_candidate(
        &self,
        _request: GenerateConceptFromCandidateRequest,
    ) -> Result<(GenerateConceptFromCandidateResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "generate_concept_from_candidate is not implemented by this adapter".to_string(),
        ))
    }

    /// Draft fill-in content for a lint gap using a web-search-capable
    /// agent. bn-xt4o — called from `kb lint --impute`. The adapter MUST
    /// turn on the runner's web-search tool access when calling the model;
    /// this is the one kb call that expects external browsing.
    ///
    /// Default implementation returns [`LlmAdapterError::Other`] so that
    /// test-double adapters and adapters that can't grant web access (e.g.
    /// a future air-gapped backend) cause `--impute` to skip the gap with
    /// a clear error rather than crash.
    ///
    /// # Errors
    ///
    /// Returns `LlmAdapterError` if the backend cannot be reached, times out,
    /// fails to parse the response, or the adapter does not implement this call.
    fn impute_gap(
        &self,
        _request: ImputeGapRequest,
    ) -> Result<(ImputeGapResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "impute_gap is not implemented by this adapter".to_string(),
        ))
    }


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
            parsed.concepts[0].source_anchors[0]
                .heading_anchor
                .as_deref(),
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

    #[test]
    fn parse_merge_response_reads_optional_category() {
        let response = parse_merge_concept_candidates_json(
            r#"{
                "groups": [
                    {
                        "canonical_name": "Tokio runtime",
                        "aliases": ["tokio"],
                        "category": "async",
                        "members": [],
                        "confident": true,
                        "rationale": null
                    },
                    {
                        "canonical_name": "Ambient concept",
                        "aliases": [],
                        "category": null,
                        "members": [],
                        "confident": true,
                        "rationale": null
                    }
                ]
            }"#,
        )
        .expect("parse merge response with categories");

        assert_eq!(response.groups.len(), 2);
        assert_eq!(response.groups[0].category.as_deref(), Some("async"));
        assert_eq!(response.groups[1].category, None);
    }

    #[test]
    fn parse_merge_response_tolerates_missing_category_key() {
        // Old responses without a `category` field must still parse — the
        // field defaults to None. This protects backwards compatibility
        // with fixtures recorded before the field was added.
        let response = parse_merge_concept_candidates_json(
            r#"{
                "groups": [
                    {
                        "canonical_name": "Borrow checker",
                        "aliases": [],
                        "members": [],
                        "confident": true,
                        "rationale": null
                    }
                ]
            }"#,
        )
        .expect("parse legacy merge response");
        assert_eq!(response.groups[0].category, None);
    }
}
