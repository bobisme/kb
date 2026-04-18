use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kb_core::fs::atomic_write;
use kb_core::{BuildRecord, EntityMetadata, NormalizedDocument, Status, hash_many};
use kb_llm::{
    ConceptCandidate, ExtractConceptsRequest, LlmAdapter, LlmAdapterError, ProvenanceRecord,
};

/// Default directory used by the concept extraction pass.
pub const CONCEPT_CANDIDATES_DIR: &str = "state/concept_candidates";

/// Output emitted by the concept extraction pass for one normalized document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptExtractionArtifact {
    /// Candidate concepts extracted from the source.
    pub candidates: Vec<ConceptCandidate>,
    /// Destination JSON file for the candidate list.
    pub output_path: PathBuf,
    /// Pretty JSON payload to persist for downstream merge passes.
    pub output_json: String,
    /// Build metadata for this pass execution.
    pub build_record: BuildRecord,
}

/// Errors produced by the concept extraction pass.
#[derive(Debug, thiserror::Error)]
pub enum ConceptExtractionError {
    #[error("llm extract_concepts failed: {0}")]
    Llm(#[from] LlmAdapterError),
    #[error("failed to serialize concept candidates: {0}")]
    Serialize(String),
    #[error("failed to read system time: {0}")]
    Time(String),
}

/// Compute the default artifact path for a source's concept candidates.
#[must_use]
pub fn concept_candidates_path(root: &Path, source_id: &str) -> PathBuf {
    root.join(CONCEPT_CANDIDATES_DIR)
        .join(format!("{source_id}.json"))
}

/// Persist the JSON payload emitted by [`run_concept_extraction_pass`].
///
/// # Errors
///
/// Returns an error when the target file cannot be written atomically.
pub fn persist_concept_candidates(output_path: &Path, output_json: &str) -> Result<()> {
    atomic_write(output_path, output_json.as_bytes())
        .with_context(|| format!("write concept candidates {}", output_path.display()))
}

/// Generate a concept-candidate artifact for a normalized document.
///
/// The pass delegates candidate extraction to `LlmAdapter::extract_concepts`, combines the
/// normalized body with the already-generated source summary, serializes the candidate list
/// to JSON for downstream merge passes, and emits a build record for stale detection.
///
/// # Errors
///
/// Returns [`ConceptExtractionError`] when the adapter call fails, the candidate payload cannot
/// be serialized, or the system clock cannot be converted into build-record timestamps.
pub fn run_concept_extraction_pass<A: LlmAdapter>(
    adapter: &A,
    document: &NormalizedDocument,
    title: &str,
    summary: Option<&str>,
    max_concepts: Option<usize>,
    output_path: impl AsRef<Path>,
) -> Result<ConceptExtractionArtifact, ConceptExtractionError> {
    let output_path = output_path.as_ref().to_path_buf();
    let request = ExtractConceptsRequest {
        title: title.to_string(),
        body: document.canonical_text.clone(),
        summary: summary.map(str::to_string),
        max_concepts,
    };

    let (response, provenance) = adapter.extract_concepts(request)?;
    let output_json = serde_json::to_string_pretty(&response.concepts)
        .map_err(|err| ConceptExtractionError::Serialize(err.to_string()))?;
    let build_record = build_record_for_concept_candidates(
        document,
        &output_path,
        &response.concepts,
        &output_json,
        &provenance,
    )?;

    Ok(ConceptExtractionArtifact {
        candidates: response.concepts,
        output_path,
        output_json,
        build_record,
    })
}

fn build_record_for_concept_candidates(
    document: &NormalizedDocument,
    output_path: &Path,
    candidates: &[ConceptCandidate],
    output_json: &str,
    provenance: &ProvenanceRecord,
) -> Result<BuildRecord, ConceptExtractionError> {
    let now = unix_time_ms()?;
    let prompt_template_hash = provenance.prompt_template_hash.to_hex();
    let prompt_render_hash = provenance.prompt_render_hash.to_hex();
    let candidate_names = candidates
        .iter()
        .map(|candidate| candidate.name.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let manifest_hash = hash_many(&[
        document.metadata.id.as_bytes(),
        b"\0",
        document.source_revision_id.as_bytes(),
        b"\0",
        output_json.as_bytes(),
        b"\0",
        candidate_names.as_bytes(),
        b"\0",
        prompt_render_hash.as_bytes(),
    ])
    .to_hex();

    Ok(BuildRecord {
        pass_name: "extract_concepts".to_string(),
        metadata: EntityMetadata {
            id: format!("build:extract-concepts:{}", document.metadata.id),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![
                document.metadata.id.clone(),
                document.source_revision_id.clone(),
                prompt_render_hash,
            ],
            model_version: Some(provenance.model.clone()),
            tool_version: Some(format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: Some(prompt_template_hash),
            dependencies: vec![document.metadata.id.clone(), document.source_revision_id.clone()],
            output_paths: vec![output_path.to_path_buf()],
            status: Status::Fresh,
        },
        input_ids: vec![document.metadata.id.clone(), document.source_revision_id.clone()],
        output_ids: vec![output_path.display().to_string()],
        manifest_hash,
    })
}

fn unix_time_ms() -> Result<u64, ConceptExtractionError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| ConceptExtractionError::Time(err.to_string()))?;
    let millis = duration
        .as_millis()
        .try_into()
        .context("system time exceeded u64 millisecond range")
        .map_err(|err| ConceptExtractionError::Time(err.to_string()))?;
    Ok(millis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use kb_core::{EntityMetadata, Hash, Status};
    use kb_llm::{
        AnswerQuestionRequest, AnswerQuestionResponse, ConceptCandidate, ExtractConceptsResponse,
        GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter, MergeConceptCandidatesRequest,
        MergeConceptCandidatesResponse, RunHealthCheckRequest, RunHealthCheckResponse,
        SourceAnchor, SummarizeDocumentRequest, SummarizeDocumentResponse, TokenUsage,
    };
    use tempfile::tempdir;

    #[derive(Debug)]
    struct FakeAdapter {
        response: Vec<ConceptCandidate>,
        provenance: ProvenanceRecord,
        expected_title: String,
        expected_body: String,
        expected_summary: Option<String>,
        expected_max_concepts: Option<usize>,
    }

    impl LlmAdapter for FakeAdapter {
        fn summarize_document(
            &self,
            _request: SummarizeDocumentRequest,
        ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept extraction test")
        }

        fn extract_concepts(
            &self,
            request: ExtractConceptsRequest,
        ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
            assert_eq!(request.title, self.expected_title);
            assert_eq!(request.body, self.expected_body);
            assert_eq!(request.summary, self.expected_summary);
            assert_eq!(request.max_concepts, self.expected_max_concepts);
            Ok((
                ExtractConceptsResponse {
                    concepts: self.response.clone(),
                },
                self.provenance.clone(),
            ))
        }

        fn merge_concept_candidates(
            &self,
            _request: MergeConceptCandidatesRequest,
        ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept extraction test")
        }

        fn answer_question(
            &self,
            _request: AnswerQuestionRequest,
        ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept extraction test")
        }

        fn generate_slides(
            &self,
            _request: GenerateSlidesRequest,
        ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept extraction test")
        }

        fn run_health_check(
            &self,
            _request: RunHealthCheckRequest,
        ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
            unreachable!("unused in concept extraction test")
        }
    }

    fn normalized_document() -> NormalizedDocument {
        NormalizedDocument {
            metadata: EntityMetadata {
                id: "normalized-doc-1".to_string(),
                created_at_millis: 1,
                updated_at_millis: 1,
                source_hashes: vec!["source-hash-1".to_string()],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: vec![],
                output_paths: vec![],
                status: Status::Fresh,
            },
            source_revision_id: "source-revision-1".to_string(),
            canonical_text: "# Ownership\n\nThe borrow checker validates references.\n".to_string(),
            normalized_assets: vec![],
            heading_ids: vec!["ownership".to_string()],
        }
    }

    fn provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            harness: "opencode".to_string(),
            harness_version: None,
            model: "openai/gpt-5.4".to_string(),
            prompt_template_name: "extract_concepts.md".to_string(),
            prompt_template_hash: Hash::from([7u8; 32]),
            prompt_render_hash: Hash::from([8u8; 32]),
            started_at: 10,
            ended_at: 20,
            latency_ms: 10,
            retries: 0,
            tokens: Some(TokenUsage {
                prompt_tokens: 42,
                completion_tokens: 12,
            }),
            cost_estimate: None,
        }
    }

    #[test]
    fn concept_extraction_pass_emits_candidate_json_and_build_record() {
        let document = normalized_document();
        let adapter = FakeAdapter {
            response: vec![ConceptCandidate {
                name: "Borrow checker".to_string(),
                aliases: vec!["borrowck".to_string()],
                definition_hint: Some("Rust's reference safety analysis.".to_string()),
                source_anchors: vec![SourceAnchor {
                    heading_anchor: Some("ownership".to_string()),
                    quote: Some("The borrow checker validates references.".to_string()),
                }],
            }],
            provenance: provenance(),
            expected_title: "Ownership Notes".to_string(),
            expected_body: document.canonical_text.clone(),
            expected_summary: Some("Rust ownership overview".to_string()),
            expected_max_concepts: Some(5),
        };
        let output_path = PathBuf::from("state/concept_candidates/source-1.json");

        let artifact = run_concept_extraction_pass(
            &adapter,
            &document,
            "Ownership Notes",
            Some("Rust ownership overview"),
            Some(5),
            &output_path,
        )
        .expect("run concept extraction pass");

        assert_eq!(artifact.candidates.len(), 1);
        assert!(artifact.output_json.contains("Borrow checker"));
        assert_eq!(artifact.output_path, output_path);
        assert_eq!(artifact.build_record.pass_name, "extract_concepts");
        assert_eq!(artifact.build_record.output_ids, vec!["state/concept_candidates/source-1.json"]);
        let expected_prompt_hash = Hash::from([7u8; 32]).to_hex();
        assert_eq!(
            artifact.build_record.metadata.prompt_template_hash.as_deref(),
            Some(expected_prompt_hash.as_str())
        );
    }

    #[test]
    fn persist_concept_candidates_writes_json_file() {
        let dir = tempdir().expect("tempdir");
        let output_path = concept_candidates_path(dir.path(), "source-1");
        persist_concept_candidates(&output_path, "[]").expect("persist concept candidates");

        let contents = std::fs::read_to_string(&output_path).expect("read candidate file");
        assert_eq!(contents, "[]");
    }
}
