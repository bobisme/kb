use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_yaml::Mapping;
use serde_yaml::Value;

use kb_core::fs::atomic_write;
use kb_core::{Artifact, Question};
use kb_llm::ProvenanceRecord;

use crate::ArtifactResult;
use crate::lexical::RetrievalPlan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSidecar {
    pub question_id: String,
    pub artifact_id: String,
    pub retrieval_plan_path: String,
    pub provenance: Option<ProvenanceRecord>,
    pub source_document_ids: Vec<String>,
    pub valid_citations: Vec<u32>,
    pub invalid_citations: Vec<u32>,
    pub has_uncertainty_banner: bool,
    /// ID of the [`kb_core::BuildRecord`] that produced this artifact, when the
    /// LLM pass succeeded. `None` for placeholder artifacts written when the
    /// backend was unavailable.
    pub build_record_id: Option<String>,
}

pub struct WriteArtifactInput<'a> {
    pub root: &'a Path,
    pub question: &'a Question,
    pub artifact: &'a Artifact,
    pub retrieval_plan: &'a RetrievalPlan,
    pub artifact_result: Option<&'a ArtifactResult>,
    pub provenance: Option<&'a ProvenanceRecord>,
    pub artifact_body: &'a str,
    /// [`kb_core::BuildRecord`] identifier for this generation, when one was
    /// emitted. The writer mirrors it into the answer frontmatter so `kb inspect`
    /// can walk the provenance chain from the artifact back to its record.
    pub build_record_id: Option<&'a str>,
}

pub struct WriteArtifactOutput {
    pub question_path: PathBuf,
    pub answer_path: PathBuf,
    pub metadata_path: PathBuf,
    pub retrieval_plan_path: PathBuf,
}

#[allow(clippy::missing_errors_doc)]
pub fn write_artifact(input: &WriteArtifactInput<'_>) -> std::io::Result<WriteArtifactOutput> {
    let question_id = &input.question.metadata.id;
    let base_dir = PathBuf::from("outputs/questions").join(question_id);

    let question_rel = base_dir.join("question.json");
    let answer_rel = base_dir.join("answer.md");
    let metadata_rel = base_dir.join("metadata.json");
    let plan_rel = base_dir.join("retrieval_plan.json");

    let plan_json = serde_json::to_string_pretty(input.retrieval_plan).map_err(json_io_err)?;
    atomic_write(input.root.join(&plan_rel), plan_json.as_bytes())?;

    let question_json =
        serde_json::to_string_pretty(input.question).map_err(json_io_err)?;
    atomic_write(input.root.join(&question_rel), question_json.as_bytes())?;

    let answer_md = render_answer_frontmatter(input).map_err(yaml_io_err)?;
    atomic_write(input.root.join(&answer_rel), answer_md.as_bytes())?;

    let source_doc_ids: Vec<String> = input
        .retrieval_plan
        .candidates
        .iter()
        .map(|c| c.id.clone())
        .collect();

    let (valid_citations, invalid_citations, has_uncertainty_banner) =
        input.artifact_result.map_or_else(
            || (Vec::new(), Vec::new(), false),
            |r| {
                (
                    r.valid_citations.clone(),
                    r.invalid_citations.clone(),
                    r.has_uncertainty_banner,
                )
            },
        );

    let sidecar = ArtifactSidecar {
        question_id: question_id.clone(),
        artifact_id: input.artifact.metadata.id.clone(),
        retrieval_plan_path: plan_rel.to_string_lossy().into_owned(),
        provenance: input.provenance.cloned(),
        source_document_ids: source_doc_ids,
        valid_citations,
        invalid_citations,
        has_uncertainty_banner,
        build_record_id: input.build_record_id.map(str::to_string),
    };

    let sidecar_json = serde_json::to_string_pretty(&sidecar).map_err(json_io_err)?;
    atomic_write(input.root.join(&metadata_rel), sidecar_json.as_bytes())?;

    Ok(WriteArtifactOutput {
        question_path: question_rel,
        answer_path: answer_rel,
        metadata_path: metadata_rel,
        retrieval_plan_path: plan_rel,
    })
}

fn render_answer_frontmatter(input: &WriteArtifactInput<'_>) -> Result<String, serde_yaml::Error> {
    let mut fm = Mapping::new();
    fm.insert(
        Value::String("id".into()),
        Value::String(input.artifact.metadata.id.clone()),
    );
    fm.insert(
        Value::String("type".into()),
        Value::String("question_answer".into()),
    );
    fm.insert(
        Value::String("question_id".into()),
        Value::String(input.question.metadata.id.clone()),
    );
    fm.insert(
        Value::String("generated_at".into()),
        Value::Number(input.artifact.metadata.created_at_millis.into()),
    );
    if let Some(prov) = input.provenance {
        fm.insert(
            Value::String("generated_by".into()),
            Value::String(format!("{}/{}", prov.harness, prov.model)),
        );
    }
    if let Some(tool) = &input.artifact.metadata.tool_version {
        fm.insert(
            Value::String("tool_version".into()),
            Value::String(tool.clone()),
        );
    }
    if let Some(build_id) = input.build_record_id {
        fm.insert(
            Value::String("build_record_id".into()),
            Value::String(build_id.to_string()),
        );
    }

    let source_ids: Vec<Value> = input
        .retrieval_plan
        .candidates
        .iter()
        .map(|c| Value::String(c.id.clone()))
        .collect();
    fm.insert(
        Value::String("source_document_ids".into()),
        Value::Sequence(source_ids),
    );

    let source_hashes: Vec<Value> = input
        .question
        .metadata
        .source_hashes
        .iter()
        .map(|h| Value::String(h.clone()))
        .collect();
    if !source_hashes.is_empty() {
        fm.insert(
            Value::String("source_revision_ids".into()),
            Value::Sequence(source_hashes),
        );
    }

    fm.insert(
        Value::String("requested_format".into()),
        Value::String(input.question.requested_format.clone()),
    );

    if let Some(model) = &input.artifact.metadata.model_version {
        fm.insert(
            Value::String("model".into()),
            Value::String(model.clone()),
        );
    }
    if input.question.requested_format == "marp" {
        fm.insert(Value::String("marp".into()), Value::Bool(true));
    }

    let yaml = serde_yaml::to_string(&fm)?;
    Ok(format!("---\n{yaml}---\n\n{}", input.artifact_body))
}

fn json_io_err(e: serde_json::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
}

fn yaml_io_err(e: serde_yaml::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use kb_core::{ArtifactKind, EntityMetadata, QuestionContext, Status};
    use tempfile::TempDir;

    fn sample_question(id: &str) -> Question {
        Question {
            metadata: EntityMetadata {
                id: id.to_string(),
                created_at_millis: 1_000_000,
                updated_at_millis: 1_000_000,
                source_hashes: Vec::new(),
                model_version: Some("gpt-5.4".into()),
                tool_version: Some("kb/0.1.0".into()),
                prompt_template_hash: None,
                dependencies: Vec::new(),
                output_paths: Vec::new(),
                status: Status::Fresh,
            },
            raw_query: "What is Rust?".into(),
            requested_format: "md".into(),
            requesting_context: QuestionContext::ProjectKb,
            retrieval_plan: "outputs/questions/q1/retrieval_plan.json".into(),
            token_budget: Some(4096),
        }
    }

    fn sample_artifact(question_id: &str) -> Artifact {
        Artifact {
            metadata: EntityMetadata {
                id: format!("artifact-{question_id}"),
                created_at_millis: 1_000_000,
                updated_at_millis: 1_000_000,
                source_hashes: Vec::new(),
                model_version: Some("gpt-5.4".into()),
                tool_version: Some("kb/0.1.0".into()),
                prompt_template_hash: None,
                dependencies: vec![question_id.to_string()],
                output_paths: Vec::new(),
                status: Status::Fresh,
            },
            question_id: question_id.to_string(),
            artifact_kind: ArtifactKind::AnswerNote,
            format: "md".into(),
            output_path: PathBuf::from("outputs/questions/q1/answer.md"),
        }
    }

    fn sample_plan() -> RetrievalPlan {
        RetrievalPlan {
            query: "What is Rust?".into(),
            token_budget: 4096,
            estimated_tokens: 500,
            candidates: vec![crate::RetrievalCandidate {
                id: "wiki/sources/rust-overview.md".into(),
                title: "Rust Overview".into(),
                score: 10,
                estimated_tokens: 500,
                reasons: vec!["title match".into()],
            }],
        }
    }

    #[test]
    fn writes_all_files_atomically() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let question = sample_question("q1");
        let artifact = sample_artifact("q1");
        let plan = sample_plan();

        let result = ArtifactResult {
            body: "Rust is a systems programming language [1].".into(),
            valid_citations: vec![1],
            invalid_citations: Vec::new(),
            has_uncertainty_banner: false,
        };

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: Some(&result),
            provenance: None,
            artifact_body: &result.body,
            build_record_id: None,
        })
        .unwrap();

        assert!(root.join(&output.question_path).exists());
        assert!(root.join(&output.answer_path).exists());
        assert!(root.join(&output.metadata_path).exists());
        assert!(root.join(&output.retrieval_plan_path).exists());
    }

    #[test]
    fn answer_md_contains_frontmatter_fields() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let question = sample_question("q2");
        let artifact = sample_artifact("q2");
        let plan = sample_plan();

        let body = "Answer text here.";
        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: body,
            build_record_id: None,
        })
        .unwrap();

        let content = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("type: question_answer"));
        assert!(content.contains("question_id: q2"));
        assert!(content.contains("generated_at:"));
        assert!(content.contains("source_document_ids:"));
        assert!(content.contains("Answer text here."));
    }

    #[test]
    fn metadata_json_contains_provenance() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let question = sample_question("q3");
        let artifact = sample_artifact("q3");
        let plan = sample_plan();

        let prov = ProvenanceRecord {
            harness: "opencode".into(),
            harness_version: None,
            model: "gpt-5.4".into(),
            prompt_template_name: "ask.md".into(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::Hash::from([0u8; 32]),
            started_at: 1000,
            ended_at: 2000,
            latency_ms: 1000,
            retries: 0,
            tokens: None,
            cost_estimate: None,
        };

        let result = ArtifactResult {
            body: "Answer.".into(),
            valid_citations: vec![1],
            invalid_citations: vec![99],
            has_uncertainty_banner: false,
        };

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: Some(&result),
            provenance: Some(&prov),
            artifact_body: &result.body,
            build_record_id: None,
        })
        .unwrap();

        let sidecar_str = std::fs::read_to_string(root.join(&output.metadata_path)).unwrap();
        let sidecar: ArtifactSidecar = serde_json::from_str(&sidecar_str).unwrap();
        assert_eq!(sidecar.question_id, "q3");
        assert!(sidecar.provenance.is_some());
        assert_eq!(sidecar.valid_citations, vec![1]);
        assert_eq!(sidecar.invalid_citations, vec![99]);
        assert_eq!(sidecar.source_document_ids, vec!["wiki/sources/rust-overview.md"]);
    }

    #[test]
    fn no_partial_files_on_success() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let question = sample_question("q4");
        let artifact = sample_artifact("q4");
        let plan = sample_plan();

        write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "body",
            build_record_id: None,
        })
        .unwrap();

        let dir = root.join("outputs/questions/q4");
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .collect();

        for entry in &entries {
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(
                !name.contains(".tmp."),
                "temporary file left behind: {name}"
            );
        }
    }

    #[test]
    fn frontmatter_includes_generated_by_with_provenance() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let question = sample_question("q5");
        let artifact = sample_artifact("q5");
        let plan = sample_plan();

        let prov = ProvenanceRecord {
            harness: "claude".into(),
            harness_version: None,
            model: "claude-3-5-sonnet".into(),
            prompt_template_name: "ask.md".into(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::Hash::from([0u8; 32]),
            started_at: 1000,
            ended_at: 2000,
            latency_ms: 1000,
            retries: 0,
            tokens: None,
            cost_estimate: None,
        };

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: Some(&prov),
            artifact_body: "body",
            build_record_id: None,
        })
        .unwrap();

        let content = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(content.contains("generated_by: claude/claude-3-5-sonnet"));
    }

    #[test]
    fn frontmatter_and_sidecar_include_build_record_id() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let question = sample_question("q6");
        let artifact = sample_artifact("q6");
        let plan = sample_plan();

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "body",
            build_record_id: Some("build:ask:q6"),
        })
        .unwrap();

        let content = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(
            content.contains("build_record_id: build:ask:q6"),
            "frontmatter should include build_record_id"
        );

        let sidecar_str = std::fs::read_to_string(root.join(&output.metadata_path)).unwrap();
        let sidecar: ArtifactSidecar = serde_json::from_str(&sidecar_str).unwrap();
        assert_eq!(sidecar.build_record_id.as_deref(), Some("build:ask:q6"));
    }
}
