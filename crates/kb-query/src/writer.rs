use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_yaml::Mapping;
use serde_yaml::Value;

use kb_core::fs::atomic_write;
use kb_core::frontmatter::read_frontmatter;
use kb_core::{Artifact, Question, extract_managed_regions};
use kb_llm::ProvenanceRecord;

use crate::ArtifactResult;
use crate::lexical::{RetrievalCandidate, RetrievalPlan};

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

    // The answer file name tracks the requested format so `--format=json` lands
    // on disk as `answer.json`, not `answer.md`. Marp still renders as
    // markdown (Marp IS markdown with a frontmatter flag), so it keeps `.md`.
    let answer_file_name = match input.question.requested_format.as_str() {
        "json" => "answer.json",
        _ => "answer.md",
    };

    let question_rel = base_dir.join("question.json");
    let answer_rel = base_dir.join(answer_file_name);
    let metadata_rel = base_dir.join("metadata.json");
    let plan_rel = base_dir.join("retrieval_plan.json");

    let plan_json = serde_json::to_string_pretty(input.retrieval_plan).map_err(json_io_err)?;
    atomic_write(input.root.join(&plan_rel), plan_json.as_bytes())?;

    let question_json =
        serde_json::to_string_pretty(input.question).map_err(json_io_err)?;
    atomic_write(input.root.join(&question_rel), question_json.as_bytes())?;

    let source_doc_ids = resolve_source_document_ids(input.root, &input.retrieval_plan.candidates);

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

    if input.question.requested_format == "json" {
        let answer_json = render_answer_json(input, &source_doc_ids, &valid_citations)
            .map_err(json_io_err)?;
        atomic_write(input.root.join(&answer_rel), answer_json.as_bytes())?;
    } else {
        let answer_md = render_answer_frontmatter(input).map_err(yaml_io_err)?;
        atomic_write(input.root.join(&answer_rel), answer_md.as_bytes())?;
    }

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

/// Serializes the answer as a structured JSON document.
///
/// Mirrors the metadata captured by the markdown answer's YAML frontmatter —
/// id, question linkage, generation timestamp, model, requested format,
/// source document IDs, retrieval candidates, citations — and wraps the
/// existing markdown body verbatim as the `body` field. Downstream tools can
/// parse this without re-rendering the answer.
fn render_answer_json(
    input: &WriteArtifactInput<'_>,
    source_document_ids: &[String],
    valid_citations: &[u32],
) -> Result<String, serde_json::Error> {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "id".into(),
        serde_json::Value::String(input.artifact.metadata.id.clone()),
    );
    obj.insert(
        "type".into(),
        serde_json::Value::String("question_answer".into()),
    );
    obj.insert(
        "question_id".into(),
        serde_json::Value::String(input.question.metadata.id.clone()),
    );
    obj.insert(
        "generated_at".into(),
        serde_json::Value::Number(input.artifact.metadata.created_at_millis.into()),
    );
    if let Some(prov) = input.provenance {
        obj.insert(
            "generated_by".into(),
            serde_json::Value::String(format!("{}/{}", prov.harness, prov.model)),
        );
    }
    if let Some(tool) = &input.artifact.metadata.tool_version {
        obj.insert(
            "tool_version".into(),
            serde_json::Value::String(tool.clone()),
        );
    }
    if let Some(build_id) = input.build_record_id {
        obj.insert(
            "build_record_id".into(),
            serde_json::Value::String(build_id.to_string()),
        );
    }
    if let Some(model) = &input.artifact.metadata.model_version {
        obj.insert("model".into(), serde_json::Value::String(model.clone()));
    }
    obj.insert(
        "requested_format".into(),
        serde_json::Value::String(input.question.requested_format.clone()),
    );
    obj.insert(
        "source_document_ids".into(),
        serde_json::Value::Array(
            source_document_ids
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
    );
    obj.insert(
        "retrieval_candidates".into(),
        serde_json::to_value(&input.retrieval_plan.candidates)?,
    );
    // Citations: surface the indices the LLM actually cited, paired with the
    // candidate path/kind. Entries are derived from the retrieval plan by
    // position (1-indexed) to match how `[n]` references resolve in the body.
    let citations: Vec<serde_json::Value> = valid_citations
        .iter()
        .filter_map(|idx| {
            let i = *idx as usize;
            if i == 0 || i > input.retrieval_plan.candidates.len() {
                return None;
            }
            let cand = &input.retrieval_plan.candidates[i - 1];
            let kind = if cand.id.starts_with("wiki/concepts/") {
                "concept"
            } else if cand.id.starts_with("wiki/sources/") {
                "source"
            } else {
                "fulldocument"
            };
            let mut c = serde_json::Map::new();
            c.insert("index".into(), serde_json::Value::Number((*idx).into()));
            c.insert("path".into(), serde_json::Value::String(cand.id.clone()));
            c.insert("kind".into(), serde_json::Value::String(kind.into()));
            Some(serde_json::Value::Object(c))
        })
        .collect();
    obj.insert("citations".into(), serde_json::Value::Array(citations));

    let source_hashes: Vec<serde_json::Value> = input
        .question
        .metadata
        .source_hashes
        .iter()
        .map(|h| serde_json::Value::String(h.clone()))
        .collect();
    if !source_hashes.is_empty() {
        obj.insert(
            "source_revision_ids".into(),
            serde_json::Value::Array(source_hashes),
        );
    }

    obj.insert(
        "body".into(),
        serde_json::Value::String(input.artifact_body.to_string()),
    );

    serde_json::to_string_pretty(&serde_json::Value::Object(obj))
}

/// Resolves retrieval candidate paths to the underlying `source_document_id`
/// values recorded in each page's frontmatter.
///
/// The retrieval plan identifies candidates by their wiki page path (e.g.
/// `wiki/sources/foo.md` or `wiki/concepts/bar.md`). Downstream consumers —
/// particularly `kb lint orphans` — expect `source_document_ids` to contain
/// real source-document identifiers that map to `normalized/<id>/`.
///
/// For each candidate this function:
/// - Reads the candidate's frontmatter.
/// - If the page is a source wiki (`source_document_id` present as string or
///   `source_document_ids` as list), returns those IDs.
/// - If the page is a concept wiki, walks its `backlinks` managed region,
///   parses each `[[wiki/sources/<slug>]]` link, and recursively extracts the
///   referenced source page's `source_document_id`.
///
/// Unreadable or unresolvable candidates are silently skipped — this mirrors
/// existing lint behaviour, which already reports missing sources.
/// Results are deduplicated and sorted for stable output.
fn resolve_source_document_ids(root: &Path, candidates: &[RetrievalCandidate]) -> Vec<String> {
    let mut resolved: BTreeSet<String> = BTreeSet::new();
    for candidate in candidates {
        collect_source_document_ids_for_path(root, &candidate.id, &mut resolved, 0);
    }
    resolved.into_iter().collect()
}

const MAX_RESOLVE_DEPTH: usize = 2;

fn collect_source_document_ids_for_path(
    root: &Path,
    rel_path: &str,
    out: &mut BTreeSet<String>,
    depth: usize,
) {
    if depth > MAX_RESOLVE_DEPTH {
        return;
    }
    let abs_path = root.join(rel_path);
    let Ok((frontmatter, body)) = read_frontmatter(&abs_path) else {
        return;
    };

    // Source pages store a single `source_document_id` string.
    if let Some(Value::String(id)) = frontmatter.get(Value::String("source_document_id".into())) {
        out.insert(id.clone());
    }

    // Some pages (including prior answer artifacts) carry a list.
    if let Some(Value::Sequence(seq)) =
        frontmatter.get(Value::String("source_document_ids".into()))
    {
        for item in seq {
            if let Value::String(s) = item {
                // Heuristic: only accept entries that aren't themselves wiki paths.
                // Paths would just re-introduce the bug this function is fixing.
                if !s.starts_with("wiki/") {
                    out.insert(s.clone());
                }
            }
        }
    }

    // Concept pages don't carry source_document_id directly — walk their
    // backlinks region to the source pages they were extracted from, and
    // recursively resolve those.
    if rel_path.starts_with("wiki/concepts/") {
        for linked in source_links_in_backlinks(&body) {
            collect_source_document_ids_for_path(root, &linked, out, depth + 1);
        }
    }
}

fn source_links_in_backlinks(body: &str) -> Vec<String> {
    let mut links = Vec::new();
    for region in extract_managed_regions(body) {
        if region.id != "backlinks" {
            continue;
        }
        let content = region.body(body);
        for line in content.lines() {
            // Look for `[[wiki/sources/<slug>]]` or `[[wiki/sources/<slug>|label]]`.
            let mut cursor = line;
            while let Some(open_idx) = cursor.find("[[") {
                let after_open = &cursor[open_idx + 2..];
                if let Some(close_idx) = after_open.find("]]") {
                    let inner = &after_open[..close_idx];
                    // Strip any `|label` suffix.
                    let target = inner.split('|').next().unwrap_or(inner).trim();
                    if let Some(stripped) = target.strip_prefix("wiki/sources/") {
                        // Normalize: the stored form omits `.md`; ensure we read the file.
                        let slug = stripped.trim_end_matches(".md");
                        if !slug.is_empty() {
                            links.push(format!("wiki/sources/{slug}.md"));
                        }
                    }
                    cursor = &after_open[close_idx + 2..];
                } else {
                    break;
                }
            }
        }
    }
    links
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

    let source_ids: Vec<Value> =
        resolve_source_document_ids(input.root, &input.retrieval_plan.candidates)
            .into_iter()
            .map(Value::String)
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

    /// Seed a source wiki page at `wiki/sources/<slug>.md` with frontmatter
    /// carrying the given `source_document_id`.
    fn seed_source_page(root: &Path, slug: &str, source_document_id: &str) {
        let path = root.join("wiki/sources").join(format!("{slug}.md"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let content = format!(
            "---\nid: {slug}\ntype: source\ntitle: {slug}\nsource_document_id: {source_document_id}\nsource_revision_id: rev-1\n---\n\n# {slug}\n",
        );
        std::fs::write(&path, content).unwrap();
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
        // Seed the source page so the resolver can recover the real doc id.
        seed_source_page(root, "rust-overview", "src-rust-overview");

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
        assert_eq!(sidecar.source_document_ids, vec!["src-rust-overview"]);
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

    /// Regression (bn-iiq): `--format=json` must produce `answer.json`, not
    /// `answer.md`, and the JSON must carry the documented envelope fields.
    #[test]
    fn json_format_writes_structured_answer_json() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Seed a source page so source_document_ids resolve to a real src-* id.
        seed_source_page(root, "rust-overview", "src-rust-overview");

        let mut question = sample_question("q-json");
        question.requested_format = "json".into();
        let mut artifact = sample_artifact("q-json");
        artifact.format = "json".into();
        artifact.artifact_kind = ArtifactKind::JsonSpec;
        artifact.output_path = PathBuf::from("outputs/questions/q-json/answer.json");
        let plan = sample_plan();

        let result = ArtifactResult {
            body: "Markdown body with a citation [1].".into(),
            valid_citations: vec![1],
            invalid_citations: vec![],
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
            build_record_id: Some("build:ask:q-json"),
        })
        .unwrap();

        // Filename must be answer.json, no stray answer.md.
        assert!(
            output.answer_path.to_string_lossy().ends_with("answer.json"),
            "answer_path should end with answer.json: {}",
            output.answer_path.display()
        );
        assert!(
            !root.join("outputs/questions/q-json/answer.md").exists(),
            "json format must not emit answer.md"
        );

        let raw = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
        assert_eq!(parsed["type"], "question_answer");
        assert_eq!(parsed["question_id"], "q-json");
        assert_eq!(parsed["requested_format"], "json");
        assert_eq!(parsed["body"], "Markdown body with a citation [1].");
        assert_eq!(parsed["build_record_id"], "build:ask:q-json");
        assert_eq!(parsed["model"], "gpt-5.4");
        assert_eq!(
            parsed["source_document_ids"],
            serde_json::json!(["src-rust-overview"])
        );
        let citations = parsed["citations"].as_array().unwrap();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0]["index"], 1);
        assert_eq!(citations[0]["path"], "wiki/sources/rust-overview.md");
        assert_eq!(citations[0]["kind"], "source");
        assert!(parsed["retrieval_candidates"].is_array());
    }

    /// Regression: retrieval candidates are wiki page paths. The promoted
    /// artifact's `source_document_ids` must contain the real `src-*` IDs
    /// pulled from each candidate's frontmatter — not the paths themselves —
    /// so `kb lint orphans` can resolve them against `normalized/<id>/`.
    #[test]
    fn source_document_ids_resolve_to_frontmatter_ids_not_paths() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // A plain source page with its own `source_document_id`.
        seed_source_page(root, "ownership-guide", "src-aaaa1111");

        // A second source page; the concept page will link to it via backlinks.
        seed_source_page(root, "borrow-checker", "src-bbbb2222");

        // A concept page whose backlinks region references the second source.
        // The resolver must walk through and pick up `src-bbbb2222`.
        let concept_path = root.join("wiki/concepts/ownership.md");
        std::fs::create_dir_all(concept_path.parent().unwrap()).unwrap();
        std::fs::write(
            &concept_path,
            "---\nid: c-ownership\nname: Ownership\n---\n\n# Ownership\n\n## Backlinks\n\
             <!-- kb:begin id=backlinks -->\n- [[wiki/sources/borrow-checker|Borrow Checker]]\n\
             <!-- kb:end id=backlinks -->\n",
        )
        .unwrap();

        let question = sample_question("q-resolve");
        let artifact = sample_artifact("q-resolve");
        let plan = RetrievalPlan {
            query: "ownership".into(),
            token_budget: 1024,
            estimated_tokens: 256,
            candidates: vec![
                crate::RetrievalCandidate {
                    id: "wiki/sources/ownership-guide.md".into(),
                    title: "Ownership Guide".into(),
                    score: 10,
                    estimated_tokens: 128,
                    reasons: vec![],
                },
                crate::RetrievalCandidate {
                    id: "wiki/concepts/ownership.md".into(),
                    title: "Ownership".into(),
                    score: 8,
                    estimated_tokens: 128,
                    reasons: vec![],
                },
            ],
        };

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "Body.",
            build_record_id: None,
        })
        .unwrap();

        // Sidecar must carry only real src-* identifiers, deduplicated + sorted.
        let sidecar_str = std::fs::read_to_string(root.join(&output.metadata_path)).unwrap();
        let sidecar: ArtifactSidecar = serde_json::from_str(&sidecar_str).unwrap();
        assert_eq!(
            sidecar.source_document_ids,
            vec!["src-aaaa1111".to_string(), "src-bbbb2222".to_string()],
            "expected resolved src-* IDs, got: {:?}",
            sidecar.source_document_ids
        );
        for entry in &sidecar.source_document_ids {
            assert!(
                !entry.starts_with("wiki/"),
                "source_document_ids must not contain wiki paths: {entry}"
            );
        }

        // Frontmatter of the answer must agree with the sidecar.
        let answer_md = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(answer_md.contains("- src-aaaa1111"));
        assert!(answer_md.contains("- src-bbbb2222"));
        assert!(!answer_md.contains("wiki/sources/ownership-guide.md"));
        assert!(!answer_md.contains("wiki/concepts/ownership.md"));
    }
}
