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
    /// Source-document IDs that actually ground the answer — the union of
    /// `source_document_id`(s) reachable from pages referenced by valid
    /// citations. Readers of the promoted wiki page should see this as
    /// "what supports this answer".
    pub source_document_ids: Vec<String>,
    /// All source-document IDs that were considered during retrieval, even
    /// those the model did not end up citing. Kept for debugging/audit — the
    /// "what was in scope" view that used to live in `source_document_ids`.
    #[serde(default)]
    pub retrieval_candidates: Vec<String>,
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
    /// Directory name under `outputs/questions/` to write into. For bn-nlw9
    /// this is `q-<id>-<slug>` (with a slug derived from the question text)
    /// or simply `q-<id>` when the slug is empty. When `None`, the writer
    /// falls back to `question.metadata.id` — preserving pre-bn-nlw9
    /// behavior for callers that haven't been updated.
    pub question_dir_name: Option<&'a str>,
    /// Wiki page paths (`wiki/sources/*.md` or `wiki/concepts/*.md`) that the
    /// model actually cited in the answer body. Used to narrow
    /// `source_document_ids` to sources that really ground the answer.
    ///
    /// When empty (e.g. placeholder artifact with no LLM run, or an answer
    /// that cited nothing), `source_document_ids` ends up empty too.
    pub cited_source_paths: &'a [String],
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
    let dir_name = input
        .question_dir_name
        .map_or_else(|| question_id.clone(), std::string::ToString::to_string);
    let base_dir = PathBuf::from("outputs/questions").join(&dir_name);

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

    let source_doc_ids = resolve_source_document_ids(input.root, input.cited_source_paths);
    let retrieval_candidate_ids = resolve_source_document_ids(
        input.root,
        &candidate_paths(&input.retrieval_plan.candidates),
    );

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
        retrieval_candidates: retrieval_candidate_ids,
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
    // bn-15w4: surface the original prompt alongside its id so JSON consumers
    // don't have to round-trip to question.json to render it.
    obj.insert(
        "question".into(),
        serde_json::Value::String(input.question.raw_query.clone()),
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

/// Resolves a list of wiki page paths to the underlying `source_document_id`
/// values recorded in each page's frontmatter.
///
/// The inputs are wiki page paths (e.g. `wiki/sources/foo.md` or
/// `wiki/concepts/bar.md`). Downstream consumers — particularly
/// `kb lint orphans` — expect `source_document_ids` to contain real
/// source-document identifiers that map to `normalized/<id>/`.
///
/// For each path this function:
/// - Reads the page's frontmatter.
/// - If the page is a source wiki (`source_document_id` present as string or
///   `source_document_ids` as list), collects those IDs.
/// - If the page is a concept wiki, reads `source_document_ids` from its
///   frontmatter when present (that is the authoritative provenance list
///   written by the concept-merge pass). Otherwise, walks its `backlinks`
///   managed region, parses each `[[wiki/sources/<slug>]]` link, and
///   recursively extracts the referenced source page's `source_document_id`.
///
/// Unreadable or unresolvable candidates are silently skipped — this mirrors
/// existing lint behaviour, which already reports missing sources.
/// Results are deduplicated and sorted for stable output.
fn resolve_source_document_ids(root: &Path, paths: &[String]) -> Vec<String> {
    let mut resolved: BTreeSet<String> = BTreeSet::new();
    for path in paths {
        collect_source_document_ids_for_path(root, path, &mut resolved, 0);
    }
    resolved.into_iter().collect()
}

fn candidate_paths(candidates: &[RetrievalCandidate]) -> Vec<String> {
    candidates.iter().map(|c| c.id.clone()).collect()
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

    // Some pages (including concept pages and prior answer artifacts) carry a
    // list of source_document_ids. This is the *authoritative* provenance list
    // written by `concept-merge` (for concepts) or by `write_artifact` itself
    // (for answer artifacts). When present, it is the ground truth — we prefer
    // it over walking the `backlinks` managed region, because backlinks also
    // include "mentioned-by" sources that did not actually contribute to the
    // page. Using backlinks indiscriminately over-broadens the answer's
    // `source_document_ids` to the point of matching `retrieval_candidates`.
    let mut found_list_entry = false;
    if let Some(Value::Sequence(seq)) =
        frontmatter.get(Value::String("source_document_ids".into()))
    {
        for item in seq {
            if let Value::String(s) = item {
                // Heuristic: only accept entries that aren't themselves wiki paths.
                // Paths would just re-introduce the bug this function is fixing.
                if !s.starts_with("wiki/") {
                    out.insert(s.clone());
                    found_list_entry = true;
                }
            }
        }
    }

    // Concept pages don't always carry `source_document_ids` in frontmatter —
    // older or hand-authored concepts may only have a `backlinks` managed
    // region. Fall back to walking backlinks *only* when the frontmatter list
    // was absent or empty, so the authoritative list wins when available.
    if !found_list_entry && rel_path.starts_with("wiki/concepts/") {
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
        resolve_source_document_ids(input.root, input.cited_source_paths)
            .into_iter()
            .map(Value::String)
            .collect();
    fm.insert(
        Value::String("source_document_ids".into()),
        Value::Sequence(source_ids),
    );

    // `retrieval_candidates` preserves the full set of sources the retrieval
    // plan considered — load-bearing debugging info that used to live in
    // `source_document_ids` before that field was narrowed to cited sources.
    let retrieval_candidate_ids: Vec<Value> = resolve_source_document_ids(
        input.root,
        &candidate_paths(&input.retrieval_plan.candidates),
    )
    .into_iter()
    .map(Value::String)
    .collect();
    fm.insert(
        Value::String("retrieval_candidates".into()),
        Value::Sequence(retrieval_candidate_ids),
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
    let question_block = format_question_blockquote(&input.question.raw_query);
    Ok(format!(
        "---\n{yaml}---\n\n{question_block}{body}",
        body = input.artifact_body,
    ))
}

/// bn-15w4: render the original `raw_query` as a markdown blockquote that gets
/// inserted between the frontmatter and the answer body, so readers opening
/// `answer.md` in Obsidian see what was asked without round-tripping to
/// `question.json`. Empty/whitespace queries collapse to nothing rather than
/// emitting an empty quote.
fn format_question_blockquote(raw_query: &str) -> String {
    let trimmed = raw_query.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut out = String::from("> **Question:**\n>\n");
    for line in trimmed.lines() {
        out.push_str("> ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out
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
                id: format!(
                    "art-{}",
                    question_id.strip_prefix("q-").unwrap_or(question_id)
                ),
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
            ..Default::default()
                }],
            fallback_reason: None,
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
            cited_source_paths: &[],
            build_record_id: None,
            question_dir_name: None,
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
            cited_source_paths: &[],
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let content = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("type: question_answer"));
        assert!(content.contains("question_id: q2"));
        assert!(content.contains("generated_at:"));
        assert!(content.contains("source_document_ids:"));
        assert!(content.contains("retrieval_candidates:"));
        assert!(content.contains("Answer text here."));
        // bn-15w4: the original question must appear above the body.
        assert!(content.contains("> **Question:**"));
        assert!(content.contains("> What is Rust?"));
    }

    /// bn-15w4: the question blockquote must precede the answer body so it
    /// reads naturally in Obsidian. Multi-line queries become a continuous
    /// blockquote rather than collapsing onto a single line.
    #[test]
    fn answer_md_renders_multiline_question_above_body() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let mut question = sample_question("q-multi");
        question.raw_query = "Line one of the question.\nLine two follows.".into();
        let artifact = sample_artifact("q-multi");
        let plan = sample_plan();

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "ANSWER_BODY",
            cited_source_paths: &[],
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let content = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        let (_, after_fm) = content.split_once("\n---\n").expect("frontmatter");
        let body_idx = after_fm.find("ANSWER_BODY").expect("body present");
        let preamble = &after_fm[..body_idx];
        assert!(preamble.contains("> Line one of the question."), "got: {preamble}");
        assert!(preamble.contains("> Line two follows."), "got: {preamble}");
        // The question must appear *before* the body, not after.
        assert!(preamble.contains("> **Question:**"));
    }

    /// bn-15w4: an empty `raw_query` collapses to nothing rather than emitting
    /// an empty `> **Question:**` block.
    #[test]
    fn answer_md_omits_empty_question_block() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let mut question = sample_question("q-empty");
        question.raw_query = "   \n  ".into();
        let artifact = sample_artifact("q-empty");
        let plan = sample_plan();

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "ANSWER",
            cited_source_paths: &[],
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let content = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(!content.contains("**Question:**"));
        assert!(content.contains("ANSWER"));
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

        let cited = vec!["wiki/sources/rust-overview.md".to_string()];
        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: Some(&result),
            provenance: Some(&prov),
            artifact_body: &result.body,
            cited_source_paths: &cited,
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let sidecar_str = std::fs::read_to_string(root.join(&output.metadata_path)).unwrap();
        let sidecar: ArtifactSidecar = serde_json::from_str(&sidecar_str).unwrap();
        assert_eq!(sidecar.question_id, "q3");
        assert!(sidecar.provenance.is_some());
        assert_eq!(sidecar.valid_citations, vec![1]);
        assert_eq!(sidecar.invalid_citations, vec![99]);
        assert_eq!(sidecar.source_document_ids, vec!["src-rust-overview"]);
        // retrieval_candidates mirrors the full retrieval scope (same single
        // candidate here, so equal to source_document_ids).
        assert_eq!(sidecar.retrieval_candidates, vec!["src-rust-overview"]);
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
            cited_source_paths: &[],
            build_record_id: None,
            question_dir_name: None,
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
            cited_source_paths: &[],
            build_record_id: None,
            question_dir_name: None,
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
            cited_source_paths: &[],
            build_record_id: Some("build:ask:q6"),
            question_dir_name: None,
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
            cited_source_paths: &["wiki/sources/rust-overview.md".to_string()],
            question_dir_name: None,
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
        // bn-15w4: JSON output mirrors the original prompt next to its id.
        assert_eq!(parsed["question"], "What is Rust?");
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
    /// Regression: the `source_document_ids` field must list only sources
    /// whose pages were *actually cited*. When both cited paths are passed
    /// (one source page, one concept page whose backlinks reach another source),
    /// the resolver must produce real `src-*` IDs — not the wiki paths themselves.
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
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/concepts/ownership.md".into(),
                    title: "Ownership".into(),
                    score: 8,
                    estimated_tokens: 128,
                    reasons: vec![],
                ..Default::default()
                },
            ],
            fallback_reason: None,
        };

        // Both pages are cited (simulated — in real use this comes from valid
        // citation indices resolved against the context manifest).
        let cited = vec![
            "wiki/sources/ownership-guide.md".to_string(),
            "wiki/concepts/ownership.md".to_string(),
        ];

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "Body.",
            cited_source_paths: &cited,
            build_record_id: None,
            question_dir_name: None,
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
        // retrieval_candidates mirrors the full retrieval scope (same here).
        assert_eq!(
            sidecar.retrieval_candidates,
            vec!["src-aaaa1111".to_string(), "src-bbbb2222".to_string()],
        );

        // Frontmatter of the answer must agree with the sidecar.
        let answer_md = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        assert!(answer_md.contains("- src-aaaa1111"));
        assert!(answer_md.contains("- src-bbbb2222"));
        assert!(!answer_md.contains("wiki/sources/ownership-guide.md"));
        assert!(!answer_md.contains("wiki/concepts/ownership.md"));
    }

    /// Core behaviour of this bone: `source_document_ids` must narrow to the
    /// sources the model actually cited, not the entire retrieval scope. The
    /// full scope remains available under `retrieval_candidates` for audit.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn source_document_ids_narrows_to_cited_sources_only() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Four source pages; only two will be "cited".
        seed_source_page(root, "alpha", "src-alpha");
        seed_source_page(root, "beta", "src-beta");
        seed_source_page(root, "gamma", "src-gamma");
        seed_source_page(root, "delta", "src-delta");

        let question = sample_question("q-narrow");
        let artifact = sample_artifact("q-narrow");
        let plan = RetrievalPlan {
            query: "anything".into(),
            token_budget: 4096,
            estimated_tokens: 400,
            candidates: vec![
                crate::RetrievalCandidate {
                    id: "wiki/sources/alpha.md".into(),
                    title: "Alpha".into(),
                    score: 10,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/sources/beta.md".into(),
                    title: "Beta".into(),
                    score: 9,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/sources/gamma.md".into(),
                    title: "Gamma".into(),
                    score: 8,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/sources/delta.md".into(),
                    title: "Delta".into(),
                    score: 7,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
            ],
            fallback_reason: None,
        };

        // Only alpha and gamma were cited by the answer.
        let cited = vec![
            "wiki/sources/alpha.md".to_string(),
            "wiki/sources/gamma.md".to_string(),
        ];

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "Body.",
            cited_source_paths: &cited,
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let sidecar_str = std::fs::read_to_string(root.join(&output.metadata_path)).unwrap();
        let sidecar: ArtifactSidecar = serde_json::from_str(&sidecar_str).unwrap();

        // Narrow: only alpha + gamma (what was cited).
        assert_eq!(
            sidecar.source_document_ids,
            vec!["src-alpha".to_string(), "src-gamma".to_string()],
        );
        // Wide: all four retrieval candidates.
        assert_eq!(
            sidecar.retrieval_candidates,
            vec![
                "src-alpha".to_string(),
                "src-beta".to_string(),
                "src-delta".to_string(),
                "src-gamma".to_string(),
            ],
        );

        // Answer frontmatter reflects the same narrowing.
        let answer_md = std::fs::read_to_string(root.join(&output.answer_path)).unwrap();
        // Extract the frontmatter only.
        let fm = answer_md
            .strip_prefix("---\n")
            .and_then(|rest| rest.split_once("\n---\n").map(|(fm, _)| fm))
            .unwrap();
        let parsed: Value = serde_yaml::from_str(fm).unwrap();
        let sdids: Vec<&str> = parsed
            .get("source_document_ids")
            .and_then(Value::as_sequence)
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(sdids, vec!["src-alpha", "src-gamma"]);
        let rcs: Vec<&str> = parsed
            .get("retrieval_candidates")
            .and_then(Value::as_sequence)
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(rcs, vec!["src-alpha", "src-beta", "src-delta", "src-gamma"]);
    }

    /// Regression for bn-2om: cited concept pages must not over-aggregate
    /// `source_document_ids` through their `backlinks` region when the concept
    /// already carries an authoritative `source_document_ids` frontmatter list.
    ///
    /// Before the fix, resolving a cited concept page merged:
    ///   1. the concept's frontmatter `source_document_ids` list (authoritative
    ///      provenance from `concept-merge`), AND
    ///   2. every source reachable via the concept's `backlinks` managed region
    ///      (which includes "mentioned by" sources, not just contributors).
    ///
    /// (2) pulled in every non-cited source that happened to mention the concept,
    /// which collapsed `source_document_ids` into `retrieval_candidates`. The
    /// fix prefers (1) when present.
    #[test]
    fn cited_concept_frontmatter_list_wins_over_backlinks_aggregation() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        // Authoritative contributor — named in the concept's frontmatter.
        seed_source_page(root, "hdfs-source", "src-hdfs-auth");
        // Noise source — only mentions the concept, reachable via backlinks.
        seed_source_page(root, "unrelated-a", "src-noise-a");
        seed_source_page(root, "unrelated-b", "src-noise-b");

        // Concept whose frontmatter authoritatively lists ONE contributor
        // (src-hdfs-auth). Its backlinks region, however, lists all three
        // sources because each mentions the concept somewhere.
        let concept_path = root.join("wiki/concepts/hdfs.md");
        std::fs::create_dir_all(concept_path.parent().unwrap()).unwrap();
        std::fs::write(
            &concept_path,
            "---\n\
             id: concept:hdfs\n\
             name: HDFS\n\
             source_document_ids:\n\
             - src-hdfs-auth\n\
             ---\n\
             \n\
             # HDFS\n\
             \n\
             <!-- kb:begin id=backlinks -->\n\
             - [[wiki/sources/hdfs-source]]\n\
             - [[wiki/sources/unrelated-a]]\n\
             - [[wiki/sources/unrelated-b]]\n\
             <!-- kb:end id=backlinks -->\n",
        )
        .unwrap();

        let question = sample_question("q-agg");
        let artifact = sample_artifact("q-agg");
        let plan = RetrievalPlan {
            query: "hdfs".into(),
            token_budget: 4096,
            estimated_tokens: 400,
            candidates: vec![crate::RetrievalCandidate {
                id: "wiki/concepts/hdfs.md".into(),
                title: "HDFS".into(),
                score: 10,
                estimated_tokens: 100,
                reasons: vec![],
            ..Default::default()
                }],
            fallback_reason: None,
        };

        let cited = vec!["wiki/concepts/hdfs.md".to_string()];
        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &question,
            artifact: &artifact,
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "Body.",
            cited_source_paths: &cited,
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let sidecar_str = std::fs::read_to_string(root.join(&output.metadata_path)).unwrap();
        let sidecar: ArtifactSidecar = serde_json::from_str(&sidecar_str).unwrap();

        // Only the authoritative contributor — the backlinks noise is ignored
        // because the frontmatter list is present and wins.
        assert_eq!(
            sidecar.source_document_ids,
            vec!["src-hdfs-auth".to_string()],
            "cited concept must narrow to its frontmatter list, got: {:?}",
            sidecar.source_document_ids,
        );
    }

    /// Regression for bn-2om: mirrors the bone's acceptance integration test.
    /// Seed a retrieval plan with 4 candidates, pretend `valid_citations = [1, 3]`
    /// by passing the corresponding two wiki paths as `cited_source_paths`.
    /// `source_document_ids` must contain exactly those two src-ids, while
    /// `retrieval_candidates` keeps all four.
    #[test]
    fn valid_citations_1_and_3_narrow_source_document_ids_to_two_entries() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();

        seed_source_page(root, "one", "src-one");
        seed_source_page(root, "two", "src-two");
        seed_source_page(root, "three", "src-three");
        seed_source_page(root, "four", "src-four");

        let plan = RetrievalPlan {
            query: "anything".into(),
            token_budget: 4096,
            estimated_tokens: 400,
            candidates: vec![
                crate::RetrievalCandidate {
                    id: "wiki/sources/one.md".into(),
                    title: "One".into(),
                    score: 10,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/sources/two.md".into(),
                    title: "Two".into(),
                    score: 9,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/sources/three.md".into(),
                    title: "Three".into(),
                    score: 8,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
                crate::RetrievalCandidate {
                    id: "wiki/sources/four.md".into(),
                    title: "Four".into(),
                    score: 7,
                    estimated_tokens: 100,
                    reasons: vec![],
                ..Default::default()
                },
            ],
            fallback_reason: None,
        };

        // Citations [1, 3] map to candidates at index 0 and 2 — i.e. "one"
        // and "three". Pass those as cited paths.
        let cited = vec![
            "wiki/sources/one.md".to_string(),
            "wiki/sources/three.md".to_string(),
        ];

        let output = write_artifact(&WriteArtifactInput {
            root,
            question: &sample_question("q-cit-1-3"),
            artifact: &sample_artifact("q-cit-1-3"),
            retrieval_plan: &plan,
            artifact_result: None,
            provenance: None,
            artifact_body: "Body.",
            cited_source_paths: &cited,
            build_record_id: None,
            question_dir_name: None,
        })
        .unwrap();

        let sidecar: ArtifactSidecar =
            serde_json::from_str(&std::fs::read_to_string(root.join(&output.metadata_path)).unwrap())
                .unwrap();
        assert_eq!(
            sidecar.source_document_ids,
            vec!["src-one".to_string(), "src-three".to_string()],
        );
        assert_eq!(
            sidecar.retrieval_candidates,
            vec![
                "src-four".to_string(),
                "src-one".to_string(),
                "src-three".to_string(),
                "src-two".to_string(),
            ],
        );
        assert_ne!(sidecar.source_document_ids, sidecar.retrieval_candidates);
    }
}
