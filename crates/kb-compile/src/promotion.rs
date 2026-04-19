use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_yaml::{Mapping, Value};

use kb_core::fs::atomic_write;
use kb_core::managed_region::rewrite_managed_region;
use kb_core::{
    BuildRecord, EntityMetadata, ReviewItem, ReviewKind, ReviewStatus, Status, hash_many,
    save_build_record, save_review_item,
};

const ANSWER_REGION_ID: &str = "answer";
const CITATIONS_REGION_ID: &str = "citations";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionInput<'a> {
    pub review_item: &'a ReviewItem,
    pub artifact_body: &'a str,
    pub promoted_at: u64,
    /// Source document IDs grounding the promoted page.
    ///
    /// These are the wiki/sources paths (or equivalent normalized document IDs)
    /// drawn from the original retrieval plan. They populate the
    /// `source_document_ids` frontmatter field so `kb lint` orphan checks pass.
    pub source_document_ids: &'a [String],
    /// Question ID that produced the answer, mirrored into `derived_from`
    /// frontmatter so the promotion chain is traceable without polluting
    /// `source_document_ids` with non-source IDs.
    pub derived_from_question_id: Option<&'a str>,
    /// Artifact ID that holds the answer body, mirrored into `derived_from`.
    pub derived_from_artifact_id: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionResult {
    pub destination: PathBuf,
    pub build_record: BuildRecord,
    pub markdown: String,
}

/// Render a promoted wiki page from a `ReviewItem` and its artifact body.
///
/// If the destination page already exists, managed regions are rewritten in place
/// while preserving all other content. Untouched regions are byte-stable.
///
/// # Errors
///
/// Returns an error if the review item has no proposed destination, or if
/// frontmatter serialization fails.
pub fn render_promotion(
    input: &PromotionInput<'_>,
    existing_page: Option<&str>,
) -> Result<PromotionResult> {
    let item = input.review_item;

    let destination = item
        .proposed_destination
        .as_ref()
        .context("ReviewItem has no proposed_destination")?
        .clone();

    if item.kind != ReviewKind::Promotion {
        bail!(
            "expected ReviewKind::Promotion, got {:?}",
            item.kind
        );
    }

    let build_record_id = format!("promote-{}", item.metadata.id);

    let frontmatter = build_frontmatter(
        item,
        &build_record_id,
        input.promoted_at,
        input.source_document_ids,
        input.derived_from_question_id,
        input.derived_from_artifact_id,
    );
    let body = build_body(input.artifact_body, &item.citations, existing_page);
    let markdown = serialize_frontmatter(&frontmatter, &body)?;

    let content_hash = hash_many(&[
        input.artifact_body.as_bytes(),
        b"\0",
        destination.to_string_lossy().as_bytes(),
    ]);

    let build_record = BuildRecord {
        metadata: EntityMetadata {
            id: build_record_id,
            created_at_millis: input.promoted_at,
            updated_at_millis: input.promoted_at,
            source_hashes: vec![content_hash.to_hex()],
            model_version: None,
            tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: None,
            dependencies: item.metadata.dependencies.clone(),
            output_paths: vec![destination.clone()],
            status: Status::Fresh,
        },
        pass_name: "promotion".to_string(),
        input_ids: item.metadata.dependencies.clone(),
        output_ids: vec![destination.to_string_lossy().into_owned()],
        manifest_hash: content_hash.to_hex(),
    };

    Ok(PromotionResult {
        destination,
        build_record,
        markdown,
    })
}

/// Execute a full promotion: render the page, write it atomically, save the
/// build record, and mark the review item as approved.
///
/// # Errors
///
/// Returns an error if the review item is not pending, the destination is missing,
/// or any I/O operation fails.
pub fn execute_promotion(
    root: &Path,
    review_item: &ReviewItem,
    promoted_at: u64,
) -> Result<PromotionResult> {
    if review_item.status != ReviewStatus::Pending {
        bail!(
            "review item {} is {:?}, expected Pending",
            review_item.metadata.id,
            review_item.status
        );
    }

    let destination = review_item
        .proposed_destination
        .as_ref()
        .context("ReviewItem has no proposed_destination")?;

    let ArtifactLoad {
        body: artifact_body,
        source_document_ids,
    } = load_artifact(root, review_item)
        .with_context(|| format!("read artifact for review {}", review_item.metadata.id))?;

    let existing_page = {
        let page_path = root.join(destination);
        if page_path.exists() {
            Some(
                std::fs::read_to_string(&page_path)
                    .with_context(|| format!("read existing page {}", page_path.display()))?,
            )
        } else {
            None
        }
    };

    let (question_id, artifact_id) = split_derived_ids(&review_item.metadata.dependencies);

    let input = PromotionInput {
        review_item,
        artifact_body: &artifact_body,
        promoted_at,
        source_document_ids: &source_document_ids,
        derived_from_question_id: question_id.as_deref(),
        derived_from_artifact_id: artifact_id.as_deref(),
    };

    let result = render_promotion(&input, existing_page.as_deref())?;

    let dest_path = root.join(&result.destination);
    atomic_write(&dest_path, result.markdown.as_bytes())
        .with_context(|| format!("write promoted page {}", dest_path.display()))?;

    save_build_record(root, &result.build_record)
        .context("save promotion build record")?;

    let mut approved = review_item.clone();
    approved.status = ReviewStatus::Approved;
    approved.metadata.updated_at_millis = promoted_at;
    save_review_item(root, &approved).context("update review item to approved")?;

    Ok(result)
}

struct ArtifactLoad {
    body: String,
    source_document_ids: Vec<String>,
}

fn load_artifact(root: &Path, item: &ReviewItem) -> Result<ArtifactLoad> {
    for dep in &item.metadata.dependencies {
        if dep.starts_with("artifact-") {
            let answer_path = root
                .join("outputs/questions")
                .join(dep.strip_prefix("artifact-").unwrap_or(dep))
                .join("answer.md");
            if answer_path.exists() {
                let raw = std::fs::read_to_string(&answer_path)
                    .with_context(|| format!("read artifact {}", answer_path.display()))?;
                return Ok(parse_artifact(&raw));
            }
        }
    }

    for output_path in &item.metadata.output_paths {
        let full = root.join(output_path);
        if full.exists() && full.extension().is_some_and(|e| e == "md") {
            let raw = std::fs::read_to_string(&full)
                .with_context(|| format!("read artifact {}", full.display()))?;
            return Ok(parse_artifact(&raw));
        }
    }

    bail!(
        "could not locate artifact body for review item {}",
        item.metadata.id
    );
}

fn parse_artifact(raw: &str) -> ArtifactLoad {
    let source_document_ids = extract_source_document_ids(raw);
    ArtifactLoad {
        body: strip_frontmatter(raw),
        source_document_ids,
    }
}

fn extract_source_document_ids(markdown: &str) -> Vec<String> {
    let Some(yaml) = extract_frontmatter_yaml(markdown) else {
        return Vec::new();
    };
    let Ok(value) = serde_yaml::from_str::<Value>(yaml) else {
        return Vec::new();
    };
    let Some(mapping) = value.as_mapping() else {
        return Vec::new();
    };
    let Some(seq) = mapping
        .get(Value::String("source_document_ids".into()))
        .and_then(Value::as_sequence)
    else {
        return Vec::new();
    };
    seq.iter()
        .filter_map(|v| v.as_str().map(ToString::to_string))
        .collect()
}

fn extract_frontmatter_yaml(markdown: &str) -> Option<&str> {
    let rest = markdown
        .strip_prefix("---\n")
        .or_else(|| markdown.strip_prefix("---\r\n"))?;
    if let Some(end) = rest.find("\n---\n") {
        return Some(&rest[..=end]);
    }
    if let Some(end) = rest.find("\r\n---\r\n") {
        return Some(&rest[..=end + 1]);
    }
    if let Some(end) = rest.find("\n---") {
        // Trailing "---" without a newline after (end of file).
        return Some(&rest[..=end]);
    }
    None
}

fn split_derived_ids(deps: &[String]) -> (Option<String>, Option<String>) {
    let mut question_id = None;
    let mut artifact_id = None;
    for dep in deps {
        if dep.starts_with("artifact-") && artifact_id.is_none() {
            artifact_id = Some(dep.clone());
        } else if dep.starts_with("question-") && question_id.is_none() {
            question_id = Some(dep.clone());
        } else if question_id.is_none() && !dep.starts_with("artifact-") {
            // Fallback: older pipelines stored the bare question id without a prefix.
            question_id = Some(dep.clone());
        }
    }
    (question_id, artifact_id)
}

fn strip_frontmatter(markdown: &str) -> String {
    if !markdown.starts_with("---\n") && !markdown.starts_with("---\r\n") {
        return markdown.to_string();
    }

    let after_first = &markdown[4..];
    if let Some(end) = after_first.find("\n---\n") {
        return after_first[end + 5..].to_string();
    }
    if let Some(end) = after_first.find("\r\n---\r\n") {
        return after_first[end + 7..].to_string();
    }

    markdown.to_string()
}

fn build_frontmatter(
    item: &ReviewItem,
    build_record_id: &str,
    promoted_at: u64,
    source_document_ids: &[String],
    derived_from_question_id: Option<&str>,
    derived_from_artifact_id: Option<&str>,
) -> Mapping {
    let mut fm = Mapping::new();
    fm.insert(
        Value::String("id".into()),
        Value::String(item.target_entity_id.clone()),
    );
    fm.insert(
        Value::String("type".into()),
        Value::String("question_answer".into()),
    );
    fm.insert(
        Value::String("promoted_at".into()),
        Value::Number(promoted_at.into()),
    );
    fm.insert(
        Value::String("build_record_id".into()),
        Value::String(build_record_id.into()),
    );
    fm.insert(
        Value::String("review_id".into()),
        Value::String(item.metadata.id.clone()),
    );

    if !item.citations.is_empty() {
        let citations: Vec<Value> = item
            .citations
            .iter()
            .map(|c| Value::String(c.clone()))
            .collect();
        fm.insert(
            Value::String("citations".into()),
            Value::Sequence(citations),
        );
    }

    // `source_document_ids` must list *real* sources (retrieval plan candidates)
    // so `kb lint`'s orphan check can resolve them against `normalized/<id>/`.
    // The question/artifact IDs are linkage, not sources, and go in `derived_from`.
    if !source_document_ids.is_empty() {
        let source_ids: Vec<Value> = source_document_ids
            .iter()
            .map(|d| Value::String(d.clone()))
            .collect();
        fm.insert(
            Value::String("source_document_ids".into()),
            Value::Sequence(source_ids),
        );
    }

    if derived_from_question_id.is_some() || derived_from_artifact_id.is_some() {
        let mut derived = Mapping::new();
        if let Some(qid) = derived_from_question_id {
            derived.insert(
                Value::String("question_id".into()),
                Value::String(qid.to_string()),
            );
        }
        if let Some(aid) = derived_from_artifact_id {
            derived.insert(
                Value::String("artifact_id".into()),
                Value::String(aid.to_string()),
            );
        }
        fm.insert(Value::String("derived_from".into()), Value::Mapping(derived));
    }

    fm
}

fn build_body(artifact_body: &str, citations: &[String], existing_page: Option<&str>) -> String {
    let answer_content = format!("\n{}\n", artifact_body.trim());
    let citations_content = render_citation_list(citations);

    existing_page.map_or_else(
        || {
            let mut body = String::new();
            body.push_str("## Answer\n");
            body.push_str(&managed_region(ANSWER_REGION_ID, &answer_content));
            body.push_str("\n\n## Citations\n");
            body.push_str(&managed_region(CITATIONS_REGION_ID, &citations_content));
            body.push('\n');
            body
        },
        |existing| {
            let mut result = strip_existing_frontmatter_body(existing);
            result = rewrite_managed_region(&result, ANSWER_REGION_ID, &answer_content)
                .unwrap_or_else(|| {
                    upsert_section(&result, "## Answer", ANSWER_REGION_ID, &answer_content)
                });
            result = rewrite_managed_region(&result, CITATIONS_REGION_ID, &citations_content)
                .unwrap_or_else(|| {
                    upsert_section(
                        &result,
                        "## Citations",
                        CITATIONS_REGION_ID,
                        &citations_content,
                    )
                });
            result
        },
    )
}

fn strip_existing_frontmatter_body(markdown: &str) -> String {
    if !markdown.starts_with("---\n") && !markdown.starts_with("---\r\n") {
        return markdown.to_string();
    }

    let after_first = &markdown[4..];
    if let Some(end) = after_first.find("\n---\n") {
        return after_first[end + 5..].to_string();
    }
    if let Some(end) = after_first.find("\r\n---\r\n") {
        return after_first[end + 7..].to_string();
    }

    markdown.to_string()
}

fn upsert_section(existing_body: &str, heading: &str, region_id: &str, content: &str) -> String {
    if let Some(updated) = rewrite_managed_region(existing_body, region_id, content) {
        return updated;
    }

    let mut body = existing_body.trim_end().to_string();
    if !body.is_empty() {
        body.push_str("\n\n");
    }
    body.push_str(heading);
    body.push('\n');
    body.push_str(&managed_region(region_id, content));
    body.push('\n');
    body
}

fn managed_region(region_id: &str, content: &str) -> String {
    let mut rendered = String::new();
    rendered.push_str("<!-- kb:begin id=");
    rendered.push_str(region_id);
    rendered.push_str(" -->");
    rendered.push_str(content);
    if !content.ends_with('\n') {
        rendered.push('\n');
    }
    rendered.push_str("<!-- kb:end id=");
    rendered.push_str(region_id);
    rendered.push_str(" -->");
    rendered
}

fn render_citation_list(citations: &[String]) -> String {
    let mut rendered = String::from("\n");
    if citations.is_empty() {
        rendered.push_str("_No citations._\n");
        return rendered;
    }
    for citation in citations {
        rendered.push_str("- ");
        rendered.push_str(citation.trim());
        rendered.push('\n');
    }
    rendered
}

fn serialize_frontmatter(frontmatter: &Mapping, body: &str) -> Result<String> {
    let yaml = serde_yaml::to_string(frontmatter).context("serialize promotion frontmatter")?;
    Ok(format!("---\n{yaml}---\n{body}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kb_core::EntityMetadata;
    use tempfile::tempdir;

    fn test_input<'a>(
        item: &'a ReviewItem,
        artifact_body: &'a str,
        promoted_at: u64,
    ) -> PromotionInput<'a> {
        PromotionInput {
            review_item: item,
            artifact_body,
            promoted_at,
            source_document_ids: &[],
            derived_from_question_id: None,
            derived_from_artifact_id: None,
        }
    }

    fn test_review_item(id: &str) -> ReviewItem {
        ReviewItem {
            metadata: EntityMetadata {
                id: id.to_string(),
                created_at_millis: 1000,
                updated_at_millis: 1000,
                source_hashes: vec!["hash-1".to_string()],
                model_version: None,
                tool_version: Some("kb/0.1.0".to_string()),
                prompt_template_hash: None,
                dependencies: vec!["question-1".to_string(), "artifact-q1".to_string()],
                output_paths: vec![
                    PathBuf::from("outputs/questions/q1/answer.md"),
                    PathBuf::from("wiki/questions/example.md"),
                ],
                status: Status::NeedsReview,
            },
            kind: ReviewKind::Promotion,
            target_entity_id: "artifact-q1".to_string(),
            proposed_destination: Some(PathBuf::from("wiki/questions/example.md")),
            citations: vec!["source-a#intro".to_string(), "source-b#details".to_string()],
            affected_pages: vec![PathBuf::from("wiki/questions/example.md")],
            created_at_millis: 1000,
            status: ReviewStatus::Pending,
            comment: "Promote answer for: What is Rust?".to_string(),
        }
    }

    #[test]
    fn render_promotion_creates_page_with_managed_regions() {
        let item = test_review_item("review-1");
        let input = test_input(&item, "Rust is a systems programming language.", 2000);

        let result = render_promotion(&input, None).expect("render");

        assert_eq!(result.destination, PathBuf::from("wiki/questions/example.md"));
        assert!(result.markdown.starts_with("---\n"));
        assert!(result.markdown.contains("type: question_answer"));
        assert!(result.markdown.contains("promoted_at:"));
        assert!(result.markdown.contains("<!-- kb:begin id=answer -->"));
        assert!(result.markdown.contains("Rust is a systems programming language."));
        assert!(result.markdown.contains("<!-- kb:end id=answer -->"));
        assert!(result.markdown.contains("<!-- kb:begin id=citations -->"));
        assert!(result.markdown.contains("- source-a#intro"));
        assert!(result.markdown.contains("- source-b#details"));
        assert!(result.markdown.contains("<!-- kb:end id=citations -->"));
    }

    #[test]
    fn render_promotion_preserves_manual_content() {
        let existing = "---\nid: old\n---\n## Answer\n<!-- kb:begin id=answer -->\nOld answer.\n<!-- kb:end id=answer -->\n\n## Notes\nKeep this manual section.\n\n## Citations\n<!-- kb:begin id=citations -->\n- old-citation\n<!-- kb:end id=citations -->\n";

        let item = test_review_item("review-2");
        let input = test_input(&item, "Updated answer text.", 3000);

        let result = render_promotion(&input, Some(existing)).expect("render");

        assert!(result.markdown.contains("Updated answer text."));
        assert!(result.markdown.contains("## Notes\nKeep this manual section."));
        assert!(result.markdown.contains("- source-a#intro"));
        assert!(!result.markdown.contains("Old answer."));
        assert!(!result.markdown.contains("old-citation"));
    }

    #[test]
    fn render_promotion_untouched_regions_are_byte_stable() {
        let existing = "---\nid: old\n---\n## Answer\n<!-- kb:begin id=answer -->\nOld answer.\n<!-- kb:end id=answer -->\n\n## Custom\n<!-- kb:begin id=custom -->\nCustom content here.\n<!-- kb:end id=custom -->\n\n## Citations\n<!-- kb:begin id=citations -->\n- old-citation\n<!-- kb:end id=citations -->\n";

        let item = test_review_item("review-3");
        let input = test_input(&item, "New answer.", 4000);

        let result = render_promotion(&input, Some(existing)).expect("render");

        assert!(result.markdown.contains(
            "<!-- kb:begin id=custom -->\nCustom content here.\n<!-- kb:end id=custom -->"
        ));
    }

    #[test]
    fn render_promotion_emits_build_record() {
        let item = test_review_item("review-4");
        let input = test_input(&item, "Answer body.", 5000);

        let result = render_promotion(&input, None).expect("render");

        assert_eq!(result.build_record.pass_name, "promotion");
        assert_eq!(result.build_record.metadata.id, "promote-review-4");
        assert_eq!(result.build_record.metadata.created_at_millis, 5000);
        assert_eq!(
            result.build_record.input_ids,
            vec!["question-1".to_string(), "artifact-q1".to_string()]
        );
        assert_eq!(
            result.build_record.output_ids,
            vec!["wiki/questions/example.md".to_string()]
        );
    }

    #[test]
    fn render_promotion_with_empty_citations() {
        let mut item = test_review_item("review-5");
        item.citations.clear();
        let input = test_input(&item, "Answer.", 6000);

        let result = render_promotion(&input, None).expect("render");

        assert!(result.markdown.contains("_No citations._"));
    }

    #[test]
    fn render_promotion_rejects_non_promotion_kind() {
        let mut item = test_review_item("review-6");
        item.kind = ReviewKind::ConceptMerge;
        let input = test_input(&item, "Answer.", 7000);

        let err = render_promotion(&input, None).expect_err("should reject non-promotion");
        assert!(err.to_string().contains("Promotion"));
    }

    #[test]
    fn render_promotion_rejects_missing_destination() {
        let mut item = test_review_item("review-7");
        item.proposed_destination = None;
        let input = test_input(&item, "Answer.", 8000);

        let err = render_promotion(&input, None).expect_err("should reject no destination");
        assert!(err.to_string().contains("proposed_destination"));
    }

    #[test]
    fn execute_promotion_writes_page_and_build_record() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();

        std::fs::create_dir_all(root.join("outputs/questions/q1")).expect("create artifact dir");
        std::fs::write(
            root.join("outputs/questions/q1/answer.md"),
            "---\nid: artifact-q1\ntype: question_answer\nsource_document_ids:\n- wiki/sources/rust.md\n- wiki/sources/cargo.md\n---\n\nRust is great.\n",
        )
        .expect("write artifact");

        let item = test_review_item("review-exec-1");
        let result = execute_promotion(root, &item, 9000).expect("execute");

        assert!(root.join(&result.destination).exists());
        let page = std::fs::read_to_string(root.join(&result.destination)).expect("read page");
        assert!(page.contains("Rust is great."));
        assert!(page.contains("<!-- kb:begin id=answer -->"));

        let record_path = root.join("state/build_records/promote-review-exec-1.json");
        assert!(record_path.exists());

        let review_path = root.join("reviews/promotions/review-exec-1.json");
        let saved: ReviewItem =
            serde_json::from_str(&std::fs::read_to_string(review_path).expect("read review"))
                .expect("parse review");
        assert_eq!(saved.status, ReviewStatus::Approved);
    }

    #[test]
    fn execute_promotion_uses_retrieval_sources_not_dependency_ids() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();

        std::fs::create_dir_all(root.join("outputs/questions/q1")).expect("create artifact dir");
        std::fs::write(
            root.join("outputs/questions/q1/answer.md"),
            "---\nid: artifact-q1\ntype: question_answer\nsource_document_ids:\n- wiki/sources/rust.md\n- wiki/sources/cargo.md\n---\n\nBody.\n",
        )
        .expect("write artifact");

        let item = test_review_item("review-exec-src-1");
        let _ = execute_promotion(root, &item, 10_000).expect("execute");

        let page_path = root.join("wiki/questions/example.md");
        let page = std::fs::read_to_string(&page_path).expect("read page");

        // Real sources from the artifact frontmatter (retrieval plan candidates)
        // end up in `source_document_ids`.
        assert!(
            page.contains("- wiki/sources/rust.md"),
            "expected retrieval source in frontmatter: {page}"
        );
        assert!(
            page.contains("- wiki/sources/cargo.md"),
            "expected retrieval source in frontmatter: {page}"
        );

        // Extract the frontmatter only so we don't pick up body/citation occurrences.
        let fm = page
            .strip_prefix("---\n")
            .and_then(|rest| rest.split_once("\n---\n").map(|(fm, _)| fm))
            .expect("frontmatter block");

        // Parse the frontmatter and check the specific sequence.
        let doc: serde_yaml::Value = serde_yaml::from_str(fm).expect("parse frontmatter");
        let sources = doc
            .get("source_document_ids")
            .and_then(Value::as_sequence)
            .expect("source_document_ids present");
        let source_strs: Vec<&str> = sources.iter().filter_map(Value::as_str).collect();
        assert_eq!(source_strs, vec!["wiki/sources/rust.md", "wiki/sources/cargo.md"]);
        for s in &source_strs {
            assert!(
                !s.starts_with("question-") && !s.starts_with("artifact-"),
                "source_document_ids must not contain question/artifact IDs: {s}"
            );
        }

        // Linkage is preserved under `derived_from`.
        let derived = doc
            .get("derived_from")
            .and_then(Value::as_mapping)
            .expect("derived_from mapping");
        assert_eq!(
            derived.get(Value::String("question_id".into())).and_then(Value::as_str),
            Some("question-1")
        );
        assert_eq!(
            derived.get(Value::String("artifact_id".into())).and_then(Value::as_str),
            Some("artifact-q1")
        );
    }

    #[test]
    fn extract_source_document_ids_from_answer_frontmatter() {
        let md = "---\nid: artifact-q1\nsource_document_ids:\n- wiki/sources/a.md\n- wiki/sources/b.md\n---\n\nBody\n";
        let ids = extract_source_document_ids(md);
        assert_eq!(ids, vec!["wiki/sources/a.md", "wiki/sources/b.md"]);
    }

    #[test]
    fn extract_source_document_ids_missing_returns_empty() {
        let md = "---\nid: artifact-q1\n---\n\nBody\n";
        assert!(extract_source_document_ids(md).is_empty());
    }

    #[test]
    fn split_derived_ids_picks_out_question_and_artifact() {
        let deps = vec![
            "question-abcd".to_string(),
            "artifact-q1".to_string(),
        ];
        let (q, a) = split_derived_ids(&deps);
        assert_eq!(q.as_deref(), Some("question-abcd"));
        assert_eq!(a.as_deref(), Some("artifact-q1"));
    }

    #[test]
    fn execute_promotion_rejects_non_pending() {
        let dir = tempdir().expect("tempdir");
        let mut item = test_review_item("review-exec-2");
        item.status = ReviewStatus::Approved;

        let err = execute_promotion(dir.path(), &item, 10_000).expect_err("should reject");
        assert!(err.to_string().contains("Pending"));
    }

    #[test]
    fn strip_frontmatter_removes_yaml_block() {
        let md = "---\nid: test\ntype: qa\n---\n\nBody content here.\n";
        assert_eq!(strip_frontmatter(md), "\nBody content here.\n");
    }

    #[test]
    fn strip_frontmatter_preserves_body_without_frontmatter() {
        let md = "Just plain text.\n";
        assert_eq!(strip_frontmatter(md), "Just plain text.\n");
    }
}
