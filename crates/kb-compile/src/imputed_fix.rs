//! Approve-time handler for `ReviewKind::ImputedFix` review items (bn-xt4o).
//!
//! The lint pass (`kb lint --impute`) writes these items with a sidecar
//! payload carrying the model's draft + cited web sources. On approve, this
//! module:
//!
//! 1. Loads the payload from `reviews/imputed_fixes/<id>.payload.json`.
//! 2. For `missing_concept` gaps: writes a fresh
//!    `wiki/concepts/<slug>.md` (refuses to overwrite existing pages).
//! 3. For `thin_concept_body` gaps: rewrites the body of an existing
//!    concept page, preserving its frontmatter and any `<!-- kb:* -->`
//!    managed regions verbatim.
//!
//! Rejected items do nothing — the impute pass just won't re-fire on the
//! same fingerprint next run.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use kb_core::fs::atomic_write;
use kb_core::{ReviewItem, ReviewKind};
use kb_lint::{ImputedFixPayload, load_imputed_fix_payload};
use serde_yaml::{Mapping, Value};

/// Result of applying an imputed-fix review item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedImputedFix {
    /// Path of the concept page that was created or updated (relative to
    /// the KB root).
    pub concept_path: PathBuf,
    /// Kind of gap that was fixed (`"missing_concept"` or
    /// `"thin_concept_body"`).
    pub gap_kind: String,
    /// Canonical concept name written to the page.
    pub concept_name: String,
    /// Web source URLs cited in the imputed draft (appended as a "Sources"
    /// section under the body so the reader can inspect provenance).
    pub cited_sources: Vec<String>,
    /// `true` when the apply wrote a new file (missing-concept); `false`
    /// when it rewrote an existing page body (thin-body).
    pub created_new_page: bool,
}

/// Apply an `ImputedFix` review item.
///
/// # Errors
///
/// Returns an error when:
/// - the item is not an `ImputedFix`,
/// - the sidecar payload cannot be read or parsed,
/// - for `missing_concept`: the target concept page already exists (collision),
/// - for `thin_concept_body`: the target concept page does not exist,
/// - the page cannot be written atomically.
pub fn apply_imputed_fix(root: &Path, item: &ReviewItem) -> Result<AppliedImputedFix> {
    if item.kind != ReviewKind::ImputedFix {
        bail!(
            "apply_imputed_fix called on a {:?} review item",
            item.kind
        );
    }
    let payload = load_imputed_fix_payload(root, item)
        .with_context(|| format!("load imputed-fix payload for '{}'", item.metadata.id))?;

    match payload.gap_kind.as_str() {
        "missing_concept" => apply_missing_concept(root, &payload),
        "thin_concept_body" => apply_thin_concept_body(root, &payload),
        other => bail!("unsupported imputed-fix gap_kind '{other}'"),
    }
}

fn apply_missing_concept(root: &Path, payload: &ImputedFixPayload) -> Result<AppliedImputedFix> {
    let concept_rel = &payload.page_path;
    let concept_path = root.join(concept_rel);

    if concept_path.exists() {
        bail!(
            "concept page already exists at {}; reject this imputed-fix and edit manually \
             or open a merge review instead",
            concept_rel.display()
        );
    }

    let slug = concept_rel
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    if slug.is_empty() {
        bail!(
            "imputed-fix payload page_path {} has no file stem",
            concept_rel.display()
        );
    }

    let mut sorted_sources: Vec<String> = payload.local_source_document_ids.clone();
    sorted_sources.sort();
    sorted_sources.dedup();

    let content = render_concept_markdown(
        &slug,
        &payload.concept_name,
        &sorted_sources,
        &payload.definition,
        &payload.sources,
        &payload.rationale,
    )?;

    atomic_write(&concept_path, content.as_bytes())
        .with_context(|| format!("write concept page {}", concept_path.display()))?;

    Ok(AppliedImputedFix {
        concept_path: concept_rel.clone(),
        gap_kind: payload.gap_kind.clone(),
        concept_name: payload.concept_name.clone(),
        cited_sources: payload.sources.iter().map(|s| s.url.clone()).collect(),
        created_new_page: true,
    })
}

fn apply_thin_concept_body(root: &Path, payload: &ImputedFixPayload) -> Result<AppliedImputedFix> {
    let concept_rel = &payload.page_path;
    let concept_path = root.join(concept_rel);

    if !concept_path.exists() {
        bail!(
            "thin-body imputed fix targets {}, but the page does not exist — \
             reject this item and re-run `kb lint --impute` on the current tree",
            concept_rel.display()
        );
    }

    let existing = fs::read_to_string(&concept_path)
        .with_context(|| format!("read concept page {}", concept_path.display()))?;
    let new_content = rewrite_body_preserving_frontmatter(
        &existing,
        &payload.concept_name,
        &payload.definition,
        &payload.sources,
        &payload.rationale,
    );

    atomic_write(&concept_path, new_content.as_bytes())
        .with_context(|| format!("rewrite concept page {}", concept_path.display()))?;

    Ok(AppliedImputedFix {
        concept_path: concept_rel.clone(),
        gap_kind: payload.gap_kind.clone(),
        concept_name: payload.concept_name.clone(),
        cited_sources: payload.sources.iter().map(|s| s.url.clone()).collect(),
        created_new_page: false,
    })
}

/// Render a brand-new concept page from an imputed fix. Frontmatter shape
/// mirrors [`kb_compile::concept_candidate::apply_concept_candidate`]'s
/// renderer so backlink / index scanners handle imputed-origin pages the
/// same as merge/candidate pages.
fn render_concept_markdown(
    slug: &str,
    canonical_name: &str,
    source_document_ids: &[String],
    definition: &str,
    sources: &[kb_llm::ImputedWebSource],
    rationale: &str,
) -> Result<String> {
    let mut fm = Mapping::new();
    fm.insert(
        Value::String("id".into()),
        Value::String(format!("concept:{slug}")),
    );
    fm.insert(
        Value::String("name".into()),
        Value::String(canonical_name.to_string()),
    );
    if !source_document_ids.is_empty() {
        let vals: Vec<Value> = source_document_ids
            .iter()
            .map(|id| Value::String(id.clone()))
            .collect();
        fm.insert(
            Value::String("source_document_ids".into()),
            Value::Sequence(vals),
        );
    }
    fm.insert(
        Value::String("origin".into()),
        Value::String("imputed".into()),
    );

    let frontmatter_yaml = serde_yaml::to_string(&fm).context("serialize concept frontmatter")?;

    let mut content = format!("---\n{frontmatter_yaml}---\n\n# {canonical_name}\n");
    let body_trimmed = definition.trim();
    if !body_trimmed.is_empty() {
        content.push('\n');
        content.push_str(body_trimmed);
        content.push('\n');
    }

    append_sources_section(&mut content, sources, rationale);
    Ok(content)
}

/// Rewrite an existing concept page's body while preserving its YAML
/// frontmatter and any `<!-- kb:begin/end -->` managed regions.
///
/// Strategy:
/// 1. Keep the frontmatter block verbatim.
/// 2. After the frontmatter, write the new body: `# <name>` heading + the
///    drafted definition + a "Sources" section listing the web URLs +
///    rationale.
/// 3. Re-append any managed regions that existed in the original page so
///    backlinks survive the rewrite untouched.
fn rewrite_body_preserving_frontmatter(
    existing: &str,
    canonical_name: &str,
    definition: &str,
    sources: &[kb_llm::ImputedWebSource],
    rationale: &str,
) -> String {
    use std::fmt::Write as _;

    let (frontmatter_block, rest) = split_frontmatter(existing);
    let managed_blocks = extract_managed_blocks(rest);

    let mut content = String::new();
    if let Some(fm) = frontmatter_block {
        content.push_str(fm);
        if !content.ends_with('\n') {
            content.push('\n');
        }
    }
    content.push('\n');
    let _ = writeln!(&mut content, "# {canonical_name}");
    let body_trimmed = definition.trim();
    if !body_trimmed.is_empty() {
        content.push('\n');
        content.push_str(body_trimmed);
        content.push('\n');
    }
    append_sources_section(&mut content, sources, rationale);

    for block in &managed_blocks {
        content.push('\n');
        content.push_str(block);
        if !content.ends_with('\n') {
            content.push('\n');
        }
    }

    content
}

fn append_sources_section(
    content: &mut String,
    sources: &[kb_llm::ImputedWebSource],
    rationale: &str,
) {
    use std::fmt::Write as _;

    if sources.is_empty() && rationale.trim().is_empty() {
        return;
    }
    content.push_str("\n## Sources (imputed)\n\n");
    if sources.is_empty() {
        content.push_str("- (no web sources cited)\n");
    } else {
        for src in sources {
            let title = src.title.trim();
            let url = src.url.trim();
            let note = src.note.trim();
            let title_display = if title.is_empty() { url } else { title };
            let note_suffix = if note.is_empty() {
                String::new()
            } else {
                format!(" — {note}")
            };
            let _ = writeln!(content, "- [{title_display}]({url}){note_suffix}");
        }
    }
    let trimmed_rationale = rationale.trim();
    if !trimmed_rationale.is_empty() {
        let _ = writeln!(content, "\n> Rationale: {trimmed_rationale}");
    }
}

/// Split an input markdown string into its frontmatter block (starting and
/// ending with `---`) and the remainder. Returns `(None, full)` when no
/// frontmatter delimiter is found at the top of the file.
fn split_frontmatter(input: &str) -> (Option<&str>, &str) {
    let stripped = input.strip_prefix("---\n");
    let Some(after_open) = stripped else {
        return (None, input);
    };
    let Some(close_idx) = after_open.find("\n---") else {
        return (None, input);
    };
    // Frontmatter block is: `---\n<body>\n---`. Keep the delimiters.
    let block_end_in_after = close_idx + "\n---".len();
    let block_end = "---\n".len() + block_end_in_after;
    let block = &input[..block_end];
    let rest_start = if input.as_bytes().get(block_end) == Some(&b'\n') {
        block_end + 1
    } else {
        block_end
    };
    let rest = &input[rest_start..];
    (Some(block), rest)
}

/// Extract every `<!-- kb:begin id=X -->...<!-- kb:end id=X -->` block from
/// `body`, preserving their text verbatim. Used to re-append managed
/// regions after a body rewrite.
fn extract_managed_blocks(body: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let lines: Vec<&str> = body.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        if trimmed.starts_with("<!-- kb:begin") {
            let start = i;
            let mut end = i;
            while end < lines.len() && !lines[end].trim_start().starts_with("<!-- kb:end") {
                end += 1;
            }
            if end < lines.len() {
                let block = lines[start..=end].join("\n");
                blocks.push(block);
                i = end + 1;
                continue;
            }
            break;
        }
        i += 1;
    }
    blocks
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use kb_core::{EntityMetadata, ReviewStatus, Status};
    use kb_llm::ImputedWebSource;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn payload(
        gap_kind: &str,
        page_path: PathBuf,
        concept_id: &str,
        concept_name: &str,
    ) -> ImputedFixPayload {
        ImputedFixPayload {
            gap_kind: gap_kind.to_string(),
            concept_name: concept_name.to_string(),
            concept_id: concept_id.to_string(),
            page_path,
            definition: "Widget is a general-purpose test concept used to validate the apply path.".to_string(),
            sources: vec![ImputedWebSource {
                url: "https://example.com/widget".to_string(),
                title: "Widget overview".to_string(),
                note: "Canonical reference.".to_string(),
            }],
            confidence: "medium".to_string(),
            rationale: "Chose the test variant.".to_string(),
            local_source_document_ids: vec!["src-alpha".to_string()],
        }
    }

    fn review_item(id: &str, page_path: PathBuf) -> ReviewItem {
        let payload_rel = PathBuf::from(format!("reviews/imputed_fixes/{id}.payload.json"));
        ReviewItem {
            metadata: EntityMetadata {
                id: id.to_string(),
                created_at_millis: 0,
                updated_at_millis: 0,
                source_hashes: vec![],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: vec![],
                output_paths: vec![payload_rel, page_path.clone()],
                status: Status::NeedsReview,
            },
            kind: ReviewKind::ImputedFix,
            target_entity_id: "concept:widget".to_string(),
            proposed_destination: Some(page_path.clone()),
            citations: vec![],
            affected_pages: vec![page_path],
            created_at_millis: 0,
            status: ReviewStatus::Pending,
            comment: String::new(),
        }
    }

    #[test]
    fn apply_missing_concept_writes_new_page() {
        let tmp = TempDir::new().unwrap();
        let page_rel = PathBuf::from("wiki/concepts/widget.md");
        let item = review_item("lint:imputed-fix:missing_concept:widget", page_rel.clone());
        let payload = payload(
            "missing_concept",
            page_rel.clone(),
            "concept:widget",
            "Widget",
        );
        kb_lint::save_imputed_fix_payload(tmp.path(), &item, &payload).unwrap();

        let applied = apply_imputed_fix(tmp.path(), &item).unwrap();
        assert!(applied.created_new_page);
        assert_eq!(applied.concept_path, page_rel);

        let written = fs::read_to_string(tmp.path().join(&page_rel)).unwrap();
        assert!(written.contains("id: concept:widget"));
        assert!(written.contains("origin: imputed"));
        assert!(written.contains("# Widget"));
        assert!(written.contains("## Sources (imputed)"));
        assert!(written.contains("https://example.com/widget"));
    }

    #[test]
    fn apply_missing_concept_refuses_to_overwrite() {
        let tmp = TempDir::new().unwrap();
        let concepts = tmp.path().join("wiki").join("concepts");
        fs::create_dir_all(&concepts).unwrap();
        let page_rel = PathBuf::from("wiki/concepts/widget.md");
        fs::write(tmp.path().join(&page_rel), "existing content").unwrap();

        let item = review_item("lint:imputed-fix:missing_concept:widget", page_rel.clone());
        let payload = payload(
            "missing_concept",
            page_rel,
            "concept:widget",
            "Widget",
        );
        kb_lint::save_imputed_fix_payload(tmp.path(), &item, &payload).unwrap();

        let err = apply_imputed_fix(tmp.path(), &item).unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn apply_thin_body_preserves_managed_regions() {
        let tmp = TempDir::new().unwrap();
        let concepts = tmp.path().join("wiki").join("concepts");
        fs::create_dir_all(&concepts).unwrap();
        let page_rel = PathBuf::from("wiki/concepts/widget.md");
        let original = "---\nid: concept:widget\nname: Widget\n---\n\n# Widget\n\nthin.\n\n<!-- kb:begin id=backlinks -->\n- [[other]]\n<!-- kb:end id=backlinks -->\n";
        fs::write(tmp.path().join(&page_rel), original).unwrap();

        let item = review_item(
            "lint:imputed-fix:thin_concept_body:widget",
            page_rel.clone(),
        );
        let payload = payload(
            "thin_concept_body",
            page_rel.clone(),
            "concept:widget",
            "Widget",
        );
        kb_lint::save_imputed_fix_payload(tmp.path(), &item, &payload).unwrap();

        let applied = apply_imputed_fix(tmp.path(), &item).unwrap();
        assert!(!applied.created_new_page);

        let written = fs::read_to_string(tmp.path().join(&page_rel)).unwrap();
        // Frontmatter preserved.
        assert!(written.starts_with("---\nid: concept:widget"));
        // New body replaced the thin content.
        assert!(written.contains("Widget is a general-purpose test concept"));
        // Managed region preserved verbatim.
        assert!(written.contains("<!-- kb:begin id=backlinks -->"));
        assert!(written.contains("- [[other]]"));
        assert!(written.contains("<!-- kb:end id=backlinks -->"));
    }

    #[test]
    fn apply_thin_body_errors_when_page_missing() {
        let tmp = TempDir::new().unwrap();
        let page_rel = PathBuf::from("wiki/concepts/widget.md");
        let item = review_item(
            "lint:imputed-fix:thin_concept_body:widget",
            page_rel.clone(),
        );
        let payload = payload(
            "thin_concept_body",
            page_rel,
            "concept:widget",
            "Widget",
        );
        kb_lint::save_imputed_fix_payload(tmp.path(), &item, &payload).unwrap();

        let err = apply_imputed_fix(tmp.path(), &item).unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }
}
