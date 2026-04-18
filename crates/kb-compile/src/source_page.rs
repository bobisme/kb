use std::path::PathBuf;

use anyhow::Result;
use kb_core::rewrite_managed_region;
use serde_yaml::{Mapping, Number, Value};

const TITLE_REGION_ID: &str = "title";
const SUMMARY_REGION_ID: &str = "summary";
const KEY_TOPICS_REGION_ID: &str = "key_topics";
const CITATIONS_REGION_ID: &str = "citations";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePageInput<'a> {
    pub page_id: &'a str,
    pub title: &'a str,
    pub source_document_id: &'a str,
    pub source_revision_id: &'a str,
    pub generated_at: u64,
    pub build_record_id: &'a str,
    pub summary: &'a str,
    pub key_topics: &'a [String],
    pub citations: &'a [String],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePageArtifact {
    pub path: PathBuf,
    pub frontmatter: Mapping,
    pub body: String,
    pub markdown: String,
}

#[must_use]
pub fn source_page_path_for_id(source_id: &str) -> PathBuf {
    PathBuf::from("wiki/sources").join(format!("{}.md", slug_for_path(source_id)))
}

/// Render or update a source wiki page with managed regions and source metadata.
///
/// # Errors
///
/// Returns an error if the existing markdown contains an unterminated frontmatter block
/// or if the generated frontmatter cannot be serialized to YAML.
pub fn render_source_page(
    input: &SourcePageInput<'_>,
    existing_markdown: Option<&str>,
) -> Result<SourcePageArtifact> {
    let frontmatter = build_frontmatter(input);
    let existing_body = existing_markdown
        .map(split_frontmatter)
        .transpose()?
        .flatten()
        .map_or_else(String::new, |(_, body)| body);
    let body = upsert_source_page_body(&existing_body, input);
    let markdown = serialize_frontmatter(&frontmatter, &body)?;

    Ok(SourcePageArtifact {
        path: source_page_path_for_id(input.source_document_id),
        frontmatter,
        body,
        markdown,
    })
}

fn build_frontmatter(input: &SourcePageInput<'_>) -> Mapping {
    let mut frontmatter = Mapping::new();
    frontmatter.insert(
        Value::String("id".into()),
        Value::String(input.page_id.into()),
    );
    frontmatter.insert(Value::String("type".into()), Value::String("source".into()));
    frontmatter.insert(
        Value::String("title".into()),
        Value::String(input.title.into()),
    );
    frontmatter.insert(
        Value::String("source_document_id".into()),
        Value::String(input.source_document_id.into()),
    );
    frontmatter.insert(
        Value::String("source_revision_id".into()),
        Value::String(input.source_revision_id.into()),
    );
    frontmatter.insert(
        Value::String("generated_at".into()),
        Value::Number(Number::from(input.generated_at)),
    );
    frontmatter.insert(
        Value::String("build_record_id".into()),
        Value::String(input.build_record_id.into()),
    );
    frontmatter
}

fn upsert_source_page_body(existing_body: &str, input: &SourcePageInput<'_>) -> String {
    let mut body = existing_body.to_string();
    body = upsert_section(
        &body,
        "# Source",
        TITLE_REGION_ID,
        &format!("\n{}\n", input.title.trim()),
    );
    body = upsert_section(
        &body,
        "## Summary",
        SUMMARY_REGION_ID,
        &format!("\n{}\n", input.summary.trim()),
    );
    body = upsert_section(
        &body,
        "## Key topics",
        KEY_TOPICS_REGION_ID,
        &render_bullet_list(input.key_topics),
    );
    upsert_section(
        &body,
        "## Citations",
        CITATIONS_REGION_ID,
        &render_bullet_list(input.citations),
    )
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

fn render_bullet_list(items: &[String]) -> String {
    let mut rendered = String::from("\n");
    if items.is_empty() {
        rendered.push_str("- _None yet._\n");
        return rendered;
    }

    for item in items {
        rendered.push_str("- ");
        rendered.push_str(item.trim());
        rendered.push('\n');
    }
    rendered
}

fn serialize_frontmatter(frontmatter: &Mapping, body: &str) -> Result<String> {
    let yaml = serde_yaml::to_string(frontmatter)?;
    Ok(format!("---\n{yaml}---\n{body}"))
}

fn split_frontmatter(markdown: &str) -> Result<Option<(String, String)>> {
    let mut lines = markdown.split_inclusive('\n');
    let Some(first_line) = lines.next() else {
        return Ok(None);
    };

    if first_line != "---\n" && first_line != "---\r\n" && first_line != "---" {
        return Ok(None);
    }

    let mut frontmatter_text = String::new();
    let mut offset = first_line.len();

    for line in lines {
        if line == "---\n" || line == "---\r\n" || line == "---" {
            let body = markdown[offset + line.len()..].to_string();
            return Ok(Some((frontmatter_text, body)));
        }

        frontmatter_text.push_str(line);
        offset += line.len();
    }

    anyhow::bail!("frontmatter block was not terminated with a closing --- line")
}

fn slug_for_path(value: &str) -> String {
    let mut slug = String::new();
    let mut last_was_hyphen = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }

    let trimmed = slug.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "source".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(key_topics: &'a [String], citations: &'a [String]) -> SourcePageInput<'a> {
        SourcePageInput {
            page_id: "wiki-source-1",
            title: "Example Source",
            source_document_id: "Source Document/1",
            source_revision_id: "source-revision-1",
            generated_at: 1_700_000_000,
            build_record_id: "build-1",
            summary: "A concise summary.",
            key_topics,
            citations,
        }
    }

    #[test]
    fn render_source_page_emits_frontmatter_and_managed_regions() {
        let key_topics = vec!["topic one".to_string(), "topic two".to_string()];
        let citations = vec!["[[wiki/concepts/rust]]".to_string()];
        let artifact = render_source_page(&input(&key_topics, &citations), None).expect("render");

        assert_eq!(
            artifact.path,
            PathBuf::from("wiki/sources/source-document-1.md")
        );
        assert_eq!(
            artifact.frontmatter.get("id"),
            Some(&Value::String("wiki-source-1".into()))
        );
        assert_eq!(
            artifact.frontmatter.get("type"),
            Some(&Value::String("source".into()))
        );
        assert_eq!(
            artifact.frontmatter.get("source_document_id"),
            Some(&Value::String("Source Document/1".into()))
        );
        assert_eq!(
            artifact.frontmatter.get("source_revision_id"),
            Some(&Value::String("source-revision-1".into()))
        );
        assert!(artifact.markdown.contains("<!-- kb:begin id=title -->"));
        assert!(artifact.markdown.contains("<!-- kb:begin id=summary -->"));
        assert!(
            artifact
                .markdown
                .contains("<!-- kb:begin id=key_topics -->")
        );
        assert!(artifact.markdown.contains("<!-- kb:begin id=citations -->"));
        assert!(artifact.markdown.contains("A concise summary."));
        assert!(artifact.markdown.contains("- topic one"));
        assert!(artifact.markdown.contains("- [[wiki/concepts/rust]]"));
    }

    #[test]
    fn render_source_page_rewrites_regions_in_place_without_clobbering_manual_content() {
        let existing = "---\nid: old\n---\n# Source\n<!-- kb:begin id=title -->\nOld Title\n<!-- kb:end id=title -->\n\n## Summary\n<!-- kb:begin id=summary -->\nOld summary.\n<!-- kb:end id=summary -->\n\n## Notes\nKeep me.\n\n## Key topics\n<!-- kb:begin id=key_topics -->\n- old topic\n<!-- kb:end id=key_topics -->\n\n## Citations\n<!-- kb:begin id=citations -->\n- old citation\n<!-- kb:end id=citations -->\n";
        let key_topics = vec!["fresh topic".to_string()];
        let citations = vec!["[[wiki/concepts/new]]".to_string()];
        let artifact =
            render_source_page(&input(&key_topics, &citations), Some(existing)).expect("render");

        assert!(artifact.body.contains("Example Source"));
        assert!(artifact.body.contains("A concise summary."));
        assert!(artifact.body.contains("- fresh topic"));
        assert!(artifact.body.contains("- [[wiki/concepts/new]]"));
        assert!(artifact.body.contains("## Notes\nKeep me."));
        assert!(!artifact.body.contains("Old Title"));
        assert!(!artifact.body.contains("Old summary."));
        assert!(!artifact.body.contains("old topic"));
        assert!(!artifact.body.contains("old citation"));
    }

    #[test]
    fn render_source_page_uses_empty_placeholder_lists() {
        let artifact = render_source_page(&input(&[], &[]), None).expect("render");
        assert!(artifact.body.contains("- _None yet._"));
    }
}
