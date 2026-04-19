use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use kb_core::fs::atomic_write;
use kb_core::rewrite_managed_region;
use regex::Regex;
use serde_yaml::Value;

const WIKI_DIR: &str = "wiki";
const CONCEPT_DIR: &str = "wiki/concepts";
const SOURCE_DIR: &str = "wiki/sources";
const NORMALIZED_DIR: &str = "normalized";

pub const BACKLINKS_REGION_ID: &str = "backlinks";
pub const REFERENCED_BY_CONCEPTS_REGION_ID: &str = "referenced_by_concepts";

const BACKLINKS_HEADING: &str = "## Backlinks";
const REFERENCED_BY_CONCEPTS_HEADING: &str = "## Referenced by concepts";

/// Updated markdown for one wiki page touched by the backlinks pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacklinksArtifact {
    /// Absolute path to the markdown file.
    pub path: PathBuf,
    /// Existing markdown from disk before update.
    pub existing_markdown: String,
    /// Markdown after managed region rewrite/upsert.
    pub updated_markdown: String,
}

impl BacklinksArtifact {
    #[must_use]
    pub fn needs_update(&self) -> bool {
        self.existing_markdown != self.updated_markdown
    }
}

/// Regenerate managed backlink regions across concept and source pages.
///
/// The pass performs two scans and then one rewrite phase:
///
/// 1. Scan every wiki page for Obsidian-style `[[...]]` wiki-links to concept pages
///    (legacy behavior — captures manual references and any auto-emitted wiki-links).
/// 2. Scan every concept page's YAML frontmatter for `sources: [{ heading_anchor, quote }]`
///    entries. Each `heading_anchor` is resolved against `normalized/<src>/metadata.json`
///    to recover the contributing `source_document_id`, which maps to a `wiki/sources/<src>.md`
///    page slug.
///
/// Concept pages receive a `backlinks` region listing referring wiki pages (both
/// wiki-link and frontmatter-source-backed references). Source pages receive a
/// `referenced_by_concepts` region listing concept pages that cited them.
///
/// # Errors
///
/// Returns an error if any target wiki file cannot be walked, read, or parsed.
pub fn run_backlinks_pass(root: &Path) -> Result<Vec<BacklinksArtifact>> {
    let concept_pages = discover_wiki_pages(root, CONCEPT_DIR)?;
    let source_pages = discover_wiki_pages(root, SOURCE_DIR)?;

    // anchor -> list of source_document_ids that own that heading (via normalized metadata)
    let anchor_to_source_docs = build_anchor_to_source_docs(root)?;

    let mut concept_backlinks: BTreeMap<String, BTreeSet<String>> = concept_pages
        .keys()
        .cloned()
        .map(|id| (id, BTreeSet::new()))
        .collect();
    let mut source_referenced_by: BTreeMap<String, BTreeSet<String>> = source_pages
        .keys()
        .cloned()
        .map(|id| (id, BTreeSet::new()))
        .collect();

    collect_wiki_link_backlinks(root, &mut concept_backlinks)?;
    collect_frontmatter_source_backlinks(
        &concept_pages,
        &source_pages,
        &anchor_to_source_docs,
        &mut concept_backlinks,
        &mut source_referenced_by,
    )?;

    let mut artifacts = Vec::with_capacity(concept_pages.len() + source_pages.len());

    for (concept_id, path) in concept_pages {
        let existing_markdown = std::fs::read_to_string(&path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        let links = concept_backlinks.remove(&concept_id).unwrap_or_default();
        let updated_markdown = upsert_section(
            &existing_markdown,
            BACKLINKS_HEADING,
            BACKLINKS_REGION_ID,
            &render_link_list(&links),
        );
        artifacts.push(BacklinksArtifact {
            path,
            existing_markdown,
            updated_markdown,
        });
    }

    for (source_id, path) in source_pages {
        let existing_markdown = std::fs::read_to_string(&path)
            .with_context(|| format!("read source page {}", path.display()))?;
        let links = source_referenced_by.remove(&source_id).unwrap_or_default();
        let updated_markdown = upsert_section(
            &existing_markdown,
            REFERENCED_BY_CONCEPTS_HEADING,
            REFERENCED_BY_CONCEPTS_REGION_ID,
            &render_link_list(&links),
        );
        artifacts.push(BacklinksArtifact {
            path,
            existing_markdown,
            updated_markdown,
        });
    }

    Ok(artifacts)
}

/// Persist backlink artifacts to disk using atomic writes.
///
/// # Errors
/// Returns an error if any atomic write fails.
pub fn persist_backlinks_artifacts(artifacts: &[BacklinksArtifact]) -> Result<()> {
    for artifact in artifacts {
        atomic_write(&artifact.path, artifact.updated_markdown.as_bytes())
            .with_context(|| format!("write {}", artifact.path.display()))?;
    }
    Ok(())
}

fn discover_wiki_pages(root: &Path, relative_dir: &str) -> Result<BTreeMap<String, PathBuf>> {
    let mut pages = BTreeMap::new();
    for path in list_markdown_files(root, relative_dir)? {
        let page_id = page_id_from_path(root, &path)?;
        pages.insert(page_id, path);
    }
    Ok(pages)
}

/// Walk `normalized/<source_document_id>/metadata.json` to build a reverse index
/// from every `heading_id` to the set of `source_document_id`s that declared it.
///
/// We use a Vec rather than a single value because two independent sources may
/// share a heading like "summary"; the caller picks the intersection with known
/// source pages.
fn build_anchor_to_source_docs(root: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let normalized_root = root.join(NORMALIZED_DIR);
    if !normalized_root.exists() {
        return Ok(map);
    }

    let entries = std::fs::read_dir(&normalized_root)
        .with_context(|| format!("read normalized dir {}", normalized_root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("walk {}", normalized_root.display()))?;
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let source_doc_id = entry.file_name().to_string_lossy().into_owned();
        let metadata_path = entry.path().join("metadata.json");
        if !metadata_path.is_file() {
            continue;
        }
        let raw = std::fs::read_to_string(&metadata_path)
            .with_context(|| format!("read {}", metadata_path.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", metadata_path.display()))?;
        let Some(headings) = parsed.get("heading_ids").and_then(|v| v.as_array()) else {
            continue;
        };
        for heading in headings {
            if let Some(h) = heading.as_str() {
                map.entry(h.to_string())
                    .or_default()
                    .push(source_doc_id.clone());
            }
        }
    }

    for docs in map.values_mut() {
        docs.sort();
        docs.dedup();
    }
    Ok(map)
}

fn collect_wiki_link_backlinks(
    root: &Path,
    concept_backlinks: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    let link_re = Regex::new(r"\[\[([^\]\r\n]+)\]\]").context("compile wikilink regex")?;

    for page in list_markdown_files(root, WIKI_DIR)? {
        let source_id = page_id_from_path(root, &page)?;
        let markdown = std::fs::read_to_string(&page)
            .with_context(|| format!("read wiki page {}", page.display()))?;

        for capture in link_re.captures_iter(&markdown) {
            let Some(raw_link) = capture.get(1).map(|c| c.as_str()) else {
                continue;
            };
            let Some(target_id) = normalize_wiki_link(raw_link) else {
                continue;
            };
            if let Some(referers) = concept_backlinks.get_mut(&target_id) {
                referers.insert(source_id.clone());
            }
        }
    }

    Ok(())
}

/// For each concept page, parse its YAML frontmatter `sources:` block, resolve
/// each `heading_anchor` to contributing source pages via normalized metadata,
/// and credit both sides of the relation.
fn collect_frontmatter_source_backlinks(
    concept_pages: &BTreeMap<String, PathBuf>,
    source_pages: &BTreeMap<String, PathBuf>,
    anchor_to_source_docs: &BTreeMap<String, Vec<String>>,
    concept_backlinks: &mut BTreeMap<String, BTreeSet<String>>,
    source_referenced_by: &mut BTreeMap<String, BTreeSet<String>>,
) -> Result<()> {
    // Build a source_document_id -> wiki/sources/<slug> map by reading source page frontmatter.
    let mut source_doc_to_page: BTreeMap<String, String> = BTreeMap::new();
    for (page_id, path) in source_pages {
        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("read source page {}", path.display()))?;
        let Some((fm, _body)) = split_frontmatter(&markdown) else {
            continue;
        };
        let Ok(parsed) = serde_yaml::from_str::<Value>(&fm) else {
            continue;
        };
        let Some(doc_id) = parsed
            .get("source_document_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        source_doc_to_page.insert(doc_id, page_id.clone());
    }

    for (concept_id, path) in concept_pages {
        let markdown = std::fs::read_to_string(path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        let Some((fm, _body)) = split_frontmatter(&markdown) else {
            continue;
        };
        let Ok(parsed) = serde_yaml::from_str::<Value>(&fm) else {
            continue;
        };
        let Some(sources) = parsed.get("sources").and_then(|v| v.as_sequence()) else {
            continue;
        };
        for entry in sources {
            let Some(anchor) = entry.get("heading_anchor").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(candidate_docs) = anchor_to_source_docs.get(anchor) else {
                continue;
            };
            for doc_id in candidate_docs {
                let Some(source_page_id) = source_doc_to_page.get(doc_id) else {
                    continue;
                };
                if let Some(referers) = concept_backlinks.get_mut(concept_id) {
                    referers.insert(source_page_id.clone());
                }
                if let Some(referers) = source_referenced_by.get_mut(source_page_id) {
                    referers.insert(concept_id.clone());
                }
            }
        }
    }

    Ok(())
}

/// Split YAML frontmatter from a markdown document. Returns `(frontmatter_yaml, body)`.
fn split_frontmatter(markdown: &str) -> Option<(String, String)> {
    let mut lines = markdown.split_inclusive('\n');
    let first = lines.next()?;
    if first != "---\n" && first != "---\r\n" && first != "---" {
        return None;
    }
    let mut fm = String::new();
    let mut offset = first.len();
    for line in lines {
        if line == "---\n" || line == "---\r\n" || line == "---" {
            let body = markdown[offset + line.len()..].to_string();
            return Some((fm, body));
        }
        fm.push_str(line);
        offset += line.len();
    }
    None
}

fn render_link_list(referers: &BTreeSet<String>) -> String {
    let mut content = String::from("\n");
    if referers.is_empty() {
        content.push_str("- _None yet._\n");
        return content;
    }
    for referer in referers {
        content.push_str("- [[");
        content.push_str(referer);
        content.push_str("]]\n");
    }
    content
}

fn upsert_section(
    existing_body: &str,
    heading: &str,
    region_id: &str,
    content: &str,
) -> String {
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

fn normalize_wiki_link(raw: &str) -> Option<String> {
    let without_alias = raw.split('|').next()?.trim();
    if without_alias.is_empty() {
        return None;
    }

    let without_anchor = without_alias
        .split('#')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;

    if without_anchor.starts_with("http://")
        || without_anchor.starts_with("https://")
        || without_anchor.starts_with("mailto:")
        || without_anchor.starts_with('#')
    {
        return None;
    }

    let mut target = without_anchor.trim_start_matches("./");
    target = target.trim_start_matches('/');
    if std::path::Path::new(target)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
    {
        target = target.trim_end_matches(".md");
    }
    if target.ends_with('/') {
        target = target.trim_end_matches('/');
    }

    if target.starts_with("wiki/concepts/") {
        Some(target.to_string())
    } else {
        None
    }
}

fn page_id_from_path(root: &Path, page: &Path) -> Result<String> {
    let relative = page
        .strip_prefix(root)
        .with_context(|| format!("{} is not under root {}", page.display(), root.display()))?;
    Ok(relative
        .to_string_lossy()
        .replace('\\', "/")
        .trim_end_matches(".md")
        .to_string())
}

fn list_markdown_files(root: &Path, relative_dir: &str) -> Result<Vec<PathBuf>> {
    let root_dir = root.join(relative_dir);
    if !root_dir.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for entry in WalkBuilder::new(&root_dir)
        .standard_filters(true)
        .require_git(false)
        .build()
    {
        let entry = entry.with_context(|| format!("walk wiki files in {}", root_dir.display()))?;

        let is_markdown = entry
            .path()
            .extension()
            .is_some_and(|ext| ext == std::ffi::OsStr::new("md"));

        if entry.file_type().is_some_and(|kind| kind.is_file()) && is_markdown {
            files.push(entry.into_path());
        }
    }

    files.sort_unstable();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &Path, markdown: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent directory");
        }
        fs::write(path, markdown).expect("write markdown fixture");
    }

    #[test]
    fn normalize_wiki_link_removes_aliases_and_anchors() {
        assert_eq!(
            normalize_wiki_link("wiki/concepts/rust"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(
            normalize_wiki_link("wiki/concepts/rust.md#section"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(
            normalize_wiki_link("wiki/concepts/rust|Rust language"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(
            normalize_wiki_link("./wiki/concepts/rust.md|Rust"),
            Some("wiki/concepts/rust".to_string())
        );
        assert_eq!(normalize_wiki_link("wiki/sources/page"), None);
        assert_eq!(normalize_wiki_link("https://example.com"), None);
    }

    #[test]
    fn run_backlinks_pass_adds_and_updates_backlinks_region() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/concepts/rust.md"),
            "# Rust\n\nOld intro for rust.\n",
        );
        write(
            &root.path().join("wiki/concepts/borrow-checker.md"),
            "# Borrow checker\n\nUses [[wiki/concepts/rust#intro|Rust]].\n",
        );
        write(
            &root.path().join("wiki/sources/page-a.md"),
            "# Page A\n\nReferences rust: [[wiki/concepts/rust|Rust]] and [[wiki/sources/other]].\n",
        );
        write(
            &root.path().join("wiki/sources/page-b.md"),
            "# Page B\n\nMentions [[wiki/concepts/rust#summary]].\n",
        );

        let mut artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        artifacts.sort_by(|a, b| a.path.cmp(&b.path));

        let rust = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/rust.md"))
            .expect("rust concept exists");

        assert!(rust.updated_markdown.contains("## Backlinks"));
        assert!(
            rust.updated_markdown
                .contains("- [[wiki/concepts/borrow-checker]]")
        );
        assert!(rust.updated_markdown.contains("- [[wiki/sources/page-a]]"));
        assert!(rust.updated_markdown.contains("- [[wiki/sources/page-b]]"));
        assert!(!rust.updated_markdown.contains("- _None yet._"));
        assert!(rust.needs_update());

        let borrow = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/borrow-checker.md"))
            .expect("borrow concept exists");
        assert!(borrow.updated_markdown.contains("- _None yet._"));
    }

    #[test]
    fn run_backlinks_pass_preserves_manual_content() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/concepts/rust.md"),
            "# Rust\n\n\nNotes section\n\n## Backlinks\n<!-- kb:begin id=backlinks -->\n- old-link\n<!-- kb:end id=backlinks -->\n",
        );
        write(
            &root.path().join("wiki/sources/page-a.md"),
            "# Page A\n\n[[wiki/concepts/rust]].\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let rust = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/rust.md"))
            .expect("rust concept exists");

        assert!(!rust.updated_markdown.contains("- old-link"));
        assert!(rust.updated_markdown.contains("- [[wiki/sources/page-a]]"));
    }

    #[test]
    fn no_references_renders_placeholder() {
        let root = tempdir().expect("tempdir");

        write(
            &root.path().join("wiki/concepts/isolated.md"),
            "# Isolated\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");
        let isolated = artifacts
            .iter()
            .find(|artifact| artifact.path.ends_with("wiki/concepts/isolated.md"))
            .expect("isolated concept exists");

        assert!(isolated.updated_markdown.contains("- _None yet._"));
    }

    #[test]
    fn frontmatter_sources_drive_cross_references_between_concepts_and_sources() {
        let root = tempdir().expect("tempdir");

        // normalized metadata: anchor "python-gil" belongs to source document "src-abc"
        write(
            &root.path().join("normalized/src-abc/metadata.json"),
            r#"{
                "metadata": {"id": "src-abc"},
                "source_revision_id": "rev-123",
                "heading_ids": ["python-gil", "alternatives"]
            }"#,
        );

        // source page pointing at src-abc / rev-123 (no wiki-links anywhere)
        write(
            &root.path().join("wiki/sources/src-abc.md"),
            "---\nid: wiki-source-src-abc\ntype: source\ntitle: Python GIL\nsource_document_id: src-abc\nsource_revision_id: rev-123\n---\n# Source\n\n## Citations\n- rev-123#python-gil\n",
        );

        // concept page whose frontmatter sources: references the anchor above
        write(
            &root.path().join("wiki/concepts/global-interpreter-lock.md"),
            "---\nid: concept:global-interpreter-lock\nname: Global Interpreter Lock\nsources:\n- heading_anchor: python-gil\n  quote: The Python GIL is a mutex\n- heading_anchor: alternatives\n  quote: async/await\n---\n\n# Global Interpreter Lock\n",
        );

        let artifacts = run_backlinks_pass(root.path()).expect("run backlink pass");

        let concept = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/global-interpreter-lock.md"))
            .expect("concept artifact");
        assert!(concept.updated_markdown.contains("## Backlinks"));
        assert!(
            concept
                .updated_markdown
                .contains("- [[wiki/sources/src-abc]]"),
            "concept should backlink to source page via frontmatter sources: resolution, got:\n{}",
            concept.updated_markdown
        );
        assert!(!concept.updated_markdown.contains("- _None yet._"));

        let source = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/src-abc.md"))
            .expect("source artifact");
        assert!(
            source
                .updated_markdown
                .contains("## Referenced by concepts")
        );
        assert!(source.updated_markdown.contains(
            "<!-- kb:begin id=referenced_by_concepts -->"
        ));
        assert!(
            source
                .updated_markdown
                .contains("- [[wiki/concepts/global-interpreter-lock]]"),
            "source should list referencing concept, got:\n{}",
            source.updated_markdown
        );
    }
}
