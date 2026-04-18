use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use kb_core::fs::atomic_write;
use kb_core::rewrite_managed_region;
use regex::Regex;

const WIKI_DIR: &str = "wiki";
const CONCEPT_DIR: &str = "wiki/concepts";
pub const BACKLINKS_REGION_ID: &str = "backlinks";
const BACKLINKS_HEADING: &str = "## Backlinks";

/// Updated markdown for one concept page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacklinksArtifact {
    /// Absolute path to the concept markdown file.
    pub path: PathBuf,
    /// Existing markdown from disk before update.
    pub existing_markdown: String,
    /// Markdown after backlink region rewrite/upsert.
    pub updated_markdown: String,
}

impl BacklinksArtifact {
    #[must_use]
    pub fn needs_update(&self) -> bool {
        self.existing_markdown != self.updated_markdown
    }
}

/// Regenerate backlink managed sections for every concept page.
///
/// The pass scans all wiki files for Obsidian-style links (`[[...]]`), builds a reverse
/// index of concept references, and updates each page under `wiki/concepts/` with a
/// `backlinks` managed region.
///
/// # Errors
///
/// Returns an error if any target wiki file cannot be walked or read.
pub fn run_backlinks_pass(root: &Path) -> Result<Vec<BacklinksArtifact>> {
    let concept_pages = discover_concept_pages(root)?;
    let mut backlinks = initial_backlink_map(&concept_pages);

    collect_backlinks(root, &mut backlinks)?;

    let mut artifacts = Vec::with_capacity(concept_pages.len());
    for (concept_id, path) in concept_pages {
        let existing_markdown = std::fs::read_to_string(&path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        let links = backlinks.remove(&concept_id).unwrap_or_default();
        let updated_markdown =
            upsert_backlinks_section(&existing_markdown, &render_backlinks_list(&links));

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

fn discover_concept_pages(root: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut pages = BTreeMap::new();
    for path in list_markdown_files(root, CONCEPT_DIR)? {
        let concept_id = page_id_from_path(root, &path)?;
        pages.insert(concept_id, path);
    }
    Ok(pages)
}

fn initial_backlink_map(
    concept_pages: &BTreeMap<String, PathBuf>,
) -> BTreeMap<String, BTreeSet<String>> {
    concept_pages
        .keys()
        .cloned()
        .map(|concept_id| (concept_id, BTreeSet::new()))
        .collect()
}

fn collect_backlinks(
    root: &Path,
    backlinks: &mut BTreeMap<String, BTreeSet<String>>,
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

            if let Some(referers) = backlinks.get_mut(&target_id) {
                referers.insert(source_id.clone());
            }
        }
    }

    Ok(())
}

fn render_backlinks_list(referers: &BTreeSet<String>) -> String {
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

fn upsert_backlinks_section(existing_body: &str, content: &str) -> String {
    if let Some(updated) = rewrite_managed_region(existing_body, BACKLINKS_REGION_ID, content) {
        return updated;
    }

    let mut body = existing_body.trim_end().to_string();
    if !body.is_empty() {
        body.push_str("\n\n");
    }
    body.push_str(BACKLINKS_HEADING);
    body.push('\n');
    body.push_str(&managed_region(BACKLINKS_REGION_ID, content));
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
}
