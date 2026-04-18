use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use kb_core::fs::atomic_write;
use serde::Deserialize;

const SOURCES_DIR: &str = "wiki/sources";
const CONCEPTS_DIR: &str = "wiki/concepts";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub title: String,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexArtifact {
    pub path: PathBuf,
    pub content: String,
}

#[derive(Deserialize)]
struct PageFrontmatter {
    title: Option<String>,
    name: Option<String>,
}

/// Generate global and per-category index pages from the wiki directory.
///
/// Produces up to three index files:
/// - `wiki/index.md` — global index linking sources and concepts
/// - `wiki/sources/index.md` — all source pages
/// - `wiki/concepts/index.md` — all concept pages
///
/// # Errors
///
/// Returns an error if wiki pages cannot be read or their frontmatter cannot be parsed.
pub fn generate_indexes(root: &Path) -> Result<Vec<IndexArtifact>> {
    let sources = discover_entries(root, SOURCES_DIR)?;
    let concepts = discover_entries(root, CONCEPTS_DIR)?;

    let mut artifacts = Vec::with_capacity(3);

    artifacts.push(IndexArtifact {
        path: root.join("wiki/index.md"),
        content: render_global_index(&sources, &concepts),
    });

    if !sources.is_empty() {
        artifacts.push(IndexArtifact {
            path: root.join("wiki/sources/index.md"),
            content: render_category_index("Sources", &sources),
        });
    }

    if !concepts.is_empty() {
        artifacts.push(IndexArtifact {
            path: root.join("wiki/concepts/index.md"),
            content: render_category_index("Concepts", &concepts),
        });
    }

    Ok(artifacts)
}

/// Persist index artifacts to disk using atomic writes.
///
/// # Errors
///
/// Returns an error if any file cannot be written.
pub fn persist_index_artifacts(artifacts: &[IndexArtifact]) -> Result<()> {
    for artifact in artifacts {
        atomic_write(&artifact.path, artifact.content.as_bytes())
            .with_context(|| format!("write index {}", artifact.path.display()))?;
    }
    Ok(())
}

fn discover_entries(root: &Path, relative_dir: &str) -> Result<Vec<IndexEntry>> {
    let dir = root.join(relative_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for result in WalkBuilder::new(&dir)
        .standard_filters(true)
        .require_git(false)
        .max_depth(Some(1))
        .build()
    {
        let entry = result.with_context(|| format!("walk {}", dir.display()))?;
        let path = entry.path();

        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        let is_markdown = path
            .extension()
            .is_some_and(|ext| ext == std::ffi::OsStr::new("md"));
        if !is_markdown {
            continue;
        }

        if path.file_name().is_some_and(|n| n == "index.md") {
            continue;
        }

        let relative_path = path
            .strip_prefix(root)
            .with_context(|| format!("{} not under root {}", path.display(), root.display()))?
            .to_string_lossy()
            .replace('\\', "/");

        let title = extract_title(path).unwrap_or_else(|| title_from_filename(path));

        entries.push(IndexEntry {
            title,
            relative_path,
        });
    }

    entries.sort_unstable_by(|a, b| a.title.cmp(&b.title));
    Ok(entries)
}

fn extract_title(path: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let yaml = extract_frontmatter_yaml(&raw)?;
    let fm: PageFrontmatter = serde_yaml::from_str(&yaml).ok()?;
    fm.title.or(fm.name).filter(|t| !t.is_empty())
}

fn extract_frontmatter_yaml(markdown: &str) -> Option<String> {
    let mut lines = markdown.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim_end() != "---" {
        return None;
    }

    let mut yaml = String::new();
    for line in lines {
        if line.trim_end() == "---" {
            return Some(yaml);
        }
        yaml.push_str(line);
    }
    None
}

fn title_from_filename(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().replace('-', " "))
        .unwrap_or_default()
}

fn render_global_index(sources: &[IndexEntry], concepts: &[IndexEntry]) -> String {
    let mut out = String::from("# Knowledge Base\n");

    out.push_str("\n## Sources\n\n");
    if sources.is_empty() {
        out.push_str("_No sources yet. Run `kb ingest` to add documents._\n");
    } else {
        write!(out, "{} source(s) indexed.\n\n", sources.len()).expect("infallible");
        for entry in sources {
            writeln!(out, "- [{}]({})", entry.title, entry.relative_path).expect("infallible");
        }
    }

    out.push_str("\n## Concepts\n\n");
    if concepts.is_empty() {
        out.push_str("_No concepts yet. Run `kb compile` after ingesting sources._\n");
    } else {
        write!(out, "{} concept(s) indexed.\n\n", concepts.len()).expect("infallible");
        for entry in concepts {
            writeln!(out, "- [{}]({})", entry.title, entry.relative_path).expect("infallible");
        }
    }

    out
}

fn render_category_index(category: &str, entries: &[IndexEntry]) -> String {
    let mut out = String::new();
    writeln!(out, "# {category}\n").expect("infallible");
    writeln!(out, "{} page(s).\n", entries.len()).expect("infallible");
    for entry in entries {
        writeln!(out, "- [{}]({})", entry.title, entry.relative_path).expect("infallible");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create dirs");
        }
        fs::write(path, content).expect("write file");
    }

    #[test]
    fn generate_indexes_creates_global_and_category_indexes() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/doc-a.md"),
            "---\ntitle: Document A\n---\n# Document A\n",
        );
        write_file(
            &root.path().join("wiki/sources/doc-b.md"),
            "---\ntitle: Document B\n---\n# Document B\n",
        );
        write_file(
            &root.path().join("wiki/concepts/rust.md"),
            "---\nname: Rust\n---\n# Rust\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        assert_eq!(artifacts.len(), 3);

        let global = &artifacts[0];
        assert!(global.path.ends_with("wiki/index.md"));
        assert!(global.content.contains("# Knowledge Base"));
        assert!(global.content.contains("[Document A]"));
        assert!(global.content.contains("[Document B]"));
        assert!(global.content.contains("[Rust]"));
        assert!(global.content.contains("2 source(s)"));
        assert!(global.content.contains("1 concept(s)"));

        let sources = &artifacts[1];
        assert!(sources.path.ends_with("wiki/sources/index.md"));
        assert!(sources.content.contains("# Sources"));
        assert!(sources.content.contains("[Document A]"));

        let concepts = &artifacts[2];
        assert!(concepts.path.ends_with("wiki/concepts/index.md"));
        assert!(concepts.content.contains("# Concepts"));
        assert!(concepts.content.contains("[Rust]"));
    }

    #[test]
    fn generate_indexes_empty_wiki_produces_placeholders() {
        let root = tempdir().expect("tempdir");

        let artifacts = generate_indexes(root.path()).expect("generate");
        assert_eq!(artifacts.len(), 1);

        let global = &artifacts[0];
        assert!(global.content.contains("_No sources yet"));
        assert!(global.content.contains("_No concepts yet"));
    }

    #[test]
    fn generate_indexes_skips_existing_index_files() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/doc-a.md"),
            "---\ntitle: Doc A\n---\n",
        );
        write_file(
            &root.path().join("wiki/sources/index.md"),
            "old index content",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        let sources_idx = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/index.md"))
            .expect("sources index");
        assert!(sources_idx.content.contains("[Doc A]"));
        assert!(!sources_idx.content.contains("old index"));
    }

    #[test]
    fn fallback_title_from_filename() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/my-document.md"),
            "No frontmatter here.\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        let global = &artifacts[0];
        assert!(global.content.contains("[my document]"));
    }

    #[test]
    fn links_use_relative_paths() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/example.md"),
            "---\ntitle: Example\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        let global = &artifacts[0];
        assert!(global.content.contains("(wiki/sources/example.md)"));
    }

    #[test]
    fn persist_index_artifacts_writes_files() {
        let root = tempdir().expect("tempdir");
        let artifacts = vec![IndexArtifact {
            path: root.path().join("wiki/index.md"),
            content: "# Test\n".to_string(),
        }];

        persist_index_artifacts(&artifacts).expect("persist");
        let content = fs::read_to_string(root.path().join("wiki/index.md")).expect("read");
        assert_eq!(content, "# Test\n");
    }

    #[test]
    fn entries_sorted_alphabetically() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/zebra.md"),
            "---\ntitle: Zebra\n---\n",
        );
        write_file(
            &root.path().join("wiki/sources/apple.md"),
            "---\ntitle: Apple\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        let global = &artifacts[0];
        let apple_pos = global.content.find("[Apple]").expect("Apple found");
        let zebra_pos = global.content.find("[Zebra]").expect("Zebra found");
        assert!(apple_pos < zebra_pos, "Apple should appear before Zebra");
    }
}
