use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use kb_core::fs::atomic_write;
use serde::Deserialize;

const SOURCES_DIR: &str = "wiki/sources";
const CONCEPTS_DIR: &str = "wiki/concepts";
const QUESTIONS_DIR: &str = "wiki/questions";

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
    question: Option<String>,
}

/// Generate global and per-category index pages from the wiki directory.
///
/// Produces up to four index files:
/// - `wiki/index.md` — global index linking sources, concepts, and (when any
///   promoted question pages exist) questions
/// - `wiki/sources/index.md` — all source pages
/// - `wiki/concepts/index.md` — all concept pages
/// - `wiki/questions/index.md` — all promoted question pages (only emitted
///   when `wiki/questions/` holds at least one non-index markdown file, so
///   KBs without promoted questions stay tidy)
///
/// # Errors
///
/// Returns an error if wiki pages cannot be read or their frontmatter cannot be parsed.
pub fn generate_indexes(root: &Path) -> Result<Vec<IndexArtifact>> {
    let sources = discover_entries(root, SOURCES_DIR)?;
    let concepts = discover_entries(root, CONCEPTS_DIR)?;
    let questions = discover_entries(root, QUESTIONS_DIR)?;

    let mut artifacts = Vec::with_capacity(4);

    let global_path = root.join("wiki/index.md");
    let global_parent = global_path
        .parent()
        .context("wiki/index.md has no parent")?
        .to_path_buf();
    artifacts.push(IndexArtifact {
        content: render_global_index(&sources, &concepts, &questions, root, &global_parent),
        path: global_path,
    });

    if !sources.is_empty() {
        let path = root.join("wiki/sources/index.md");
        let parent = path
            .parent()
            .context("wiki/sources/index.md has no parent")?
            .to_path_buf();
        artifacts.push(IndexArtifact {
            content: render_category_index("Sources", &sources, root, &parent),
            path,
        });
    }

    if !concepts.is_empty() {
        let path = root.join("wiki/concepts/index.md");
        let parent = path
            .parent()
            .context("wiki/concepts/index.md has no parent")?
            .to_path_buf();
        artifacts.push(IndexArtifact {
            content: render_category_index("Concepts", &concepts, root, &parent),
            path,
        });
    }

    if !questions.is_empty() {
        let path = root.join("wiki/questions/index.md");
        let parent = path
            .parent()
            .context("wiki/questions/index.md has no parent")?
            .to_path_buf();
        artifacts.push(IndexArtifact {
            content: render_category_index("Questions", &questions, root, &parent),
            path,
        });
    }

    Ok(artifacts)
}

/// Compute a forward-slash markdown link target for an entry whose path is
/// stored relative to `root`, rewriting it to be relative to `index_parent`
/// so that markdown renderers (Obsidian, GitHub, VS Code preview) resolve it
/// correctly.
fn link_target(entry: &IndexEntry, root: &Path, index_parent: &Path) -> String {
    let abs_target = root.join(&entry.relative_path);
    let rel = relative_from(&abs_target, index_parent)
        .unwrap_or_else(|| PathBuf::from(&entry.relative_path));
    rel.to_string_lossy().replace('\\', "/")
}

/// Compute `target` expressed relative to `base`, walking up with `..` as
/// needed. Returns `None` if the two paths have different absolute/relative
/// kinds (so prefixes cannot be compared).
fn relative_from(target: &Path, base: &Path) -> Option<PathBuf> {
    if target.is_absolute() != base.is_absolute() {
        return None;
    }

    let mut t = target.components();
    let mut b = base.components();
    loop {
        match (t.clone().next(), b.clone().next()) {
            (Some(tc), Some(bc)) if tc == bc => {
                t.next();
                b.next();
            }
            _ => break,
        }
    }

    let mut rel = PathBuf::new();
    for _ in b {
        rel.push("..");
    }
    for c in t {
        rel.push(c.as_os_str());
    }
    if rel.as_os_str().is_empty() {
        rel.push(".");
    }
    Some(rel)
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
    fm.title
        .or(fm.question)
        .or(fm.name)
        .filter(|t| !t.is_empty())
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

fn render_global_index(
    sources: &[IndexEntry],
    concepts: &[IndexEntry],
    questions: &[IndexEntry],
    root: &Path,
    index_parent: &Path,
) -> String {
    let mut out = String::from("# Knowledge Base\n");

    out.push_str("\n## Sources\n\n");
    if sources.is_empty() {
        out.push_str("_No sources yet. Run `kb ingest` to add documents._\n");
    } else {
        write!(out, "{} source(s) indexed.\n\n", sources.len()).expect("infallible");
        for entry in sources {
            writeln!(
                out,
                "- [{}]({})",
                entry.title,
                link_target(entry, root, index_parent)
            )
            .expect("infallible");
        }
    }

    out.push_str("\n## Concepts\n\n");
    if concepts.is_empty() {
        out.push_str("_No concepts yet. Run `kb compile` after ingesting sources._\n");
    } else {
        write!(out, "{} concept(s) indexed.\n\n", concepts.len()).expect("infallible");
        for entry in concepts {
            writeln!(
                out,
                "- [{}]({})",
                entry.title,
                link_target(entry, root, index_parent)
            )
            .expect("infallible");
        }
    }

    // Questions section is suppressed entirely when no promoted question
    // pages exist, to keep freshly-initialized KBs uncluttered.
    if !questions.is_empty() {
        out.push_str("\n## Questions\n\n");
        write!(out, "{} question(s) indexed.\n\n", questions.len()).expect("infallible");
        for entry in questions {
            writeln!(
                out,
                "- [{}]({})",
                entry.title,
                link_target(entry, root, index_parent)
            )
            .expect("infallible");
        }
    }

    out
}

fn render_category_index(
    category: &str,
    entries: &[IndexEntry],
    root: &Path,
    index_parent: &Path,
) -> String {
    let mut out = String::new();
    writeln!(out, "# {category}\n").expect("infallible");
    writeln!(out, "{} page(s).\n", entries.len()).expect("infallible");
    for entry in entries {
        writeln!(
            out,
            "- [{}]({})",
            entry.title,
            link_target(entry, root, index_parent)
        )
        .expect("infallible");
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
        // No promoted questions → no Questions section.
        assert!(!global.content.contains("## Questions"));

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
        assert!(!global.content.contains("## Questions"));
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
    fn global_index_links_are_relative_to_wiki_dir() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/example.md"),
            "---\ntitle: Example\n---\n",
        );
        write_file(
            &root.path().join("wiki/concepts/foo.md"),
            "---\nname: Foo\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        let global = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/index.md"))
            .expect("global index");

        // Link targets must be relative to the index file's parent directory
        // (`wiki/`), not the KB root.
        assert!(
            global.content.contains("(sources/example.md)"),
            "expected file-relative source link; got:\n{}",
            global.content
        );
        assert!(
            global.content.contains("(concepts/foo.md)"),
            "expected file-relative concept link; got:\n{}",
            global.content
        );
        assert!(
            !global.content.contains("(wiki/sources/"),
            "global index must not contain wiki/-prefixed links"
        );
        assert!(
            !global.content.contains("(wiki/concepts/"),
            "global index must not contain wiki/-prefixed links"
        );
    }

    #[test]
    fn category_index_links_are_relative_to_category_dir() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/example.md"),
            "---\ntitle: Example\n---\n",
        );
        write_file(
            &root.path().join("wiki/concepts/foo.md"),
            "---\nname: Foo\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");

        let sources = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/sources/index.md"))
            .expect("sources index");
        assert!(
            sources.content.contains("(example.md)"),
            "expected bare filename in sources index; got:\n{}",
            sources.content
        );
        assert!(!sources.content.contains("(wiki/"));
        assert!(!sources.content.contains("(sources/"));

        let concepts = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/concepts/index.md"))
            .expect("concepts index");
        assert!(
            concepts.content.contains("(foo.md)"),
            "expected bare filename in concepts index; got:\n{}",
            concepts.content
        );
        assert!(!concepts.content.contains("(wiki/"));
        assert!(!concepts.content.contains("(concepts/"));
    }

    #[test]
    fn every_index_link_resolves_from_its_parent_dir() {
        // Integration-style check: build a wiki, generate + persist indexes,
        // parse each markdown link, and assert the target exists when joined
        // with the index file's parent directory.
        let root = tempdir().expect("tempdir");
        for name in ["alpha", "beta", "gamma"] {
            write_file(
                &root.path().join(format!("wiki/sources/src-{name}.md")),
                &format!("---\ntitle: {name}\n---\n"),
            );
        }
        for name in ["one", "two"] {
            write_file(
                &root.path().join(format!("wiki/concepts/{name}.md")),
                &format!("---\nname: {name}\n---\n"),
            );
        }
        for name in ["q-one", "q-two"] {
            write_file(
                &root.path().join(format!("wiki/questions/{name}.md")),
                &format!("---\ntitle: {name}\n---\n"),
            );
        }

        let artifacts = generate_indexes(root.path()).expect("generate");
        persist_index_artifacts(&artifacts).expect("persist");

        let link_re = regex::Regex::new(r"\]\(([^)]+)\)").expect("regex");
        for artifact in &artifacts {
            let parent = artifact.path.parent().expect("index has parent");
            let content = fs::read_to_string(&artifact.path).expect("read index");
            let mut link_count = 0usize;
            for cap in link_re.captures_iter(&content) {
                let target = &cap[1];
                link_count += 1;
                let resolved = parent.join(target);
                assert!(
                    resolved.exists(),
                    "link {target:?} in {:?} does not resolve ({resolved:?} missing)",
                    artifact.path
                );
            }
            assert!(
                link_count > 0,
                "expected at least one link in {:?}",
                artifact.path
            );
        }
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

    #[test]
    fn generate_indexes_emits_questions_section_and_index_page() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/doc.md"),
            "---\ntitle: Doc\n---\n",
        );
        write_file(
            &root.path().join("wiki/questions/what-is-rust.md"),
            "---\ntitle: What is Rust?\n---\n",
        );
        write_file(
            &root.path().join("wiki/questions/why-async.md"),
            "---\nquestion: Why async?\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");

        // Global index must list both questions with paths relative to wiki/.
        let global = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/index.md"))
            .expect("global index");
        assert!(
            global.content.contains("## Questions"),
            "expected Questions section; got:\n{}",
            global.content
        );
        assert!(global.content.contains("2 question(s)"));
        assert!(global.content.contains("[What is Rust?]"));
        assert!(global.content.contains("[Why async?]"));
        assert!(global.content.contains("(questions/what-is-rust.md)"));
        assert!(global.content.contains("(questions/why-async.md)"));

        // Per-category index must exist with file-relative links.
        let questions = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/questions/index.md"))
            .expect("questions index");
        assert!(questions.content.contains("# Questions"));
        assert!(questions.content.contains("[What is Rust?]"));
        assert!(questions.content.contains("(what-is-rust.md)"));
        assert!(!questions.content.contains("(wiki/"));
        assert!(!questions.content.contains("(questions/"));
    }

    #[test]
    fn generate_indexes_omits_questions_when_none_exist() {
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/sources/doc.md"),
            "---\ntitle: Doc\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        // No wiki/questions/index.md artifact when there are zero questions.
        assert!(
            !artifacts
                .iter()
                .any(|a| a.path.ends_with("wiki/questions/index.md")),
            "should not emit questions index when no promoted questions exist"
        );

        let global = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/index.md"))
            .expect("global index");
        assert!(
            !global.content.contains("## Questions"),
            "global index must not contain Questions section when empty"
        );
    }

    #[test]
    fn question_title_falls_back_to_question_field() {
        // Promoted question pages sometimes use `question:` instead of
        // `title:` — the index must surface whichever is present.
        let root = tempdir().expect("tempdir");
        write_file(
            &root.path().join("wiki/questions/slug.md"),
            "---\nquestion: How does caching work?\n---\n",
        );

        let artifacts = generate_indexes(root.path()).expect("generate");
        let global = artifacts
            .iter()
            .find(|a| a.path.ends_with("wiki/index.md"))
            .expect("global index");
        assert!(global.content.contains("[How does caching work?]"));
    }
}
