use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::emit_json;

/// One node in the directory listing.
#[derive(Debug, Serialize)]
pub struct LsEntry {
    /// Path relative to the KB root, with `/` separators.
    path: String,
    /// `"dir"` or `"file"`.
    kind: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<Self>,
}

#[derive(Debug, Serialize)]
struct LsPayload {
    root: String,
    entries: Vec<LsEntry>,
}

pub fn run_ls(root: &Path, json: bool) -> Result<()> {
    let entries = collect_entries(root, root)?;

    if json {
        let payload = LsPayload {
            root: root.display().to_string(),
            entries,
        };
        return emit_json("ls", payload);
    }

    println!("{}", root.display());
    print_tree(&entries, "");
    Ok(())
}

/// Walk `dir` and collect non-hidden entries, sorted directories-first then
/// alphabetically. Recurses into directories. Symlinks and special files are
/// skipped because the KB tree shouldn't contain them; if it does, treating
/// them as opaque (omitted) is safer than chasing a cycle.
fn collect_entries(root: &Path, dir: &Path) -> Result<Vec<LsEntry>> {
    let mut raw: Vec<(PathBuf, bool)> = Vec::new();

    let read = match fs::read_dir(dir) {
        Ok(read) => read,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(err).with_context(|| format!("reading {}", dir.display()));
        }
    };

    for entry in read {
        let entry = entry.with_context(|| format!("reading {}", dir.display()))?;
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if name_str.starts_with('.') {
            continue;
        }
        let file_type = entry.file_type().with_context(|| {
            format!("statting {}", entry.path().display())
        })?;
        if file_type.is_dir() {
            raw.push((entry.path(), true));
        } else if file_type.is_file() {
            raw.push((entry.path(), false));
        }
        // Symlinks and other types intentionally ignored.
    }

    raw.sort_by(|a, b| match (a.1, b.1) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.0.file_name().cmp(&b.0.file_name()),
    });

    let mut out = Vec::with_capacity(raw.len());
    for (path, is_dir) in raw {
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if is_dir {
            let children = collect_entries(root, &path)?;
            out.push(LsEntry {
                path: rel,
                kind: "dir",
                children,
            });
        } else {
            out.push(LsEntry {
                path: rel,
                kind: "file",
                children: Vec::new(),
            });
        }
    }
    Ok(out)
}

fn print_tree(entries: &[LsEntry], prefix: &str) {
    let total = entries.len();
    for (idx, entry) in entries.iter().enumerate() {
        let last = idx + 1 == total;
        let branch = if last { "└── " } else { "├── " };
        let name = leaf_name(&entry.path);
        let suffix = if entry.kind == "dir" { "/" } else { "" };
        println!("{prefix}{branch}{name}{suffix}");
        if entry.kind == "dir" && !entry.children.is_empty() {
            let child_prefix = format!("{prefix}{}", if last { "    " } else { "│   " });
            print_tree(&entry.children, &child_prefix);
        }
    }
}

fn leaf_name(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn skips_dot_entries_at_every_level() {
        let tmp = tempdir().expect("temp root");
        let root = tmp.path();
        fs::write(root.join("kb.toml"), "").expect("write kb.toml");
        fs::create_dir_all(root.join(".kb/state")).expect("create .kb/state");
        fs::write(root.join(".kb/state/jobs.json"), "{}").expect("write jobs.json");
        fs::create_dir_all(root.join("wiki/sources")).expect("create wiki/sources");
        fs::write(root.join("wiki/sources/.hidden.md"), "").expect("write hidden");
        fs::write(root.join("wiki/sources/page.md"), "").expect("write page");

        let entries = collect_entries(root, root).expect("collect");
        let names: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(names, vec!["wiki", "kb.toml"]);

        let wiki = &entries[0];
        assert_eq!(wiki.kind, "dir");
        assert_eq!(wiki.children.len(), 1);
        assert_eq!(wiki.children[0].path, "wiki/sources");
        let sources = &wiki.children[0];
        assert_eq!(sources.children.len(), 1);
        assert_eq!(sources.children[0].path, "wiki/sources/page.md");
    }

    #[test]
    fn sorts_directories_before_files() {
        let tmp = tempdir().expect("temp root");
        let root = tmp.path();
        fs::write(root.join("zzz.md"), "").expect("write zzz");
        fs::write(root.join("aaa.md"), "").expect("write aaa");
        fs::create_dir_all(root.join("mmm")).expect("create mmm");

        let entries = collect_entries(root, root).expect("collect");
        let names: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(names, vec!["mmm", "aaa.md", "zzz.md"]);
    }

    #[test]
    fn missing_root_returns_empty() {
        let tmp = tempdir().expect("temp root");
        let missing = tmp.path().join("nope");
        assert!(
            collect_entries(&missing, &missing)
                .expect("collect missing")
                .is_empty()
        );
    }
}
