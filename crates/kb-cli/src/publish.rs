use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::Regex;
use serde::Serialize;

use crate::config::PublishTargetConfig;

#[derive(Debug, Serialize)]
pub struct PublishReport {
    pub target: String,
    pub dest: String,
    pub dry_run: bool,
    pub files: Vec<PublishFileResult>,
}

#[derive(Debug, Serialize)]
pub struct PublishFileResult {
    pub source: String,
    pub dest: String,
    pub outcome: &'static str,
}

pub fn run_publish(
    root: &Path,
    target_name: &str,
    target_cfg: &PublishTargetConfig,
    dry_run: bool,
    json: bool,
) -> Result<()> {
    let dest_root = resolve_dest(root, &target_cfg.path);
    let filter = target_cfg.filter.as_deref().unwrap_or("");

    let candidates = collect_matching_files(root, filter)?;

    let mut results = Vec::new();

    for rel in &candidates {
        let src = root.join(rel);
        let dest = dest_root.join(rel);

        let source_str = rel.to_string_lossy().into_owned();
        let dest_str = dest.to_string_lossy().into_owned();

        if dry_run {
            results.push(PublishFileResult {
                source: source_str,
                dest: dest_str,
                outcome: "would_copy",
            });
            continue;
        }

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = fs::read_to_string(&src)?;
        let rewritten = rewrite_links(&content, root, rel);
        fs::write(&dest, rewritten.as_bytes())?;

        results.push(PublishFileResult {
            source: source_str,
            dest: dest_str,
            outcome: "copied",
        });
    }

    let report = PublishReport {
        target: target_name.to_string(),
        dest: dest_root.to_string_lossy().into_owned(),
        dry_run,
        files: results,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report.files.is_empty() {
        println!(
            "publish[{target_name}]: no files matched filter '{filter}' — nothing to publish"
        );
    } else {
        for f in &report.files {
            println!("{} {} -> {}", f.outcome, f.source, f.dest);
        }
        println!(
            "publish[{target_name}]: {} file(s) {}",
            report.files.len(),
            if dry_run { "would be published" } else { "published" }
        );
    }

    Ok(())
}

fn resolve_dest(root: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

/// Walk `root` recursively and return relative paths of files matching `filter`.
///
/// Filter syntax:
/// - `""` — match all files
/// - `"prefix/**"` — all files under prefix/
/// - `"prefix/*"` — files directly under prefix/ (no subdirectories)
/// - exact path — single file
fn collect_matching_files(root: &Path, filter: &str) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_recursive(root, root, filter, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_recursive(
    root: &Path,
    dir: &Path,
    filter: &str,
    out: &mut Vec<PathBuf>,
) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err.into()),
    };

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;

        let rel = path
            .strip_prefix(root)
            .map_err(|_| anyhow::anyhow!("path escapes root"))?
            .to_path_buf();
        let rel_str = rel.to_string_lossy();

        if file_type.is_dir() {
            // Only recurse into directory if the filter could possibly match under it
            if filter.is_empty() || dir_could_match(&rel_str, filter) {
                collect_recursive(root, &path, filter, out)?;
            }
        } else if file_type.is_file() && filter_matches(&rel_str, filter) {
            out.push(rel);
        }
    }

    Ok(())
}

fn dir_could_match(dir_rel: &str, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    if let Some(prefix) = filter.strip_suffix("/**") {
        return prefix.starts_with(dir_rel)
            || dir_rel.starts_with(prefix)
            || dir_rel.starts_with(&format!("{prefix}/"));
    }
    if let Some(prefix) = filter.strip_suffix("/*") {
        return dir_rel == prefix;
    }
    // exact filter — only if it lives under this dir
    filter.starts_with(&format!("{dir_rel}/"))
}

fn filter_matches(rel_path: &str, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    if let Some(prefix) = filter.strip_suffix("/**") {
        return rel_path.starts_with(&format!("{prefix}/")) || rel_path == prefix;
    }
    if let Some(prefix) = filter.strip_suffix("/*") {
        let rest = rel_path.strip_prefix(&format!("{prefix}/")).unwrap_or("");
        return !rest.is_empty() && !rest.contains('/');
    }
    rel_path == filter
}

/// Rewrite relative `[text](path)` markdown links to absolute KB-rooted paths.
///
/// Absolute URLs, fragment-only links, and already-absolute paths are left unchanged.
/// This ensures published copies remain navigable even outside the KB directory.
fn rewrite_links(content: &str, kb_root: &Path, source_rel: &Path) -> String {
    let source_dir = source_rel.parent().unwrap_or_else(|| Path::new(""));
    let re = Regex::new(r"\[([^\]\n]*)\]\(([^)\n]*)\)").expect("static regex");

    re.replace_all(content, |caps: &regex::Captures| {
        let text = &caps[1];
        let href = &caps[2];

        if href.is_empty()
            || href.starts_with("http://")
            || href.starts_with("https://")
            || href.starts_with('/')
            || href.starts_with('#')
        {
            return caps[0].to_string();
        }

        let (path_part, fragment) = href
            .split_once('#')
            .map(|(p, f)| (p, format!("#{f}")))
            .unwrap_or((href, String::new()));

        if path_part.is_empty() {
            return caps[0].to_string();
        }

        let abs = kb_root.join(source_dir).join(path_part);
        format!("[{text}]({}{})", abs.display(), fragment)
    })
    .into_owned()
}

/// Plain-string error message for an unknown publish target. Callers wrap
/// this in whatever error type fits their pipeline (`anyhow::anyhow!` for
/// system-failure paths, `ValidationError` for the CLI dispatch that
/// rejects before writing any job manifest — see bn-1jx).
#[must_use]
pub fn target_not_found_message(target_name: &str, available: &[String]) -> String {
    if available.is_empty() {
        format!(
            "publish target '{target_name}' not found — \
             add a [publish.targets.{target_name}] section to kb.toml"
        )
    } else {
        let list = available.join(", ");
        format!(
            "publish target '{target_name}' not found — \
             available targets: {list}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn filter_matches_glob_double_star() {
        assert!(filter_matches("wiki/topic.md", "wiki/**"));
        assert!(filter_matches("wiki/sub/topic.md", "wiki/**"));
        assert!(!filter_matches("outputs/answer.md", "wiki/**"));
    }

    #[test]
    fn filter_matches_glob_single_star() {
        assert!(filter_matches("wiki/topic.md", "wiki/*"));
        assert!(!filter_matches("wiki/sub/topic.md", "wiki/*"));
    }

    #[test]
    fn filter_matches_empty_matches_all() {
        assert!(filter_matches("wiki/topic.md", ""));
        assert!(filter_matches("outputs/answer.md", ""));
    }

    #[test]
    fn filter_matches_exact() {
        assert!(filter_matches("wiki/topic.md", "wiki/topic.md"));
        assert!(!filter_matches("wiki/other.md", "wiki/topic.md"));
    }

    #[test]
    fn rewrite_links_makes_relative_absolute() {
        let root = PathBuf::from("/kb");
        let source_rel = Path::new("wiki/topic.md");
        let content = "[see also](related.md) and [external](https://example.com)";
        let result = rewrite_links(content, &root, source_rel);
        assert!(result.contains("/kb/wiki/related.md"), "got: {result}");
        assert!(result.contains("https://example.com"), "got: {result}");
    }

    #[test]
    fn rewrite_links_preserves_fragment() {
        let root = PathBuf::from("/kb");
        let source_rel = Path::new("wiki/topic.md");
        let content = "[heading](other.md#section)";
        let result = rewrite_links(content, &root, source_rel);
        assert!(result.contains("/kb/wiki/other.md#section"), "got: {result}");
    }

    #[test]
    fn rewrite_links_leaves_absolute_unchanged() {
        let root = PathBuf::from("/kb");
        let source_rel = Path::new("wiki/topic.md");
        let content = "[abs](/absolute/path.md)";
        let result = rewrite_links(content, &root, source_rel);
        assert_eq!(result, content);
    }

    #[test]
    fn publish_creates_files_at_dest() {
        let kb = tempdir().expect("kb dir");
        let dest = tempdir().expect("dest dir");

        let wiki = kb.path().join("wiki");
        fs::create_dir_all(&wiki).expect("create wiki");
        fs::write(wiki.join("topic.md"), "# Topic\n[link](other.md)\n").expect("write wiki file");

        let target_cfg = crate::config::PublishTargetConfig {
            path: dest.path().to_string_lossy().into_owned(),
            filter: Some("wiki/**".to_string()),
            format: None,
        };

        run_publish(kb.path(), "notes", &target_cfg, false, false).expect("publish");

        let published = dest.path().join("wiki").join("topic.md");
        assert!(published.exists(), "published file should exist");
        let content = fs::read_to_string(&published).expect("read published");
        // relative link should have been rewritten to absolute
        assert!(
            content.contains(kb.path().to_string_lossy().as_ref()),
            "link should be absolute: {content}"
        );
    }

    #[test]
    fn publish_dry_run_does_not_write() {
        let kb = tempdir().expect("kb dir");
        let dest = tempdir().expect("dest dir");

        let wiki = kb.path().join("wiki");
        fs::create_dir_all(&wiki).expect("create wiki");
        fs::write(wiki.join("a.md"), "# A\n").expect("write");

        let target_cfg = crate::config::PublishTargetConfig {
            path: dest.path().to_string_lossy().into_owned(),
            filter: Some("wiki/**".to_string()),
            format: None,
        };

        run_publish(kb.path(), "notes", &target_cfg, true, false).expect("dry run");

        let published = dest.path().join("wiki").join("a.md");
        assert!(!published.exists(), "dry-run should not create files");
    }

    #[test]
    fn publish_is_idempotent() {
        let kb = tempdir().expect("kb dir");
        let dest = tempdir().expect("dest dir");

        let wiki = kb.path().join("wiki");
        fs::create_dir_all(&wiki).expect("create wiki");
        fs::write(wiki.join("a.md"), "# A\n").expect("write");

        let target_cfg = crate::config::PublishTargetConfig {
            path: dest.path().to_string_lossy().into_owned(),
            filter: Some("wiki/**".to_string()),
            format: None,
        };

        run_publish(kb.path(), "notes", &target_cfg, false, false).expect("first publish");
        run_publish(kb.path(), "notes", &target_cfg, false, false).expect("second publish");

        let published = dest.path().join("wiki").join("a.md");
        assert!(published.exists(), "file should exist after both runs");
    }
}
