use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use kb_core::{NORMALIZED_SUBDIR, normalized_dir};
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

    let mut candidates = collect_matching_files(root, filter)?;
    // Implicit extra include: any `normalized/<src>/assets/<file>` that is
    // referenced by the published wiki tree needs to travel with it,
    // otherwise the `../../normalized/<src>/assets/<file>` refs on the
    // wiki source pages (set up in Part B) break on the target.
    //
    // We mirror the tree as-is (Approach 1): assets land at the same
    // relative path on the target, so no wiki-page rewriting is needed at
    // publish time. Users don't get a knob to disable this — a publish
    // that loses image files isn't the behaviour anyone wants.
    append_unique(&mut candidates, collect_asset_files(root)?);
    candidates.sort();

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

        if is_asset_path(rel) {
            // Asset files may be binary (png, jpg, etc.) — copy bytes
            // verbatim without the markdown link-rewriting pass.
            fs::copy(&src, &dest)?;
        } else {
            let content = fs::read_to_string(&src)?;
            let rewritten = rewrite_links(&content, root, rel);
            fs::write(&dest, rewritten.as_bytes())?;
        }

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

/// Walk `root`/normalized/ and collect every file under
/// `normalized/<src>/assets/`. These travel alongside the wiki tree so that
/// `../../normalized/<src>/assets/<basename>` references (set up by Part B)
/// resolve on the published target.
fn collect_asset_files(root: &Path) -> Result<Vec<PathBuf>> {
    let normalized_root = normalized_dir(root);
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&normalized_root) else {
        // No normalized tree yet (e.g. a freshly-initialized KB). Nothing to
        // publish under assets/.
        return Ok(out);
    };
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() {
            continue;
        }
        let assets_dir = entry.path().join("assets");
        if !assets_dir.is_dir() {
            continue;
        }
        collect_files_under(root, &assets_dir, &mut out)?;
    }
    Ok(out)
}

/// Recursively collect every file under `dir`, returning paths relative to
/// `root`. Used for mirroring the asset tree without applying the publish
/// filter.
fn collect_files_under(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files_under(root, &path, out)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map_err(|_| anyhow::anyhow!("asset path escapes root"))?
            .to_path_buf();
        out.push(rel);
    }
    Ok(())
}

/// Append items from `extra` to `list`, skipping any paths already present.
/// Used to merge the implicit asset tree with the filter-matched wiki files.
fn append_unique(list: &mut Vec<PathBuf>, extra: Vec<PathBuf>) {
    for item in extra {
        if !list.iter().any(|existing| existing == &item) {
            list.push(item);
        }
    }
}

/// True for paths shaped like `.kb/normalized/<src>/assets/<...>` — the
/// implicit asset tree we carry along with every publish.
fn is_asset_path(rel: &Path) -> bool {
    let mut components = rel.components();
    let Some(first) = components.next() else {
        return false;
    };
    if first.as_os_str() != std::ffi::OsStr::new(kb_core::KB_DIR) {
        return false;
    }
    let Some(second) = components.next() else {
        return false;
    };
    if second.as_os_str() != std::ffi::OsStr::new(NORMALIZED_SUBDIR) {
        return false;
    }
    // skip <src>
    if components.next().is_none() {
        return false;
    }
    let Some(fourth) = components.next() else {
        return false;
    };
    fourth.as_os_str() == std::ffi::OsStr::new("assets")
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

    /// bn-1geb Part C acceptance: `normalized/<src>/assets/*` files ride
    /// along with a `wiki/**` publish so the wiki pages'
    /// `../../normalized/<src>/assets/<name>` image refs (set up by
    /// Part B) resolve on the target. Byte content must be preserved —
    /// the asset branch must not run content through the markdown link
    /// rewriter (which would corrupt a PNG).
    #[test]
    fn publish_copies_assets_to_target() {
        let kb = tempdir().expect("kb dir");
        let dest = tempdir().expect("dest dir");

        // Seed a minimal ingest+compile shape by hand: the publish walker
        // only needs `wiki/**` + `normalized/<src>/assets/*` on disk.
        let wiki_sources = kb.path().join("wiki/sources");
        fs::create_dir_all(&wiki_sources).expect("mkdir wiki/sources");
        fs::write(
            wiki_sources.join("src-abc.md"),
            "# Source\n\n![fig](../../normalized/src-abc/assets/diagram.png)\n",
        )
        .expect("write wiki source");

        let assets = normalized_dir(kb.path()).join("src-abc/assets");
        fs::create_dir_all(&assets).expect("mkdir assets");
        // Use raw bytes with a NUL byte embedded — that way any accidental
        // `read_to_string` + link-rewrite pass would fail with an InvalidData
        // error, making bugs loud.
        let png_bytes: &[u8] = b"\x89PNG\r\n\x1a\nfake\0image\0bytes";
        fs::write(assets.join("diagram.png"), png_bytes).expect("write png");

        let target_cfg = crate::config::PublishTargetConfig {
            path: dest.path().to_string_lossy().into_owned(),
            filter: Some("wiki/**".to_string()),
            format: None,
        };

        run_publish(kb.path(), "notes", &target_cfg, false, false).expect("publish");

        // Wiki page was published…
        let wiki_out = dest.path().join("wiki/sources/src-abc.md");
        assert!(wiki_out.is_file(), "wiki source page should exist");

        // …AND the asset rode along at the mirrored path.
        let asset_out = dest
            .path()
            .join(kb_core::KB_DIR)
            .join("normalized/src-abc/assets/diagram.png");
        assert!(
            asset_out.is_file(),
            "asset should be mirrored into target at {}",
            asset_out.display()
        );
        let published_bytes = fs::read(&asset_out).expect("read published asset");
        assert_eq!(
            published_bytes, png_bytes,
            "asset bytes must survive publish unchanged (no text rewriting on binaries)"
        );
    }

    /// Publishing a tree with no assets must still succeed — the implicit
    /// asset collection step must tolerate a missing `normalized/` dir.
    #[test]
    fn publish_without_any_assets_still_succeeds() {
        let kb = tempdir().expect("kb dir");
        let dest = tempdir().expect("dest dir");

        let wiki = kb.path().join("wiki");
        fs::create_dir_all(&wiki).expect("create wiki");
        fs::write(wiki.join("only.md"), "# Only\n").expect("write md");

        let target_cfg = crate::config::PublishTargetConfig {
            path: dest.path().to_string_lossy().into_owned(),
            filter: Some("wiki/**".to_string()),
            format: None,
        };

        run_publish(kb.path(), "notes", &target_cfg, false, false)
            .expect("publish with no assets");
        assert!(dest.path().join("wiki/only.md").is_file());
    }

    #[test]
    fn is_asset_path_identifies_normalized_assets_tree() {
        assert!(is_asset_path(Path::new(".kb/normalized/src-abc/assets/pic.png")));
        assert!(is_asset_path(Path::new(
            ".kb/normalized/src-abc/assets/sub/nested.png"
        )));
        assert!(!is_asset_path(Path::new(".kb/normalized/src-abc/source.md")));
        // Legacy layout must no longer match — `kb migrate` moved it away.
        assert!(!is_asset_path(Path::new("normalized/src-abc/assets/pic.png")));
        assert!(!is_asset_path(Path::new("wiki/sources/src-abc.md")));
        assert!(!is_asset_path(Path::new("other/foo.md")));
    }
}
