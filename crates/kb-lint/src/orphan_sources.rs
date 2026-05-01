//! Orphan-source lint (bn-asr2).
//!
//! A *source page* is "orphaned" when no concept page cites it AND no
//! other source page links to it via a wiki-link. Orphans accumulate as
//! the KB shifts focus — the underlying source is still useful raw
//! material but no rendered page points at it, so it'll never be
//! discovered through normal navigation.
//!
//! This is a structural check — no LLM, no I/O outside `wiki/`. The
//! report is intentionally a warning (not an error): some sources are
//! deliberately archival and the user can either link them from a
//! concept page, remove them, or list them in `[lint.orphans]
//! exempt_globs` so the lint stops complaining.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use kb_core::extract_src_id_references;

use crate::{
    IssueKind, LintIssue, MissingCitationsLevel, markdown_files_under, relative_to_root,
};

const WIKI_SOURCES_DIR: &str = "wiki/sources";
const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";

/// `[lint.orphans]` runtime config.
///
/// `exempt_globs` are matched against the source page's path *relative
/// to the KB root* (e.g. `wiki/sources/archive/legal-2019.md`). Globs
/// use the standard syntax: `**` for any path segments, `*` for any
/// single segment characters, `?` for one char.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrphanSourcesConfig {
    pub enabled: bool,
    pub level: MissingCitationsLevel,
    pub exempt_globs: Vec<String>,
}

impl Default for OrphanSourcesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: MissingCitationsLevel::Warn,
            exempt_globs: Vec::new(),
        }
    }
}

/// Walk `wiki/sources/` and report each source page that is neither
/// cited by any concept nor linked from any other source.
///
/// # Errors
/// Returns an error when wiki pages cannot be read or `exempt_globs`
/// contains an invalid glob expression.
pub fn detect_orphan_sources(
    root: &Path,
    config: &OrphanSourcesConfig,
) -> Result<Vec<LintIssue>> {
    let sources_dir = root.join(WIKI_SOURCES_DIR);
    if !sources_dir.exists() {
        return Ok(Vec::new());
    }
    let exempt_set = build_exempt_set(&config.exempt_globs)?;
    let source_pages = enumerate_source_pages(&sources_dir, root)?;
    if source_pages.is_empty() {
        return Ok(Vec::new());
    }

    let cited_ids = collect_concept_citations(root)?;
    let linked_paths = collect_source_wiki_links(root, &source_pages)?;

    let mut issues = Vec::new();
    for entry in &source_pages {
        let rel_str = entry.rel_path.to_string_lossy().into_owned();
        if exempt_set.is_match(&rel_str) {
            continue;
        }
        // Skip the auto-generated index page.
        if entry
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.eq_ignore_ascii_case("index.md"))
        {
            continue;
        }
        let cited = entry.src_id.as_deref().is_some_and(|id| cited_ids.contains(id));
        let linked = linked_paths.contains(&entry.rel_path);
        if cited || linked {
            continue;
        }
        issues.push(LintIssue {
            severity: config.level.severity(),
            kind: IssueKind::OrphanSource,
            referring_page: rel_str.clone(),
            line: 0,
            target: entry
                .src_id
                .clone()
                .unwrap_or_else(|| entry.path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string()),
            message: "orphan source — not cited by any concept and not linked from any other source".to_string(),
            suggested_fix: Some(
                "consider linking from a concept page or removing the source (or list it under `[lint.orphans] exempt_globs`)"
                    .to_string(),
            ),
        });
    }
    Ok(issues)
}

#[derive(Debug, Clone)]
struct SourcePageEntry {
    /// Absolute path on disk.
    path: PathBuf,
    /// Path relative to the KB root, e.g. `wiki/sources/src-abc.md`.
    rel_path: PathBuf,
    /// Bare src-id parsed from the filename, e.g. `src-abc` from
    /// `src-abc.md` or `src-abc-some-slug.md`. `None` when the filename
    /// doesn't follow the convention (rare; shouldn't happen for
    /// kb-managed sources but we don't crash on hand-edited files).
    src_id: Option<String>,
}

fn enumerate_source_pages(sources_dir: &Path, root: &Path) -> Result<Vec<SourcePageEntry>> {
    let mut out = Vec::new();
    for path in markdown_files_under(sources_dir)? {
        let rel = relative_to_root(root, &path);
        let src_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(parse_src_id_from_stem);
        out.push(SourcePageEntry {
            path,
            rel_path: rel,
            src_id,
        });
    }
    Ok(out)
}

/// `src-abc.md` → `Some("src-abc")`; `src-abc-some-slug.md` →
/// `Some("src-abc")`. Only accepts stems that begin with `src-`.
fn parse_src_id_from_stem(stem: &str) -> Option<String> {
    if !stem.starts_with("src-") {
        return None;
    }
    // Find the boundary between the id (`src-<token>`) and an optional slug.
    // The id is `src-` + a single token of `[A-Za-z0-9]+`. Anything after a
    // second `-` belongs to the slug.
    let rest = &stem[4..];
    let token_end = rest
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_alphanumeric())
        .map_or(rest.len(), |(idx, _)| idx);
    if token_end == 0 {
        return None;
    }
    Some(format!("src-{}", &rest[..token_end]))
}

/// Walk `wiki/concepts/**/*.md` and harvest every src-id mentioned in a
/// citation. Concept pages may use the bracket form, paren form, or
/// `[src-id#section]` — the parser handles all three.
fn collect_concept_citations(root: &Path) -> Result<BTreeSet<String>> {
    let concepts_dir = root.join(WIKI_CONCEPTS_DIR);
    if !concepts_dir.exists() {
        return Ok(BTreeSet::new());
    }
    let mut ids = BTreeSet::new();
    for path in markdown_files_under(&concepts_dir)? {
        let body = fs::read_to_string(&path)
            .with_context(|| format!("read concept page {}", path.display()))?;
        for r in extract_src_id_references(&body) {
            ids.insert(r.src_id);
        }
    }
    Ok(ids)
}

/// Walk every source page and collect the *relative paths* of any other
/// source page they wiki-link to. Format: `[[wiki/sources/<stem>]]` or
/// `[[wiki/sources/<stem>.md]]`.
///
/// The result lets `detect_orphan_sources` ask "is this source page
/// linked from anywhere?" by checking membership against its own
/// `rel_path`. Self-links don't count — a page linking to itself is not
/// the same as being discoverable from somewhere else.
fn collect_source_wiki_links(
    _root: &Path,
    source_pages: &[SourcePageEntry],
) -> Result<BTreeSet<PathBuf>> {
    let mut linked = BTreeSet::new();
    let known_paths: BTreeSet<&Path> = source_pages.iter().map(|p| p.rel_path.as_path()).collect();
    let known_stems: BTreeSet<String> = source_pages
        .iter()
        .filter_map(|p| {
            p.path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
        })
        .collect();

    let re = wiki_link_regex();
    for source in source_pages {
        let body = fs::read_to_string(&source.path)
            .with_context(|| format!("read source page {}", source.path.display()))?;
        for cap in re.captures_iter(&body) {
            let Some(target) = cap.get(1) else { continue };
            let raw = target.as_str().trim();
            // Strip optional `.md` suffix and any `#anchor`.
            let no_anchor = raw.split('#').next().unwrap_or(raw);
            let stripped = no_anchor.trim_end_matches(".md").trim_end_matches('/');
            if stripped.is_empty() {
                continue;
            }
            // Two forms accepted:
            //   [[wiki/sources/<stem>]]  -> match on rel_path
            //   [[<stem>]]               -> match on stem alone (legacy/short form)
            let target_rel: Option<PathBuf> = stripped.strip_prefix("wiki/sources/").map_or_else(
                || {
                    if known_stems.contains(stripped) {
                        Some(PathBuf::from(format!("wiki/sources/{stripped}.md")))
                    } else {
                        None
                    }
                },
                |rest| Some(PathBuf::from(format!("wiki/sources/{rest}.md"))),
            );
            let Some(rel) = target_rel else { continue };
            // Skip self-links — they don't make a page discoverable.
            if rel == source.rel_path {
                continue;
            }
            if known_paths.contains(rel.as_path()) {
                linked.insert(rel);
            }
        }
    }
    Ok(linked)
}

fn wiki_link_regex() -> &'static regex::Regex {
    use std::sync::LazyLock;
    static RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\[\[([^\]\r\n|]+)(?:\|[^\]\r\n]*)?\]\]")
            .expect("valid wiki-link regex")
    });
    &RE
}

fn build_exempt_set(globs: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for raw in globs {
        let glob =
            Glob::new(raw).with_context(|| format!("invalid glob in [lint.orphans] exempt_globs: {raw}"))?;
        builder.add(glob);
    }
    builder
        .build()
        .context("build glob set for [lint.orphans] exempt_globs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IssueSeverity;
    use std::fs;
    use tempfile::tempdir;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        fs::write(&p, body).expect("write");
    }

    #[test]
    fn cited_source_is_not_orphan() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# Source\nbody.\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nSome claim [src-abc].\n",
        );
        let issues =
            detect_orphan_sources(root, &OrphanSourcesConfig::default()).expect("run");
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn uncited_unlinked_source_fires_warning() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# Source\nbody.\n");
        // No concepts and no other sources.
        let issues =
            detect_orphan_sources(root, &OrphanSourcesConfig::default()).expect("run");
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].kind, IssueKind::OrphanSource);
        assert_eq!(issues[0].severity, IssueSeverity::Warning);
        assert!(
            issues[0].message.contains("orphan source"),
            "{:?}",
            issues[0].message
        );
        assert_eq!(issues[0].referring_page, "wiki/sources/src-abc.md");
        assert_eq!(issues[0].target, "src-abc");
    }

    #[test]
    fn linked_only_source_is_not_orphan() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        // src-abc is referenced via wiki-link from src-def, so it has
        // an entry point even without a citation.
        write(root, "wiki/sources/src-abc.md", "# Source A\n");
        write(
            root,
            "wiki/sources/src-def.md",
            "# Source D\nSee [[wiki/sources/src-abc]].\n",
        );
        // A concept page cites src-def so it doesn't itself become orphan.
        write(root, "wiki/concepts/x.md", "# X\n[src-def]\n");
        let issues =
            detect_orphan_sources(root, &OrphanSourcesConfig::default()).expect("run");
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    #[test]
    fn exempt_glob_suppresses_orphan_warning() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/archive/src-old.md", "# Old\n");
        let cfg = OrphanSourcesConfig {
            exempt_globs: vec!["wiki/sources/archive/**".to_string()],
            ..OrphanSourcesConfig::default()
        };
        let issues = detect_orphan_sources(root, &cfg).expect("run");
        assert!(issues.is_empty(), "exempted path should not fire: {issues:?}");
    }

    #[test]
    fn level_error_promotes_severity() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# Source\n");
        let cfg = OrphanSourcesConfig {
            level: MissingCitationsLevel::Error,
            ..OrphanSourcesConfig::default()
        };
        let issues = detect_orphan_sources(root, &cfg).expect("run");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
    }

    #[test]
    fn slugged_filename_resolves_to_src_id() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        // Production-style: src-abc-some-slug.md is the same source as src-abc.
        write(root, "wiki/sources/src-abc-some-slug.md", "# Source\n");
        write(root, "wiki/concepts/foo.md", "# Foo\n[src-abc]\n");
        let issues =
            detect_orphan_sources(root, &OrphanSourcesConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn index_md_under_sources_is_not_reported() {
        // wiki/sources/index.md is auto-generated and wouldn't make
        // sense to flag.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/index.md", "# Sources\n");
        let issues =
            detect_orphan_sources(root, &OrphanSourcesConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn parse_src_id_from_stem_handles_known_shapes() {
        assert_eq!(parse_src_id_from_stem("src-abc"), Some("src-abc".into()));
        assert_eq!(
            parse_src_id_from_stem("src-abc-some-slug"),
            Some("src-abc".into())
        );
        assert_eq!(parse_src_id_from_stem("not-a-src"), None);
        assert_eq!(parse_src_id_from_stem("src-"), None);
    }

    #[test]
    fn invalid_glob_is_reported() {
        let bad = vec!["[".to_string()]; // unclosed bracket
        let err = build_exempt_set(&bad).expect_err("expected glob parse error");
        let msg = format!("{err:#}");
        assert!(msg.contains("invalid glob"), "got: {msg}");
    }
}
