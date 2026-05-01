//! Stale-citation lint (bn-asr2).
//!
//! Walks every wiki page (concepts and sources alike) and finds every
//! `[src-id]` / `[src-id#section]` / `(src-id)` mention. Each one must
//! resolve to a source page on disk under `wiki/sources/`. If it
//! doesn't, the citation is "stale" — the source was renamed or removed
//! but the citation wasn't updated. Emits an error.
//!
//! The src-id parser is shared with the citation-verification lint via
//! [`kb_core::extract_src_id_references`] so the two checks stay in
//! lock-step on what counts as a citation.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use kb_core::extract_src_id_references;

use crate::{
    IssueKind, LintIssue, MissingCitationsLevel, line_number_at, markdown_files_under,
    relative_to_root,
};

const WIKI_DIR: &str = "wiki";
const WIKI_SOURCES_DIR: &str = "wiki/sources";

/// `[lint.stale_citations]` runtime config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleCitationsConfig {
    pub enabled: bool,
    pub level: MissingCitationsLevel,
}

impl Default for StaleCitationsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: MissingCitationsLevel::Error,
        }
    }
}

/// Find every `[src-id]`-style citation in concept and source pages and
/// flag any that don't resolve to a known source on disk.
///
/// # Errors
/// Returns an error when wiki pages cannot be read.
pub fn detect_stale_citations(
    root: &Path,
    config: &StaleCitationsConfig,
) -> Result<Vec<LintIssue>> {
    let wiki = root.join(WIKI_DIR);
    if !wiki.exists() {
        return Ok(Vec::new());
    }
    let known_ids = enumerate_source_ids(root)?;
    let mut issues = Vec::new();

    for page in markdown_files_under(&wiki)? {
        let body = fs::read_to_string(&page)
            .with_context(|| format!("read wiki page {}", page.display()))?;
        let rel_page = relative_to_root(root, &page).to_string_lossy().into_owned();
        let mut seen_for_page: BTreeSet<String> = BTreeSet::new();
        for r in extract_src_id_references(&body) {
            if known_ids.contains(&r.src_id) {
                continue;
            }
            // De-dup per page so one missing source doesn't spam every
            // mention. The first offset is the actionable one anyway.
            if !seen_for_page.insert(r.src_id.clone()) {
                continue;
            }
            let line = line_number_at(&body, r.offset);
            issues.push(LintIssue {
                severity: config.level.severity(),
                kind: IssueKind::StaleCitation,
                referring_page: rel_page.clone(),
                line,
                target: r.src_id.clone(),
                message: format!(
                    "citation `[{}]` does not resolve to any source under `wiki/sources/`",
                    r.src_id,
                ),
                suggested_fix: Some(format!(
                    "verify the source still exists; either restore `wiki/sources/{}*.md` or remove the citation",
                    r.src_id,
                )),
            });
        }
    }
    Ok(issues)
}

/// Enumerate every `src-<id>` known to the on-disk source tree. The id
/// is parsed from the filename stem — both `src-abc.md` and
/// `src-abc-some-slug.md` map to `src-abc`.
fn enumerate_source_ids(root: &Path) -> Result<BTreeSet<String>> {
    let dir = root.join(WIKI_SOURCES_DIR);
    if !dir.exists() {
        return Ok(BTreeSet::new());
    }
    let mut ids = BTreeSet::new();
    for path in markdown_files_under(&dir)? {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && let Some(id) = parse_src_id_from_stem(stem)
        {
            ids.insert(id);
        }
    }
    Ok(ids)
}

fn parse_src_id_from_stem(stem: &str) -> Option<String> {
    if !stem.starts_with("src-") {
        return None;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IssueSeverity;
    use tempfile::tempdir;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(&p, body).expect("write");
    }

    #[test]
    fn valid_citation_is_clean() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        write(root, "wiki/concepts/foo.md", "# Foo\nSee [src-abc].\n");
        let issues =
            detect_stale_citations(root, &StaleCitationsConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn stale_citation_fires_error() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nKnown [src-abc].\nMissing [src-zzz] here.\n",
        );
        let issues =
            detect_stale_citations(root, &StaleCitationsConfig::default()).expect("run");
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].kind, IssueKind::StaleCitation);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
        assert_eq!(issues[0].target, "src-zzz");
        assert_eq!(issues[0].referring_page, "wiki/concepts/foo.md");
        // Line of the missing citation is 3 (1-based: title, known, missing).
        assert_eq!(issues[0].line, 3);
    }

    #[test]
    fn citation_with_section_suffix_resolves_to_id() {
        // bn-3rzz: `[src-id#section]` form. The id alone resolves
        // even when the section anchor is invalid.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nSee [src-abc#methods].\n",
        );
        let issues =
            detect_stale_citations(root, &StaleCitationsConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn citation_with_unknown_trailing_token_still_resolves() {
        // bn-3ij3 may add `p.7`/`pp.7-9` suffix tokens. Parser must be
        // tolerant: the bare src-id still resolves.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nPage [src-abc p.7] and [src-abc pp.7-9].\n",
        );
        let issues =
            detect_stale_citations(root, &StaleCitationsConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn slugged_source_filename_resolves_id() {
        // Production-style `src-abc-some-slug.md` — the citation form
        // is bare `[src-abc]`.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc-the-paper.md", "# A\n");
        write(root, "wiki/concepts/foo.md", "# Foo\n[src-abc]\n");
        let issues =
            detect_stale_citations(root, &StaleCitationsConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn level_warn_demotes_severity() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/concepts/foo.md", "# Foo\n[src-zzz]\n");
        let cfg = StaleCitationsConfig {
            level: MissingCitationsLevel::Warn,
            ..StaleCitationsConfig::default()
        };
        let issues = detect_stale_citations(root, &cfg).expect("run");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Warning);
    }

    #[test]
    fn duplicate_mentions_dedup_per_page() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nFirst [src-zzz] then [src-zzz] again.\n",
        );
        let issues =
            detect_stale_citations(root, &StaleCitationsConfig::default()).expect("run");
        assert_eq!(issues.len(), 1, "expected one issue per (page, src-id) pair");
    }
}
