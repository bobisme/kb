//! Content-drift lint (bn-asr2).
//!
//! When a concept body cites a known source page with a quoted span —
//! `"…" [src-xyz]` — but the span no longer appears verbatim in
//! `src-xyz`'s body, the source has drifted out from under the
//! citation. This is content rot, not a missing pointer: the source
//! page exists, just doesn't say what the concept claims it says.
//!
//! The shape of this check looks redundant with the existing
//! `unverified-quote` lint (bn-166d). They share the same span-matching
//! kernel — `kb_core::is_quote_present` — but report against different
//! axes:
//!
//! - `unverified-quote` covers BOTH "source missing" and "quote missing
//!   from source".
//! - `stale-citation` (bn-asr2) covers "source missing" with a sharper
//!   error.
//! - `drift` (this module) focuses purely on the "source exists, quote
//!   doesn't" case so the user can tune the warning rate independently
//!   (e.g. silence drift in noisy KBs while still gating on stale
//!   citations).
//!
//! Skipping the `SourceNotFound` case here means a KB enabling all three
//! gets exactly one signal per defect — no double-reporting.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use kb_core::frontmatter::read_frontmatter;

use crate::{
    IssueKind, LintIssue, MissingCitationsLevel, count_lines_before, line_number_at,
    load_source_body_for_id, markdown_files_under, relative_to_root, truncate_for_message,
};

const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";

/// `[lint.drift]` runtime config.
///
/// `fuzz_per_100_chars` mirrors
/// [`kb_core::DEFAULT_FUZZ_PER_100_CHARS`] — the same per-100-char edit
/// budget the citation-verification lint uses. Keeping them parallel
/// (rather than sharing one knob) lets users dial drift more strictly
/// than verification when they care about exact prose preservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftConfig {
    pub enabled: bool,
    pub level: MissingCitationsLevel,
    pub fuzz_per_100_chars: u32,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: MissingCitationsLevel::Warn,
            fuzz_per_100_chars: kb_core::DEFAULT_FUZZ_PER_100_CHARS,
        }
    }
}

/// Walk concept pages and report quote-citation drift.
///
/// For each `(quote, src-id)` pair where the source resolves but the
/// quote no longer appears verbatim, emit a `LintIssue`. Source-not-
/// found cases are intentionally skipped — those belong to the
/// stale-citation check.
///
/// # Errors
/// Returns an error when concept pages cannot be read.
pub fn detect_drift(root: &Path, config: &DriftConfig) -> Result<Vec<LintIssue>> {
    let concepts_dir = root.join(WIKI_CONCEPTS_DIR);
    if !concepts_dir.exists() {
        return Ok(Vec::new());
    }
    let mut source_cache: BTreeMap<String, Option<String>> = BTreeMap::new();
    let mut issues = Vec::new();

    for page in markdown_files_under(&concepts_dir)? {
        let raw = fs::read_to_string(&page)
            .with_context(|| format!("read concept page {}", page.display()))?;
        // Skip pages with malformed frontmatter — the citation-verification
        // and missing-citations lints already report those.
        let Ok((_, body)) = read_frontmatter(&page) else {
            continue;
        };
        let body_offset = raw.len().saturating_sub(body.len());
        let folded_body = kb_core::fold_smart_quotes(&body);
        let pairs = kb_core::extract_quote_citations(&body);

        for pair in pairs {
            let normalized_quote = kb_core::normalize_for_match(&pair.quote);
            let source_body = source_cache
                .entry(pair.src_id.clone())
                .or_insert_with(|| load_source_body_for_id(root, &pair.src_id));
            // Drift is only meaningful when the source resolves. A
            // missing source is a stale-citation problem; we let that
            // lint speak to it.
            let Some(content) = source_body.as_ref() else {
                continue;
            };
            let normalized_source = kb_core::normalize_for_match(content);
            match kb_core::is_quote_present(
                &normalized_quote,
                &normalized_source,
                config.fuzz_per_100_chars,
            ) {
                kb_core::QuoteMatch::Exact | kb_core::QuoteMatch::Fuzzy { .. } => continue,
                kb_core::QuoteMatch::NotFound => {}
            }
            let rel_page = relative_to_root(root, &page).to_string_lossy().into_owned();
            let line =
                line_number_at(&folded_body, pair.offset) + count_lines_before(&raw, body_offset);
            issues.push(LintIssue {
                severity: config.level.severity(),
                kind: IssueKind::Drift,
                referring_page: rel_page,
                line,
                target: pair.src_id.clone(),
                message: format!(
                    "drift: quote `\"{}\"` no longer appears verbatim in `{}` (fuzz {}/100 chars)",
                    truncate_for_message(&pair.quote, 80),
                    pair.src_id,
                    config.fuzz_per_100_chars,
                ),
                suggested_fix: Some(
                    "the source page changed; either update the quote to match, drop the quote, or reach for an unmodified revision"
                        .to_string(),
                ),
            });
        }
    }
    Ok(issues)
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
    fn verbatim_quote_does_not_drift() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(
            root,
            "wiki/sources/src-abc.md",
            "# Source\nThe answer is 42 in this paper.\n",
        );
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nThey said \"the answer is 42\" [src-abc].\n",
        );
        let issues = detect_drift(root, &DriftConfig::default()).expect("run");
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn drifted_quote_fires_warning() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(
            root,
            "wiki/sources/src-abc.md",
            "# Source\nThe answer used to be different.\n",
        );
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nThey said \"the answer is 42\" [src-abc].\n",
        );
        let issues = detect_drift(root, &DriftConfig::default()).expect("run");
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].kind, IssueKind::Drift);
        assert_eq!(issues[0].severity, IssueSeverity::Warning);
        assert_eq!(issues[0].target, "src-abc");
    }

    #[test]
    fn missing_source_does_not_emit_drift() {
        // That's stale-citation territory.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nThey said \"the answer is 42\" [src-abc].\n",
        );
        let issues = detect_drift(root, &DriftConfig::default()).expect("run");
        assert!(
            issues.is_empty(),
            "drift should skip missing-source cases: {issues:?}"
        );
    }

    #[test]
    fn fuzz_knob_admits_close_match() {
        // A 100-char quote with one character changed should fail at
        // fuzz=0 and pass at fuzz=2.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        // 100-char source.
        let source = "abcdefghij".repeat(10);
        // Same length, but with a one-char swap in the middle.
        let mut quote_chars: Vec<char> = source.chars().collect();
        quote_chars[50] = 'Z';
        let quote: String = quote_chars.iter().collect();

        write(root, "wiki/sources/src-abc.md", &format!("# S\n{source}\n"));
        write(
            root,
            "wiki/concepts/foo.md",
            &format!("# Foo\n\"{quote}\" [src-abc]\n"),
        );

        let strict = DriftConfig {
            fuzz_per_100_chars: 0,
            ..DriftConfig::default()
        };
        let strict_issues = detect_drift(root, &strict).expect("run");
        assert_eq!(
            strict_issues.len(),
            1,
            "fuzz=0 should reject one-char drift: {strict_issues:?}"
        );

        let lax = DriftConfig {
            fuzz_per_100_chars: 2,
            ..DriftConfig::default()
        };
        let lax_issues = detect_drift(root, &lax).expect("run");
        assert!(
            lax_issues.is_empty(),
            "fuzz=2 should admit one-char drift on a 100-char quote: {lax_issues:?}"
        );
    }

    #[test]
    fn level_error_promotes_severity() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# Source\nDifferent text.\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\n\"the answer is 42\" [src-abc]\n",
        );
        let cfg = DriftConfig {
            level: MissingCitationsLevel::Error,
            ..DriftConfig::default()
        };
        let issues = detect_drift(root, &cfg).expect("run");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Error);
    }
}
