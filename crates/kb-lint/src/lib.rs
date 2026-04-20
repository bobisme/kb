#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use kb_core::{
    BuildRecord, EntityMetadata, Manifest, ReviewItem, ReviewKind, ReviewStatus, Status,
    build_records_dir, extract_managed_regions, frontmatter::read_frontmatter,
    load_build_record, slug_from_title,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_yaml::Value;

pub mod contradictions;
pub use contradictions::{ContradictionsConfig, check_contradictions, detect_contradictions_issues};

pub mod impute;
pub use impute::{
    ImputeConfig, ImputeKindSelector, ImputeOutcome, ImputedFixPayload, ImputedItem,
    load_imputed_fix_payload, outcomes_to_lint_issues, run_impute_pass, save_imputed_fix_payload,
};

const WIKI_DIR: &str = "wiki";
const MD_EXT: &str = "md";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintRule {
    BrokenLinks,
    Orphans,
    StaleRevision,
    StaleArtifacts,
    MissingCitations,
    MissingConcepts,
    /// Cross-source factual inconsistencies in a concept's cited quotes.
    /// LLM-powered and expensive — never runs as part of `LintRule::All`.
    /// Callers must select it explicitly via `--check contradictions`.
    Contradictions,
    All,
}

impl LintRule {
    /// Parse a CLI rule selector.
    ///
    /// # Errors
    /// Returns an error when the caller requests an unsupported lint rule.
    pub fn parse(input: Option<&str>) -> Result<Self> {
        match input {
            None => Ok(Self::All),
            Some("broken-links" | "broken_links" | "brokenlinks") => Ok(Self::BrokenLinks),
            Some("orphans" | "orphan" | "orphan-pages" | "orphan_pages") => Ok(Self::Orphans),
            Some(
                "stale-revision" | "stale_revision" | "stale-revisions" | "stale_revisions",
            ) => Ok(Self::StaleRevision),
            Some("stale" | "stale-artifacts" | "stale_artifacts") => Ok(Self::StaleArtifacts),
            Some("missing-citations" | "missing_citations" | "missingcitations") => {
                Ok(Self::MissingCitations)
            }
            Some(
                "missing-concepts" | "missing_concepts" | "missingconcepts" | "concept-candidates"
                | "concept_candidates",
            ) => Ok(Self::MissingConcepts),
            Some(
                "contradictions" | "contradiction" | "cross-source-contradictions"
                | "cross_source_contradictions",
            ) => Ok(Self::Contradictions),
            Some(other) => Err(anyhow!("unsupported lint rule: {other}")),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BrokenLinks => "broken-links",
            Self::Orphans => "orphans",
            Self::StaleRevision => "stale-revision",
            Self::StaleArtifacts => "stale",
            Self::MissingCitations => "missing-citations",
            Self::MissingConcepts => "missing-concepts",
            Self::Contradictions => "contradictions",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LintReport {
    pub rule: String,
    pub issue_count: usize,
    pub issues: Vec<LintIssue>,
}

impl LintReport {
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == IssueSeverity::Error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LintIssue {
    pub severity: IssueSeverity,
    pub kind: IssueKind,
    pub referring_page: String,
    pub line: usize,
    pub target: String,
    pub message: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    MissingPage,
    MissingAnchor,
    SourceDocumentMissing,
    SourceRevisionMissing,
    SourceRevisionStale,
    OutputMissing,
    BuildRecordMissing,
    ManifestMismatch,
    FrontmatterBuildRecordMissing,
    BuildRecordOutputMissing,
    MissingCitations,
    InvalidFrontmatter,
    ConceptCandidate,
    /// Concept has quotes from ≥ 2 different source documents that make
    /// contradictory claims (per the LLM judge).
    Contradiction,
    /// Concept page exists but its body is empty or too short to be
    /// useful. Surfaced by the `thin_concepts` discovery pass used by
    /// `kb lint --impute` (bn-xt4o).
    ThinConceptBody,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingCitationsLevel {
    Warn,
    Error,
}

impl MissingCitationsLevel {
    #[must_use]
    pub const fn severity(self) -> IssueSeverity {
        match self {
            Self::Warn => IssueSeverity::Warning,
            Self::Error => IssueSeverity::Error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintOptions {
    pub require_citations: bool,
    pub missing_citations_level: MissingCitationsLevel,
    pub missing_concepts: MissingConceptsConfig,
}

impl Default for LintOptions {
    fn default() -> Self {
        Self {
            require_citations: true,
            missing_citations_level: MissingCitationsLevel::Warn,
            missing_concepts: MissingConceptsConfig::default(),
        }
    }
}

/// Run the selected lint rule against the KB at `root`.
///
/// # Errors
/// Returns an error when the lint pass cannot read KB state or wiki files.
pub fn run_lint(root: &Path, rule: LintRule) -> Result<LintReport> {
    run_lint_with_options(root, rule, &LintOptions::default())
}

/// Run the selected lint rule with caller-supplied options.
///
/// # Errors
/// Returns an error when the lint pass cannot read KB state or wiki files.
pub fn run_lint_with_options(
    root: &Path,
    rule: LintRule,
    options: &LintOptions,
) -> Result<LintReport> {
    let mut issues = Vec::new();

    if matches!(rule, LintRule::BrokenLinks | LintRule::All) {
        issues.extend(run_broken_links(root)?);
    }
    if matches!(rule, LintRule::Orphans | LintRule::All) {
        issues.extend(detect_orphan_pages(root)?);
    }
    if matches!(rule, LintRule::StaleRevision | LintRule::All) {
        issues.extend(detect_stale_revisions(root)?);
    }
    if matches!(rule, LintRule::StaleArtifacts | LintRule::All) {
        issues.extend(detect_stale_artifacts(root)?);
    }
    if options.require_citations && matches!(rule, LintRule::MissingCitations | LintRule::All) {
        issues.extend(detect_missing_citations(root, options)?);
    }
    if options.missing_concepts.enabled
        && matches!(rule, LintRule::MissingConcepts | LintRule::All)
    {
        issues.extend(detect_missing_concepts_issues(root, &options.missing_concepts)?);
    }

    Ok(LintReport {
        rule: rule.as_str().to_string(),
        issue_count: issues.len(),
        issues,
    })
}

// ---------------------------------------------------------------------------
// Broken-link checking
// ---------------------------------------------------------------------------

fn run_broken_links(root: &Path) -> Result<Vec<LintIssue>> {
    let pages = collect_wiki_pages(root)?;
    let registry = LinkRegistry::build(root, &pages)?;
    let mut issues = Vec::new();

    for page in &pages {
        let content = fs::read_to_string(&page.path)
            .with_context(|| format!("read wiki page {}", page.path.display()))?;
        for issue in scan_page_for_broken_links(page, &content, &registry) {
            issues.push(issue);
        }
    }

    Ok(issues)
}

#[derive(Debug, Clone)]
struct WikiPageRecord {
    path: PathBuf,
    page_id: String,
}

fn collect_wiki_pages(root: &Path) -> Result<Vec<WikiPageRecord>> {
    let wiki_root = root.join(WIKI_DIR);
    let mut pages = Vec::new();
    if !wiki_root.exists() {
        return Ok(pages);
    }
    visit_markdown_files(root, &wiki_root, &mut pages)?;
    pages.sort_by(|a, b| a.page_id.cmp(&b.page_id));
    Ok(pages)
}

fn visit_markdown_files(root: &Path, dir: &Path, pages: &mut Vec<WikiPageRecord>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("scan directory {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            visit_markdown_files(root, &path, pages)?;
            continue;
        }
        if path.extension().is_some_and(|ext| ext == MD_EXT)
            && let Some(page_id) = path_to_page_id(root, &path)
        {
            pages.push(WikiPageRecord { path, page_id });
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct LinkRegistry {
    pages: BTreeSet<String>,
    anchors: BTreeMap<String, BTreeSet<String>>,
}

impl LinkRegistry {
    fn build(root: &Path, pages: &[WikiPageRecord]) -> Result<Self> {
        let mut registry = Self::default();

        for page in pages {
            registry.pages.insert(page.page_id.clone());
            let content = fs::read_to_string(&page.path)
                .with_context(|| format!("read wiki page {}", page.path.display()))?;
            let mut ids = collect_anchor_ids(&content);
            ids.sort();
            ids.dedup();
            registry
                .anchors
                .insert(page.page_id.clone(), ids.into_iter().collect());
        }

        let manifest = Manifest::load(root)?;
        for artifact_path in manifest.artifacts.keys() {
            if let Some(page_id) = manifest_path_to_page_id(artifact_path) {
                registry.pages.insert(page_id);
            }
        }

        Ok(registry)
    }

    fn has_page(&self, page_id: &str) -> bool {
        self.pages.contains(page_id)
    }

    fn has_anchor(&self, page_id: &str, anchor: &str) -> bool {
        self.anchors
            .get(page_id)
            .is_some_and(|anchors| anchors.contains(anchor))
    }

    fn suggest_page(&self, page_id: &str) -> Option<String> {
        best_match(page_id, self.pages.iter().map(String::as_str))
    }

    fn suggest_anchor(&self, page_id: &str, anchor: &str) -> Option<String> {
        self.anchors.get(page_id).and_then(|anchors| {
            if anchors.len() == 1 {
                return anchors.iter().next().cloned();
            }
            best_match(anchor, anchors.iter().map(String::as_str))
        })
    }
}

fn collect_anchor_ids(content: &str) -> Vec<String> {
    let mut anchors = Vec::new();

    for region in extract_managed_regions(content) {
        anchors.push(region.id.to_string());
    }

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(title) = heading_title(trimmed) {
            let slug = slug_from_title(title);
            if !slug.is_empty() {
                anchors.push(slug);
            }
        }
    }

    anchors
}

fn heading_title(line: &str) -> Option<&str> {
    let hashes = line.chars().take_while(|ch| *ch == '#').count();
    if hashes == 0 {
        return None;
    }

    let remainder = line[hashes..].trim();
    if remainder.is_empty() {
        return None;
    }

    Some(remainder.trim_end_matches('#').trim())
}

fn scan_page_for_broken_links(
    page: &WikiPageRecord,
    content: &str,
    registry: &LinkRegistry,
) -> Vec<LintIssue> {
    let wikilink_re = Regex::new(r"\[\[([^\]\r\n]+)\]\]").expect("valid wikilink regex");
    let markdown_link_re =
        Regex::new(r"(!?)\[[^\]]*\]\(([^)]+)\)").expect("valid markdown link regex");

    let mut issues = Vec::new();
    let mut in_frontmatter = false;
    let mut past_frontmatter_start = false;
    let mut in_code_block = false;

    for (idx, line) in content.lines().enumerate() {
        let line_number = idx + 1;

        // Detect and skip YAML frontmatter at the very start of the document.
        if !past_frontmatter_start {
            past_frontmatter_start = true;
            if line.trim_end() == "---" {
                in_frontmatter = true;
                continue;
            }
        } else if in_frontmatter {
            if line.trim_end() == "---" {
                in_frontmatter = false;
            }
            continue;
        }

        // Toggle fenced code block state on lines that start with ```.
        let trimmed_start = line.trim_start();
        if trimmed_start.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }

        // Strip inline code spans (text between matching single backticks) so
        // that `[[inside-code]]` does not trigger the wiki-link scanner.
        let scan_line = strip_inline_code_spans(line);

        for capture in wikilink_re.captures_iter(&scan_line) {
            let Some(raw_target) = capture.get(1).map(|m| m.as_str()) else {
                continue;
            };
            if let Some(issue) = validate_link(raw_target, page, line_number, registry) {
                issues.push(issue);
            }
        }

        for capture in markdown_link_re.captures_iter(&scan_line) {
            if capture.get(1).is_some_and(|m| m.as_str() == "!") {
                continue;
            }
            let Some(raw_target) = capture.get(2).map(|m| m.as_str()) else {
                continue;
            };
            let Some(destination) = parse_markdown_destination(raw_target) else {
                continue;
            };
            if let Some(issue) = validate_link(&destination, page, line_number, registry) {
                issues.push(issue);
            }
        }
    }

    issues
}

/// Replace inline code spans (text between single backticks) with spaces while
/// preserving line length and byte positions. Non-span backticks are left as-is.
fn strip_inline_code_spans(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            // Find the matching closing backtick on this line.
            if let Some(close_offset) = bytes[i + 1..].iter().position(|&b| b == b'`') {
                let close = i + 1 + close_offset;
                // Replace the entire span (including both backticks) with spaces
                // of equal byte length to keep offsets stable.
                for _ in i..=close {
                    out.push(' ');
                }
                i = close + 1;
                continue;
            }
            // Unmatched backtick: copy it and stop trying to strip further spans.
            out.push('`');
            i += 1;
            // Copy the rest of the line verbatim.
            while i < bytes.len() {
                out.push(bytes[i] as char);
                i += 1;
            }
            break;
        }
        // Copy the current UTF-8 codepoint.
        let ch_start = i;
        // Advance to next codepoint boundary.
        i += 1;
        while i < bytes.len() && (bytes[i] & 0b1100_0000) == 0b1000_0000 {
            i += 1;
        }
        out.push_str(&line[ch_start..i]);
    }
    out
}

fn validate_link(
    raw_target: &str,
    page: &WikiPageRecord,
    line: usize,
    registry: &LinkRegistry,
) -> Option<LintIssue> {
    let target = normalize_link_target(raw_target, &page.page_id)?;

    if !registry.has_page(&target.page_id) {
        let suggestion = registry.suggest_page(&target.page_id);
        return Some(LintIssue {
            severity: IssueSeverity::Error,
            kind: IssueKind::MissingPage,
            referring_page: format!("{}.md", page.page_id),
            line,
            target: render_target(&target),
            message: format!("unresolved wiki page target `{}`", target.page_id),
            suggested_fix: suggestion.map(|page_id| {
                render_target(&NormalizedTarget {
                    page_id,
                    anchor: target.anchor.clone(),
                })
            }),
        });
    }

    if let Some(anchor) = &target.anchor
        && !registry.has_anchor(&target.page_id, anchor)
    {
        let suggestion = registry.suggest_anchor(&target.page_id, anchor);
        return Some(LintIssue {
            severity: IssueSeverity::Error,
            kind: IssueKind::MissingAnchor,
            referring_page: format!("{}.md", page.page_id),
            line,
            target: render_target(&target),
            message: format!(
                "unresolved section anchor `{anchor}` in target `{}`",
                target.page_id
            ),
            suggested_fix: suggestion.map(|anchor| {
                render_target(&NormalizedTarget {
                    page_id: target.page_id.clone(),
                    anchor: Some(anchor),
                })
            }),
        });
    }

    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NormalizedTarget {
    page_id: String,
    anchor: Option<String>,
}

fn normalize_link_target(raw_target: &str, current_page_id: &str) -> Option<NormalizedTarget> {
    let target = raw_target.trim();
    if target.is_empty() {
        return None;
    }

    let target = target.split('|').next().unwrap_or(target).trim();
    if target.starts_with("http://")
        || target.starts_with("https://")
        || target.starts_with("mailto:")
    {
        return None;
    }

    let (raw_path, raw_anchor) = match target.split_once('#') {
        Some((path, anchor)) => (path.trim(), Some(anchor.trim().to_string())),
        None => (target, None),
    };

    let page_id = if raw_path.is_empty() {
        current_page_id.to_string()
    } else {
        normalize_page_reference(raw_path, current_page_id)?
    };

    Some(NormalizedTarget {
        page_id,
        anchor: raw_anchor.filter(|anchor| !anchor.is_empty()),
    })
}

fn normalize_page_reference(raw_path: &str, current_page_id: &str) -> Option<String> {
    let mut path = raw_path.trim().trim_matches('<').trim_matches('>').trim();
    if path.is_empty() {
        return None;
    }

    if let Some(stripped) = path.strip_suffix(".md") {
        path = stripped;
    }

    let resolved = if path.starts_with(WIKI_DIR) {
        clean_relative_path(Path::new(path))
    } else {
        let current = Path::new(current_page_id);
        let parent = current.parent().unwrap_or_else(|| Path::new(""));
        clean_relative_path(&parent.join(path))
    };

    let normalized = resolved.to_string_lossy().replace('\\', "/");
    (!normalized.is_empty()).then_some(normalized)
}

fn clean_relative_path(path: &Path) -> PathBuf {
    let mut cleaned = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                cleaned.pop();
            }
            Component::Normal(part) => cleaned.push(part),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    cleaned
}

fn parse_markdown_destination(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with('<') {
        let end = trimmed.find('>')?;
        return Some(trimmed[..=end].trim().to_string());
    }

    Some(
        trimmed
            .split_whitespace()
            .next()
            .unwrap_or(trimmed)
            .to_string(),
    )
}

fn render_target(target: &NormalizedTarget) -> String {
    target.anchor.as_ref().map_or_else(
        || target.page_id.clone(),
        |anchor| format!("{}#{anchor}", target.page_id),
    )
}

fn best_match<'a>(query: &str, candidates: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut best: Option<(&str, usize)> = None;
    for candidate in candidates {
        let score = levenshtein(query, candidate);
        match best {
            Some((_, best_score)) if score >= best_score => {}
            _ => best = Some((candidate, score)),
        }
    }

    best.and_then(|(candidate, score)| {
        let threshold = (query.len().max(candidate.len()) / 2).max(3);
        (score <= threshold).then(|| candidate.to_string())
    })
}

fn levenshtein(left: &str, right: &str) -> usize {
    if left == right {
        return 0;
    }
    if left.is_empty() {
        return right.chars().count();
    }
    if right.is_empty() {
        return left.chars().count();
    }

    let right_chars: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right_chars.len()).collect();
    let mut current = vec![0; right_chars.len() + 1];

    for (left_index, left_char) in left.chars().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_char) in right_chars.iter().enumerate() {
            let substitution_cost = usize::from(left_char != *right_char);
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + substitution_cost);
        }
        previous.clone_from_slice(&current);
    }

    previous[right_chars.len()]
}

fn path_to_page_id(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    manifest_path_to_page_id(relative)
}

fn manifest_path_to_page_id(path: &Path) -> Option<String> {
    if path.extension().is_none_or(|ext| ext != MD_EXT) {
        return None;
    }
    let relative = path.to_string_lossy().replace('\\', "/");
    relative.strip_suffix(".md").map(ToOwned::to_owned)
}

// ---------------------------------------------------------------------------
// Orphan page detection
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct NormalizedMetadata {
    source_revision_id: String,
}

fn detect_orphan_pages(root: &Path) -> Result<Vec<LintIssue>> {
    let mut issues = Vec::new();
    for page in markdown_files_under(&root.join("wiki"))? {
        let (frontmatter, _) = match read_frontmatter(&page) {
            Ok(v) => v,
            Err(err) => {
                issues.push(LintIssue {
                    kind: IssueKind::InvalidFrontmatter,
                    severity: IssueSeverity::Error,
                    referring_page: relative_to_root(root, &page)
                        .to_string_lossy()
                        .into_owned(),
                    line: 0,
                    target: String::new(),
                    message: format!("invalid YAML frontmatter: {err}"),
                    suggested_fix: None,
                });
                continue;
            }
        };

        let doc_ids =
            frontmatter_string_list(&frontmatter, "source_document_id", "source_document_ids");
        let rev_ids =
            frontmatter_string_list(&frontmatter, "source_revision_id", "source_revision_ids");

        if doc_ids.is_empty() && rev_ids.is_empty() {
            continue;
        }

        // bn-1yvn: `kb forget` flags promoted question pages with an
        // `orphaned_sources:` frontmatter list rather than rewriting them.
        // A doc-id listed there is a user-acknowledged orphan — suppress the
        // orphan error for that specific src-id. Errors for doc-ids NOT in
        // the marker list still fire.
        let acknowledged_sources: BTreeSet<String> =
            frontmatter_string_list(&frontmatter, "orphaned_source", "orphaned_sources")
                .into_iter()
                .collect();

        let rel_page = relative_to_root(root, &page);
        let rel_page_str = rel_page.to_string_lossy().into_owned();
        let existing_docs: Vec<_> = doc_ids
            .iter()
            .filter(|doc_id| {
                normalized_metadata_for_doc(root, doc_id)
                    .ok()
                    .flatten()
                    .is_some()
            })
            .cloned()
            .collect();

        let missing_docs: Vec<&String> = doc_ids
            .iter()
            .filter(|doc_id| !existing_docs.contains(*doc_id))
            .collect();

        for missing_doc in &missing_docs {
            if acknowledged_sources.contains(*missing_doc) {
                continue;
            }
            issues.push(LintIssue {
                severity: IssueSeverity::Error,
                kind: IssueKind::SourceDocumentMissing,
                referring_page: rel_page_str.clone(),
                line: 0,
                target: (*missing_doc).clone(),
                message: format!("referenced source document '{missing_doc}' is missing"),
                suggested_fix: None,
            });
        }

        // Revision-level orphan: page references revision(s), but NO live
        // revision is available for any of the page's source documents (i.e.
        // the underlying normalized/<doc>/ dirs are all gone). This is the
        // "truly orphaned" case — the source is no longer in the KB.
        //
        // The related-but-distinct case — live revisions exist but none of
        // them match the page's recorded revision ids — is a *stale revision*
        // (source re-ingested, page not yet recompiled) and lives in its own
        // `stale_revision` lint class. See `detect_stale_revisions`.
        //
        // bn-1yvn: if every missing doc-id is acknowledged via
        // `orphaned_sources`, the revision-missing signal is just the
        // downstream echo of that acknowledged state; suppress it. If *any*
        // missing doc is un-acknowledged, the revision errors still fire.
        if !rev_ids.is_empty() {
            let live_revision_ids: BTreeSet<_> = existing_docs
                .iter()
                .filter_map(|doc_id| {
                    normalized_metadata_for_doc(root, doc_id)
                        .ok()
                        .flatten()
                        .map(|metadata| metadata.source_revision_id)
                })
                .collect();

            let all_missing_acknowledged = !missing_docs.is_empty()
                && missing_docs
                    .iter()
                    .all(|doc_id| acknowledged_sources.contains(*doc_id));

            if live_revision_ids.is_empty() && !all_missing_acknowledged {
                for revision_id in &rev_ids {
                    issues.push(LintIssue {
                        severity: IssueSeverity::Error,
                        kind: IssueKind::SourceRevisionMissing,
                        referring_page: rel_page_str.clone(),
                        line: 0,
                        target: revision_id.clone(),
                        message: format!(
                            "referenced source revision '{revision_id}' is no longer present"
                        ),
                        suggested_fix: None,
                    });
                }
            }
        }
    }
    Ok(issues)
}

// ---------------------------------------------------------------------------
// Stale revision detection
// ---------------------------------------------------------------------------
//
// A page has "stale revisions" when:
//   - its source document(s) still exist (normalized/<doc>/metadata.json is
//     present), AND
//   - the page's recorded `source_revision_id(s)` do not match the current
//     normalized revision for any of those documents.
//
// This is a "please recompile" signal, not an "orphaned page" signal: the
// source is still in the KB, just at a newer revision. Keeping it as its own
// lint class (rather than folding it into `orphans`) gives users a precise
// `--check stale_revision` filter and avoids the misleading "orphans" label
// on pages whose sources are very much present.

fn detect_stale_revisions(root: &Path) -> Result<Vec<LintIssue>> {
    let mut issues = Vec::new();
    for page in markdown_files_under(&root.join("wiki"))? {
        let (frontmatter, _) = match read_frontmatter(&page) {
            Ok(v) => v,
            Err(err) => {
                issues.push(LintIssue {
                    kind: IssueKind::InvalidFrontmatter,
                    severity: IssueSeverity::Error,
                    referring_page: relative_to_root(root, &page)
                        .to_string_lossy()
                        .into_owned(),
                    line: 0,
                    target: String::new(),
                    message: format!("invalid YAML frontmatter: {err}"),
                    suggested_fix: None,
                });
                continue;
            }
        };

        let doc_ids =
            frontmatter_string_list(&frontmatter, "source_document_id", "source_document_ids");
        let rev_ids =
            frontmatter_string_list(&frontmatter, "source_revision_id", "source_revision_ids");

        if rev_ids.is_empty() {
            continue;
        }

        let rel_page = relative_to_root(root, &page);
        let rel_page_str = rel_page.to_string_lossy().into_owned();

        let live_revision_ids: BTreeSet<_> = doc_ids
            .iter()
            .filter_map(|doc_id| {
                normalized_metadata_for_doc(root, doc_id)
                    .ok()
                    .flatten()
                    .map(|metadata| metadata.source_revision_id)
            })
            .collect();

        // No live revisions — this is an *orphan* concern, handled by
        // `detect_orphan_pages`. Avoid double-reporting here.
        if live_revision_ids.is_empty() {
            continue;
        }

        if rev_ids
            .iter()
            .all(|revision_id| !live_revision_ids.contains(revision_id))
        {
            issues.push(LintIssue {
                severity: IssueSeverity::Error,
                kind: IssueKind::SourceRevisionStale,
                referring_page: rel_page_str,
                line: 0,
                target: rev_ids.join(", "),
                message: format!(
                    "page references stale source revision(s): {}",
                    rev_ids.join(", ")
                ),
                suggested_fix: None,
            });
        }
    }
    Ok(issues)
}

// ---------------------------------------------------------------------------
// Stale artifact detection
// ---------------------------------------------------------------------------

fn detect_stale_artifacts(root: &Path) -> Result<Vec<LintIssue>> {
    let manifest = Manifest::load(root)?;
    let mut issues = Vec::new();

    for (artifact_path, manifest_record) in &manifest.artifacts {
        let output_path = root.join(artifact_path);
        let artifact_str = artifact_path.to_string_lossy().into_owned();
        if !output_path.exists() {
            issues.push(LintIssue {
                severity: IssueSeverity::Error,
                kind: IssueKind::OutputMissing,
                referring_page: artifact_str.clone(),
                line: 0,
                target: artifact_str.clone(),
                message: "manifest entry points to a file that no longer exists".to_string(),
                suggested_fix: None,
            });
        }

        match load_build_record(root, &manifest_record.metadata.id)? {
            None => issues.push(LintIssue {
                severity: IssueSeverity::Error,
                kind: IssueKind::BuildRecordMissing,
                referring_page: artifact_str,
                line: 0,
                target: manifest_record.metadata.id.clone(),
                message: format!(
                    "manifest references missing build record '{}'",
                    manifest_record.metadata.id
                ),
                suggested_fix: None,
            }),
            Some(current) if !same_build_fingerprint(manifest_record, &current) => {
                issues.push(LintIssue {
                    severity: IssueSeverity::Error,
                    kind: IssueKind::ManifestMismatch,
                    referring_page: artifact_str,
                    line: 0,
                    target: manifest_record.metadata.id.clone(),
                    message: format!(
                        "manifest build record '{}' diverges from state/build_records",
                        manifest_record.metadata.id
                    ),
                    suggested_fix: None,
                });
            }
            Some(_) => {}
        }
    }

    for page in markdown_files_under(&root.join("wiki"))? {
        let (frontmatter, _) = match read_frontmatter(&page) {
            Ok(v) => v,
            Err(err) => {
                issues.push(LintIssue {
                    kind: IssueKind::InvalidFrontmatter,
                    severity: IssueSeverity::Error,
                    referring_page: relative_to_root(root, &page)
                        .to_string_lossy()
                        .into_owned(),
                    line: 0,
                    target: String::new(),
                    message: format!("invalid YAML frontmatter: {err}"),
                    suggested_fix: None,
                });
                continue;
            }
        };
        let Some(build_record_id) = frontmatter_string(&frontmatter, "build_record_id") else {
            continue;
        };
        if load_build_record(root, &build_record_id)?.is_none() {
            issues.push(LintIssue {
                severity: IssueSeverity::Error,
                kind: IssueKind::FrontmatterBuildRecordMissing,
                referring_page: relative_to_root(root, &page).to_string_lossy().into_owned(),
                line: 0,
                target: build_record_id.clone(),
                message: format!("page references missing build record '{build_record_id}'"),
                suggested_fix: None,
            });
        }
    }

    for record in load_all_build_records(root)? {
        if let Some(issue) = build_record_output_missing(root, &record) {
            issues.push(issue);
        }
    }

    Ok(issues)
}

/// Flag a build record whose declared outputs are all missing from disk.
///
/// A build record is considered fresh iff at least one declared output path
/// exists. `output_ids` are synthetic graph identities (e.g.
/// `wiki-source-src-xxxx`) and are **not** filesystem paths, so existence
/// checks must resolve through `metadata.output_paths`.
fn build_record_output_missing(root: &Path, record: &BuildRecord) -> Option<LintIssue> {
    if record.metadata.output_paths.is_empty() {
        return None;
    }
    let all_missing = record
        .metadata
        .output_paths
        .iter()
        .all(|p| !root.join(p).exists());
    if !all_missing {
        return None;
    }
    let target_path = record
        .metadata
        .output_paths
        .first()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(LintIssue {
        severity: IssueSeverity::Error,
        kind: IssueKind::BuildRecordOutputMissing,
        referring_page: target_path,
        line: 0,
        target: record.metadata.id.clone(),
        message: format!(
            "build record '{}' points to a missing output",
            record.metadata.id
        ),
        suggested_fix: None,
    })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn markdown_files_under(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !root.exists() {
        return Ok(files);
    }
    collect_markdown_files(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_markdown_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("scan {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_markdown_files(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            files.push(path);
        }
    }
    Ok(())
}

fn load_all_build_records(root: &Path) -> Result<Vec<BuildRecord>> {
    let dir = build_records_dir(root);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("scan {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("read build record {}", path.display()))?;
        let record = serde_json::from_str(&raw)
            .with_context(|| format!("deserialize build record {}", path.display()))?;
        records.push(record);
    }
    Ok(records)
}

fn same_build_fingerprint(left: &BuildRecord, right: &BuildRecord) -> bool {
    left.manifest_hash == right.manifest_hash
        && left.input_ids == right.input_ids
        && left.output_ids == right.output_ids
        && left.pass_name == right.pass_name
}

fn normalized_metadata_for_doc(root: &Path, doc_id: &str) -> Result<Option<NormalizedMetadata>> {
    let path = root.join("normalized").join(doc_id).join("metadata.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read normalized metadata {}", path.display()))?;
    let metadata = serde_json::from_str(&raw)
        .with_context(|| format!("deserialize normalized metadata {}", path.display()))?;
    Ok(Some(metadata))
}

fn detect_missing_citations(root: &Path, options: &LintOptions) -> Result<Vec<LintIssue>> {
    let mut issues = Vec::new();

    for page in markdown_files_under(&root.join(WIKI_DIR))? {
        // Skip auto-generated index pages (e.g. wiki/concepts/index.md,
        // wiki/sources/index.md). These are navigational listings, not source
        // material; requiring citations on them is a category error. Other
        // lint checks (broken-links, etc.) still evaluate index pages.
        if page
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("index.md"))
        {
            continue;
        }
        // Skip promoted question pages (wiki/questions/<slug>.md). These
        // pages encode citations via frontmatter `citations:` list plus
        // `[N]` bracket references in the body — a different structural
        // shape than the source-page model this check was built for.
        // See bn-3gs. Other lint checks (broken-links, orphans, etc.)
        // still evaluate question pages.
        let rel_for_skip = relative_to_root(root, &page);
        if rel_for_skip.starts_with("wiki/questions") {
            continue;
        }
        let raw = fs::read_to_string(&page)
            .with_context(|| format!("read wiki page {}", page.display()))?;
        let (frontmatter, body) = match read_frontmatter(&page) {
            Ok(v) => v,
            Err(err) => {
                issues.push(LintIssue {
                    kind: IssueKind::InvalidFrontmatter,
                    severity: IssueSeverity::Error,
                    referring_page: relative_to_root(root, &page)
                        .to_string_lossy()
                        .into_owned(),
                    line: 0,
                    target: String::new(),
                    message: format!("invalid YAML frontmatter: {err}"),
                    suggested_fix: None,
                });
                continue;
            }
        };
        let doc_ids =
            frontmatter_string_list(&frontmatter, "source_document_id", "source_document_ids");
        let rev_ids =
            frontmatter_string_list(&frontmatter, "source_revision_id", "source_revision_ids");
        let has_sources = !(doc_ids.is_empty() && rev_ids.is_empty());
        // Concept pages (under wiki/concepts/) store provenance as a structured
        // `sources:` list in frontmatter — that is the established shape written
        // by render_concept_page. Treat a non-empty `sources:` list as
        // satisfying the body-citation requirement for concept pages.
        let rel_path = relative_to_root(root, &page);
        let is_concept_page = rel_path.starts_with("wiki/concepts");
        let has_frontmatter_sources_list =
            frontmatter_has_non_empty_sequence(&frontmatter, "sources");
        let has_citations = if is_concept_page && has_frontmatter_sources_list {
            true
        } else {
            page_has_citations(&body)
        };

        if has_sources && has_citations {
            continue;
        }

        let rel_page = relative_to_root(root, &page).to_string_lossy().into_owned();
        let body_offset = raw.len().saturating_sub(body.len());

        for region in extract_managed_regions(&body)
            .into_iter()
            .filter(|region| is_synthetic_region(&frontmatter, region.id))
        {
            let target = format!("{rel_page}#{}", region.id);
            let line = line_number_at(&raw, body_offset + region.full_start);
            let message = match (has_sources, has_citations) {
                (false, false) => "synthetic region is missing source ids in frontmatter and citations in the page body".to_string(),
                (false, true) => "synthetic region is missing source ids in frontmatter".to_string(),
                (true, false) => "synthetic region is missing citations in the page body".to_string(),
                (true, true) => continue,
            };

            issues.push(LintIssue {
                severity: options.missing_citations_level.severity(),
                kind: IssueKind::MissingCitations,
                referring_page: rel_page.clone(),
                line,
                target,
                message,
                suggested_fix: Some(
                    "add source_* frontmatter ids and at least one citation link in the page body"
                        .to_string(),
                ),
            });
        }
    }

    Ok(issues)
}

fn is_synthetic_region(frontmatter: &serde_yaml::Mapping, region_id: &str) -> bool {
    let has_direct_source_ids = frontmatter
        .contains_key(Value::String("source_document_id".to_string()))
        || frontmatter.contains_key(Value::String("source_revision_id".to_string()));
    let has_plural_source_ids = frontmatter
        .contains_key(Value::String("source_document_ids".to_string()))
        || frontmatter.contains_key(Value::String("source_revision_ids".to_string()));
    let source_page_shape = has_direct_source_ids || has_plural_source_ids;

    if source_page_shape {
        !matches!(region_id, "title" | "citations")
    } else {
        !matches!(region_id, "citations")
    }
}

fn page_has_citations(body: &str) -> bool {
    extract_managed_regions(body)
        .into_iter()
        .find(|region| region.id == "citations")
        .is_some_and(|region| citations_exist(region.body(body)))
        || citations_exist(body)
}

fn citations_exist(text: &str) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty()
            && trimmed != "- _None yet._"
            && (trimmed.contains("[[") || (trimmed.contains('[') && trimmed.contains("](")))
    })
}

fn line_number_at(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

fn frontmatter_string_list(
    frontmatter: &serde_yaml::Mapping,
    singular: &str,
    plural: &str,
) -> Vec<String> {
    frontmatter
        .get(Value::String(plural.to_string()))
        .map_or_else(
            || frontmatter_string(frontmatter, singular).map_or_else(Vec::new, |value| vec![value]),
            yaml_string_list,
        )
}

fn yaml_string_list(value: &Value) -> Vec<String> {
    match value {
        Value::Sequence(values) => values
            .iter()
            .filter_map(|value| match value {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        Value::String(s) => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn frontmatter_has_non_empty_sequence(frontmatter: &serde_yaml::Mapping, key: &str) -> bool {
    match frontmatter.get(Value::String(key.to_string())) {
        Some(Value::Sequence(values)) => !values.is_empty(),
        _ => false,
    }
}

fn frontmatter_string(frontmatter: &serde_yaml::Mapping, key: &str) -> Option<String> {
    match frontmatter.get(Value::String(key.to_string())) {
        Some(Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn relative_to_root(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root)
        .map_or_else(|_| path.to_path_buf(), Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kb_core::{BuildRecord, EntityMetadata, Manifest, Status, save_build_record};
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::tempdir;

    fn metadata(id: &str) -> EntityMetadata {
        EntityMetadata {
            id: id.to_string(),
            created_at_millis: 1,
            updated_at_millis: 1,
            source_hashes: vec![],
            model_version: None,
            tool_version: Some("kb/test".to_string()),
            prompt_template_hash: None,
            dependencies: vec![],
            output_paths: vec![],
            status: Status::Fresh,
        }
    }

    fn build_record(id: &str, manifest_hash: &str, output: &str) -> BuildRecord {
        let mut meta = metadata(id);
        // Mirror production: `output_paths` carries the real filesystem paths,
        // while `output_ids` are synthetic graph identities.
        meta.output_paths = vec![PathBuf::from(output)];
        BuildRecord {
            metadata: meta,
            pass_name: "test".to_string(),
            input_ids: vec!["normalized/doc-1".to_string()],
            output_ids: vec![format!("wiki-{id}")],
            manifest_hash: manifest_hash.to_string(),
        }
    }

    #[test]
    fn broken_link_lint_reports_missing_page_and_anchor() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        Manifest::default().save(root).expect("save manifest");

        fs::write(
            root.join("wiki/concepts/rust.md"),
            "# Rust\n<!-- kb:begin id=summary -->\nSummary\n<!-- kb:end id=summary -->\n",
        )
        .expect("write rust page");
        fs::write(
            root.join("wiki/sources/page.md"),
            "# Page\nBroken [[wiki/concepts/rsut]].\nBroken [[wiki/concepts/rust#summry]].\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::BrokenLinks).expect("run lint");
        assert_eq!(report.issue_count, 2);
        assert_eq!(report.issues[0].kind, IssueKind::MissingPage);
        assert_eq!(report.issues[0].line, 2);
        assert_eq!(
            report.issues[0].suggested_fix.as_deref(),
            Some("wiki/concepts/rust")
        );
        assert_eq!(report.issues[1].kind, IssueKind::MissingAnchor);
        assert_eq!(report.issues[1].line, 3);
        assert_eq!(
            report.issues[1].suggested_fix.as_deref(),
            Some("wiki/concepts/rust#summary")
        );
    }

    #[test]
    fn broken_link_lint_ignores_wikilinks_in_yaml_frontmatter() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        Manifest::default().save(root).expect("save manifest");

        fs::write(
            root.join("wiki/sources/page.md"),
            "---\nsources:\n  - quote: \"See also: [[raft-overview]]\"\n---\n# Page\n\nBroken [[wiki/concepts/bar]].\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::BrokenLinks).expect("run lint");
        assert_eq!(
            report.issue_count, 1,
            "expected only body wiki-link reported, got: {report:?}"
        );
        assert_eq!(report.issues[0].line, 7);
        assert_eq!(report.issues[0].target, "wiki/concepts/bar");
    }

    #[test]
    fn broken_link_lint_ignores_wikilinks_in_fenced_code_blocks() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        Manifest::default().save(root).expect("save manifest");

        fs::write(
            root.join("wiki/sources/page.md"),
            "# Page\n\n```\n[[inside-code]]\n```\n\nBroken [[wiki/concepts/missing]].\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::BrokenLinks).expect("run lint");
        assert_eq!(
            report.issue_count, 1,
            "expected only body wiki-link reported, got: {report:?}"
        );
        assert_eq!(report.issues[0].line, 7);
        assert_eq!(report.issues[0].target, "wiki/concepts/missing");
    }

    #[test]
    fn broken_link_lint_ignores_wikilinks_in_inline_code_spans() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        Manifest::default().save(root).expect("save manifest");

        fs::write(
            root.join("wiki/sources/page.md"),
            "# Page\n\nExample: `[[inline-code]]` and real [[wiki/concepts/missing]].\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::BrokenLinks).expect("run lint");
        assert_eq!(
            report.issue_count, 1,
            "expected only body wiki-link reported, got: {report:?}"
        );
        assert_eq!(report.issues[0].line, 3);
        assert_eq!(report.issues[0].target, "wiki/concepts/missing");
    }

    #[test]
    fn broken_link_lint_understands_relative_markdown_links() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        Manifest::default().save(root).expect("save manifest");

        fs::write(
            root.join("wiki/concepts/borrow-checker.md"),
            "# Borrow checker\n",
        )
        .expect("write concept page");
        fs::write(
            root.join("wiki/sources/page.md"),
            "# Page\nSee [concept](../concepts/borrow-checker.md).\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::BrokenLinks).expect("run lint");
        assert!(report.is_clean(), "expected no lint issues: {report:?}");
    }

    #[test]
    fn parse_markdown_destination_ignores_titles() {
        assert_eq!(
            parse_markdown_destination("wiki/concepts/rust.md#summary \"Rust\""),
            Some("wiki/concepts/rust.md#summary".to_string())
        );
        assert_eq!(
            parse_markdown_destination("<wiki/concepts/rust.md#summary>"),
            Some("<wiki/concepts/rust.md#summary>".to_string())
        );
    }

    fn write_normalized_metadata(root: &Path, doc_id: &str, revision_id: &str) {
        let dir = root.join("normalized").join(doc_id);
        fs::create_dir_all(&dir).expect("create normalized dir");
        fs::write(
            dir.join("metadata.json"),
            format!("{{\"source_revision_id\":\"{revision_id}\"}}"),
        )
        .expect("write normalized metadata");
    }

    #[test]
    fn orphan_pages_are_reported_when_source_document_is_missing() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/sources/orphan.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Orphan\n",
        )
        .expect("write page");

        let report = run_lint(root, LintRule::Orphans).expect("lint report");
        assert_eq!(report.issue_count, 2);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == IssueKind::SourceDocumentMissing)
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == IssueKind::SourceRevisionMissing)
        );
        // Orphans must NOT report SourceRevisionStale — that belongs to
        // `stale_revision` as of bn-2tq.
        assert!(
            report
                .issues
                .iter()
                .all(|issue| issue.kind != IssueKind::SourceRevisionStale),
            "orphans should not report stale-revision issues: {report:?}"
        );
    }

    #[test]
    fn orphans_does_not_fire_on_stale_revision_when_source_is_present() {
        // Regression: bn-2tq. Page references rev-old, but the source was
        // re-ingested at rev-new and the page hasn't been recompiled. The
        // source is present, just at a newer revision — that's a
        // `stale_revision`, not an `orphan`.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/sources/src-a.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-old\n---\n# A\n",
        )
        .expect("write page");
        write_normalized_metadata(root, "doc-1", "rev-new");

        let report = run_lint(root, LintRule::Orphans).expect("lint report");
        assert!(
            report.is_clean(),
            "orphans must not fire when source is present: {report:?}"
        );
    }

    #[test]
    fn stale_revision_fires_when_revision_differs_from_live() {
        // bn-2tq: page revision differs from the normalized document's
        // current revision. `stale_revision` should fire.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/sources/src-a.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-old\n---\n# A\n",
        )
        .expect("write page");
        write_normalized_metadata(root, "doc-1", "rev-new");

        let report = run_lint(root, LintRule::StaleRevision).expect("lint report");
        assert_eq!(report.issue_count, 1, "{report:?}");
        assert_eq!(report.issues[0].kind, IssueKind::SourceRevisionStale);
        assert_eq!(report.issues[0].severity, IssueSeverity::Error);
        assert!(report.has_errors());
    }

    #[test]
    fn stale_revision_does_not_fire_when_revision_matches() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/sources/src-a.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# A\n",
        )
        .expect("write page");
        write_normalized_metadata(root, "doc-1", "rev-1");

        let report = run_lint(root, LintRule::StaleRevision).expect("lint report");
        assert!(
            report.is_clean(),
            "stale_revision must not fire on fresh pages: {report:?}"
        );
    }

    #[test]
    fn stale_revision_does_not_fire_when_source_is_orphaned() {
        // When the source document is entirely gone, that's an `orphans`
        // concern (SourceRevisionMissing). `stale_revision` should stay
        // quiet so we don't double-report.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/sources/src-a.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# A\n",
        )
        .expect("write page");
        // No normalized/doc-1/ — document is orphaned.

        let stale = run_lint(root, LintRule::StaleRevision).expect("lint report");
        assert!(
            stale.is_clean(),
            "stale_revision must not fire on orphans: {stale:?}"
        );
    }

    #[test]
    fn orphans_suppressed_when_orphaned_sources_marker_acknowledges_missing_src() {
        // bn-1yvn: `kb forget` stamps `orphaned_sources: [src-X]` on promoted
        // question pages instead of rewriting them. The orphans lint must
        // trust that marker and stay quiet for the acknowledged src-id.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/questions/how.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\n\
             id: q-1\n\
             title: How\n\
             source_document_ids:\n  - src-X\n\
             source_revision_ids:\n  - rev-X\n\
             orphaned_sources:\n  - src-X\n\
             ---\n\n# q\n",
        )
        .expect("write page");
        // No normalized/src-X/ — doc is gone, but user acknowledged it.

        let report = run_lint(root, LintRule::Orphans).expect("lint report");
        assert!(
            report.is_clean(),
            "orphans must honor orphaned_sources marker: {report:?}"
        );
    }

    #[test]
    fn orphans_still_fire_when_no_orphaned_sources_marker_is_present() {
        // bn-1yvn: the suppression is keyed strictly on the marker. A page
        // with no `orphaned_sources:` field still reports missing docs.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/questions/how.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\n\
             id: q-1\n\
             title: How\n\
             source_document_ids:\n  - src-X\n\
             source_revision_ids:\n  - rev-X\n\
             ---\n\n# q\n",
        )
        .expect("write page");

        let report = run_lint(root, LintRule::Orphans).expect("lint report");
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == IssueKind::SourceDocumentMissing
                    && issue.target == "src-X"),
            "unmarked orphan must still fire: {report:?}"
        );
    }

    #[test]
    fn orphans_fire_for_unflagged_src_even_when_another_src_is_acknowledged() {
        // bn-1yvn: if the page flags src-X but ALSO references src-Y which
        // is missing and NOT flagged, the lint must fire for src-Y only.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let page = root.join("wiki/questions/how.md");
        fs::create_dir_all(page.parent().expect("parent")).expect("create wiki dir");
        fs::write(
            &page,
            "---\n\
             id: q-1\n\
             title: How\n\
             source_document_ids:\n  - src-X\n  - src-Y\n\
             source_revision_ids:\n  - rev-X\n  - rev-Y\n\
             orphaned_sources:\n  - src-X\n\
             ---\n\n# q\n",
        )
        .expect("write page");
        // Neither src-X nor src-Y has a normalized dir — both are gone.

        let report = run_lint(root, LintRule::Orphans).expect("lint report");
        let missing_doc_targets: Vec<&str> = report
            .issues
            .iter()
            .filter(|issue| issue.kind == IssueKind::SourceDocumentMissing)
            .map(|issue| issue.target.as_str())
            .collect();
        assert_eq!(
            missing_doc_targets,
            vec!["src-Y"],
            "only the unflagged src should fire: {report:?}"
        );
    }

    #[test]
    fn lint_rule_parse_accepts_stale_revision_aliases() {
        assert_eq!(
            LintRule::parse(Some("stale-revision")).expect("parse"),
            LintRule::StaleRevision,
        );
        assert_eq!(
            LintRule::parse(Some("stale_revision")).expect("parse"),
            LintRule::StaleRevision,
        );
        assert_eq!(
            LintRule::parse(Some("stale-revisions")).expect("parse"),
            LintRule::StaleRevision,
        );
        // `stale` is still stale-artifacts, not stale-revision.
        assert_eq!(
            LintRule::parse(Some("stale")).expect("parse"),
            LintRule::StaleArtifacts,
        );
    }

    #[test]
    fn stale_check_uses_output_paths_not_output_ids_for_existence() {
        // Regression: bn-2bv. `output_ids` holds synthetic graph IDs like
        // `wiki-source-src-xxxx` which are NOT filesystem paths. Existence
        // checks must resolve through `metadata.output_paths`.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let output_rel = PathBuf::from("wiki/sources/src-04aca3d1.md");
        let output_abs = root.join(&output_rel);
        fs::create_dir_all(output_abs.parent().expect("parent")).expect("create output dir");
        fs::write(
            &output_abs,
            "---\nbuild_record_id: build:source-summary:src-04aca3d1\n---\n# Src\n",
        )
        .expect("write output");

        let mut record = build_record(
            "build:source-summary:src-04aca3d1",
            "manifest-a",
            output_rel.to_str().expect("utf-8"),
        );
        // Mirror the real shape: synthetic id, real path.
        record.output_ids = vec!["wiki-source-src-04aca3d1".to_string()];
        record.metadata.output_paths = vec![output_rel.clone()];

        Manifest {
            artifacts: BTreeMap::from([(output_rel, record.clone())]),
        }
        .save(root)
        .expect("save manifest");
        save_build_record(root, &record).expect("save build record");

        // Healthy KB: no BuildRecordOutputMissing.
        let report = run_lint(root, LintRule::StaleArtifacts).expect("lint report");
        assert!(
            report
                .issues
                .iter()
                .all(|i| i.kind != IssueKind::BuildRecordOutputMissing),
            "unexpected BuildRecordOutputMissing on healthy KB: {:?}",
            report.issues
        );

        // Delete the real file — the finding should now fire.
        fs::remove_file(&output_abs).expect("remove output");
        let report = run_lint(root, LintRule::StaleArtifacts).expect("lint report");
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.kind == IssueKind::BuildRecordOutputMissing),
            "expected BuildRecordOutputMissing after deleting output: {:?}",
            report.issues
        );
    }

    #[test]
    fn stale_artifacts_are_reported_when_manifest_and_build_record_diverge() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let output = root.join("wiki/sources/doc-1.md");
        fs::create_dir_all(output.parent().expect("parent")).expect("create output dir");
        fs::write(&output, "---\nbuild_record_id: build-1\n---\n# Doc\n").expect("write output");

        let manifest_record = build_record("build-1", "manifest-a", "wiki/sources/doc-1.md");
        Manifest {
            artifacts: BTreeMap::from([(PathBuf::from("wiki/sources/doc-1.md"), manifest_record)]),
        }
        .save(root)
        .expect("save manifest");
        save_build_record(
            root,
            &build_record("build-1", "manifest-b", "wiki/sources/doc-1.md"),
        )
        .expect("save build record");

        let report = run_lint(root, LintRule::StaleArtifacts).expect("lint report");
        assert_eq!(report.issue_count, 1);
        assert_eq!(report.issues[0].kind, IssueKind::ManifestMismatch);
    }

    #[test]
    fn missing_citations_warns_for_synthetic_regions_without_links() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/sources")).expect("create wiki dir");
        fs::write(
            root.join("wiki/sources/doc.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Source\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert_eq!(report.issue_count, 1);
        assert_eq!(report.issues[0].kind, IssueKind::MissingCitations);
        assert_eq!(report.issues[0].severity, IssueSeverity::Warning);
        assert!(!report.has_errors());
    }

    #[test]
    fn missing_citations_allows_extractive_regions_without_links() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/sources")).expect("create wiki dir");
        fs::write(
            root.join("wiki/sources/doc.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Source\n\n<!-- kb:begin id=title -->\nDocument title\n<!-- kb:end id=title -->\n\n## Citations\n<!-- kb:begin id=citations -->\n- _None yet._\n<!-- kb:end id=citations -->\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert!(report.is_clean(), "expected no issues: {report:?}");
    }

    #[test]
    fn invalid_frontmatter_is_reported_and_scan_continues() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        Manifest::default().save(root).expect("save manifest");

        // Bad frontmatter: unterminated quoted scalar.
        fs::write(
            root.join("wiki/concepts/broken.md"),
            "---\n'a: unterminated\n---\n# Hi\n",
        )
        .expect("write broken page");

        // Valid page with a broken wiki-link (broken-links check must still run).
        fs::write(
            root.join("wiki/sources/page.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Page\n\nSee [[wiki/concepts/missing]].\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::All).expect("lint should not bail");

        // InvalidFrontmatter reported (once per frontmatter-consuming pass; today
        // that's orphans, stale-revision, stale-artifacts, and missing-citations).
        let invalid: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.kind == IssueKind::InvalidFrontmatter)
            .collect();
        assert!(
            !invalid.is_empty(),
            "expected at least one InvalidFrontmatter issue, got: {:?}",
            report.issues
        );
        for issue in &invalid {
            assert_eq!(issue.severity, IssueSeverity::Error);
            assert_eq!(issue.line, 0);
            assert!(issue.referring_page.ends_with("broken.md"));
            assert!(issue.message.contains("invalid YAML frontmatter"));
        }

        // Broken-links check still ran.
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.kind == IssueKind::MissingPage),
            "broken-links check should still run, issues: {:?}",
            report.issues
        );

        // Orphans check still ran (doc-1 is missing).
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.kind == IssueKind::SourceDocumentMissing),
            "orphans check should still run, issues: {:?}",
            report.issues
        );

        // And has_errors is true.
        assert!(report.has_errors());
    }

    #[test]
    fn missing_citations_accepts_frontmatter_sources_on_concept_pages() {
        // Regression: bn-358. Concept pages store provenance as a structured
        // `sources:` list in frontmatter (written by render_concept_page). That
        // structured data IS the citation, so body-citation markers should not
        // be required when `sources:` + `source_document_ids:` are both
        // populated.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::write(
            root.join("wiki/concepts/rust.md"),
            "---\nsource_document_ids:\n  - doc-1\nsources:\n  - heading_anchor: intro\n    quote: \"Rust is a systems language.\"\n---\n# Rust\n\n<!-- kb:begin id=summary -->\nA short definition.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write concept page");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert!(
            report.is_clean(),
            "concept page with frontmatter sources should not warn: {report:?}"
        );
    }

    #[test]
    fn missing_citations_flags_concept_pages_missing_both_sources_and_citations() {
        // Regression guard: a concept page with neither `source_document_ids`
        // nor `sources:` (and no body citations) must still trigger the warning.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::write(
            root.join("wiki/concepts/rust.md"),
            "---\ngenerated_by: builder\n---\n# Rust\n\n<!-- kb:begin id=summary -->\nA short definition.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write concept page");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert!(
            !report.is_clean(),
            "concept page with no grounding should still warn: {report:?}"
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == IssueKind::MissingCitations),
            "expected MissingCitations issue, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn missing_citations_still_requires_body_citations_for_source_pages() {
        // Unchanged behavior: source pages under wiki/sources/ must have body
        // citations even if frontmatter has `sources:`.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");
        fs::write(
            root.join("wiki/sources/doc.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\nsources:\n  - heading_anchor: intro\n    quote: \"Rust is a systems language.\"\n---\n# Source\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write source page");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert!(
            !report.is_clean(),
            "source page without body citations should still warn: {report:?}"
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.kind == IssueKind::MissingCitations),
            "expected MissingCitations issue, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn missing_citations_skips_wiki_index_pages() {
        // Regression: bn-308. Auto-generated index pages (wiki/concepts/index.md,
        // wiki/sources/index.md) are navigational listings, not source material.
        // They should never trigger missing-citations warnings, even when their
        // body contains synthetic managed regions without citations.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create concepts dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");

        // Index page shaped like a source page (would normally warn) — must be skipped.
        fs::write(
            root.join("wiki/concepts/index.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Concepts\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write concepts index");
        fs::write(
            root.join("wiki/sources/index.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Sources\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write sources index");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert!(
            report.is_clean(),
            "index pages must not trigger missing-citations: {report:?}"
        );

        // Sanity: a non-index page with the same shape still warns.
        fs::write(
            root.join("wiki/sources/doc.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Source\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write non-index source page");
        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert_eq!(report.issue_count, 1);
        assert_eq!(report.issues[0].kind, IssueKind::MissingCitations);
        assert!(
            report.issues[0].referring_page.ends_with("doc.md"),
            "non-index page should still warn: {:?}",
            report.issues
        );
    }

    #[test]
    fn missing_citations_skips_wiki_question_pages() {
        // Regression: bn-3gs. Promoted question pages under wiki/questions/
        // encode citations via a frontmatter `citations:` list plus `[N]`
        // bracket references in the body — a different structural shape than
        // the source-page model this check was built for. They must not
        // trigger missing-citations warnings, even when their body contains
        // synthetic managed regions without inline wiki-links.
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/questions")).expect("create questions dir");
        fs::create_dir_all(root.join("wiki/sources")).expect("create sources dir");

        // Question page shaped like a source page (would normally warn) — must be skipped.
        fs::write(
            root.join("wiki/questions/what-is-x.md"),
            "---\ntype: question_answer\nsource_document_id: doc-1\nsource_revision_id: rev-1\ncitations:\n  - id: 1\n    source: doc-1\n---\n# What is X?\n\n## Summary\n<!-- kb:begin id=summary -->\nAn answer referencing [1].\n<!-- kb:end id=summary -->\n",
        )
        .expect("write question page");

        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert!(
            report.is_clean(),
            "question pages must not trigger missing-citations: {report:?}"
        );

        // Sanity: a non-question page with the same shape still warns.
        fs::write(
            root.join("wiki/sources/doc.md"),
            "---\nsource_document_id: doc-1\nsource_revision_id: rev-1\n---\n# Source\n\n## Summary\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write non-question source page");
        let report = run_lint(root, LintRule::MissingCitations).expect("lint report");
        assert_eq!(report.issue_count, 1);
        assert_eq!(report.issues[0].kind, IssueKind::MissingCitations);
        assert!(
            report.issues[0].referring_page.ends_with("doc.md"),
            "non-question source page should still warn: {:?}",
            report.issues
        );
    }

    #[test]
    fn missing_citations_can_be_promoted_to_error() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/concepts")).expect("create wiki dir");
        fs::write(
            root.join("wiki/concepts/rust.md"),
            "---\nsource_revision_ids:\n  - rev-1\ngenerated_by: builder\n---\n# Rust\n\n<!-- kb:begin id=summary -->\nSynthetic summary.\n<!-- kb:end id=summary -->\n",
        )
        .expect("write concept page");

        let report = run_lint_with_options(
            root,
            LintRule::MissingCitations,
            &LintOptions {
                require_citations: true,
                missing_citations_level: MissingCitationsLevel::Error,
                missing_concepts: MissingConceptsConfig::default(),
            },
        )
        .expect("lint report");
        assert_eq!(report.issue_count, 1);
        assert_eq!(report.issues[0].severity, IssueSeverity::Error);
        assert!(report.has_errors());
    }
}

// ---------------------------------------------------------------------------
// Missing-concept detection (bn-31lt)
// ---------------------------------------------------------------------------
//
// Walks every `normalized/<doc>/source.md` body, extracts capitalized
// multi-word spans and backtick-quoted identifiers, and emits a review item of
// kind `concept_candidate` for terms that:
//
// 1. appear in at least `min_sources` distinct source documents,
// 2. have at least `min_mentions` total mentions across the corpus,
// 3. are not already the name/alias of an existing concept page,
// 4. are not English stopwords or generic filler.
//
// This is a cheap, regex-based first pass — noisy but fast. The cross-source
// filter (rule #1) is what prevents every "Smith" in a single document from
// being flagged.

/// Configuration for the `missing_concepts` lint check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingConceptsConfig {
    /// Whether the check runs at all. Default: `true`.
    pub enabled: bool,
    /// A candidate term must appear in at least this many distinct source
    /// documents to be flagged. Default: `3`.
    pub min_sources: usize,
    /// A candidate term must have at least this many total mentions across
    /// the corpus. Default: `5`.
    pub min_mentions: usize,
}

impl Default for MissingConceptsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_sources: 3,
            min_mentions: 5,
        }
    }
}

/// A single candidate term flagged by the missing-concepts lint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptCandidateHit {
    /// The surface form most commonly used for the candidate (e.g.
    /// `FooBar System` or `Byzantine Fault Tolerance`).
    pub name: String,
    /// Sorted, deduplicated list of source-document ids in which the term
    /// appears.
    pub source_ids: Vec<String>,
    /// Total mention count across the corpus.
    pub mention_count: usize,
}

/// Capitalized multi-word span extractor.
///
/// Matches 2–4 consecutive Titlecase or CamelCase words (e.g. `Rust
/// Language`, `Byzantine Fault Tolerance`, `FooBar System`). Single
/// Titlecase words are intentionally *not* matched — too noisy
/// (sentence-initial words, proper nouns like "The", "Mr.", etc.).
///
/// The inner word pattern `[A-Z][A-Za-z]+` allows internal uppercase
/// letters so CamelCase identifiers flow through — the bone spec called
/// this out explicitly: "catches proper nouns and CamelCase".
fn multiword_term_regex() -> Regex {
    Regex::new(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}\b")
        .expect("valid multiword term regex")
}

/// Backtick-quoted identifier extractor.
///
/// Matches anything between single backticks that looks like a code identifier
/// — letters, digits, `_`, `-`, `.`, `:`, `/`. Bounds the match length to a
/// sensible identifier size to avoid capturing code snippets.
fn backtick_identifier_regex() -> Regex {
    Regex::new(r"`([A-Za-z][A-Za-z0-9_\-./:]{2,63})`").expect("valid backtick identifier regex")
}

/// Lowercased English stopwords + common filler used to discard candidates
/// whose first token is a throwaway English word. Importing the query-side
/// list wholesale would drag `kb-llm` + `tokio` into `kb-lint`; the set below
/// is a straight copy of the hand-picked list in `kb-query::lexical::STOPWORDS`
/// plus a handful of extra capitalized-sentence-start words that creep into
/// the multi-word regex (e.g. "The Next", "This Means"). Keep it small — it's
/// a noise filter, not a content filter.
const MISSING_CONCEPTS_STOPWORDS: &[&str] = &[
    // Core (mirrors kb-query::lexical::STOPWORDS; copied rather than depended
    // on to keep kb-lint's dep tree small).
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "of", "in", "on", "at", "by", "for", "with", "to", "from", "and", "or", "but", "not", "it",
    "its", "this", "that", "these", "those", "as", "if", "so", "such", "do", "does", "did", "can",
    "will", "would", "should", "could", "how", "why", "when", "where", "which", "whose", "whom",
    "also", "then", "than", "some", "any", "most", "many", "about", "across", "around", "between",
    "during", "into", "onto", "over", "under", "while",
    // Extras: sentence-initial capitalized words that leak through the
    // multi-word regex because they're followed by another Titlecase word.
    "there", "here", "we", "you", "they", "our", "your", "their", "his", "her",
    "section", "chapter", "figure", "table", "page", "example", "note", "see",
    "use", "using", "used", "one", "two", "three", "four", "five", "first",
    "second", "third", "last", "next", "new", "old", "both", "either", "neither",
];

/// Entry point that emits `LintIssue`s for the `kb lint --check
/// missing-concepts` surface. Mirrors the review-item-producing
/// [`check_missing_concepts`] but renders the findings as warnings so the
/// existing lint reporter can display them.
fn detect_missing_concepts_issues(
    root: &Path,
    config: &MissingConceptsConfig,
) -> Result<Vec<LintIssue>> {
    let hits = check_missing_concepts_raw(root, config)?;
    Ok(hits
        .into_iter()
        .map(|hit| LintIssue {
            severity: IssueSeverity::Warning,
            kind: IssueKind::ConceptCandidate,
            referring_page: String::new(),
            line: 0,
            target: hit.name.clone(),
            message: format!(
                "concept candidate '{}' mentioned in {} source(s) ({} total mentions) but no concept page exists",
                hit.name,
                hit.source_ids.len(),
                hit.mention_count,
            ),
            suggested_fix: Some(format!(
                "create wiki/concepts/{}.md (or approve review item 'concept-candidate:{}')",
                slug_from_title(&hit.name),
                slug_from_title(&hit.name),
            )),
        })
        .collect())
}

/// Scan normalized source bodies and return one `ReviewItem` per concept
/// candidate that passes the `min_sources` / `min_mentions` thresholds and is
/// not already covered by a concept page.
///
/// # Errors
///
/// Returns an error when the `normalized/` or `wiki/concepts/` directories
/// cannot be scanned, a file cannot be read, or the system clock cannot be
/// queried for timestamps.
pub fn check_missing_concepts(
    root: &Path,
    config: &MissingConceptsConfig,
) -> Result<Vec<ReviewItem>> {
    let hits = check_missing_concepts_raw(root, config)?;
    if hits.is_empty() {
        return Ok(Vec::new());
    }

    let now = unix_time_ms()?;
    let mut items = Vec::with_capacity(hits.len());
    for hit in hits {
        items.push(build_concept_candidate_review_item(&hit, now));
    }
    Ok(items)
}

fn build_concept_candidate_review_item(hit: &ConceptCandidateHit, now: u64) -> ReviewItem {
    let slug = slug_from_title(&hit.name);
    let id = format!("lint:concept-candidate:{slug}");
    let destination = PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{slug}.md"));

    // Hash the candidate name + source ids so save_review_item's
    // rejected-item dedup works across re-runs against the same corpus.
    let mut fingerprint_parts: Vec<&[u8]> = vec![hit.name.as_bytes()];
    for src in &hit.source_ids {
        fingerprint_parts.push(src.as_bytes());
    }
    let fingerprint = kb_core::hash_many(&fingerprint_parts).to_hex();

    let comment = format!(
        "Term '{}' is mentioned in {} source(s) ({} total mention(s)) but has no concept page. \
         Sources: {}. Approve to draft wiki/concepts/{}.md from the mentions (LLM step \
         currently stubbed — see bone bn-31lt).",
        hit.name,
        hit.source_ids.len(),
        hit.mention_count,
        hit.source_ids.join(", "),
        slug,
    );

    ReviewItem {
        metadata: EntityMetadata {
            id: id.clone(),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![fingerprint],
            model_version: None,
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: None,
            dependencies: hit.source_ids.clone(),
            output_paths: vec![destination.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptCandidate,
        target_entity_id: id,
        proposed_destination: Some(destination.clone()),
        citations: hit.source_ids.clone(),
        affected_pages: vec![destination],
        created_at_millis: now,
        status: ReviewStatus::Pending,
        comment,
    }
}

/// Public wrapper around the missing-concepts extractor.
///
/// The impute module (and other callers that need the per-hit source ids
/// and mention counts) can reuse the same filtering logic as the lint and
/// review-queue paths.
///
/// # Errors
///
/// Returns an error when the KB directory cannot be scanned.
pub fn check_missing_concepts_hits(
    root: &Path,
    config: &MissingConceptsConfig,
) -> Result<Vec<ConceptCandidateHit>> {
    check_missing_concepts_raw(root, config)
}

/// Core extractor + filter. Shared by `detect_missing_concepts_issues` (lint
/// surface) and `check_missing_concepts` (review-queue writer). Deterministic:
/// the returned list is sorted by (source count desc, mention count desc,
/// name asc) so lint output and review-item ordering are stable across runs.
fn check_missing_concepts_raw(
    root: &Path,
    config: &MissingConceptsConfig,
) -> Result<Vec<ConceptCandidateHit>> {
    if !config.enabled {
        return Ok(Vec::new());
    }

    let normalized_root = root.join("normalized");
    if !normalized_root.exists() {
        return Ok(Vec::new());
    }

    // Build the exclusion set: concept names + aliases (lowercased, trimmed).
    let existing_terms = load_existing_concept_terms(root)?;

    let multiword_re = multiword_term_regex();
    let backtick_re = backtick_identifier_regex();

    // candidate_key -> (canonical_display_form, source_ids_set, total_mentions)
    let mut tally: BTreeMap<String, CandidateAgg> = BTreeMap::new();

    for entry in fs::read_dir(&normalized_root)
        .with_context(|| format!("scan normalized dir {}", normalized_root.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Ok(doc_id) = entry.file_name().into_string() else {
            continue;
        };
        let source_path = entry.path().join("source.md");
        if !source_path.exists() {
            continue;
        }
        let body = fs::read_to_string(&source_path)
            .with_context(|| format!("read normalized source {}", source_path.display()))?;
        let scan_body = strip_code_blocks(&body);

        // Scan paragraph-by-paragraph so the multi-word regex's `\s+`
        // connector can't bridge two paragraphs (e.g. a trailing
        // "FooBar System" at the end of paragraph N joining with a
        // sentence-opening "The FooBar" at the start of paragraph N+1 into
        // the junk candidate "FooBar System The FooBar"). Paragraphs are
        // separated by one or more blank lines.
        for paragraph in split_paragraphs(&scan_body) {
            for cap in multiword_re.find_iter(paragraph) {
                let raw = cap.as_str().trim();
                // The multi-word regex is greedy, so "The FooBar System"
                // matches as a single span. Peel stopword tokens from both
                // ends so phrases like "The FooBar System" surface as
                // "FooBar System" and "Tree Storage Engines The" surface as
                // "Tree Storage Engines". Stop as soon as the remaining
                // phrase still has at least two words; if peeling leaves
                // fewer than two words, drop the match entirely (single-word
                // candidates are out-of-scope — too noisy).
                let Some(term) = strip_leading_stopwords(raw) else {
                    continue;
                };
                if !accept_candidate_term(&term, &existing_terms) {
                    continue;
                }
                record_hit(&mut tally, &term, &doc_id);
            }
        }

        for cap in backtick_re.captures_iter(&body) {
            let Some(m) = cap.get(1) else { continue };
            let term = m.as_str().trim();
            if !accept_candidate_term(term, &existing_terms) {
                continue;
            }
            record_hit(&mut tally, term, &doc_id);
        }
    }

    let mut hits: Vec<ConceptCandidateHit> = tally
        .into_iter()
        .filter_map(|(_key, agg)| {
            let source_count = agg.source_ids.len();
            if source_count < config.min_sources {
                return None;
            }
            if agg.mention_count < config.min_mentions {
                return None;
            }
            Some(ConceptCandidateHit {
                name: agg.display,
                source_ids: agg.source_ids.into_iter().collect(),
                mention_count: agg.mention_count,
            })
        })
        .collect();

    // Deterministic ordering: source count desc, then mention count desc,
    // then name asc. Stable across runs so snapshots / test assertions
    // don't flap on equal-weight candidates.
    hits.sort_by(|a, b| {
        b.source_ids
            .len()
            .cmp(&a.source_ids.len())
            .then_with(|| b.mention_count.cmp(&a.mention_count))
            .then_with(|| a.name.cmp(&b.name))
    });

    Ok(hits)
}

struct CandidateAgg {
    display: String,
    source_ids: BTreeSet<String>,
    mention_count: usize,
}

fn record_hit(tally: &mut BTreeMap<String, CandidateAgg>, term: &str, doc_id: &str) {
    let key = term.to_ascii_lowercase();
    let entry = tally.entry(key).or_insert_with(|| CandidateAgg {
        display: term.to_string(),
        source_ids: BTreeSet::new(),
        mention_count: 0,
    });
    entry.source_ids.insert(doc_id.to_string());
    entry.mention_count += 1;
}

/// Load every existing concept's name and aliases (lowercased) so the lint
/// doesn't flag terms that are already covered.
fn load_existing_concept_terms(root: &Path) -> Result<BTreeSet<String>> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    let concepts_dir = root.join(WIKI_CONCEPTS_DIR);
    if !concepts_dir.exists() {
        return Ok(set);
    }
    for entry in fs::read_dir(&concepts_dir)
        .with_context(|| format!("scan concept pages dir {}", concepts_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        if path.file_name().is_some_and(|n| n == "index.md") {
            continue;
        }
        let Ok((frontmatter, _)) = read_frontmatter(&path) else {
            continue;
        };
        if let Some(name) = frontmatter.get(Value::String("name".to_string())).and_then(Value::as_str) {
            set.insert(name.trim().to_ascii_lowercase());
        }
        if let Some(Value::Sequence(aliases)) =
            frontmatter.get(Value::String("aliases".to_string()))
        {
            for alias in aliases {
                if let Some(s) = alias.as_str() {
                    set.insert(s.trim().to_ascii_lowercase());
                }
            }
        }
        // Also suppress flagging based on the file stem, since the stem is
        // the canonical slug and occasionally the only identity a page has
        // (e.g. historical pages without a `name:` frontmatter key).
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            set.insert(stem.trim().to_ascii_lowercase());
        }
    }
    Ok(set)
}

/// Gate a raw candidate string against length bounds and the set of
/// already-covered concept terms. Leading-stopword peeling is done by
/// [`strip_leading_stopwords`] before this check is called.
fn accept_candidate_term(term: &str, existing: &BTreeSet<String>) -> bool {
    let trimmed = term.trim();
    if trimmed.len() < 3 || trimmed.len() > 80 {
        return false;
    }
    // Reject pure-lowercase-alpha words (e.g. `country`, `price`, `append`,
    // `template`). These come from the backtick-identifier regex admitting
    // plain English words in inline backticks — noise for a KB. A real
    // identifier has at least one capital letter, digit, or non-alpha
    // character (`/`, `.`, `-`, `_`, `:`) so this filter only drops the
    // pure-lowercase-English-word case. Multi-word phrases contain a space
    // (not `[a-z]`) so they pass through untouched.
    if !trimmed.is_empty() && trimmed.bytes().all(|b| b.is_ascii_lowercase()) {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    if existing.contains(&lowered) {
        return false;
    }
    // Reject all-stopword phrases (rare after leading-peel, but defends
    // against edge cases like "Next One" that slipped through peeling).
    let all_stop = trimmed.split_whitespace().all(is_stopword);
    if all_stop {
        return false;
    }
    true
}

/// Peel stopword tokens from both ends of a multi-word candidate so phrases
/// like `"The FooBar System"` surface as `"FooBar System"` and
/// `"Tree Storage Engines The"` surface as `"Tree Storage Engines"`.
/// Returns `None` when the remaining phrase has fewer than two tokens
/// (single-word candidates are out-of-scope for this check — too noisy).
fn strip_leading_stopwords(raw: &str) -> Option<String> {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }
    let mut start = 0;
    while start < tokens.len() && is_stopword(tokens[start]) {
        start += 1;
    }
    let mut end = tokens.len();
    while end > start && is_stopword(tokens[end - 1]) {
        end -= 1;
    }
    let rest = &tokens[start..end];
    if rest.len() < 2 {
        return None;
    }
    Some(rest.join(" "))
}

/// Split a scan body into paragraphs on blank-line boundaries. Used by the
/// multi-word regex pass so `\s+` can't bridge two paragraphs.
///
/// A "blank line" here is any run of one-or-more lines that contain only
/// whitespace. Returned paragraphs are the non-blank spans between those
/// boundaries; empty strings are filtered out so consumers don't have to.
fn split_paragraphs(body: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut start: Option<usize> = None;
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Find end of current line.
        let line_start = i;
        let mut line_end = i;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let line = &body[line_start..line_end];
        let blank = line.chars().all(char::is_whitespace);
        if blank {
            if let Some(s) = start.take() {
                let para = body[s..line_start].trim_end_matches('\n');
                if !para.is_empty() {
                    out.push(para);
                }
            }
        } else if start.is_none() {
            start = Some(line_start);
        }
        // Advance past the newline (if any).
        i = if line_end < bytes.len() {
            line_end + 1
        } else {
            line_end
        };
    }
    if let Some(s) = start {
        let para = body[s..].trim_end_matches('\n');
        if !para.is_empty() {
            out.push(para);
        }
    }
    out
}

fn is_stopword(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    MISSING_CONCEPTS_STOPWORDS.contains(&lower.as_str())
}

/// Replace fenced ``` ``` code blocks with blank space so the multi-word
/// regex doesn't pick up code identifiers (those are handled by the backtick
/// regex separately). Inline single-backtick spans are left alone — the
/// backtick regex picks identifiers out of them and the multi-word regex
/// skips them naturally because they aren't `TitleCase` phrases.
fn strip_code_blocks(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut in_fence = false;
    for line in body.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        } else if !in_fence {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Duplicate concept detection
// ---------------------------------------------------------------------------

const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";
const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.85;
/// Minimum word-bigram overlap required between two concept definition hints
/// before a name-similarity hit will be treated as a near-duplicate. Chosen
/// deliberately low: genuine duplicates tend to restate the same idea, so any
/// non-trivial overlap clears the bar, while structurally distinct concepts
/// that merely share a topic word ("consensus", "causal") score near zero.
const DEFAULT_DEFINITION_SIMILARITY_THRESHOLD: f64 = 0.15;

/// Common technical words that dominate concept names in a distributed-systems
/// / programming-languages corpus and therefore drive bigram false positives
/// (e.g. every "... consensus" concept matches every other "... consensus").
/// We strip these before name-similarity scoring so the comparison happens on
/// the distinguishing part of the term.
///
/// Keep this list *small* and lowercase. Words here must be topic markers, not
/// discriminators — adding something like "checker" would break genuine dups
/// like "Borrow check" vs "Borrow checker".
const STOP_WORDS: &[&str] = &[
    "consensus",
    "causal",
    "lock",
    "data",
    "store",
    "system",
    "tree",
    "model",
    "protocol",
];

/// Configuration for the duplicate-concepts lint check.
#[derive(Debug, Clone)]
pub struct DuplicateConceptsConfig {
    /// Name/alias similarity threshold for flagging pairs (0.0–1.0). Default: 0.85.
    pub similarity_threshold: f64,
    /// Definition-text similarity threshold (word-bigram Dice). A pair must
    /// clear both `similarity_threshold` on names *and* this value on
    /// definitions to be flagged. Default: 0.15.
    ///
    /// When both concepts lack a definition hint, this check is bypassed so
    /// we still catch duplicates discovered before bodies are populated.
    pub definition_similarity_threshold: f64,
}

impl Default for DuplicateConceptsConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            definition_similarity_threshold: DEFAULT_DEFINITION_SIMILARITY_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone)]
struct ConceptRecord {
    id: String,
    name: String,
    aliases: Vec<String>,
    /// First paragraph of the concept body (the "definition hint" rendered by
    /// the merge pipeline). Empty when the page has no body.
    definition_hint: String,
}

/// Scan concept pages under `wiki/concepts/` and return a `ReviewItem` for every
/// near-duplicate pair.
///
/// Two concepts are near-duplicates when the highest bigram-similarity score
/// across all pairwise comparisons of their names and aliases meets or exceeds
/// `config.similarity_threshold`.
///
/// # Errors
///
/// Returns an error when the concept directory cannot be scanned, a concept page
/// cannot be parsed, or the system clock cannot be queried for timestamps.
pub fn check_duplicate_concepts(
    root: &Path,
    config: &DuplicateConceptsConfig,
) -> Result<Vec<ReviewItem>> {
    let concepts = load_concepts(root)?;
    let now = unix_time_ms()?;
    let mut items = Vec::new();

    for (i, a) in concepts.iter().enumerate() {
        for b in concepts.iter().skip(i + 1) {
            let Some((term_a, term_b, score)) = best_term_pair(a, b) else {
                continue;
            };
            if score < config.similarity_threshold {
                continue;
            }

            // Require definition-text overlap in addition to name/alias
            // similarity. Pairs like "Byzantine consensus" vs "Raft consensus"
            // share the topic word but describe different ideas — their
            // definition hints have near-zero n-gram overlap.
            //
            // If neither side has a definition (fresh concepts with empty
            // bodies, or synthetic tests), we fall back to name-only to
            // avoid regressing the pre-definition behavior.
            let have_defs = !a.definition_hint.is_empty() && !b.definition_hint.is_empty();
            if have_defs {
                let def_score = definition_similarity(&a.definition_hint, &b.definition_hint);
                if def_score < config.definition_similarity_threshold {
                    continue;
                }
            }

            items.push(build_review_item(a, b, &term_a, &term_b, score, now));
        }
    }

    Ok(items)
}

fn load_concepts(root: &Path) -> Result<Vec<ConceptRecord>> {
    let dir = root.join(WIKI_CONCEPTS_DIR);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in std::fs::read_dir(&dir)
        .with_context(|| format!("scan concept pages directory {}", dir.display()))?
    {
        let path = entry
            .with_context(|| format!("read directory entry in {}", dir.display()))?
            .path();

        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        if path.file_name().is_some_and(|n| n == "index.md") {
            continue;
        }

        match parse_concept_page(&path) {
            Ok(record) if !record.name.is_empty() => records.push(record),
            Ok(_) => {}
            Err(err) => tracing::warn!("skipping {}: {err}", path.display()),
        }
    }

    records.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(records)
}

fn parse_concept_page(path: &Path) -> Result<ConceptRecord> {
    let (frontmatter, body) = read_frontmatter(path)
        .with_context(|| format!("read frontmatter from {}", path.display()))?;

    let id = frontmatter
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let name = frontmatter
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let aliases = frontmatter
        .get("aliases")
        .and_then(Value::as_sequence)
        .map(|seq| {
            seq.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    let definition_hint = extract_definition_hint(&body);

    Ok(ConceptRecord {
        id,
        name,
        aliases,
        definition_hint,
    })
}

/// Extract the first non-heading paragraph from a concept page body.
///
/// Concept pages render as:
///
/// ```text
/// # Concept name
///
/// <definition hint paragraph(s)>
///
/// ## Backlinks
/// <!-- kb:begin id=backlinks -->
/// ...
/// ```
///
/// We return the paragraph text (joined), stripped of surrounding whitespace.
/// Managed regions (like the backlinks block) and any `## ...` section are
/// excluded so auto-generated backlinks don't contaminate the similarity score.
fn extract_definition_hint(body: &str) -> String {
    let mut collected: Vec<&str> = Vec::new();
    let mut started = false;
    for raw in body.lines() {
        let line = raw.trim();

        // Stop as soon as we hit a new section (## Backlinks, ## Sources, ...).
        // Also stop at any kb-managed region marker.
        if line.starts_with("## ") || line.starts_with("<!-- kb:begin") {
            break;
        }

        // Skip the leading '# Name' heading and any blank lines preceding the
        // first paragraph.
        if !started {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            started = true;
        } else if line.is_empty() {
            // Allow blank lines inside the first section (between paragraphs)
            // but stop as soon as we've collected at least one paragraph and
            // hit the *second* blank line gap — prevents pulling in unrelated
            // trailing content that isn't explicitly sectioned.
            if collected.last().is_some_and(|p: &&str| p.is_empty()) {
                break;
            }
            collected.push("");
            continue;
        }

        collected.push(line);
    }

    collected
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn all_terms(concept: &ConceptRecord) -> Vec<String> {
    let mut terms = vec![concept.name.clone()];
    terms.extend(concept.aliases.iter().cloned());
    terms
}

fn best_term_pair(a: &ConceptRecord, b: &ConceptRecord) -> Option<(String, String, f64)> {
    let a_terms = all_terms(a);
    let b_terms = all_terms(b);

    let mut best: Option<(String, String, f64)> = None;
    for ta in &a_terms {
        for tb in &b_terms {
            // Strip common topic words ("consensus", "causal", …) before
            // scoring. When a term is *entirely* stop-words after stripping,
            // fall back to the raw normalized form so legitimate single-word
            // names (e.g. "Ownership") still compare.
            let na = normalize_term(ta);
            let nb = normalize_term(tb);
            let sa = strip_stop_words(&na);
            let sb = strip_stop_words(&nb);
            let (la, lb) = match (sa.is_empty(), sb.is_empty()) {
                (false, false) => (sa, sb),
                _ => (na, nb),
            };
            let score = bigram_similarity(&la, &lb);
            let is_better = best.as_ref().is_none_or(|entry| score > entry.2);
            if is_better {
                best = Some((ta.clone(), tb.clone(), score));
            }
        }
    }

    best
}

fn normalize_term(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Remove every whole-word occurrence of a `STOP_WORDS` entry from an
/// already-lowercased, whitespace-normalized term. Operates at word granularity
/// so "lockfile" survives while "lock" is stripped.
fn strip_stop_words(s: &str) -> String {
    s.split_whitespace()
        .filter(|w| !STOP_WORDS.contains(w))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Dice coefficient over *word* bigrams (plus unigrams as a fallback) for two
/// definition-hint strings. Word-level is a better fit than char-level for
/// longer text because it captures phrase overlap without being dominated by
/// common character sequences.
#[allow(clippy::cast_precision_loss)]
fn definition_similarity(a: &str, b: &str) -> f64 {
    let a_words: Vec<String> = a
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect();
    let b_words: Vec<String> = b
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect();

    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    // Use word bigrams when both sides have at least two words; otherwise
    // fall back to word unigrams so single-sentence fragments still compare.
    let build_grams = |words: &[String]| -> Vec<String> {
        if words.len() >= 2 {
            words.windows(2).map(|w| format!("{} {}", w[0], w[1])).collect()
        } else {
            words.to_vec()
        }
    };

    let a_grams = build_grams(&a_words);
    let mut b_remaining = build_grams(&b_words);

    if a_grams.is_empty() || b_remaining.is_empty() {
        return 0.0;
    }

    let total = a_grams.len() + b_remaining.len();
    let mut intersection: usize = 0;
    for g in &a_grams {
        if let Some(pos) = b_remaining.iter().position(|x| x == g) {
            intersection += 1;
            b_remaining.swap_remove(pos);
        }
    }

    2.0 * intersection as f64 / total as f64
}

/// Dice coefficient over character bigrams for two pre-normalized strings.
#[allow(clippy::cast_precision_loss)]
fn bigram_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.len() < 2 || b_chars.len() < 2 {
        return 0.0;
    }

    let a_bigrams: Vec<(char, char)> = a_chars.windows(2).map(|w| (w[0], w[1])).collect();
    let mut b_remaining: Vec<(char, char)> = b_chars.windows(2).map(|w| (w[0], w[1])).collect();

    let total = a_bigrams.len() + b_remaining.len();
    let mut intersection: usize = 0;
    for bg in &a_bigrams {
        if let Some(pos) = b_remaining.iter().position(|x| x == bg) {
            intersection += 1;
            b_remaining.swap_remove(pos);
        }
    }

    2.0 * intersection as f64 / total as f64
}

fn build_review_item(
    a: &ConceptRecord,
    b: &ConceptRecord,
    term_a: &str,
    term_b: &str,
    score: f64,
    now: u64,
) -> ReviewItem {
    let review_path = PathBuf::from("reviews/merges").join(format!(
        "lint-duplicate-concepts-{}-{}.json",
        slug_from_title(&a.id),
        slug_from_title(&b.id)
    ));

    ReviewItem {
        metadata: EntityMetadata {
            id: format!("lint:duplicate-concepts:{}:{}", a.id, b.id),
            created_at_millis: now,
            updated_at_millis: now,
            source_hashes: vec![],
            model_version: None,
            tool_version: Some(format!(
                "{}/{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION")
            )),
            prompt_template_hash: None,
            dependencies: vec![a.id.clone(), b.id.clone()],
            output_paths: vec![review_path.clone()],
            status: Status::NeedsReview,
        },
        kind: ReviewKind::ConceptMerge,
        target_entity_id: a.id.clone(),
        proposed_destination: Some(review_path),
        citations: Vec::new(),
        affected_pages: vec![
            PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{}.md", slug_from_title(&a.name))),
            PathBuf::from(WIKI_CONCEPTS_DIR).join(format!("{}.md", slug_from_title(&b.name))),
        ],
        created_at_millis: now,
        status: ReviewStatus::Pending,
        comment: format!(
            "Near-duplicate of '{}' (similarity: {score:.2}; matched: '{}' \u{2248} '{}')",
            b.id, term_a, term_b
        ),
    }
}

fn unix_time_ms() -> Result<u64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before Unix epoch")?;
    duration
        .as_millis()
        .try_into()
        .context("system time exceeds u64 millisecond range")
}

// ---------------------------------------------------------------------------
// Thin-concept-body discovery (bn-xt4o)
// ---------------------------------------------------------------------------
//
// Lightweight, non-LLM scan over `wiki/concepts/*.md`. A concept is "thin"
// when its body (everything after the frontmatter + leading `# Title`
// heading, excluding `<!-- kb:* -->` managed regions) contains fewer than
// `min_body_words` words. Used by `kb lint --impute` to pick concepts that
// are worth asking a web-search agent to backfill.

/// Configuration for the thin-concept-body discovery pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinConceptBodyConfig {
    /// A concept body with strictly fewer than this many words is flagged
    /// as thin. Default: `12` — just barely enough for a single-sentence
    /// general definition.
    pub min_body_words: usize,
}

impl Default for ThinConceptBodyConfig {
    fn default() -> Self {
        Self { min_body_words: 12 }
    }
}

/// One thin-body concept. The path is relative to the KB root; the
/// `body_words` count is the post-trim post-managed-region word count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinConceptHit {
    /// Relative path of the concept page.
    pub page_path: PathBuf,
    /// `id` frontmatter field (`concept:<slug>`). Empty when the page has
    /// no `id:` key — we still emit the hit so the user sees it, but the
    /// impute apply step relies on the `name` field and file stem.
    pub concept_id: String,
    /// `name` frontmatter field (falls back to the file stem).
    pub name: String,
    /// Current body text (trimmed, managed-regions stripped). Handed to the
    /// LLM as `existing_body` so the model can decide whether to rewrite or
    /// extend.
    pub body: String,
    /// Number of words in `body`.
    pub body_words: usize,
}

/// Walk `wiki/concepts/*.md` and flag concepts with too-short bodies.
///
/// A concept is thin when its body has fewer than `min_body_words` words.
/// Skips `index.md` and any file that fails frontmatter parsing (matches
/// the rest of kb-lint's "best effort" approach to malformed inputs).
///
/// # Errors
///
/// Returns an error when the concepts dir cannot be scanned or a file
/// cannot be read.
pub fn detect_thin_concept_bodies(
    root: &Path,
    config: &ThinConceptBodyConfig,
) -> Result<Vec<ThinConceptHit>> {
    let concepts_dir = root.join(WIKI_CONCEPTS_DIR);
    if !concepts_dir.exists() {
        return Ok(Vec::new());
    }
    let mut hits = Vec::new();
    for entry in fs::read_dir(&concepts_dir)
        .with_context(|| format!("scan concepts dir {}", concepts_dir.display()))?
    {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        if path.file_name().is_some_and(|n| n == "index.md") {
            continue;
        }

        let Ok((fm, full_body)) = read_frontmatter(&path) else {
            continue;
        };

        let body = extract_concept_body_text(&full_body);
        let word_count = body.split_whitespace().count();
        if word_count >= config.min_body_words {
            continue;
        }

        let concept_id = fm
            .get(Value::String("id".into()))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let name_from_fm = fm
            .get(Value::String("name".into()))
            .and_then(Value::as_str)
            .map(str::to_string);
        let name = name_from_fm.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .replace('-', " ")
        });

        let rel_path = path.strip_prefix(root).unwrap_or(&path).to_path_buf();

        hits.push(ThinConceptHit {
            page_path: rel_path,
            concept_id,
            name,
            body,
            body_words: word_count,
        });
    }
    hits.sort_by(|a, b| a.page_path.cmp(&b.page_path));
    Ok(hits)
}

/// Strip the leading `# Heading` line (if any), drop `<!-- kb:begin/end -->`
/// managed regions, and return the remaining body content trimmed of
/// leading/trailing whitespace.
fn extract_concept_body_text(full_body: &str) -> String {
    // Drop kb-managed regions — those are auto-generated (backlinks, etc.)
    // and don't count as human-authored body text. Keep the text simple:
    // line-based filter, not a proper markdown parser.
    let mut in_managed = false;
    let mut lines = Vec::new();
    for line in full_body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("<!-- kb:begin") {
            in_managed = true;
            continue;
        }
        if trimmed.starts_with("<!-- kb:end") {
            in_managed = false;
            continue;
        }
        if in_managed {
            continue;
        }
        lines.push(line);
    }

    // Drop the first leading blank lines, then the first `# ...` line if
    // present — we want the body text, not the page title.
    let mut cursor = 0;
    while cursor < lines.len() && lines[cursor].trim().is_empty() {
        cursor += 1;
    }
    if cursor < lines.len() {
        let stripped = lines[cursor].trim_start();
        if stripped.starts_with("# ") {
            cursor += 1;
        }
    }
    lines[cursor..].join("\n").trim().to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod duplicate_concept_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn write_concept(dir: &Path, slug: &str, name: &str, aliases: &[&str]) -> PathBuf {
        write_concept_with_body(dir, slug, name, aliases, "")
    }

    fn write_concept_with_body(
        dir: &Path,
        slug: &str,
        name: &str,
        aliases: &[&str],
        body: &str,
    ) -> PathBuf {
        use std::fmt::Write as _;
        let path = dir.join(format!("{slug}.md"));
        let mut content = format!("---\nid: concept:{slug}\nname: {name}\n");
        if !aliases.is_empty() {
            content.push_str("aliases:\n");
            for alias in aliases {
                writeln!(content, "  - {alias}").unwrap();
            }
        }
        content.push_str("---\n\n");
        writeln!(content, "# {name}").unwrap();
        if !body.is_empty() {
            content.push('\n');
            content.push_str(body);
            content.push('\n');
        }
        fs::write(&path, &content).unwrap();
        path
    }

    fn setup_kb(root: &TempDir) -> PathBuf {
        let concepts_dir = root.path().join("wiki/concepts");
        fs::create_dir_all(&concepts_dir).unwrap();
        concepts_dir
    }

    #[test]
    fn empty_dir_returns_no_items() {
        let dir = TempDir::new().unwrap();
        setup_kb(&dir);
        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(items.is_empty());
    }

    #[test]
    fn missing_concepts_dir_returns_no_items() {
        let dir = TempDir::new().unwrap();
        // Don't create the wiki/concepts directory
        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(items.is_empty());
    }

    #[test]
    fn exact_duplicate_names_flagged() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept(&concepts_dir, "borrow-checker-a", "Borrow checker", &[]);
        write_concept(&concepts_dir, "borrow-checker-b", "Borrow checker", &[]);

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert_eq!(items.len(), 1, "expected one duplicate pair");
        assert_eq!(items[0].kind, ReviewKind::ConceptMerge);
        assert!(items[0].comment.contains("similarity: 1.00"));
    }

    #[test]
    fn near_duplicate_names_flagged() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept(&concepts_dir, "borrow-check", "Borrow Check", &[]);
        write_concept(&concepts_dir, "borrow-checker", "Borrow checker", &[]);

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert_eq!(items.len(), 1, "near-duplicate names should be flagged");
        let comment = &items[0].comment;
        assert!(comment.contains("Borrow Check") || comment.contains("Borrow checker"));
    }

    #[test]
    fn distinct_concepts_not_flagged() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept(&concepts_dir, "borrow-checker", "Borrow checker", &[]);
        write_concept(&concepts_dir, "lifetime", "Lifetime", &[]);
        write_concept(&concepts_dir, "ownership", "Ownership", &[]);

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(
            items.is_empty(),
            "distinct concepts should produce no review items, got: {items:?}"
        );
    }

    #[test]
    fn alias_overlap_flagged() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        // Concept A's name matches concept B's alias
        write_concept(&concepts_dir, "borrowck", "borrowck", &[]);
        write_concept(
            &concepts_dir,
            "borrow-checker",
            "Borrow checker",
            &["borrowck"],
        );

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert_eq!(items.len(), 1, "alias overlap should be flagged");
        assert!(items[0].comment.contains("similarity: 1.00"));
    }

    #[test]
    fn threshold_controls_sensitivity() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept(&concepts_dir, "concept-a", "Rust language", &[]);
        write_concept(&concepts_dir, "concept-b", "Rust lang", &[]);

        // At high threshold, should not flag
        let strict = DuplicateConceptsConfig {
            similarity_threshold: 0.99,
            ..DuplicateConceptsConfig::default()
        };
        let strict_items = check_duplicate_concepts(dir.path(), &strict).expect("strict check");

        // At low threshold, should flag
        let lenient = DuplicateConceptsConfig {
            similarity_threshold: 0.5,
            ..DuplicateConceptsConfig::default()
        };
        let lenient_items = check_duplicate_concepts(dir.path(), &lenient).expect("lenient check");

        assert!(
            lenient_items.len() >= strict_items.len(),
            "lower threshold should produce at least as many items as higher threshold"
        );
        assert!(
            !lenient_items.is_empty(),
            "similar names should be flagged at low threshold"
        );
    }

    #[test]
    fn review_item_has_correct_structure() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept(&concepts_dir, "borrow-check", "Borrow Check", &[]);
        write_concept(&concepts_dir, "borrow-checker", "Borrow checker", &[]);

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert_eq!(items.len(), 1);

        let item = &items[0];
        assert!(item.metadata.id.starts_with("lint:duplicate-concepts:"));
        assert_eq!(item.kind, ReviewKind::ConceptMerge);
        assert!(!item.target_entity_id.is_empty());
        assert!(item.comment.contains("similarity:"));
        assert!(item.comment.contains("matched:"));
        assert!(item.metadata.tool_version.is_some());
        assert_eq!(item.metadata.status, Status::NeedsReview);
    }

    #[test]
    fn index_page_is_skipped() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        // Write an index page that should be ignored
        fs::write(
            concepts_dir.join("index.md"),
            "---\ntitle: Concepts Index\n---\n\n# Index\n",
        )
        .unwrap();
        write_concept(&concepts_dir, "ownership", "Ownership", &[]);

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(items.is_empty(), "single concept should produce no items");
    }

    #[test]
    fn bigram_similarity_exact_match_is_one() {
        assert!((bigram_similarity("borrow checker", "borrow checker") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bigram_similarity_unrelated_strings_is_low() {
        let score = bigram_similarity("borrow checker", "lifetime");
        assert!(score < 0.3, "unrelated strings: {score}");
    }

    #[test]
    fn bigram_similarity_near_duplicate_is_high() {
        let score = bigram_similarity("borrow checker", "borrow check");
        assert!(score > 0.85, "near-duplicate strings: {score}");
    }

    #[test]
    fn bigram_similarity_short_strings() {
        // Exact match shortcut fires before length check
        assert!((bigram_similarity("a", "a") - 1.0).abs() < 1e-10);
        // One side too short → no bigrams possible → 0
        assert!(bigram_similarity("ab", "a") < f64::EPSILON);
        assert!(bigram_similarity("a", "ab") < f64::EPSILON);
    }

    // ---- stop-word filter -------------------------------------------------

    #[test]
    fn strip_stop_words_removes_common_topic_words() {
        assert_eq!(strip_stop_words("raft consensus"), "raft");
        assert_eq!(strip_stop_words("causal order"), "order");
        // Non-stop-word compounds survive intact.
        assert_eq!(strip_stop_words("borrow checker"), "borrow checker");
        // Multi-word stripping.
        assert_eq!(strip_stop_words("distributed consensus protocol"), "distributed");
    }

    #[test]
    fn strip_stop_words_preserves_subword_matches() {
        // "lockfile" is a single word — only whole-word "lock" is stripped.
        assert_eq!(strip_stop_words("lockfile"), "lockfile");
        assert_eq!(strip_stop_words("datastore"), "datastore");
    }

    // ---- definition similarity --------------------------------------------

    #[test]
    fn definition_similarity_unrelated_definitions_low() {
        // Byzantine-consensus vs Raft-consensus paragraphs from the pass-7
        // corpus. They share the word "consensus"/"algorithm" but almost no
        // phrase-level overlap.
        let byzantine = "Consensus under adversarial or malicious faults, typically requiring \
                         3F+1 nodes to tolerate F Byzantine failures.";
        let raft = "A replicated-log consensus algorithm designed to be easier to understand \
                    than Paxos.";
        let score = definition_similarity(byzantine, raft);
        assert!(
            score < 0.15,
            "byzantine vs raft definitions should be dissimilar, got {score}"
        );
    }

    #[test]
    fn definition_similarity_overlapping_definitions_high() {
        let a = "Tokio is an asynchronous runtime for the Rust programming language that \
                 provides the building blocks for writing network applications.";
        let b = "The tokio runtime is an asynchronous runtime for the Rust programming \
                 language used to build network applications.";
        let score = definition_similarity(a, b);
        assert!(
            score > 0.3,
            "overlapping definitions should score high, got {score}"
        );
    }

    // ---- extract_definition_hint -----------------------------------------

    #[test]
    fn extract_definition_hint_reads_first_paragraph() {
        let body = "\n# Byzantine consensus\n\n\
                    Consensus under adversarial faults.\n\n\
                    ## Backlinks\n<!-- kb:begin id=backlinks -->\n- [[x]]\n\
                    <!-- kb:end id=backlinks -->\n";
        let hint = extract_definition_hint(body);
        assert_eq!(hint, "Consensus under adversarial faults.");
    }

    #[test]
    fn extract_definition_hint_empty_body() {
        assert_eq!(extract_definition_hint(""), "");
        assert_eq!(extract_definition_hint("\n# Heading\n\n"), "");
    }

    // ---- false-positive regressions (the bones) --------------------------

    #[test]
    fn byzantine_vs_raft_is_not_flagged() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept_with_body(
            &concepts_dir,
            "byzantine-consensus",
            "Byzantine consensus",
            &["Byzantine fault-tolerant consensus", "BFT consensus"],
            "Consensus under adversarial or malicious faults, typically requiring 3F+1 \
             nodes to tolerate F Byzantine failures.",
        );
        write_concept_with_body(
            &concepts_dir,
            "raft-consensus",
            "Raft consensus",
            &["Raft", "Raft leader election"],
            "A replicated-log consensus algorithm designed to be easier to understand \
             than Paxos.",
        );

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(
            items.is_empty(),
            "Byzantine vs Raft should NOT flag (false positive), got: {:?}",
            items.iter().map(|i| &i.comment).collect::<Vec<_>>()
        );
    }

    #[test]
    fn crdt_vs_happens_before_is_not_flagged() {
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept_with_body(
            &concepts_dir,
            "conflict-free-replicated-data-type",
            "Conflict-free Replicated Data Type",
            &["CRDT", "CRDTs", "causal ordering", "reliable causal delivery"],
            "A network delivery property ensuring operations are delivered in causal order, \
             which operation-based CRDTs rely on for correct convergence.",
        );
        write_concept_with_body(
            &concepts_dir,
            "happens-before-relation",
            "Happens-before relation",
            &["causal order", "causality"],
            "The partial order over distributed events that captures whether one event could \
             have causally influenced another.",
        );

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(
            items.is_empty(),
            "CRDT vs happens-before should NOT flag (false positive), got: {:?}",
            items.iter().map(|i| &i.comment).collect::<Vec<_>>()
        );
    }

    #[test]
    fn true_positive_tokio_still_flags() {
        // A genuine duplicate: "Tokio" (with alias "tokio runtime") vs a
        // separately-extracted "Tokio runtime" concept. The alias vs name
        // comparison hits 1.0 and the definitions share phrases like
        // "asynchronous runtime" and "Rust programming language".
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept_with_body(
            &concepts_dir,
            "tokio",
            "Tokio",
            &["tokio runtime"],
            "Tokio is an asynchronous runtime for the Rust programming language that provides \
             the building blocks for writing network applications.",
        );
        write_concept_with_body(
            &concepts_dir,
            "tokio-runtime",
            "Tokio runtime",
            &[],
            "The Tokio runtime is an asynchronous runtime for the Rust programming language \
             used to build network applications.",
        );

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert_eq!(
            items.len(),
            1,
            "Tokio vs tokio runtime should flag as duplicate, got: {items:?}"
        );
    }

    #[test]
    fn high_name_similarity_but_different_definitions_is_not_flagged() {
        // Defense-in-depth: two concepts whose names are near-identical but
        // whose bodies describe different ideas should NOT flag once
        // definition gating is in place. (If someone regresses the gate, this
        // test catches it before it ships.)
        let dir = TempDir::new().unwrap();
        let concepts_dir = setup_kb(&dir);
        write_concept_with_body(
            &concepts_dir,
            "write-amplification",
            "Write amplification",
            &[],
            "The ratio of physical bytes written to storage versus logical bytes written by the \
             application, typical of log-structured merge trees.",
        );
        write_concept_with_body(
            &concepts_dir,
            "write-verification",
            "Write verification",
            &[],
            "Checksumming or re-reading a written record to confirm durability before \
             acknowledging the client.",
        );

        let items = check_duplicate_concepts(dir.path(), &DuplicateConceptsConfig::default())
            .expect("check duplicates");
        assert!(
            items.is_empty(),
            "distinct-topic near-identical names should not flag when definitions \
             disagree, got: {:?}",
            items.iter().map(|i| &i.comment).collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod missing_concepts_tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_normalized_source(root: &Path, doc_id: &str, body: &str) {
        let dir = root.join("normalized").join(doc_id);
        fs::create_dir_all(&dir).unwrap();
        // metadata.json is not required by the missing-concepts walker but
        // keeping it aligned with production shape prevents surprises.
        fs::write(
            dir.join("metadata.json"),
            format!("{{\"source_revision_id\":\"rev-{doc_id}\"}}"),
        )
        .unwrap();
        fs::write(dir.join("source.md"), body).unwrap();
    }

    fn write_concept(root: &Path, slug: &str, name: &str, aliases: &[&str]) {
        use std::fmt::Write as _;
        let dir = root.join("wiki/concepts");
        fs::create_dir_all(&dir).unwrap();
        let mut content = format!("---\nid: concept:{slug}\nname: {name}\n");
        if !aliases.is_empty() {
            content.push_str("aliases:\n");
            for a in aliases {
                writeln!(content, "  - {a}").unwrap();
            }
        }
        content.push_str("---\n\n");
        writeln!(content, "# {name}").unwrap();
        fs::write(dir.join(format!("{slug}.md")), content).unwrap();
    }

    #[test]
    fn returns_empty_when_no_normalized_dir() {
        let dir = TempDir::new().unwrap();
        let hits = check_missing_concepts_raw(dir.path(), &MissingConceptsConfig::default())
            .expect("run check");
        assert!(hits.is_empty());
    }

    #[test]
    fn term_in_three_sources_is_flagged() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        // 'FooBarSystem' appears in three sources with five+ mentions total.
        write_normalized_source(
            root,
            "doc-a",
            "# A\n\nThe FooBar System is interesting. FooBar System has many parts.\n",
        );
        write_normalized_source(
            root,
            "doc-b",
            "# B\n\nFooBar System unique aspects. The FooBar System model.\n",
        );
        write_normalized_source(
            root,
            "doc-c",
            "# C\n\nAnother look at FooBar System.\n",
        );

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            hits.iter().any(|h| h.name == "FooBar System"),
            "expected 'FooBar System' to be flagged, got: {hits:?}"
        );
        let hit = hits.iter().find(|h| h.name == "FooBar System").unwrap();
        assert_eq!(hit.source_ids.len(), 3);
        assert!(hit.mention_count >= 5);
    }

    #[test]
    fn term_in_only_two_sources_is_not_flagged() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_normalized_source(root, "doc-a", "BetaGamma Cluster is important. BetaGamma Cluster again.\n");
        write_normalized_source(root, "doc-b", "BetaGamma Cluster explained. BetaGamma Cluster design. BetaGamma Cluster rules.\n");
        // Third source has a different capitalization case, not the term.
        write_normalized_source(root, "doc-c", "Nothing here.\n");

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name == "BetaGamma Cluster"),
            "term in only 2 sources must not flag: {hits:?}"
        );
    }

    #[test]
    fn existing_concept_name_is_excluded() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_concept(root, "widget-engine", "Widget Engine", &[]);
        write_normalized_source(root, "doc-a", "Widget Engine is fast.\nWidget Engine excels.\n");
        write_normalized_source(root, "doc-b", "Widget Engine usage. Widget Engine tips.\n");
        write_normalized_source(root, "doc-c", "Widget Engine internals.\n");

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name.eq_ignore_ascii_case("widget engine")),
            "existing concept must not be flagged: {hits:?}"
        );
    }

    #[test]
    fn existing_concept_alias_is_excluded() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_concept(root, "wgt", "Wgt", &["Widget Engine"]);
        write_normalized_source(root, "doc-a", "Widget Engine is fast.\nWidget Engine excels.\n");
        write_normalized_source(root, "doc-b", "Widget Engine usage. Widget Engine tips.\n");
        write_normalized_source(root, "doc-c", "Widget Engine internals.\n");

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name.eq_ignore_ascii_case("widget engine")),
            "alias match must suppress candidate: {hits:?}"
        );
    }

    #[test]
    fn stopword_starter_is_filtered() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        // 'The Next' starts with a stopword and should never fire.
        write_normalized_source(root, "doc-a", "The Next part. The Next section.\nThe Next block.\n");
        write_normalized_source(root, "doc-b", "The Next concept. The Next part.\nThe Next idea.\n");
        write_normalized_source(root, "doc-c", "The Next step. The Next thing. The Next loop.\n");

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name.to_lowercase().starts_with("the ")),
            "stopword-starter must not be flagged: {hits:?}"
        );
    }

    #[test]
    fn backtick_identifier_is_flagged_when_threshold_met() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_normalized_source(
            root,
            "doc-a",
            "Use `my_special_ident` to configure things.\n`my_special_ident` is strict.\n",
        );
        write_normalized_source(
            root,
            "doc-b",
            "Example: `my_special_ident` for the token.\n`my_special_ident` wins.\n",
        );
        write_normalized_source(
            root,
            "doc-c",
            "You also configure `my_special_ident` here.\n",
        );

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            hits.iter().any(|h| h.name == "my_special_ident"),
            "backtick identifier must be picked up: {hits:?}"
        );
    }

    #[test]
    fn disabled_config_returns_empty() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_normalized_source(root, "doc-a", "Common Widget appears. Common Widget again.\n");
        write_normalized_source(root, "doc-b", "Common Widget works. Common Widget here.\n");
        write_normalized_source(root, "doc-c", "Common Widget is fine.\n");

        let cfg = MissingConceptsConfig {
            enabled: false,
            ..MissingConceptsConfig::default()
        };
        let hits = check_missing_concepts_raw(root, &cfg).expect("run");
        assert!(hits.is_empty());
    }

    #[test]
    fn check_missing_concepts_emits_review_item() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_normalized_source(
            root,
            "doc-a",
            "The FooBar System is useful. FooBar System does X.\n",
        );
        write_normalized_source(
            root,
            "doc-b",
            "FooBar System usage. FooBar System designed.\n",
        );
        write_normalized_source(root, "doc-c", "Another view of FooBar System.\n");

        let items =
            check_missing_concepts(root, &MissingConceptsConfig::default()).expect("check");
        let hit = items
            .iter()
            .find(|i| i.metadata.id == "lint:concept-candidate:foobar-system")
            .unwrap_or_else(|| {
                panic!(
                    "expected lint:concept-candidate:foobar-system review item, got: {:?}",
                    items.iter().map(|i| &i.metadata.id).collect::<Vec<_>>()
                )
            });
        assert_eq!(hit.kind, ReviewKind::ConceptCandidate);
        assert_eq!(hit.status, ReviewStatus::Pending);
        assert_eq!(hit.metadata.status, Status::NeedsReview);
        assert_eq!(hit.metadata.dependencies.len(), 3);
        assert_eq!(
            hit.proposed_destination.as_deref(),
            Some(Path::new("wiki/concepts/foobar-system.md"))
        );
        assert!(hit.comment.contains("FooBar System"));
        assert!(hit.comment.contains("doc-a"));
    }

    #[test]
    fn code_block_contents_are_ignored_for_multiword_regex() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        // A plain-English phrase appears only inside fenced code blocks. The
        // multi-word regex should ignore it; with no other mentions it must
        // not be flagged.
        let body = "# Title\n\nSome prose.\n\n```\nActual Rust Code Here\nActual Rust Code Here\nActual Rust Code Here\n```\n";
        write_normalized_source(root, "doc-a", body);
        write_normalized_source(root, "doc-b", body);
        write_normalized_source(root, "doc-c", body);

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name == "Actual Rust Code Here"),
            "fenced code block should be ignored: {hits:?}"
        );
    }

    #[test]
    fn split_paragraphs_handles_blank_runs() {
        let body = "alpha\nbeta\n\ngamma\n\n\ndelta\n";
        let paras = split_paragraphs(body);
        assert_eq!(paras, vec!["alpha\nbeta", "gamma", "delta"]);
    }

    #[test]
    fn strip_leading_stopwords_peels_both_ends() {
        let out = strip_leading_stopwords("The FooBar System The").expect("some");
        assert_eq!(out, "FooBar System");
        let out = strip_leading_stopwords("Query Optimization The").expect("some");
        assert_eq!(out, "Query Optimization");
        // All stopwords → None.
        assert!(strip_leading_stopwords("The The The").is_none());
        // After peeling leaves fewer than 2 tokens → None.
        assert!(strip_leading_stopwords("The Widget The").is_none());
    }

    #[test]
    fn multiword_does_not_cross_paragraph_boundary() {
        // 'FooBar System' at the end of paragraph N must not merge with
        // 'The FooBar' at the start of paragraph N+1 into the junk candidate
        // 'FooBar System The FooBar'. Three sources are required to clear
        // the min_sources=3 / min_mentions=5 thresholds.
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let body = "FooBar System enables X.\n\nThe FooBar System is great.\nFooBar System rules.\n";
        write_normalized_source(root, "doc-a", body);
        write_normalized_source(root, "doc-b", body);
        write_normalized_source(root, "doc-c", body);

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name.contains("The FooBar")
                || h.name == "FooBar System The FooBar"),
            "paragraph-crossing candidate must not be produced: {hits:?}"
        );
        assert!(
            hits.iter().any(|h| h.name == "FooBar System"),
            "expected 'FooBar System' candidate, got: {hits:?}"
        );
    }

    #[test]
    fn trailing_stopword_is_peeled() {
        // "Query Optimization\n\nThe next paragraph" must not produce
        // 'Optimization The' or 'Query Optimization The' — the trailing
        // 'The' belongs to the next paragraph.
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let body = "Tree Storage Engines\n\nThe next paragraph.\nTree Storage Engines again.\nQuery Optimization\n\nThe wrap.\nTree Storage Engines and Query Optimization.\n";
        write_normalized_source(root, "doc-a", body);
        write_normalized_source(root, "doc-b", body);
        write_normalized_source(root, "doc-c", body);

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        for h in &hits {
            let name_lower = h.name.to_lowercase();
            assert!(
                !name_lower.ends_with(" the"),
                "candidate must not end with a stopword: {h:?}"
            );
            assert_ne!(
                h.name, "Optimization The",
                "cross-paragraph stopword peel failed: {hits:?}"
            );
            assert_ne!(
                h.name, "Query Optimization The",
                "trailing-stopword peel failed: {hits:?}"
            );
            assert_ne!(
                h.name, "Tree Storage Engines The",
                "trailing-stopword peel failed: {hits:?}"
            );
        }
    }

    #[test]
    fn backtick_pure_lowercase_english_is_rejected() {
        // `country` in inline backticks is a plain English word, not a
        // code identifier — it should never surface as a concept candidate.
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let body = "Set `country` on the row.\n`country` is a column.\n`country` filters apply.\n";
        write_normalized_source(root, "doc-a", body);
        write_normalized_source(root, "doc-b", body);
        write_normalized_source(root, "doc-c", body);

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        assert!(
            !hits.iter().any(|h| h.name == "country"),
            "pure-lowercase-English backtick word must be rejected: {hits:?}"
        );
    }

    #[test]
    fn backtick_real_identifiers_survive() {
        // Identifiers with a capital letter, digit, or non-alpha character
        // are legitimate code identifiers and must still be picked up.
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        // Each identifier appears in 3 sources with >=5 mentions each.
        let body = "The `recLSN` pointer, `go/types` package, `FooBar` struct, and `LSM_tree` all matter.\nRepeat: `recLSN` `go/types` `FooBar` `LSM_tree`.\n";
        write_normalized_source(root, "doc-a", body);
        write_normalized_source(root, "doc-b", body);
        write_normalized_source(root, "doc-c", body);

        let hits =
            check_missing_concepts_raw(root, &MissingConceptsConfig::default()).expect("run");
        for ident in &["recLSN", "go/types", "FooBar", "LSM_tree"] {
            assert!(
                hits.iter().any(|h| h.name == *ident),
                "expected identifier '{ident}' to be flagged: {hits:?}"
            );
        }
    }

    #[test]
    fn lint_issues_are_warnings_not_errors() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write_normalized_source(
            root,
            "doc-a",
            "The FooBar System is useful. FooBar System does X.\n",
        );
        write_normalized_source(
            root,
            "doc-b",
            "FooBar System usage. FooBar System designed.\n",
        );
        write_normalized_source(root, "doc-c", "Another view of FooBar System.\n");

        let report =
            run_lint_with_options(root, LintRule::MissingConcepts, &LintOptions::default())
                .expect("lint");
        assert!(!report.issues.is_empty(), "expected at least one issue");
        assert!(
            report
                .issues
                .iter()
                .all(|i| i.severity == IssueSeverity::Warning),
            "missing-concepts should emit warnings, not errors: {report:?}"
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.kind == IssueKind::ConceptCandidate),
            "expected ConceptCandidate kind: {report:?}"
        );
    }
}
