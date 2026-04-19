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

const WIKI_DIR: &str = "wiki";
const MD_EXT: &str = "md";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintRule {
    BrokenLinks,
    Orphans,
    StaleArtifacts,
    MissingCitations,
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
            Some("stale" | "stale-artifacts" | "stale_artifacts") => Ok(Self::StaleArtifacts),
            Some("missing-citations" | "missing_citations" | "missingcitations") => {
                Ok(Self::MissingCitations)
            }
            Some(other) => Err(anyhow!("unsupported lint rule: {other}")),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BrokenLinks => "broken-links",
            Self::Orphans => "orphans",
            Self::StaleArtifacts => "stale",
            Self::MissingCitations => "missing-citations",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LintOptions {
    pub require_citations: bool,
    pub missing_citations_level: MissingCitationsLevel,
}

impl Default for LintOptions {
    fn default() -> Self {
        Self {
            require_citations: true,
            missing_citations_level: MissingCitationsLevel::Warn,
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
    if matches!(rule, LintRule::StaleArtifacts | LintRule::All) {
        issues.extend(detect_stale_artifacts(root)?);
    }
    if options.require_citations && matches!(rule, LintRule::MissingCitations | LintRule::All) {
        issues.extend(detect_missing_citations(root, *options)?);
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

    for (idx, line) in content.lines().enumerate() {
        let line_number = idx + 1;

        for capture in wikilink_re.captures_iter(line) {
            let Some(raw_target) = capture.get(1).map(|m| m.as_str()) else {
                continue;
            };
            if let Some(issue) = validate_link(raw_target, page, line_number, registry) {
                issues.push(issue);
            }
        }

        for capture in markdown_link_re.captures_iter(line) {
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

        for missing_doc in doc_ids
            .iter()
            .filter(|doc_id| !existing_docs.contains(*doc_id))
        {
            issues.push(LintIssue {
                severity: IssueSeverity::Error,
                kind: IssueKind::SourceDocumentMissing,
                referring_page: rel_page_str.clone(),
                line: 0,
                target: missing_doc.clone(),
                message: format!("referenced source document '{missing_doc}' is missing"),
                suggested_fix: None,
            });
        }

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

            if live_revision_ids.is_empty() {
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
            } else if rev_ids
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

fn detect_missing_citations(root: &Path, options: LintOptions) -> Result<Vec<LintIssue>> {
    let mut issues = Vec::new();

    for page in markdown_files_under(&root.join(WIKI_DIR))? {
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
        // that's orphans, stale-artifacts, and missing-citations).
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
            },
        )
        .expect("lint report");
        assert_eq!(report.issue_count, 1);
        assert_eq!(report.issues[0].severity, IssueSeverity::Error);
        assert!(report.has_errors());
    }
}

// ---------------------------------------------------------------------------
// Duplicate concept detection
// ---------------------------------------------------------------------------

const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";
const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.85;

/// Configuration for the duplicate-concepts lint check.
#[derive(Debug, Clone)]
pub struct DuplicateConceptsConfig {
    /// Similarity threshold for flagging pairs (0.0–1.0). Default: 0.85.
    pub similarity_threshold: f64,
}

impl Default for DuplicateConceptsConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone)]
struct ConceptRecord {
    id: String,
    name: String,
    aliases: Vec<String>,
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
            if let Some((term_a, term_b, score)) = best_term_pair(a, b) {
                if score >= config.similarity_threshold {
                    items.push(build_review_item(a, b, &term_a, &term_b, score, now));
                }
            }
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
    let (frontmatter, _body) = read_frontmatter(path)
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

    Ok(ConceptRecord { id, name, aliases })
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
            let score = bigram_similarity(&normalize_term(ta), &normalize_term(tb));
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod duplicate_concept_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn write_concept(dir: &Path, slug: &str, name: &str, aliases: &[&str]) -> PathBuf {
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
        };
        let strict_items = check_duplicate_concepts(dir.path(), &strict).expect("strict check");

        // At low threshold, should flag
        let lenient = DuplicateConceptsConfig {
            similarity_threshold: 0.5,
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
}
