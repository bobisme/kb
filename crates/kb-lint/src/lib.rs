#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use kb_core::{Manifest, extract_managed_regions, slug_from_title};
use regex::Regex;
use serde::Serialize;

const WIKI_DIR: &str = "wiki";
const MD_EXT: &str = "md";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintRule {
    BrokenLinks,
}

impl LintRule {
    /// Parse a CLI rule selector.
    ///
    /// # Errors
    /// Returns an error when the caller requests an unsupported lint rule.
    pub fn parse(input: Option<&str>) -> Result<Self> {
        match input {
            None | Some("broken-links" | "broken_links" | "brokenlinks") => {
                Ok(Self::BrokenLinks)
            }
            Some(other) => Err(anyhow!("unsupported lint rule: {other}")),
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BrokenLinks => "broken-links",
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LintIssue {
    pub kind: IssueKind,
    pub referring_page: String,
    pub line: usize,
    pub target: String,
    pub message: String,
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    MissingPage,
    MissingAnchor,
}

/// Run the selected lint rule against the KB at `root`.
///
/// # Errors
/// Returns an error when the lint pass cannot read KB state or wiki files.
pub fn run_lint(root: &Path, rule: LintRule) -> Result<LintReport> {
    match rule {
        LintRule::BrokenLinks => run_broken_links_lint(root),
    }
}

fn run_broken_links_lint(root: &Path) -> Result<LintReport> {
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

    Ok(LintReport {
        rule: LintRule::BrokenLinks.as_str().to_string(),
        issue_count: issues.len(),
        issues,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

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
        assert_eq!(report.issues[0].suggested_fix.as_deref(), Some("wiki/concepts/rust"));
        assert_eq!(report.issues[1].kind, IssueKind::MissingAnchor);
        assert_eq!(report.issues[1].line, 3);
        assert_eq!(report.issues[1].suggested_fix.as_deref(), Some("wiki/concepts/rust#summary"));
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
}
