use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kb_core::fs::atomic_write;
use kb_core::{extract_managed_regions, frontmatter::read_frontmatter};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use tracing::warn;

const INDEX_SUBPATH: &str = "state/indexes/lexical.json";
const WIKI_SOURCES: &str = "wiki/sources";
const WIKI_CONCEPTS: &str = "wiki/concepts";

const WEIGHT_TITLE: usize = 4;
const WEIGHT_ALIAS: usize = 3;
const WEIGHT_HEADING: usize = 2;
const WEIGHT_SUMMARY: usize = 1;
const DEFAULT_RETRIEVAL_TOP_K: usize = 25;
const APPROX_CHARS_PER_TOKEN: usize = 4;
const MIN_ENTRY_TOKEN_ESTIMATE: u32 = 32;

/// A single page's indexed data for lexical search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalEntry {
    /// Relative path from KB root, e.g. `wiki/sources/foo.md`.
    pub id: String,
    pub title: String,
    pub aliases: Vec<String>,
    pub headings: Vec<String>,
    pub summary: String,
}

/// Full-text lexical search index stored at `state/indexes/lexical.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LexicalIndex {
    pub entries: Vec<LexicalEntry>,
}

/// A ranked search result from [`LexicalIndex::search`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Page ID (relative path from KB root).
    pub id: String,
    pub title: String,
    /// Higher scores indicate better matches.
    pub score: usize,
    /// Reasons explaining why this result ranked where it did.
    #[serde(default)]
    pub reasons: Vec<String>,
}

/// A persisted retrieval plan for a question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalPlan {
    pub query: String,
    pub token_budget: u32,
    pub estimated_tokens: u32,
    pub candidates: Vec<RetrievalCandidate>,
}

/// A ranked candidate selected for retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalCandidate {
    pub id: String,
    pub title: String,
    pub score: usize,
    pub estimated_tokens: u32,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScoreAnalysis {
    score: usize,
    reasons: Vec<String>,
}

impl LexicalIndex {
    /// Load the lexical index from `state/indexes/lexical.json`.
    ///
    /// Returns an empty index when the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but cannot be read or deserialized.
    pub fn load(root: &Path) -> Result<Self> {
        let path = index_path(root);
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("read lexical index {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("deserialize lexical index {}", path.display()))
    }

    /// Persist the lexical index to `state/indexes/lexical.json`.
    ///
    /// # Errors
    ///
    /// Returns an error when the index cannot be serialized or written.
    pub fn save(&self, root: &Path) -> Result<()> {
        let path = index_path(root);
        let mut json = serde_json::to_vec_pretty(self).context("serialize lexical index")?;
        json.push(b'\n');
        atomic_write(&path, &json)
            .with_context(|| format!("write lexical index {}", path.display()))
    }

    /// Search for pages matching the query string.
    ///
    /// Returns up to `top_k` results ranked by score descending.
    /// Field weights: title (4) > alias (3) > heading (2) > summary (1).
    #[must_use]
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<(ScoreAnalysis, &LexicalEntry)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let analysis = analyze_entry(entry, &query_tokens);
                if analysis.score > 0 {
                    Some((analysis, entry))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.score.cmp(&a.0.score).then_with(|| a.1.id.cmp(&b.1.id)));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(analysis, entry)| SearchResult {
                id: entry.id.clone(),
                title: entry.title.clone(),
                score: analysis.score,
                reasons: analysis.reasons,
            })
            .collect()
    }

    /// Build a deterministic, budgeted retrieval plan for a question.
    #[must_use]
    pub fn plan_retrieval(&self, query: &str, token_budget: u32) -> RetrievalPlan {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() || token_budget == 0 {
            return RetrievalPlan {
                query: query.to_string(),
                token_budget,
                estimated_tokens: 0,
                candidates: Vec::new(),
            };
        }

        let mut scored: Vec<(ScoreAnalysis, &LexicalEntry)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let analysis = analyze_entry(entry, &query_tokens);
                if analysis.score > 0 {
                    Some((analysis, entry))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.0.score.cmp(&a.0.score).then_with(|| a.1.id.cmp(&b.1.id)));

        let mut estimated_tokens = 0_u32;
        let mut candidates = Vec::new();

        for (analysis, entry) in scored.into_iter().take(DEFAULT_RETRIEVAL_TOP_K) {
            let entry_tokens = estimate_entry_tokens(entry);
            if estimated_tokens.saturating_add(entry_tokens) > token_budget {
                continue;
            }

            estimated_tokens = estimated_tokens.saturating_add(entry_tokens);
            candidates.push(RetrievalCandidate {
                id: entry.id.clone(),
                title: entry.title.clone(),
                score: analysis.score,
                estimated_tokens: entry_tokens,
                reasons: analysis.reasons,
            });
        }

        RetrievalPlan {
            query: query.to_string(),
            token_budget,
            estimated_tokens,
            candidates,
        }
    }
}

/// Build a lexical index by scanning wiki pages under `root`.
///
/// Scans `wiki/sources/*.md` (source pages) and `wiki/concepts/*.md` (concept pages).
///
/// # Errors
///
/// Returns an error when a wiki directory cannot be scanned.
pub fn build_lexical_index(root: &Path) -> Result<LexicalIndex> {
    let mut entries = Vec::new();

    scan_wiki_dir(&root.join(WIKI_SOURCES), |path| {
        match index_source_page(path, root) {
            Ok(entry) => entries.push(entry),
            Err(err) => warn!("lexical index: skipping {}: {err}", path.display()),
        }
    })?;

    scan_wiki_dir(&root.join(WIKI_CONCEPTS), |path| {
        match index_concept_page(path, root) {
            Ok(entry) => entries.push(entry),
            Err(err) => warn!("lexical index: skipping {}: {err}", path.display()),
        }
    })?;

    entries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(LexicalIndex { entries })
}

fn index_path(root: &Path) -> PathBuf {
    root.join(INDEX_SUBPATH)
}

fn scan_wiki_dir(dir: &Path, mut visit: impl FnMut(&Path)) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("scan {}", dir.display()))? {
        let path = entry
            .with_context(|| format!("read directory entry in {}", dir.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "md") {
            paths.push(path);
        }
    }
    paths.sort();
    for path in &paths {
        visit(path);
    }
    Ok(())
}

fn index_source_page(path: &Path, root: &Path) -> Result<LexicalEntry> {
    let (frontmatter, body) =
        read_frontmatter(path).with_context(|| format!("parse {}", path.display()))?;

    let title = frontmatter
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let summary = extract_managed_regions(&body)
        .into_iter()
        .find(|r| r.id == "summary")
        .map(|r| r.body(&body).trim().to_string())
        .unwrap_or_default();

    let headings = extract_headings(&body);

    Ok(LexicalEntry {
        id: relative_id(path, root),
        title,
        aliases: Vec::new(),
        headings,
        summary,
    })
}

fn index_concept_page(path: &Path, root: &Path) -> Result<LexicalEntry> {
    let (frontmatter, body) =
        read_frontmatter(path).with_context(|| format!("parse {}", path.display()))?;

    let title = frontmatter
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
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

    let headings = extract_headings(&body);

    Ok(LexicalEntry {
        id: relative_id(path, root),
        title,
        aliases,
        headings,
        summary: String::new(),
    })
}

fn relative_id(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn extract_headings(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|line| {
            if !line.starts_with('#') {
                return None;
            }
            let text = line.trim_start_matches('#').trim();
            if text.is_empty() {
                None
            } else {
                Some(text.to_string())
            }
        })
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|s| s.len() > 1)
        .collect()
}

fn count_occurrences(tokens: &[String], target: &str) -> usize {
    tokens.iter().filter(|t| t.as_str() == target).count()
}

fn analyze_entry(entry: &LexicalEntry, query_tokens: &[String]) -> ScoreAnalysis {
    let title_tokens = tokenize(&entry.title);
    let alias_tokens: Vec<String> = entry.aliases.iter().flat_map(|a| tokenize(a)).collect();
    let heading_tokens: Vec<String> = entry.headings.iter().flat_map(|h| tokenize(h)).collect();
    let summary_tokens = tokenize(&entry.summary);

    let mut reasons = Vec::new();
    let mut score = 0;

    score += collect_reasons(
        &mut reasons,
        "title",
        &title_tokens,
        query_tokens,
        WEIGHT_TITLE,
    );
    score += collect_reasons(
        &mut reasons,
        "alias",
        &alias_tokens,
        query_tokens,
        WEIGHT_ALIAS,
    );
    score += collect_reasons(
        &mut reasons,
        "heading",
        &heading_tokens,
        query_tokens,
        WEIGHT_HEADING,
    );
    score += collect_reasons(
        &mut reasons,
        "summary",
        &summary_tokens,
        query_tokens,
        WEIGHT_SUMMARY,
    );

    ScoreAnalysis { score, reasons }
}

fn collect_reasons(
    reasons: &mut Vec<String>,
    field: &str,
    field_tokens: &[String],
    query_tokens: &[String],
    weight: usize,
) -> usize {
    let mut field_score = 0;
    let mut seen = Vec::new();

    for token in query_tokens {
        if seen.contains(token) {
            continue;
        }
        seen.push(token.clone());

        let query_count = count_occurrences(query_tokens, token);
        let field_count = count_occurrences(field_tokens, token);
        if field_count == 0 {
            continue;
        }

        let contribution = field_count * query_count * weight;
        field_score += contribution;
        reasons.push(format!(
            "{field} matched '{token}' {field_count}x (+{contribution})"
        ));
    }

    field_score
}

fn estimate_entry_tokens(entry: &LexicalEntry) -> u32 {
    let chars = entry.title.len()
        + entry.aliases.iter().map(String::len).sum::<usize>()
        + entry.headings.iter().map(String::len).sum::<usize>()
        + entry.summary.len();

    let estimated = chars.div_ceil(APPROX_CHARS_PER_TOKEN);
    u32::try_from(estimated)
        .unwrap_or(u32::MAX)
        .max(MIN_ENTRY_TOKEN_ESTIMATE)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_source_page(dir: &Path, slug: &str, title: &str, summary: &str) {
        let content = format!(
            "---\nid: wiki-source-{slug}\ntype: source\ntitle: {title}\n---\n\
             \n# Source\n<!-- kb:begin id=title -->\n{title}\n<!-- kb:end id=title -->\n\
             \n## Summary\n<!-- kb:begin id=summary -->\n{summary}\n<!-- kb:end id=summary -->\n"
        );
        fs::write(dir.join(format!("{slug}.md")), content).unwrap();
    }

    fn write_concept_page(dir: &Path, slug: &str, name: &str, aliases: &[&str]) {
        use std::fmt::Write as _;
        let mut content = format!("---\nid: concept:{slug}\nname: {name}\n");
        if !aliases.is_empty() {
            content.push_str("aliases:\n");
            for alias in aliases {
                writeln!(content, "  - {alias}").unwrap();
            }
        }
        writeln!(content, "---\n\n# {name}\n\n## Overview\n\nAbout {name}.").unwrap();
        fs::write(dir.join(format!("{slug}.md")), content).unwrap();
    }

    #[test]
    fn build_index_scans_source_and_concept_pages() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        let concepts = root.join("wiki/concepts");
        fs::create_dir_all(&sources).unwrap();
        fs::create_dir_all(&concepts).unwrap();

        write_source_page(
            &sources,
            "rust-book",
            "The Rust Programming Language",
            "Memory safety.",
        );
        write_concept_page(&concepts, "borrow-checker", "Borrow checker", &["borrowck"]);

        let index = build_lexical_index(root).unwrap();
        assert_eq!(index.entries.len(), 2);
        let ids: Vec<&str> = index.entries.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"wiki/concepts/borrow-checker.md"));
        assert!(ids.contains(&"wiki/sources/rust-book.md"));
    }

    #[test]
    fn search_ranks_title_match_above_summary_match() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();

        write_source_page(
            &sources,
            "rust-overview",
            "Rust Overview",
            "A general overview.",
        );
        write_source_page(
            &sources,
            "memory",
            "Memory Safety",
            "Rust enables memory safety.",
        );

        let index = build_lexical_index(root).unwrap();
        let results = index.search("rust", 10);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "wiki/sources/rust-overview.md");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn search_returns_empty_for_no_match() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("wiki/sources")).unwrap();
        write_source_page(
            &root.join("wiki/sources"),
            "rust-book",
            "The Rust Book",
            "About Rust.",
        );

        let index = build_lexical_index(root).unwrap();
        assert!(index.search("python", 10).is_empty());
    }

    #[test]
    fn alias_tokens_contribute_to_score() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let concepts = root.join("wiki/concepts");
        fs::create_dir_all(&concepts).unwrap();

        write_concept_page(
            &concepts,
            "borrow-checker",
            "Memory Validator",
            &["borrowck", "borrow checker"],
        );

        let index = build_lexical_index(root).unwrap();
        let results = index.search("borrowck", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "wiki/concepts/borrow-checker.md");
    }

    #[test]
    fn index_round_trips_through_disk() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(&sources, "test", "Test Page", "A test summary.");

        let index = build_lexical_index(root).unwrap();
        index.save(root).unwrap();
        let loaded = LexicalIndex::load(root).unwrap();
        assert_eq!(index, loaded);
    }

    #[test]
    fn load_returns_empty_when_absent() {
        let dir = tempdir().unwrap();
        let index = LexicalIndex::load(dir.path()).unwrap();
        assert!(index.entries.is_empty());
    }

    #[test]
    fn build_index_tolerates_missing_wiki_dirs() {
        let dir = tempdir().unwrap();
        let index = build_lexical_index(dir.path()).unwrap();
        assert!(index.entries.is_empty());
    }

    #[test]
    fn top_k_limits_results() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        for i in 0..5 {
            write_source_page(
                &sources,
                &format!("rust-{i}"),
                &format!("Rust Guide {i}"),
                "Rust.",
            );
        }
        let index = build_lexical_index(root).unwrap();
        let results = index.search("rust", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn source_page_summary_is_indexed() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(
            &sources,
            "ownership",
            "Ownership",
            "Rust ownership rules prevent dangling pointers.",
        );

        let index = build_lexical_index(root).unwrap();
        let results = index.search("dangling", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "wiki/sources/ownership.md");
    }

    #[test]
    fn tokenize_splits_on_non_alphanumeric() {
        let tokens = tokenize("hello-world foo_bar baz");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
        assert!(tokens.contains(&"baz".to_string()));
    }

    #[test]
    fn tokenize_filters_single_chars() {
        let tokens = tokenize("a b c hello");
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"hello".to_string()));
    }

    #[test]
    fn retrieval_plan_respects_budget_and_preserves_order() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(&sources, "rust-overview", "Rust Overview", "Rust memory safety.");
        write_source_page(&sources, "rust-borrow", "Rust Borrowing", "Borrow checker and Rust.");

        let index = build_lexical_index(root).unwrap();
        let budget = estimate_entry_tokens(&index.entries[0]);
        let plan = index.plan_retrieval("rust", budget);

        assert_eq!(plan.token_budget, budget);
        assert_eq!(plan.candidates.len(), 1);
        assert!(plan.estimated_tokens <= budget);
        assert_eq!(plan.candidates[0].id, "wiki/sources/rust-borrow.md");
    }

    #[test]
    fn retrieval_plan_includes_ranking_reasons() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let concepts = root.join("wiki/concepts");
        fs::create_dir_all(&concepts).unwrap();
        write_concept_page(&concepts, "borrow-checker", "Borrow checker", &["borrowck"]);

        let index = build_lexical_index(root).unwrap();
        let plan = index.plan_retrieval("borrowck checker", 1_000);

        assert_eq!(plan.candidates.len(), 1);
        assert!(plan.candidates[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("alias matched 'borrowck'")));
        assert!(plan.candidates[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("title matched 'checker'")));
    }
}
