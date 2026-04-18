use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kb_core::fs::atomic_write;
use kb_core::{
    extract_managed_regions, frontmatter::read_frontmatter, managed_region::slug_from_title,
    read_normalized_document,
};
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

/// A budgeted context payload assembled from retrieval candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssembledContext {
    pub text: String,
    pub token_budget: u32,
    pub estimated_tokens: u32,
    pub manifest: Vec<ContextManifestEntry>,
}

/// Maps a span in [`AssembledContext::text`] back to a source location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextManifestEntry {
    pub start_offset: usize,
    pub end_offset: usize,
    pub source_id: String,
    pub anchor: Option<String>,
    pub chunk_kind: ContextChunkKind,
}

/// The kind of chunk inserted into assembled retrieval context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextChunkKind {
    FullDocument,
    Summary,
    Section,
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

/// Assemble full-text retrieval context from a ranked retrieval plan.
///
/// Source wiki pages prefer the backing `normalized/<doc-id>/source.md` body when possible.
/// If a full document does not fit inside the remaining budget, the assembler falls back to a
/// summary chunk plus the highest-scoring sections for that document, each wrapped in a stable
/// delimiter that preserves citation anchors.
///
/// # Errors
///
/// Returns an error when a referenced retrieval candidate cannot be read.
pub fn assemble_context(root: &Path, plan: &RetrievalPlan) -> Result<AssembledContext> {
    if plan.token_budget == 0 || plan.candidates.is_empty() {
        return Ok(AssembledContext {
            text: String::new(),
            token_budget: plan.token_budget,
            estimated_tokens: 0,
            manifest: Vec::new(),
        });
    }

    let query_tokens = tokenize(&plan.query);
    let mut text = String::new();
    let mut manifest = Vec::new();
    let mut estimated_tokens = 0_u32;

    for candidate in &plan.candidates {
        let document = load_context_document(root, &candidate.id)?;
        let full_chunk = render_context_chunk(
            &document.source_id,
            None,
            ContextChunkKind::FullDocument,
            &document.full_text,
        );
        let full_tokens = estimate_text_tokens(&full_chunk);

        if estimated_tokens.saturating_add(full_tokens) <= plan.token_budget {
            push_chunk(
                &mut text,
                &mut manifest,
                &document.source_id,
                None,
                ContextChunkKind::FullDocument,
                &full_chunk,
            );
            estimated_tokens = estimated_tokens.saturating_add(full_tokens);
            continue;
        }

        if !document.summary.is_empty() {
            let summary_chunk = render_context_chunk(
                &document.source_id,
                Some("summary"),
                ContextChunkKind::Summary,
                &document.summary,
            );
            let summary_tokens = estimate_text_tokens(&summary_chunk);
            if estimated_tokens.saturating_add(summary_tokens) <= plan.token_budget {
                push_chunk(
                    &mut text,
                    &mut manifest,
                    &document.source_id,
                    Some("summary"),
                    ContextChunkKind::Summary,
                    &summary_chunk,
                );
                estimated_tokens = estimated_tokens.saturating_add(summary_tokens);
            }
        }

        let mut sections: Vec<(usize, usize, &ContextSection)> = document
            .sections
            .iter()
            .enumerate()
            .map(|(index, section)| {
                (
                    score_section(section, &query_tokens),
                    index,
                    section,
                )
            })
            .filter(|(score, _, _)| *score > 0)
            .collect();
        sections.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        for (_, _, section) in sections {
            let anchor = section.anchor.as_deref().or(Some("section"));
            let section_chunk = render_context_chunk(
                &document.source_id,
                anchor,
                ContextChunkKind::Section,
                &section.content,
            );
            let section_tokens = estimate_text_tokens(&section_chunk);
            if estimated_tokens.saturating_add(section_tokens) > plan.token_budget {
                continue;
            }

            push_chunk(
                &mut text,
                &mut manifest,
                &document.source_id,
                anchor,
                ContextChunkKind::Section,
                &section_chunk,
            );
            estimated_tokens = estimated_tokens.saturating_add(section_tokens);
        }
    }

    Ok(AssembledContext {
        text,
        token_budget: plan.token_budget,
        estimated_tokens,
        manifest,
    })
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextDocument {
    source_id: String,
    full_text: String,
    summary: String,
    sections: Vec<ContextSection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextSection {
    anchor: Option<String>,
    content: String,
}

fn load_context_document(root: &Path, candidate_id: &str) -> Result<ContextDocument> {
    let page_path = root.join(candidate_id);
    let (frontmatter, body) = read_frontmatter(&page_path)
        .with_context(|| format!("parse retrieval candidate {}", page_path.display()))?;

    if candidate_id.starts_with(WIKI_SOURCES) {
        let source_id = frontmatter
            .get("source_document_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let summary = extract_summary(&body);

        if !source_id.is_empty()
            && let Ok(document) = read_normalized_document(root, source_id)
        {
            let full_text = document.canonical_text.trim().to_string();
            let sections = split_markdown_sections(&full_text);
            return Ok(ContextDocument {
                source_id: candidate_id.to_string(),
                full_text,
                summary,
                sections,
            });
        }

        let full_text = body.trim().to_string();
        let sections = split_markdown_sections(&full_text);
        return Ok(ContextDocument {
            source_id: candidate_id.to_string(),
            full_text,
            summary,
            sections,
        });
    }

    let full_text = body.trim().to_string();
    let summary = extract_summary(&body);
    let sections = split_markdown_sections(&full_text);
    Ok(ContextDocument {
        source_id: candidate_id.to_string(),
        full_text,
        summary,
        sections,
    })
}

fn extract_summary(body: &str) -> String {
    extract_managed_regions(body)
        .into_iter()
        .find(|region| region.id == "summary")
        .map(|region| region.body(body).trim().to_string())
        .unwrap_or_default()
}

fn split_markdown_sections(text: &str) -> Vec<ContextSection> {
    let mut sections = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        if line.starts_with('#') {
            if let Some(section) = finalize_section(current_heading.take(), &current_lines) {
                sections.push(section);
            }
            current_heading = Some(line.to_string());
            current_lines.clear();
            current_lines.push(line);
        } else if current_heading.is_some() {
            current_lines.push(line);
        }
    }

    if let Some(section) = finalize_section(current_heading, &current_lines) {
        sections.push(section);
    }

    sections
}

fn finalize_section(heading_line: Option<String>, lines: &[&str]) -> Option<ContextSection> {
    let heading_line = heading_line?;
    let heading_text = heading_line.trim_start_matches('#').trim();
    if heading_text.is_empty() {
        return None;
    }

    Some(ContextSection {
        anchor: Some(slug_from_title(heading_text)),
        content: lines.join("\n").trim().to_string(),
    })
}

fn score_section(section: &ContextSection, query_tokens: &[String]) -> usize {
    let anchor_tokens = section
        .anchor
        .as_deref()
        .map(tokenize)
        .unwrap_or_default();
    let section_tokens = tokenize(&section.content);
    let mut score = 0;
    score += query_tokens
        .iter()
        .map(|token| count_occurrences(&anchor_tokens, token) * WEIGHT_HEADING)
        .sum::<usize>();
    score += query_tokens
        .iter()
        .map(|token| count_occurrences(&section_tokens, token))
        .sum::<usize>();
    score
}

fn render_context_chunk(
    source_id: &str,
    anchor: Option<&str>,
    chunk_kind: ContextChunkKind,
    content: &str,
) -> String {
    let anchor_suffix = anchor.map_or_else(String::new, |value| format!("#{value}"));
    let kind_label = match chunk_kind {
        ContextChunkKind::FullDocument => "full_document",
        ContextChunkKind::Summary => "summary",
        ContextChunkKind::Section => "section",
    };
    format!(
        "<<<kb-source id=\"{source_id}{anchor_suffix}\" kind=\"{kind_label}\">>>\n{content}\n<<<kb-end>>>\n"
    )
}

fn push_chunk(
    text: &mut String,
    manifest: &mut Vec<ContextManifestEntry>,
    source_id: &str,
    anchor: Option<&str>,
    chunk_kind: ContextChunkKind,
    chunk: &str,
) {
    let start_offset = text.len();
    text.push_str(chunk);
    let end_offset = text.len();
    manifest.push(ContextManifestEntry {
        start_offset,
        end_offset,
        source_id: source_id.to_string(),
        anchor: anchor.map(str::to_string),
        chunk_kind,
    });
}

fn estimate_text_tokens(text: &str) -> u32 {
    let estimated = text.len().div_ceil(APPROX_CHARS_PER_TOKEN);
    u32::try_from(estimated).unwrap_or(u32::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use kb_core::{EntityMetadata, NormalizedDocument, Status, write_normalized_document};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn write_source_page(dir: &Path, slug: &str, title: &str, summary: &str) {
        let content = format!(
            "---\nid: wiki-source-{slug}\ntype: source\ntitle: {title}\nsource_document_id: {slug}\nsource_revision_id: rev-{slug}\n---\n\
             \n# Source\n<!-- kb:begin id=title -->\n{title}\n<!-- kb:end id=title -->\n\
             \n## Summary\n<!-- kb:begin id=summary -->\n{summary}\n<!-- kb:end id=summary -->\n"
        );
        fs::write(dir.join(format!("{slug}.md")), content).unwrap();
    }

    fn write_normalized_source(root: &Path, id: &str, text: &str, heading_ids: &[&str]) {
        let document = NormalizedDocument {
            metadata: EntityMetadata {
                id: id.to_string(),
                created_at_millis: 1,
                updated_at_millis: 1,
                source_hashes: vec!["hash".to_string()],
                model_version: None,
                tool_version: None,
                prompt_template_hash: None,
                dependencies: Vec::new(),
                output_paths: Vec::new(),
                status: Status::Fresh,
            },
            source_revision_id: format!("rev-{id}"),
            canonical_text: text.to_string(),
            normalized_assets: Vec::<PathBuf>::new(),
            heading_ids: heading_ids.iter().map(|value| (*value).to_string()).collect(),
        };
        write_normalized_document(root, &document).unwrap();
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

    #[test]
    fn assemble_context_prefers_full_normalized_source_when_it_fits() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(&sources, "rust-guide", "Rust Guide", "Short summary.");
        write_normalized_source(
            root,
            "rust-guide",
            "# Ownership\n\nOwnership keeps memory safe.\n\n## Borrowing\n\nBorrowing rules.\n",
            &["ownership", "borrowing"],
        );

        let plan = RetrievalPlan {
            query: "ownership".to_string(),
            token_budget: 500,
            estimated_tokens: 0,
            candidates: vec![RetrievalCandidate {
                id: "wiki/sources/rust-guide.md".to_string(),
                title: "Rust Guide".to_string(),
                score: 10,
                estimated_tokens: 20,
                reasons: vec!["title matched 'ownership' 1x (+4)".to_string()],
            }],
        };

        let assembled = assemble_context(root, &plan).unwrap();
        assert!(assembled.text.contains("Ownership keeps memory safe."));
        assert!(!assembled.text.contains("Short summary."));
        assert_eq!(assembled.manifest.len(), 1);
        assert_eq!(assembled.manifest[0].chunk_kind, ContextChunkKind::FullDocument);
        assert_eq!(assembled.manifest[0].anchor, None);
        assert!(assembled.estimated_tokens <= assembled.token_budget);
    }

    #[test]
    fn assemble_context_falls_back_to_summary_and_top_sections_under_budget() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(
            &sources,
            "rust-guide",
            "Rust Guide",
            "Borrowing summary for quick retrieval.",
        );
        write_normalized_source(
            root,
            "rust-guide",
            "# Ownership\n\nOwnership keeps memory safe with moves and detailed compiler rules that span multiple paragraphs for retrieval budget pressure.\n\n## Borrowing\n\nBorrowing allows shared references.\n\n## Lifetimes\n\nLifetimes connect reference validity across functions, structs, traits, and longer examples that intentionally consume more context budget than the focused borrowing section.\n",
            &["ownership", "borrowing", "lifetimes"],
        );

        let full_chunk = render_context_chunk(
            "wiki/sources/rust-guide.md",
            None,
            ContextChunkKind::FullDocument,
            "# Ownership\n\nOwnership keeps memory safe with moves and detailed compiler rules that span multiple paragraphs for retrieval budget pressure.\n\n## Borrowing\n\nBorrowing allows shared references.\n\n## Lifetimes\n\nLifetimes connect reference validity across functions, structs, traits, and longer examples that intentionally consume more context budget than the focused borrowing section.",
        );
        let summary_chunk = render_context_chunk(
            "wiki/sources/rust-guide.md",
            Some("summary"),
            ContextChunkKind::Summary,
            "Borrowing summary for quick retrieval.",
        );
        let borrowing_chunk = render_context_chunk(
            "wiki/sources/rust-guide.md",
            Some("borrowing"),
            ContextChunkKind::Section,
            "## Borrowing\n\nBorrowing allows shared references.",
        );
        let budget = estimate_text_tokens(&summary_chunk)
            .saturating_add(estimate_text_tokens(&borrowing_chunk));
        assert!(budget < estimate_text_tokens(&full_chunk));
        let plan = RetrievalPlan {
            query: "borrowing references".to_string(),
            token_budget: budget,
            estimated_tokens: 0,
            candidates: vec![RetrievalCandidate {
                id: "wiki/sources/rust-guide.md".to_string(),
                title: "Rust Guide".to_string(),
                score: 10,
                estimated_tokens: 20,
                reasons: vec!["summary matched 'borrowing' 1x (+1)".to_string()],
            }],
        };

        let assembled = assemble_context(root, &plan).unwrap();
        assert!(assembled.text.contains("Borrowing summary for quick retrieval."));
        assert!(assembled.text.contains("Borrowing allows shared references."));
        assert!(!assembled.text.contains("Lifetimes connect reference validity."));
        assert_eq!(assembled.manifest.len(), 2);
        assert_eq!(assembled.manifest[0].chunk_kind, ContextChunkKind::Summary);
        assert_eq!(assembled.manifest[0].anchor.as_deref(), Some("summary"));
        assert_eq!(assembled.manifest[1].chunk_kind, ContextChunkKind::Section);
        assert_eq!(assembled.manifest[1].anchor.as_deref(), Some("borrowing"));
        assert!(assembled.manifest[0].end_offset <= assembled.manifest[1].start_offset);
        assert!(assembled.estimated_tokens <= assembled.token_budget);
    }
}
