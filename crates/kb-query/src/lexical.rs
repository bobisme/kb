use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kb_core::fs::atomic_write;
use kb_core::{
    extract_managed_regions, frontmatter::read_frontmatter, managed_region::slug_from_title,
    read_normalized_document, state_dir,
};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use tracing::warn;

const INDEX_REL: [&str; 2] = ["indexes", "lexical.json"];
const WIKI_SOURCES: &str = "wiki/sources";
const WIKI_CONCEPTS: &str = "wiki/concepts";

const WEIGHT_TITLE: usize = 4;
const WEIGHT_ALIAS: usize = 3;
const WEIGHT_HEADING: usize = 2;
const WEIGHT_SUMMARY: usize = 1;
const DEFAULT_RETRIEVAL_TOP_K: usize = 25;

/// Soft cap on semantic-only tail candidates in hybrid retrieval.
///
/// Kept as a separate constant so a future change to the lexical default
/// doesn't silently change hybrid behavior.
pub const DEFAULT_RETRIEVAL_TOP_K_FOR_HYBRID: usize = DEFAULT_RETRIEVAL_TOP_K;
const APPROX_CHARS_PER_TOKEN: usize = 4;
const MIN_ENTRY_TOKEN_ESTIMATE: u32 = 32;

/// Minimum number of positively-scored candidates that must be produced
/// before the low-coverage fallback is skipped. See [`LexicalIndex::plan_retrieval`].
const MIN_CANDIDATES_BEFORE_FALLBACK: usize = 3;
/// Minimum top-candidate score required to avoid the low-coverage fallback.
/// Any positive match (score >= 1) keeps the fallback from firing.
const MIN_TOP_SCORE_BEFORE_FALLBACK: usize = 1;
/// Upper bound on the number of source-summary pages the fallback will add on
/// top of the three index pages.
const FALLBACK_SOURCE_SUMMARY_LIMIT: usize = 5;

/// Reason string stamped on each candidate the low-coverage fallback adds.
pub const FALLBACK_CANDIDATE_REASON: &str =
    "fallback: low-coverage (added to provide baseline context)";
/// Value stored in [`RetrievalPlan::fallback_reason`] when the low-coverage
/// fallback fires.
pub const FALLBACK_REASON_LOW_COVERAGE: &str = "low-coverage";

/// Relative paths of the three auto-generated wiki index pages the fallback
/// always tries to include (in this order).
const FALLBACK_INDEX_PAGES: &[&str] = &[
    "wiki/index.md",
    "wiki/concepts/index.md",
    "wiki/sources/index.md",
];

/// Common English stopwords stripped from query tokens before scoring.
///
/// Kept small and hand-picked: we only want to drop terms that add noise to
/// lexical scoring ("is", "the", "how", ...) without dropping short technical
/// terms ("raft", "sgd") or content words ("work", "thing"). Indexed text is
/// NOT filtered — stopwords remain in the index; we simply refuse to score
/// them on the query side.
///
/// The list is intentionally bounded (~70 words) and focuses on:
/// - articles and copulas: "a", "the", "is", "are", ...
/// - question stems: "how", "why", "when", "where", "which", ...
/// - auxiliary verbs and modals: "do", "does", "can", "would", ...
/// - mild quantifiers and conjunctions: "some", "any", "also", "then", ...
/// - common prepositions: "about", "into", "between", ...
pub const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "of", "in", "on", "at", "by", "for", "with", "to", "from", "and", "or", "but", "not", "it",
    "its", "this", "that", "these", "those", "as", "if", "so", "such", "do", "does", "did", "can",
    "will", "would", "should", "could", "how", "why", "when", "where", "which", "whose", "whom",
    "also", "then", "than", "some", "any", "most", "many", "about", "across", "around", "between",
    "during", "into", "onto", "over", "under", "while",
];

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
///
/// `Eq` is intentionally not derived: candidates may carry an
/// `Option<f32>` semantic score which has no total ordering. Direct
/// equality comparison was never used on `RetrievalPlan` outside tests
/// asserting individual fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalPlan {
    pub query: String,
    pub token_budget: u32,
    pub estimated_tokens: u32,
    pub candidates: Vec<RetrievalCandidate>,
    /// Populated when the low-coverage fallback expanded the candidate set.
    /// `None` for normal, well-scored plans; legacy plans without this field
    /// round-trip as `None` via `serde(default)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

/// A ranked candidate selected for retrieval.
///
/// The legacy `score` field is the lexical score used by `LexicalIndex`'s
/// scoring loop. Hybrid retrieval populates the additional optional
/// `semantic_score` / `semantic_rank` / `lexical_rank` fields and surfaces
/// the per-tier reasons through `reasons`. JSON consumers that pre-date
/// the hybrid layer continue to round-trip cleanly thanks to
/// `skip_serializing_if = "Option::is_none"`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalCandidate {
    pub id: String,
    pub title: String,
    pub score: usize,
    pub estimated_tokens: u32,
    pub reasons: Vec<String>,
    /// Semantic similarity score in `[0, 1]`. Set by the hybrid layer when
    /// a candidate participated in the semantic ranked list. `None` for
    /// lexical-only or fallback-added candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_score: Option<f32>,
    /// 1-indexed rank in the semantic ranked list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_rank: Option<usize>,
    /// 1-indexed rank in the lexical ranked list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<usize>,
    /// Reciprocal Rank Fusion score combining lexical and semantic ranks.
    /// Populated by `plan_retrieval_hybrid` after both tiers run; this is
    /// the value the candidate list is sorted by. `None` for lexical-only
    /// runs (`KB_SEMANTIC=0` or no semantic hits) where `score` already
    /// drives the order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_score: Option<f32>,
    /// Personalized-PageRank score from the citation graph (bn-32od).
    /// `None` when the structural tier is disabled or the candidate
    /// didn't appear in the PPR result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_score: Option<f32>,
    /// 1-indexed rank in the structural ranked list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_rank: Option<usize>,
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
    ///
    /// Stopwords ("is", "the", "for", ...) are stripped from the query before
    /// scoring — they would otherwise dominate ranking with noisy matches.
    /// A query that contains nothing but stopwords returns an empty result
    /// set; callers that want to distinguish "no matches" from "reduced to
    /// stopwords" should call [`tokenize_query`] themselves.
    #[must_use]
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_tokens = tokenize_query(query);
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
    ///
    /// Stopwords are filtered out of the query tokens before scoring so that
    /// common words ("is", "the", "what", ...) don't dominate the plan's
    /// ranking reasons. If the filter removes every token, the plan is empty.
    ///
    /// When regular scoring produces fewer than
    /// [`MIN_CANDIDATES_BEFORE_FALLBACK`] positive candidates, or the top
    /// candidate's score is below [`MIN_TOP_SCORE_BEFORE_FALLBACK`], a
    /// low-coverage fallback kicks in: up to three auto-generated wiki index
    /// pages (see [`FALLBACK_INDEX_PAGES`]) and up to
    /// [`FALLBACK_SOURCE_SUMMARY_LIMIT`] source summaries (most recently
    /// modified first) are appended as baseline context. Added candidates are
    /// stamped with [`FALLBACK_CANDIDATE_REASON`] and the plan's
    /// [`RetrievalPlan::fallback_reason`] is set to
    /// [`FALLBACK_REASON_LOW_COVERAGE`]. Pages that don't exist under `root`
    /// are quietly skipped — freshly-initialized KBs may not have them yet.
    ///
    /// The `root` argument is only read for the fallback's existence checks
    /// and mtime-based ordering; it is otherwise unused, so callers with no
    /// on-disk KB can pass any path.
    #[must_use]
    pub fn plan_retrieval(&self, query: &str, token_budget: u32, root: &Path) -> RetrievalPlan {
        let query_tokens = tokenize_query(query);
        if query_tokens.is_empty() || token_budget == 0 {
            return RetrievalPlan {
                query: query.to_string(),
                token_budget,
                estimated_tokens: 0,
                candidates: Vec::new(),
                fallback_reason: None,
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
        let mut included_ids: Vec<String> = Vec::new();

        let top_score = scored.first().map_or(0, |(analysis, _)| analysis.score);
        let positive_candidate_count = scored.len();

        for (analysis, entry) in scored.into_iter().take(DEFAULT_RETRIEVAL_TOP_K) {
            let entry_tokens = estimate_entry_tokens(entry);
            if estimated_tokens.saturating_add(entry_tokens) > token_budget {
                continue;
            }

            estimated_tokens = estimated_tokens.saturating_add(entry_tokens);
            included_ids.push(entry.id.clone());
            candidates.push(RetrievalCandidate {
                id: entry.id.clone(),
                title: entry.title.clone(),
                score: analysis.score,
                estimated_tokens: entry_tokens,
                reasons: analysis.reasons,
                semantic_score: None,
                semantic_rank: None,
                lexical_rank: None,

                fused_score: None,
                structural_score: None,
                structural_rank: None,
            });
        }

        // Low-coverage fallback: when scoring found too few (or too weak)
        // candidates, append baseline context so the LLM has at least the
        // three auto-generated index pages plus a handful of source
        // summaries to reason from. This makes high-level meta queries
        // like "what is this wiki about?" answerable instead of returning
        // zero citations. See bn-1yvv.
        let fallback_reason = if positive_candidate_count < MIN_CANDIDATES_BEFORE_FALLBACK
            || top_score < MIN_TOP_SCORE_BEFORE_FALLBACK
        {
            append_low_coverage_fallback(
                root,
                token_budget,
                &mut candidates,
                &mut estimated_tokens,
                &mut included_ids,
            );
            Some(FALLBACK_REASON_LOW_COVERAGE.to_string())
        } else {
            None
        };

        RetrievalPlan {
            query: query.to_string(),
            token_budget,
            estimated_tokens,
            candidates,
            fallback_reason,
        }
    }
}

/// Append the low-coverage fallback candidates to `candidates`.
///
/// Adds, in order:
/// 1. Each existing page in [`FALLBACK_INDEX_PAGES`].
/// 2. Up to [`FALLBACK_SOURCE_SUMMARY_LIMIT`] source-summary pages from
///    `wiki/sources/`, ordered by file mtime descending (most recent first),
///    with filename as a stable tie-breaker.
///
/// Candidates already present in `included_ids` are skipped so the fallback
/// never duplicates a page the main scorer already picked. Individual
/// candidates that would push `estimated_tokens` past `token_budget` are
/// skipped; the fallback never silently inflates the token usage beyond the
/// caller's cap.
///
/// Each appended candidate is stamped with [`FALLBACK_CANDIDATE_REASON`] so
/// introspection tools (and the retrieval plan JSON) can show why the page
/// was included even though it had zero query match.
fn append_low_coverage_fallback(
    root: &Path,
    token_budget: u32,
    candidates: &mut Vec<RetrievalCandidate>,
    estimated_tokens: &mut u32,
    included_ids: &mut Vec<String>,
) {
    let try_push = |rel_id: &str,
                    candidates: &mut Vec<RetrievalCandidate>,
                    estimated_tokens: &mut u32,
                    included_ids: &mut Vec<String>| {
        if included_ids.iter().any(|id| id == rel_id) {
            return;
        }
        let abs = root.join(rel_id);
        if !abs.is_file() {
            return;
        }
        let tokens = estimate_path_tokens(&abs);
        if estimated_tokens.saturating_add(tokens) > token_budget {
            return;
        }
        *estimated_tokens = estimated_tokens.saturating_add(tokens);
        included_ids.push(rel_id.to_string());
        candidates.push(RetrievalCandidate {
            id: rel_id.to_string(),
            title: fallback_title_for(&abs, rel_id),
            score: 0,
            estimated_tokens: tokens,
            reasons: vec![FALLBACK_CANDIDATE_REASON.to_string()],
            semantic_score: None,
            semantic_rank: None,
            lexical_rank: None,

            fused_score: None,
            structural_score: None,
            structural_rank: None,
        });
    };

    for index_page in FALLBACK_INDEX_PAGES {
        try_push(index_page, candidates, estimated_tokens, included_ids);
    }

    for rel_id in collect_fallback_source_summaries(root) {
        try_push(&rel_id, candidates, estimated_tokens, included_ids);
    }
}

/// Collect up to [`FALLBACK_SOURCE_SUMMARY_LIMIT`] source-summary page
/// relative paths ordered by mtime descending (newest first), with filename
/// as a tie-breaker for deterministic output when mtimes match (common on
/// filesystems that only track second-granularity mtimes, or during tests
/// that write files in a tight loop).
///
/// `wiki/sources/index.md` is skipped because it's already covered by
/// [`FALLBACK_INDEX_PAGES`]. A missing directory returns an empty list so
/// freshly-initialized KBs don't panic here.
fn collect_fallback_source_summaries(root: &Path) -> Vec<String> {
    let dir = root.join(WIKI_SOURCES);
    let Ok(read_dir) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut entries: Vec<(std::time::SystemTime, String, String)> = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        let file_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };
        if file_name == "index.md" {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        let rel_id = format!("{WIKI_SOURCES}/{file_name}");
        entries.push((mtime, file_name, rel_id));
    }

    // Sort newest first, with filename ascending as tie-breaker.
    entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    entries.truncate(FALLBACK_SOURCE_SUMMARY_LIMIT);
    entries.into_iter().map(|(_, _, rel)| rel).collect()
}

/// Public token estimator for the hybrid layer. Mirrors the
/// `estimate_path_tokens` heuristic so semantic-only tail candidates
/// budget identically to fallback candidates.
#[must_use]
pub fn estimate_path_tokens_for_hybrid(path: &Path) -> u32 {
    estimate_path_tokens(path)
}

/// Public fallback-title resolver for the hybrid layer.
#[must_use]
pub fn fallback_title_for_hybrid(path: &Path, rel_id: &str) -> String {
    fallback_title_for(path, rel_id)
}

/// Estimate token usage for a file on disk using the same ~4-chars-per-token
/// ratio the rest of the module uses, falling back to
/// [`MIN_ENTRY_TOKEN_ESTIMATE`] when the file can't be read (e.g. permission
/// error) so the fallback still accounts for its presence in the budget.
fn estimate_path_tokens(path: &Path) -> u32 {
    let Ok(meta) = fs::metadata(path) else {
        return MIN_ENTRY_TOKEN_ESTIMATE;
    };
    let bytes = usize::try_from(meta.len()).unwrap_or(usize::MAX);
    let estimated = bytes.div_ceil(APPROX_CHARS_PER_TOKEN);
    u32::try_from(estimated)
        .unwrap_or(u32::MAX)
        .max(MIN_ENTRY_TOKEN_ESTIMATE)
}

/// Best-effort title for a fallback candidate. Prefers the frontmatter
/// `title` (sources) or `name` (concepts) so the plan JSON reads nicely;
/// falls back to the relative path when frontmatter can't be parsed.
fn fallback_title_for(path: &Path, rel_id: &str) -> String {
    if let Ok((frontmatter, _)) = read_frontmatter(path) {
        if let Some(title) = frontmatter.get("title").and_then(Value::as_str) {
            if !title.is_empty() {
                return title.to_string();
            }
        }
        if let Some(name) = frontmatter.get("name").and_then(Value::as_str) {
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    rel_id.to_string()
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

    let query_tokens = tokenize_query(&plan.query);
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
    state_dir(root).join(INDEX_REL[0]).join(INDEX_REL[1])
}

/// Returns the filesystem path for the lexical search index under `root`.
///
/// The file is written at `state/indexes/lexical.json`. The path is returned
/// even when the file does not yet exist.
#[must_use]
pub fn lexical_index_path(root: &Path) -> PathBuf {
    index_path(root)
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

    // bn-2qda: Captions injected by the compile `captions` pass live in
    // managed regions whose ids start with `caption-`. Merge their bodies
    // into the indexed `summary` so query terms inside an image caption
    // surface a hit on the source page. The summary field is the only
    // "long" text in a `LexicalEntry`, so this is the cheapest place to
    // pick up captions without touching the entry shape.
    let regions = extract_managed_regions(&body);
    let summary_body = regions
        .iter()
        .find(|r| r.id == "summary")
        .map(|r| r.body(&body).trim().to_string())
        .unwrap_or_default();

    let caption_bodies: Vec<String> = regions
        .iter()
        .filter(|r| r.id.starts_with("caption-"))
        .map(|r| r.body(&body).trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let summary = if caption_bodies.is_empty() {
        summary_body
    } else if summary_body.is_empty() {
        caption_bodies.join("\n\n")
    } else {
        format!("{summary_body}\n\n{}", caption_bodies.join("\n\n"))
    };

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
        .flat_map(split_camel_case)
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() > 1)
        .collect()
}

/// Tokenize a query string and drop common English stopwords.
///
/// Use this for query-side tokenization in search and retrieval. Indexed text
/// should keep using [`tokenize`] so that stopwords stay in the index — we
/// only want to avoid scoring them on the query side. Matching is
/// case-insensitive because [`tokenize`] lowercases tokens before we compare.
#[must_use]
pub fn tokenize_query(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .filter(|token| !is_stopword(token))
        .collect()
}

/// Return `true` when `text` has at least one token but every token is a
/// stopword.
///
/// Callers use this to decide whether an empty search result should be
/// framed as "no matches" or as "your query is entirely stopwords". Queries
/// with no tokens at all (blank, pure punctuation) return `false` so the
/// caller can fall through to its usual empty-results handling.
#[must_use]
pub fn query_reduced_to_stopwords(text: &str) -> bool {
    let raw = tokenize(text);
    !raw.is_empty() && raw.iter().all(|token| is_stopword(token))
}

fn is_stopword(token: &str) -> bool {
    STOPWORDS.iter().any(|sw| sw.eq_ignore_ascii_case(token))
}

/// Split a word on `CamelCase` boundaries, e.g. `SourceDocument` → `Source`, `Document`.
fn split_camel_case(word: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = word.chars().collect();
    for i in 1..chars.len() {
        if chars[i].is_uppercase() && chars[i - 1].is_lowercase() {
            parts.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    parts.push(chars[start..].iter().collect());
    parts
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
        let plan = index.plan_retrieval("rust", budget, root);

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
        let plan = index.plan_retrieval("borrowck checker", 1_000, root);

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
                semantic_score: None,
                semantic_rank: None,
                lexical_rank: None,

                fused_score: None,
                structural_score: None,
                structural_rank: None,
            }],
            fallback_reason: None,
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
                semantic_score: None,
                semantic_rank: None,
                lexical_rank: None,

                fused_score: None,
                structural_score: None,
                structural_rank: None,
            }],
            fallback_reason: None,
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

    #[test]
    fn tokenize_splits_camel_case() {
        let tokens = tokenize("SourceDocument");
        assert_eq!(tokens, vec!["source", "document"]);
    }

    #[test]
    fn tokenize_splits_multi_part_camel_case() {
        let tokens = tokenize("SourceRevisionId");
        assert_eq!(tokens, vec!["source", "revision", "id"]);
    }

    #[test]
    fn tokenize_preserves_lowercase_words() {
        let tokens = tokenize("hello world");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn search_finds_doc_with_camel_case_query() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();

        write_source_page(
            &sources,
            "ingest",
            "Source Ingestion",
            "How ingestion works.",
        );

        let index = build_lexical_index(root).unwrap();
        let results = index.search("SourceDocument ID vs SourceRevision ID", 10);

        assert!(!results.is_empty(), "should find Source Ingestion doc");
        assert_eq!(results[0].id, "wiki/sources/ingest.md");
    }

    #[test]
    fn tokenize_query_strips_stopwords() {
        assert_eq!(
            tokenize_query("the Raft is fast"),
            vec!["raft".to_string(), "fast".to_string()]
        );
    }

    #[test]
    fn tokenize_query_returns_empty_when_only_stopwords() {
        let tokens = tokenize_query("a an the");
        assert!(
            tokens.is_empty(),
            "stopword-only query should tokenize to empty, got {tokens:?}"
        );
    }

    #[test]
    fn tokenize_query_matches_stopwords_case_insensitively() {
        assert!(tokenize_query("The IS Are").is_empty());
    }

    #[test]
    fn tokenize_query_keeps_short_technical_terms() {
        // Stopword list must not drop short technical terms like "raft" or
        // "sgd" — they are shorter than many stopwords but not generic.
        let tokens = tokenize_query("Raft SGD");
        assert!(tokens.contains(&"raft".to_string()));
        assert!(tokens.contains(&"sgd".to_string()));
    }

    #[test]
    fn query_reduced_to_stopwords_detects_stopword_only_queries() {
        assert!(query_reduced_to_stopwords("is"));
        assert!(query_reduced_to_stopwords("the and or"));
        assert!(!query_reduced_to_stopwords("Raft safety"));
        assert!(!query_reduced_to_stopwords("the Raft"));
        // Empty / whitespace / pure punctuation: no tokens at all — not a
        // stopword-only query, just an empty one.
        assert!(!query_reduced_to_stopwords(""));
        assert!(!query_reduced_to_stopwords("   "));
        assert!(!query_reduced_to_stopwords("!!!"));
    }

    #[test]
    fn stopwords_do_not_contribute_to_scoring() {
        // A body that mentions "is" 5 times and "raft" 1 time, scored
        // against the query "Raft", must count only the 1 raft match —
        // not be flooded by the 5 stopword hits.
        let entry = LexicalEntry {
            id: "wiki/sources/raft.md".to_string(),
            title: "Raft".to_string(),
            aliases: Vec::new(),
            headings: Vec::new(),
            summary: "is is is is is raft".to_string(),
        };
        let query_tokens = tokenize_query("Raft");
        let analysis = analyze_entry(&entry, &query_tokens);
        // Only raft matches should score: title (1 * WEIGHT_TITLE) +
        // summary (1 * WEIGHT_SUMMARY). No "is" contributions.
        assert_eq!(analysis.score, WEIGHT_TITLE + WEIGHT_SUMMARY);
        assert!(
            analysis
                .reasons
                .iter()
                .all(|reason| !reason.contains("'is'")),
            "scoring reasons must not reference stopword 'is': {:?}",
            analysis.reasons
        );
    }

    #[test]
    fn search_with_stopword_prefix_scores_same_as_without() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(
            &sources,
            "raft",
            "Raft Protocol",
            "The Raft consensus protocol.",
        );

        let index = build_lexical_index(root).unwrap();
        let with_stopword = index.search("the Raft protocol", 10);
        let without_stopword = index.search("Raft protocol", 10);
        assert_eq!(
            with_stopword.len(),
            without_stopword.len(),
            "stopword prefix must not change result count"
        );
        assert_eq!(
            with_stopword[0].score, without_stopword[0].score,
            "stopword prefix must not change score"
        );
    }

    #[test]
    fn plan_retrieval_reasons_exclude_stopwords() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(
            &sources,
            "raft",
            "Raft Safety",
            "Raft is a consensus algorithm and it is safe.",
        );

        let index = build_lexical_index(root).unwrap();
        let plan = index.plan_retrieval("What is Raft?", 1_000, root);
        assert_eq!(plan.candidates.len(), 1);
        for reason in &plan.candidates[0].reasons {
            assert!(
                !reason.contains("'is'")
                    && !reason.contains("'what'")
                    && !reason.contains("'a'"),
                "stopword reason leaked into retrieval plan: {reason}"
            );
        }
    }

    #[test]
    fn search_returns_empty_for_stopword_only_query() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(&sources, "raft", "Raft", "Raft is a protocol.");

        let index = build_lexical_index(root).unwrap();
        assert!(index.search("is", 10).is_empty());
        assert!(index.search("the and or", 10).is_empty());
    }

    #[test]
    fn tokenize_query_strips_question_stem_words() {
        // Interrogatives and auxiliary question verbs must be filtered so that
        // queries like "How does X work?" score only the content words.
        let tokens = tokenize_query("How does X work when Y?");
        assert!(!tokens.contains(&"how".to_string()));
        assert!(!tokens.contains(&"does".to_string()));
        assert!(!tokens.contains(&"when".to_string()));
        // Content words must be preserved.
        assert!(tokens.contains(&"work".to_string()));
    }

    #[test]
    fn tokenize_query_strips_why_where_which_whom() {
        let tokens = tokenize_query("Why where which whose whom raft");
        assert_eq!(tokens, vec!["raft".to_string()]);
    }

    #[test]
    fn tokenize_query_strips_mild_quantifiers_and_conjunctions() {
        let tokens = tokenize_query("also then than some any most many raft");
        assert_eq!(tokens, vec!["raft".to_string()]);
    }

    #[test]
    fn tokenize_query_strips_common_prepositions() {
        let tokens =
            tokenize_query("about across around between during into onto over under while raft");
        assert_eq!(tokens, vec!["raft".to_string()]);
    }

    #[test]
    fn tokenize_query_keeps_content_words_like_work() {
        // "work", "thing", and similar content words must NOT be filtered.
        // They're legitimate scoring signals in queries like "how does X work".
        let tokens = tokenize_query("how does X work");
        assert!(tokens.contains(&"work".to_string()));
        let tokens = tokenize_query("what is the main thing");
        assert!(tokens.contains(&"main".to_string()));
        assert!(tokens.contains(&"thing".to_string()));
    }

    #[test]
    fn plan_retrieval_reasons_exclude_question_stems() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(
            &sources,
            "scheduling",
            "Process Scheduling",
            "How the scheduler decides when to run which process.",
        );

        let index = build_lexical_index(root).unwrap();
        let plan = index.plan_retrieval("How does scheduling differ from Y?", 1_000, root);
        assert_eq!(plan.candidates.len(), 1);
        for reason in &plan.candidates[0].reasons {
            assert!(
                !reason.contains("'how'")
                    && !reason.contains("'does'")
                    && !reason.contains("'when'")
                    && !reason.contains("'which'")
                    && !reason.contains("'from'"),
                "question-stem stopword leaked into retrieval plan: {reason}"
            );
        }
    }

    #[test]
    fn stopwords_list_stays_bounded() {
        // Guard against unbounded growth; we explicitly do NOT want to become
        // a generic English stopword filter. Cap roughly matches the design
        // target in bn-1jl (~70 words).
        assert!(
            STOPWORDS.len() <= 70,
            "STOPWORDS grew past the design cap of 70 (now {}): review additions for scope creep",
            STOPWORDS.len()
        );
    }

    /// Write a placeholder index page at `rel_path` under `root`. Content
    /// doesn't matter for fallback selection — we only check existence — but
    /// we still write valid frontmatter + body so `fallback_title_for` can
    /// extract a title.
    fn write_index_page(root: &Path, rel_path: &str, title: &str) {
        let abs = root.join(rel_path);
        fs::create_dir_all(abs.parent().unwrap()).unwrap();
        let content = format!("---\ntitle: {title}\n---\n\n# {title}\n\nIndex page body.\n");
        fs::write(&abs, content).unwrap();
    }

    #[test]
    fn low_coverage_fallback_adds_index_pages_and_source_summaries() {
        // Seed a KB whose only source has a summary that won't match the
        // query. Regular scoring produces zero candidates, so the fallback
        // must kick in and add (1) every existing index page and (2) the
        // source summary as a baseline context candidate.
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(
            &sources,
            "borrow-checker",
            "Borrow Checker",
            "Ownership moves and lifetimes.",
        );
        write_index_page(root, "wiki/index.md", "Wiki");
        write_index_page(root, "wiki/concepts/index.md", "Concepts");
        write_index_page(root, "wiki/sources/index.md", "Sources");

        let index = build_lexical_index(root).unwrap();
        // Query uses only meta terms that don't appear in the summary.
        let plan = index.plan_retrieval("what is this wiki about?", 100_000, root);

        assert_eq!(plan.fallback_reason.as_deref(), Some("low-coverage"));

        let ids: Vec<&str> = plan.candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"wiki/index.md"), "missing wiki/index.md in {ids:?}");
        assert!(
            ids.contains(&"wiki/concepts/index.md"),
            "missing wiki/concepts/index.md in {ids:?}"
        );
        assert!(
            ids.contains(&"wiki/sources/index.md"),
            "missing wiki/sources/index.md in {ids:?}"
        );
        assert!(
            ids.contains(&"wiki/sources/borrow-checker.md"),
            "missing source summary in {ids:?}"
        );

        // Every fallback-added candidate must carry the dedicated reason so
        // introspection can explain its presence.
        for candidate in &plan.candidates {
            if candidate.score == 0 {
                assert_eq!(
                    candidate.reasons,
                    vec![FALLBACK_CANDIDATE_REASON.to_string()],
                    "fallback candidate {} missing reason stamp",
                    candidate.id
                );
            }
        }
    }

    #[test]
    fn low_coverage_fallback_skips_missing_index_pages() {
        // Freshly-initialized KBs may not have any of the three index pages
        // yet. The fallback must tolerate their absence instead of panicking
        // or inserting broken references.
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(&sources, "alpha", "Alpha", "Totally unrelated.");

        let index = build_lexical_index(root).unwrap();
        let plan = index.plan_retrieval("hdfs federation", 100_000, root);

        assert_eq!(plan.fallback_reason.as_deref(), Some("low-coverage"));
        let ids: Vec<&str> = plan.candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(!ids.contains(&"wiki/index.md"));
        assert!(!ids.contains(&"wiki/concepts/index.md"));
        assert!(!ids.contains(&"wiki/sources/index.md"));
        // The source summary still shows up as a baseline contributor.
        assert!(
            ids.contains(&"wiki/sources/alpha.md"),
            "missing source summary in {ids:?}"
        );
    }

    #[test]
    fn strong_matches_do_not_trigger_fallback() {
        // Seed a KB with enough well-scoring candidates that the fallback
        // thresholds are never tripped. The plan must omit `fallback_reason`
        // and MUST NOT include any of the index pages even though they
        // exist on disk.
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        for slug in &["raft-one", "raft-two", "raft-three", "raft-four"] {
            write_source_page(
                &sources,
                slug,
                &format!("Raft {slug}"),
                "Raft consensus protocol details.",
            );
        }
        write_index_page(root, "wiki/index.md", "Wiki");
        write_index_page(root, "wiki/concepts/index.md", "Concepts");
        write_index_page(root, "wiki/sources/index.md", "Sources");

        let index = build_lexical_index(root).unwrap();
        let plan = index.plan_retrieval("raft", 100_000, root);

        assert!(plan.fallback_reason.is_none(), "fallback fired unexpectedly");
        let ids: Vec<&str> = plan.candidates.iter().map(|c| c.id.as_str()).collect();
        for index_page in FALLBACK_INDEX_PAGES {
            assert!(
                !ids.contains(index_page),
                "index page {index_page} leaked into a strong-match plan: {ids:?}"
            );
        }
        // Every candidate here came from real scoring; none carries the
        // fallback reason stamp.
        for candidate in &plan.candidates {
            assert!(
                !candidate.reasons.iter().any(|r| r == FALLBACK_CANDIDATE_REASON),
                "scored candidate {} wrongly stamped with fallback reason",
                candidate.id,
            );
        }
    }

    #[test]
    fn low_coverage_fallback_caps_source_summaries_at_five() {
        // When more than FALLBACK_SOURCE_SUMMARY_LIMIT source summaries
        // exist, the fallback must keep at most five of them. Exact
        // selection order is tested separately via `collect_fallback_source_summaries`.
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        for slug in &["a", "b", "c", "d", "e", "f", "g"] {
            write_source_page(&sources, slug, slug, "Unrelated body.");
        }

        let index = build_lexical_index(root).unwrap();
        let plan = index.plan_retrieval("zzzz-no-match", 100_000, root);
        assert_eq!(plan.fallback_reason.as_deref(), Some("low-coverage"));

        let source_ids: Vec<&str> = plan
            .candidates
            .iter()
            .filter(|c| c.id.starts_with("wiki/sources/") && c.id != "wiki/sources/index.md")
            .map(|c| c.id.as_str())
            .collect();

        assert_eq!(
            source_ids.len(),
            FALLBACK_SOURCE_SUMMARY_LIMIT,
            "fallback kept {} source summaries, expected exactly {}",
            source_ids.len(),
            FALLBACK_SOURCE_SUMMARY_LIMIT,
        );
    }

    #[test]
    fn collect_fallback_source_summaries_orders_newest_first() {
        // Directly exercise the mtime-sorted summary picker. Writing each
        // page with a spaced-out sleep between writes gives us monotonically
        // increasing mtimes regardless of filesystem granularity.
        use std::thread::sleep;
        use std::time::Duration;

        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();

        // Write in chronological order: a (oldest) ... g (newest).
        for slug in &["a", "b", "c", "d", "e", "f", "g"] {
            write_source_page(&sources, slug, slug, "Body.");
            sleep(Duration::from_millis(20));
        }

        let picked = collect_fallback_source_summaries(root);
        assert_eq!(picked.len(), FALLBACK_SOURCE_SUMMARY_LIMIT);
        // Newest 5 should be the last five writes: g, f, e, d, c (in mtime
        // descending order).
        assert_eq!(
            picked,
            vec![
                "wiki/sources/g.md".to_string(),
                "wiki/sources/f.md".to_string(),
                "wiki/sources/e.md".to_string(),
                "wiki/sources/d.md".to_string(),
                "wiki/sources/c.md".to_string(),
            ],
        );
    }

    #[test]
    fn low_coverage_fallback_skips_duplicates_already_in_plan() {
        // If scoring already included `wiki/sources/alpha.md` as a positive
        // match, the fallback must NOT add a duplicate copy — even if it
        // would otherwise pick that file when iterating source summaries.
        let dir = tempdir().unwrap();
        let root = dir.path();
        let sources = root.join("wiki/sources");
        fs::create_dir_all(&sources).unwrap();
        write_source_page(&sources, "alpha", "Alpha", "Raft safety discussion.");

        let index = build_lexical_index(root).unwrap();
        // Query matches "raft" (found in summary) — 1 positive candidate,
        // which is below MIN_CANDIDATES_BEFORE_FALLBACK, so fallback fires.
        let plan = index.plan_retrieval("raft", 100_000, root);
        assert_eq!(plan.fallback_reason.as_deref(), Some("low-coverage"));

        let count = plan
            .candidates
            .iter()
            .filter(|c| c.id == "wiki/sources/alpha.md")
            .count();
        assert_eq!(count, 1, "duplicate fallback entry for wiki/sources/alpha.md");
    }
}
