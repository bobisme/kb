//! Hybrid retrieval: lexical + semantic ranked lists fused with Reciprocal
//! Rank Fusion. Sits on top of [`LexicalIndex`] without changing its scoring.
//!
//! Flow:
//!
//! 1. Run `LexicalIndex::search(query, limit)` — always.
//! 2. If semantic is enabled (config + `KB_SEMANTIC` env), open the
//!    embedding store, embed the query, and run `knn_search`.
//! 3. Filter the semantic list against threshold guards
//!    (`MIN_SEMANTIC_SCORE`, `MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL`) so junk
//!    out-of-corpus matches never reach fusion.
//! 4. RRF-fuse and return [`HybridResult`]s annotated with both ranks and
//!    a vector of human-readable reasons.
//!
//! When the embedding store doesn't exist or semantic search fails, we log
//! once and degrade to lexical-only behavior — `kb ask` keeps working.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use rusqlite::Connection;
use tracing::warn;

use crate::lexical::{LexicalIndex, RetrievalCandidate, RetrievalPlan, SearchResult};
use crate::semantic::{
    EmbeddingBackend, HashEmbedBackend, embed::embedding_db_path, search::knn_search,
};

/// RRF constant `k`. Pulled directly from the standard literature and
/// matched to bones-search.
pub const RRF_K: usize = 60;

/// Hard floor on a semantic hit's score before it is included in fusion.
///
/// The design doc lifted the `0.15` constant from `bones-search`, where
/// it's tuned for ML-quality embeddings whose cosine for unrelated text
/// hovers near 0. With kb v1's hash-embed backend, unrelated text shares
/// enough character-n-gram surface for cosine to land in `[0.05, 0.35]`,
/// so a 0.15 floor would let pure noise fuse with lexical hits. We raise
/// the floor to `0.35` for the hash backend; when Phase 2 ships an ML
/// backend the constant will likely move back down (or become
/// backend-relative). Flagged as a deviation in the bone report.
pub const MIN_SEMANTIC_SCORE: f32 = 0.35;

/// Stricter floor when lexical returned zero hits.
///
/// The semantic tier is the only signal in that case, so the top hit
/// needs to be visibly higher than the noise floor before we present it
/// as a real match. Hash-embed's morphological hits land in `[0.45,
/// 0.55]`; pure noise against the same corpus hovers in `[0.20, 0.40]`.
/// A floor of `0.45` catches the morphological match while still
/// rejecting most of the noise. Same rationale as [`MIN_SEMANTIC_SCORE`]
/// applies — the bones constant of 0.20 is too low for the hash backend.
pub const MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL: f32 = 0.45;

static SEMANTIC_DEGRADED_WARNED: AtomicBool = AtomicBool::new(false);

/// A single hybrid retrieval result.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct HybridResult {
    /// Page id, matches `LexicalEntry::id` and the semantic store's `item_id`.
    pub item_id: String,
    /// Page title (filled from the lexical index when available).
    pub title: String,
    /// Fused RRF score in `(0, 1]`.
    pub score: f32,
    /// Lexical RRF contribution (0.0 when not in the lexical list).
    pub lexical_score: f32,
    /// Semantic RRF contribution (0.0 when not in the semantic list).
    pub semantic_score: f32,
    /// 1-indexed lexical rank (or `usize::MAX` when absent).
    pub lexical_rank: usize,
    /// 1-indexed semantic rank (or `usize::MAX` when absent).
    pub semantic_rank: usize,
    /// Underlying lexical scoring score, copied from the lexical search
    /// result for explanation. 0 when absent.
    pub lexical_raw_score: usize,
    /// Underlying semantic similarity score in `[0, 1]`. 0 when absent.
    pub semantic_raw_score: f32,
    /// Per-tier reasons assembled for human display.
    pub reasons: Vec<String>,
}

/// Tunable hybrid-search parameters.
#[derive(Debug, Clone, Copy)]
pub struct HybridOptions {
    /// Fuse against the semantic tier.
    pub semantic_enabled: bool,
    /// `k` in `1 / (k + rank)`.
    pub rrf_k: usize,
    /// Minimum semantic score for a hit to enter fusion.
    pub min_semantic_score: f32,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            semantic_enabled: true,
            rrf_k: RRF_K,
            min_semantic_score: MIN_SEMANTIC_SCORE,
        }
    }
}

/// Run hybrid retrieval with default options.
///
/// Reads the lexical index from disk, embeds the query with the
/// always-available hash backend, and KNN-searches the embedding store at
/// `.kb/state/embeddings.db` (when present). When `KB_SEMANTIC=0` is set,
/// behaves identically to a pure-lexical search.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded. Semantic
/// failures degrade to lexical-only with a one-shot warning.
pub fn hybrid_search(root: &Path, query: &str, limit: usize) -> Result<Vec<HybridResult>> {
    hybrid_search_with_options(root, query, limit, HybridOptions::default())
}

/// Run hybrid retrieval with explicit options. Tests and config-driven
/// callers use this to tweak `min_semantic_score` or short-circuit
/// semantic search without touching env vars.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded.
pub fn hybrid_search_with_options(
    root: &Path,
    query: &str,
    limit: usize,
    options: HybridOptions,
) -> Result<Vec<HybridResult>> {
    let lexical_index = LexicalIndex::load(root).context("load lexical index")?;
    hybrid_search_with_index(root, &lexical_index, query, limit, options)
}

/// Hybrid retrieval that reuses a caller-owned lexical index. Used by
/// long-lived processes (`kb serve`) that load the lexical index once at
/// startup.
///
/// # Errors
///
/// Returns an error when the semantic store cannot be opened. Most
/// semantic failures degrade to lexical-only.
pub fn hybrid_search_with_index(
    root: &Path,
    lexical_index: &LexicalIndex,
    query: &str,
    limit: usize,
    options: HybridOptions,
) -> Result<Vec<HybridResult>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let lexical_hits = lexical_index.search(query, limit);

    let semantic_enabled = options.semantic_enabled && semantic_env_enabled();
    let semantic_hits = if semantic_enabled {
        match run_semantic(root, query, limit) {
            Ok(hits) => hits,
            Err(err) => {
                if !SEMANTIC_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
                    warn!("semantic retrieval unavailable; using lexical only: {err:#}");
                }
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let semantic_filtered = filter_semantic(
        semantic_hits,
        lexical_hits.is_empty(),
        options.min_semantic_score,
    );

    Ok(fuse(
        &lexical_hits,
        &semantic_filtered,
        limit,
        options.rrf_k,
    ))
}

fn run_semantic(root: &Path, query: &str, limit: usize) -> Result<Vec<SemanticHit>> {
    let db_path = embedding_db_path(root);
    if !db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open(&db_path)
        .with_context(|| format!("open embedding db {}", db_path.display()))?;
    let backend = HashEmbedBackend::new();
    let qvec = backend
        .embed(query)
        .context("embed query for semantic search")?;
    let raw = knn_search(&conn, &qvec, limit).context("knn search")?;
    Ok(raw
        .into_iter()
        .map(|hit| SemanticHit {
            item_id: hit.item_id,
            score: hit.score,
        })
        .collect())
}

#[derive(Debug, Clone)]
struct SemanticHit {
    item_id: String,
    score: f32,
}

fn filter_semantic(
    hits: Vec<SemanticHit>,
    lexical_empty: bool,
    min_semantic_score: f32,
) -> Vec<SemanticHit> {
    if hits.is_empty() {
        return hits;
    }
    if lexical_empty && hits[0].score < MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL {
        return Vec::new();
    }
    hits.into_iter()
        .filter(|hit| hit.score >= min_semantic_score)
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn rank_to_score(rank: usize, k: usize) -> f32 {
    if rank == usize::MAX {
        0.0
    } else {
        1.0 / (k as f32 + rank as f32)
    }
}

#[allow(clippy::cast_precision_loss)]
fn fuse(
    lexical: &[SearchResult],
    semantic: &[SemanticHit],
    limit: usize,
    k: usize,
) -> Vec<HybridResult> {
    let mut lexical_meta: HashMap<&str, (usize, &SearchResult)> = HashMap::new();
    for (i, hit) in lexical.iter().enumerate() {
        lexical_meta.insert(hit.id.as_str(), (i + 1, hit));
    }

    let mut semantic_meta: HashMap<&str, (usize, f32)> = HashMap::new();
    for (i, hit) in semantic.iter().enumerate() {
        semantic_meta.insert(hit.item_id.as_str(), (i + 1, hit.score));
    }

    // Deterministic union of candidate ids: lexical order first, then
    // semantic-only ids in their semantic order, then a sorted fallback as
    // tie-breaker so identical inputs always produce identical outputs.
    let mut order: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for hit in lexical {
        if seen.insert(hit.id.clone()) {
            order.push(hit.id.clone());
        }
    }
    for hit in semantic {
        if seen.insert(hit.item_id.clone()) {
            order.push(hit.item_id.clone());
        }
    }

    let mut results: Vec<HybridResult> = order
        .into_iter()
        .filter_map(|id| {
            let lex = lexical_meta.get(id.as_str()).copied();
            let sem = semantic_meta.get(id.as_str()).copied();
            if lex.is_none() && sem.is_none() {
                return None;
            }
            let lexical_rank = lex.map_or(usize::MAX, |(rank, _)| rank);
            let semantic_rank = sem.map_or(usize::MAX, |(rank, _)| rank);

            let lexical_score = rank_to_score(lexical_rank, k);
            let semantic_score = rank_to_score(semantic_rank, k);
            let score = lexical_score + semantic_score;

            let title = lex.map_or_else(|| id.clone(), |(_, hit)| hit.title.clone());
            let lexical_raw_score = lex.map_or(0, |(_, hit)| hit.score);
            let semantic_raw_score = sem.map_or(0.0, |(_, score)| score);

            let mut reasons = Vec::new();
            if let Some((rank, hit)) = lex {
                reasons.push(format!(
                    "lexical match (rank {}, score {})",
                    rank, hit.score
                ));
                reasons.extend(hit.reasons.iter().cloned());
            }
            if let Some((rank, sim)) = sem {
                reasons.push(format!("semantic match (rank {rank}, score {sim:.2})"));
            }

            Some(HybridResult {
                item_id: id,
                title,
                score,
                lexical_score,
                semantic_score,
                lexical_rank,
                semantic_rank,
                lexical_raw_score,
                semantic_raw_score,
                reasons,
            })
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.item_id.cmp(&b.item_id))
    });
    results.truncate(limit);
    results
}

/// Build a budgeted retrieval plan that fuses lexical and semantic ranking.
///
/// Behavior:
///
/// 1. Run [`LexicalIndex::plan_retrieval`] to obtain the lexical-only base
///    plan (preserving its budgeting, low-coverage fallback, and reasons).
/// 2. Enrich each lexical candidate with `semantic_score` / `semantic_rank`
///    when the hybrid layer also returned it, plus its lexical rank.
/// 3. Append semantic-only candidates (present in the semantic ranked list
///    but missed by lexical) at the tail of `candidates`, subject to the
///    remaining token budget. These carry `score = 0` (no lexical match)
///    so the existing scoring sort isn't disturbed; their semantic
///    contribution shows in the `semantic_score` field and reasons.
///
/// `KB_SEMANTIC=0` short-circuits step 2 and 3, producing the legacy
/// lexical-only plan unchanged.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded.
pub fn plan_retrieval_hybrid(
    root: &Path,
    query: &str,
    token_budget: u32,
    options: HybridOptions,
) -> Result<RetrievalPlan> {
    let lexical_index = LexicalIndex::load(root).context("load lexical index")?;
    let mut plan = lexical_index.plan_retrieval(query, token_budget, root);

    let semantic_enabled = options.semantic_enabled && semantic_env_enabled();
    if !semantic_enabled {
        return Ok(plan);
    }

    // Treat the already-budgeted lexical candidate count as the soft cap
    // for the semantic-only tail. We never want to silently bloat the
    // candidate set far past the lexical's natural top-K.
    let semantic_limit = plan
        .candidates
        .len()
        .max(crate::lexical::DEFAULT_RETRIEVAL_TOP_K_FOR_HYBRID);

    let semantic_hits = match run_semantic(root, query, semantic_limit) {
        Ok(hits) => filter_semantic(
            hits,
            plan.candidates.is_empty(),
            options.min_semantic_score,
        ),
        Err(err) => {
            if !SEMANTIC_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
                warn!("semantic retrieval unavailable; using lexical only: {err:#}");
            }
            Vec::new()
        }
    };

    if semantic_hits.is_empty() {
        return Ok(plan);
    }

    let semantic_ranks: std::collections::HashMap<&str, (usize, f32)> = semantic_hits
        .iter()
        .enumerate()
        .map(|(i, hit)| (hit.item_id.as_str(), (i + 1, hit.score)))
        .collect();

    // Step 2: enrich existing candidates with semantic + lexical ranks.
    for (lex_rank, candidate) in plan.candidates.iter_mut().enumerate() {
        candidate.lexical_rank = Some(lex_rank + 1);
        if let Some((sem_rank, sem_score)) = semantic_ranks.get(candidate.id.as_str()).copied() {
            candidate.semantic_rank = Some(sem_rank);
            candidate.semantic_score = Some(sem_score);
            candidate
                .reasons
                .push(format!("semantic match (rank {sem_rank}, score {sem_score:.2})"));
        }
    }

    // Step 3: append semantic-only tail candidates respecting the budget.
    let lexical_ids: std::collections::HashSet<String> =
        plan.candidates.iter().map(|c| c.id.clone()).collect();
    for (i, hit) in semantic_hits.iter().enumerate() {
        if lexical_ids.contains(&hit.item_id) {
            continue;
        }
        let abs = root.join(&hit.item_id);
        if !abs.is_file() {
            continue;
        }
        let entry_tokens = crate::lexical::estimate_path_tokens_for_hybrid(&abs);
        if plan
            .estimated_tokens
            .saturating_add(entry_tokens)
            > plan.token_budget
        {
            continue;
        }
        plan.estimated_tokens = plan.estimated_tokens.saturating_add(entry_tokens);
        plan.candidates.push(RetrievalCandidate {
            id: hit.item_id.clone(),
            title: crate::lexical::fallback_title_for_hybrid(&abs, &hit.item_id),
            score: 0,
            estimated_tokens: entry_tokens,
            reasons: vec![format!(
                "semantic match (rank {}, score {:.2})",
                i + 1,
                hit.score
            )],
            semantic_score: Some(hit.score),
            semantic_rank: Some(i + 1),
            lexical_rank: None,
            fused_score: None,
        });
    }

    // Step 4: compute fused RRF score per candidate from its (lexical_rank,
    // semantic_rank) pair, then sort by it. Without this the plan would
    // keep the lexical-only order even though the candidate set has been
    // augmented with semantic-only tail hits — confusing for users who
    // see `semantic match (rank 1, score 0.59)` on a candidate sitting
    // below several lower-ranked lexical-only entries.
    for candidate in &mut plan.candidates {
        let lex = candidate
            .lexical_rank
            .map_or(0.0, |r| rank_to_score(r, RRF_K));
        let sem = candidate
            .semantic_rank
            .map_or(0.0, |r| rank_to_score(r, RRF_K));
        candidate.fused_score = Some(lex + sem);
    }
    plan.candidates.sort_by(|a, b| {
        b.fused_score
            .unwrap_or(0.0)
            .partial_cmp(&a.fused_score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    Ok(plan)
}

fn semantic_env_enabled() -> bool {
    !matches!(
        std::env::var("KB_SEMANTIC").ok().as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::SearchResult;

    fn lex(id: &str, score: usize) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            title: id.to_string(),
            score,
            reasons: Vec::new(),
        }
    }

    fn sem(id: &str, score: f32) -> SemanticHit {
        SemanticHit {
            item_id: id.to_string(),
            score,
        }
    }

    #[test]
    fn rrf_fuse_sums_reciprocal_ranks() {
        let lexical = vec![lex("a", 9), lex("b", 5), lex("c", 1)];
        let semantic = vec![sem("c", 0.9), sem("a", 0.7), sem("d", 0.5)];

        let fused = fuse(&lexical, &semantic, 4, 60);
        assert_eq!(fused.len(), 4);
        // 'a' is rank 1 in lexical and rank 2 in semantic — best fused score.
        assert_eq!(fused[0].item_id, "a");
        // Both 'a' and 'c' have rank-1+rank-2-or-3 contributions, so 'a'
        // (rank 1+2) beats 'c' (rank 3+1) by formula:
        // 1/61 + 1/62 ≈ 0.03253 vs 1/63 + 1/61 ≈ 0.03227.
        assert_eq!(fused[1].item_id, "c");
    }

    #[test]
    fn rrf_drops_items_not_present_in_either_list() {
        let lexical = vec![lex("a", 9)];
        let semantic = vec![sem("b", 0.6)];
        let fused = fuse(&lexical, &semantic, 5, 60);
        assert_eq!(fused.len(), 2);
        let ids: Vec<&str> = fused.iter().map(|r| r.item_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn filter_semantic_drops_below_threshold() {
        // Default min is MIN_SEMANTIC_SCORE; only hits >= that survive.
        let hits = vec![sem("a", 0.7), sem("b", 0.40), sem("c", 0.10)];
        let filtered = filter_semantic(hits, false, MIN_SEMANTIC_SCORE);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].item_id, "a");
        assert_eq!(filtered[1].item_id, "b");
    }

    #[test]
    fn filter_semantic_strict_when_lexical_empty() {
        // Top hit must be >= MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL when there's
        // no lexical safety net. The 0.45 floor blocks pure-noise
        // hash-embed scores that hover in [0.05, 0.40].
        let hits = vec![sem("a", 0.40), sem("b", 0.36)];
        let filtered = filter_semantic(hits, true, MIN_SEMANTIC_SCORE);
        assert!(filtered.is_empty());
    }

    #[test]
    fn filter_semantic_keeps_high_top_when_lexical_empty() {
        let hits = vec![sem("a", 0.85), sem("b", 0.60), sem("c", 0.10)];
        let filtered = filter_semantic(hits, true, MIN_SEMANTIC_SCORE);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn fused_reasons_include_both_tiers() {
        let mut a = lex("a", 9);
        a.reasons.push("title contains alpha".to_string());
        let lexical = vec![a];
        let semantic = vec![sem("a", 0.62)];
        let fused = fuse(&lexical, &semantic, 5, 60);
        let reasons = &fused[0].reasons;
        assert!(reasons.iter().any(|r| r.contains("lexical match")));
        assert!(reasons.iter().any(|r| r.contains("title contains alpha")));
        assert!(reasons.iter().any(|r| r.contains("semantic match")));
    }
}
