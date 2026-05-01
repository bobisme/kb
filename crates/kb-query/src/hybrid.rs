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
use crate::semantic::rerank::{Reranker, RerankSettings};
use crate::semantic::{
    EmbeddingBackend, HashEmbedBackend, embed::embedding_db_path,
    search::{aggregate_chunks_to_items, knn_search},
};
use crate::structural::{StructuralOptions, StructuralScorer};

/// RRF constant `k`. Pulled directly from the standard literature and
/// matched to bones-search.
pub const RRF_K: usize = 60;

/// Hard floor on a semantic hit's score before it is included in fusion.
///
/// Hash-embed-tuned default. Unrelated text shares enough character-n-gram
/// surface for cosine to land in `[0.05, 0.35]`, so a lower floor would
/// let pure noise fuse with lexical hits.
///
/// `MiniLM` unrelated text typically lands below `0.15`; for that backend,
/// see [`MINILM_MIN_SEMANTIC_SCORE`]. [`SemanticBackendKind::default_min_semantic_score`]
/// is the canonical lookup — read it instead of hardcoding the constant.
pub const MIN_SEMANTIC_SCORE: f32 = HASH_MIN_SEMANTIC_SCORE;

/// Hash-embed floor for fusion (bn-3qsj).
pub const HASH_MIN_SEMANTIC_SCORE: f32 = 0.35;

/// `MiniLM`-tuned floor for fusion. Matches `bones-search` (bn-2xbd).
pub const MINILM_MIN_SEMANTIC_SCORE: f32 = 0.15;

/// Stricter floor when lexical returned zero hits.
///
/// The semantic tier is the only signal in that case, so the top hit
/// needs to be visibly higher than the noise floor before we present it
/// as a real match. Hash-embed's morphological hits land in `[0.45,
/// 0.55]`; pure noise against the same corpus hovers in `[0.20, 0.40]`.
/// A floor of `0.45` catches the morphological match while still
/// rejecting most of the noise.
///
/// `MiniLM` has cleaner separation; see [`MINILM_MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL`].
pub const MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL: f32 = HASH_MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL;

/// Hash-embed top-no-lexical floor.
pub const HASH_MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL: f32 = 0.45;

/// `MiniLM`-tuned top-no-lexical floor. Matches `bones-search` (bn-2xbd).
pub const MINILM_MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL: f32 = 0.20;

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
    /// Structural RRF contribution (0.0 when not in the structural list).
    /// bn-32od.
    #[serde(default)]
    pub structural_score: f32,
    /// 1-indexed lexical rank (or `usize::MAX` when absent).
    pub lexical_rank: usize,
    /// 1-indexed semantic rank (or `usize::MAX` when absent).
    pub semantic_rank: usize,
    /// 1-indexed structural rank (or `usize::MAX` when absent). bn-32od.
    #[serde(default)]
    pub structural_rank: usize,
    /// Underlying lexical scoring score, copied from the lexical search
    /// result for explanation. 0 when absent.
    pub lexical_raw_score: usize,
    /// Underlying semantic similarity score in `[0, 1]`. 0 when absent.
    pub semantic_raw_score: f32,
    /// Underlying personalized-PageRank score from the citation graph.
    /// 0.0 when the structural tier is disabled or the candidate didn't
    /// participate. bn-32od.
    #[serde(default)]
    pub structural_raw_score: f32,
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
    /// Minimum semantic score for a hit to enter fusion. Hash and `MiniLM`
    /// have very different score distributions; pick the right floor with
    /// [`Self::for_backend`] (or
    /// [`crate::SemanticBackendKind::default_min_semantic_score`])
    /// instead of hardcoding [`MIN_SEMANTIC_SCORE`].
    pub min_semantic_score: f32,
    /// Stricter floor on the *top* semantic hit when lexical returned
    /// nothing. Same backend-relative tuning as [`Self::min_semantic_score`].
    pub min_semantic_top_score_no_lexical: f32,
    /// Cross-encoder rerank knobs (bn-1cp2). The `RerankSettings` is the
    /// `Copy` subset of the rerank config — the actual model lives on the
    /// caller-side and is plumbed through the `_with_reranker` function
    /// variants. When `enabled = false` (the default), the hybrid pipeline
    /// behaves exactly like before bn-1cp2.
    pub rerank: RerankSettings,
    /// Citation-graph + personalized-PageRank tier (bn-32od). When
    /// `enabled = true` (the default), the third RRF tier reads the graph
    /// at `<root>/.kb/state/graph.db` and biases ranks toward sources
    /// connected to the lexical/semantic seeds. Setting `enabled = false`
    /// makes hybrid retrieval return identical results to the pre-bone
    /// behavior.
    pub structural: StructuralOptions,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            semantic_enabled: true,
            rrf_k: RRF_K,
            min_semantic_score: MIN_SEMANTIC_SCORE,
            min_semantic_top_score_no_lexical: MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL,
            rerank: RerankSettings {
                enabled: false,
                top_k: crate::semantic::rerank::DEFAULT_TOP_K,
                keep: crate::semantic::rerank::DEFAULT_KEEP,
            },
            structural: StructuralOptions {
                enabled: true,
                damping: crate::structural::DEFAULT_DAMPING,
                max_iterations: crate::structural::DEFAULT_MAX_ITERATIONS,
                epsilon: crate::structural::DEFAULT_EPSILON,
            },
        }
    }
}

impl HybridOptions {
    /// Construct options with thresholds calibrated for the given backend
    /// (bn-2xbd). `MiniLM` lands in a much tighter cosine band than hash, so
    /// a single global floor either lets junk through or filters real
    /// matches — picking per-backend defaults avoids both failure modes.
    #[must_use]
    pub const fn for_backend(kind: crate::SemanticBackendKind) -> Self {
        Self {
            semantic_enabled: true,
            rrf_k: RRF_K,
            min_semantic_score: kind.default_min_semantic_score(),
            min_semantic_top_score_no_lexical: kind.default_min_semantic_top_score_no_lexical(),
            rerank: RerankSettings {
                enabled: false,
                top_k: crate::semantic::rerank::DEFAULT_TOP_K,
                keep: crate::semantic::rerank::DEFAULT_KEEP,
            },
            structural: StructuralOptions {
                enabled: true,
                damping: crate::structural::DEFAULT_DAMPING,
                max_iterations: crate::structural::DEFAULT_MAX_ITERATIONS,
                epsilon: crate::structural::DEFAULT_EPSILON,
            },
        }
    }

    /// Return a copy with rerank settings overridden. Convenience for
    /// callers that build options first and then layer in rerank from a
    /// separate config section.
    #[must_use]
    pub const fn with_rerank(mut self, rerank: RerankSettings) -> Self {
        self.rerank = rerank;
        self
    }

    /// Return a copy with structural settings overridden. Mirrors
    /// [`Self::with_rerank`] for the bn-32od structural tier.
    #[must_use]
    pub const fn with_structural(mut self, structural: StructuralOptions) -> Self {
        self.structural = structural;
        self
    }
}

/// Run hybrid retrieval with default options.
///
/// Reads the lexical index from disk, embeds the query with the
/// always-available hash backend, and KNN-searches the embedding store at
/// `.kb/state/embeddings.db` (when present). When `KB_SEMANTIC=0` is set,
/// behaves identically to a pure-lexical search.
///
/// Callers wired to `kb.toml [semantic]` (the kb-cli binary, kb-compile)
/// should use [`hybrid_search_with_backend`] instead so the configured
/// backend is used to embed the query — otherwise stored `MiniLM` vectors
/// would be queried with hash-embed coordinates and similarity collapses.
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
/// Uses the always-available hash backend to embed the query. Callers
/// wired to `kb.toml [semantic]` should prefer [`hybrid_search_with_backend`]
/// so the query embedding matches the stored corpus vectors.
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

/// Hybrid retrieval that reuses a caller-owned lexical index.
///
/// Used by long-lived processes (`kb serve`) that load the lexical index
/// once at startup. Embeds the query with [`HashEmbedBackend`] — see
/// [`hybrid_search_with_index_and_backend`] for the config-aware variant.
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
    let backend = HashEmbedBackend::new();
    hybrid_search_with_index_and_backend(root, lexical_index, query, limit, options, &backend)
}

/// Hybrid retrieval honoring a caller-provided embedding backend.
///
/// `kb-cli` and `kb-compile` build a [`super::semantic::SemanticBackend`]
/// from `kb.toml [semantic]` and pass it through here; the backend used to
/// embed the corpus at compile time must match the backend used to embed
/// the query, otherwise the cosine space is incoherent.
///
/// Equivalent to [`hybrid_search_with_backend_and_reranker`] with
/// `reranker = None`. Callers that want cross-encoder rerank (bn-1cp2)
/// should use the `_and_reranker` variant.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded.
pub fn hybrid_search_with_backend(
    root: &Path,
    query: &str,
    limit: usize,
    options: HybridOptions,
    backend: &dyn EmbeddingBackend,
) -> Result<Vec<HybridResult>> {
    hybrid_search_with_backend_and_reranker(root, query, limit, options, backend, None)
}

/// `hybrid_search_with_backend` plus optional cross-encoder rerank
/// (bn-1cp2). See [`hybrid_search_with_index_and_backend_and_reranker`]
/// for the semantics.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded.
pub fn hybrid_search_with_backend_and_reranker(
    root: &Path,
    query: &str,
    limit: usize,
    options: HybridOptions,
    backend: &dyn EmbeddingBackend,
    reranker: Option<&dyn Reranker>,
) -> Result<Vec<HybridResult>> {
    let lexical_index = LexicalIndex::load(root).context("load lexical index")?;
    hybrid_search_with_index_and_backend_and_reranker(
        root,
        &lexical_index,
        query,
        limit,
        options,
        backend,
        reranker,
    )
}

/// Caller-owned-index variant of [`hybrid_search_with_backend`].
///
/// Equivalent to [`hybrid_search_with_index_and_backend_and_reranker`]
/// with `reranker = None`. Callers wanting cross-encoder rerank
/// (bn-1cp2) should use the `_and_reranker` variant.
///
/// # Errors
///
/// Returns an error when the semantic store cannot be opened.
pub fn hybrid_search_with_index_and_backend(
    root: &Path,
    lexical_index: &LexicalIndex,
    query: &str,
    limit: usize,
    options: HybridOptions,
    backend: &dyn EmbeddingBackend,
) -> Result<Vec<HybridResult>> {
    hybrid_search_with_index_and_backend_and_reranker(
        root,
        lexical_index,
        query,
        limit,
        options,
        backend,
        None,
    )
}

/// `hybrid_search_with_index_and_backend` plus optional cross-encoder rerank.
///
/// (bn-1cp2.) When `reranker` is `Some` and `options.rerank.enabled` is
/// true, the fused top-`limit` list is fed to the reranker and
/// re-ordered. The result is then truncated to
/// `min(limit, options.rerank.keep)`.
///
/// We score the fused list (not the unbounded top-K) because callers of
/// this function (the `kb search` CLI, the `/search` web endpoint) ask
/// for a specific `limit` of results — they want to display N hits,
/// not feed K candidates to a downstream LLM. The plan-based ask path
/// uses [`plan_retrieval_hybrid_with_backend_and_reranker`] which honors
/// `rerank.top_k` independently.
///
/// # Errors
///
/// Returns an error when the semantic store cannot be opened.
pub fn hybrid_search_with_index_and_backend_and_reranker(
    root: &Path,
    lexical_index: &LexicalIndex,
    query: &str,
    limit: usize,
    options: HybridOptions,
    backend: &dyn EmbeddingBackend,
    reranker: Option<&dyn Reranker>,
) -> Result<Vec<HybridResult>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    // When rerank is on, pull a wider candidate pool from each tier so
    // the cross-encoder has room to surface hits that would otherwise
    // fall outside the user-requested `limit`. The fused list is
    // truncated back down by `fuse(...)` to `pool_limit`, then by the
    // rerank pass to the final `keep`. With rerank off this is identical
    // to the pre-bn-1cp2 behavior.
    let want_rerank = options.rerank.enabled && reranker.is_some();
    let pool_limit = if want_rerank {
        options.rerank.effective_top_k().max(limit)
    } else {
        limit
    };

    let lexical_hits = lexical_index.search(query, pool_limit);

    let semantic_enabled = options.semantic_enabled && semantic_env_enabled();
    let semantic_hits = if semantic_enabled {
        match run_semantic(root, query, pool_limit, backend) {
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
        options.min_semantic_top_score_no_lexical,
    );

    // bn-32od: structural tier. Seed PPR on the lexical+semantic seed
    // union, run on the in-memory citation graph, and pass through to
    // the fuse loop. When the tier is disabled or the graph is empty,
    // `structural_hits` is empty and the fuse degenerates back to the
    // pre-bone two-tier behavior.
    let structural_seeds = collect_seed_ids(&lexical_hits, &semantic_filtered);
    let structural_hits = run_structural(root, &structural_seeds, options.structural, pool_limit);

    let mut fused = fuse(
        &lexical_hits,
        &semantic_filtered,
        &structural_hits,
        pool_limit,
        options.rrf_k,
    );

    if want_rerank
        && let Some(reranker) = reranker
        && !fused.is_empty()
    {
        apply_rerank_to_search(
            &mut fused,
            query,
            reranker,
            lexical_index,
            options.rerank,
        );
    }

    if fused.len() > limit {
        fused.truncate(limit);
    }
    Ok(fused)
}

/// `apply_rerank` for the [`HybridResult`]-shaped search path.
///
/// Mirrors [`apply_rerank`] but operates on `Vec<HybridResult>` because
/// `kb search` and the web `/search` endpoint don't go through
/// `RetrievalCandidate`. Score is written into the `score` field
/// (overwriting the fused RRF score) so downstream sorts honor the
/// post-rerank order.
fn apply_rerank_to_search(
    fused: &mut Vec<HybridResult>,
    query: &str,
    reranker: &dyn Reranker,
    lexical_index: &LexicalIndex,
    settings: RerankSettings,
) {
    let top_k = settings.effective_top_k();
    let keep = settings.effective_keep();
    if fused.len() > top_k {
        fused.truncate(top_k);
    }

    let summaries: HashMap<&str, &str> = lexical_index
        .entries
        .iter()
        .map(|e| (e.id.as_str(), e.summary.as_str()))
        .collect();

    let texts: Vec<String> = fused
        .iter()
        .map(|c| candidate_text_for_rerank_hybrid(c, &summaries))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let scores = match reranker.score_batch(query, &text_refs) {
        Ok(scores) => scores,
        Err(err) => {
            if !RERANK_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
                warn!("cross-encoder rerank failed; using fused order: {err:#}");
            }
            return;
        }
    };
    if scores.len() != fused.len() {
        if !RERANK_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
            warn!(
                "cross-encoder rerank returned {} scores for {} candidates; using fused order",
                scores.len(),
                fused.len()
            );
        }
        return;
    }

    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| fused[a.0].item_id.cmp(&fused[b.0].item_id))
    });

    let original = std::mem::take(fused);
    *fused = indexed
        .into_iter()
        .take(keep)
        .map(|(idx, score)| {
            let mut hit = original[idx].clone();
            hit.score = score;
            hit.reasons
                .push(format!("cross-encoder rerank (score {score:.4})"));
            hit
        })
        .collect();
}

fn candidate_text_for_rerank_hybrid(
    hit: &HybridResult,
    summaries: &HashMap<&str, &str>,
) -> String {
    let summary = summaries
        .get(hit.item_id.as_str())
        .copied()
        .unwrap_or("")
        .trim();
    if summary.is_empty() {
        hit.title.clone()
    } else if hit.title.is_empty() {
        summary.to_string()
    } else {
        format!("{}: {}", hit.title, summary)
    }
}

fn run_semantic(
    root: &Path,
    query: &str,
    limit: usize,
    backend: &dyn EmbeddingBackend,
) -> Result<Vec<SemanticHit>> {
    let db_path = embedding_db_path(root);
    if !db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open(&db_path)
        .with_context(|| format!("open embedding db {}", db_path.display()))?;
    let qvec = backend
        .embed(query)
        .context("embed query for semantic search")?;
    // bn-3rzz: pull more chunks than `limit` so the aggregator has room to
    // collapse them down to `limit` distinct items. Many sources have
    // multiple high-scoring chunks; without overscan we'd surface fewer
    // items than requested.
    let chunk_limit = limit.saturating_mul(4).max(limit);
    let raw_chunks = knn_search(&conn, &qvec, chunk_limit).context("knn search")?;
    let items = aggregate_chunks_to_items(raw_chunks, limit);
    Ok(items
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

/// Structural-tier hit produced by personalized `PageRank` over the
/// citation graph (bn-32od). The shape mirrors [`SemanticHit`] so the
/// fuse loop can treat all three tiers uniformly.
#[derive(Debug, Clone)]
struct StructuralHit {
    item_id: String,
    score: f32,
}

/// Build the seed set for personalized `PageRank`: the union of every
/// lexical hit and every semantic hit, deduplicated and order-preserved
/// (lexical first, then semantic-only). Lexical winners contribute the
/// strongest evidence, so they go first; semantic-only ids extend the
/// seed set with morphologically similar pages the lexical tier missed.
fn collect_seed_ids(lexical: &[SearchResult], semantic: &[SemanticHit]) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<String> = Vec::new();
    for hit in lexical {
        if seen.insert(hit.id.clone()) {
            out.push(hit.id.clone());
        }
    }
    for hit in semantic {
        if seen.insert(hit.item_id.clone()) {
            out.push(hit.item_id.clone());
        }
    }
    out
}

/// Run the structural tier and convert the result into [`StructuralHit`]s.
///
/// Returns an empty vector when the tier is disabled, the graph file
/// doesn't exist, or PPR returns no nodes. Any error opening the graph
/// database is logged once and degrades gracefully — the lexical and
/// semantic tiers are still useful on their own. `pool_limit` caps the
/// number of structural hits we feed into the fuse so PPR's long tail
/// of microscopic-rank nodes doesn't bloat the candidate set.
fn run_structural(
    root: &Path,
    seeds: &[String],
    options: StructuralOptions,
    pool_limit: usize,
) -> Vec<StructuralHit> {
    if !options.enabled || pool_limit == 0 || !structural_env_enabled() {
        return Vec::new();
    }
    let scorer = match StructuralScorer::load(root, options) {
        Ok(scorer) => scorer,
        Err(err) => {
            if !STRUCTURAL_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
                warn!("structural tier unavailable; using lex+sem only: {err:#}");
            }
            return Vec::new();
        }
    };
    if scorer.edge_count() == 0 {
        return Vec::new();
    }
    let result = scorer.score(seeds);
    result
        .ranked
        .into_iter()
        .take(pool_limit)
        .map(|(item_id, score)| StructuralHit { item_id, score })
        .collect()
}

static STRUCTURAL_DEGRADED_WARNED: AtomicBool = AtomicBool::new(false);

fn filter_semantic(
    hits: Vec<SemanticHit>,
    lexical_empty: bool,
    min_semantic_score: f32,
    min_semantic_top_score_no_lexical: f32,
) -> Vec<SemanticHit> {
    if hits.is_empty() {
        return hits;
    }
    if lexical_empty && hits[0].score < min_semantic_top_score_no_lexical {
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
    structural: &[StructuralHit],
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

    let mut structural_meta: HashMap<&str, (usize, f32)> = HashMap::new();
    for (i, hit) in structural.iter().enumerate() {
        structural_meta.insert(hit.item_id.as_str(), (i + 1, hit.score));
    }

    // Deterministic union of candidate ids: lexical order first, then
    // semantic-only ids in their semantic order, then structural-only ids
    // in their PPR order, then a sorted fallback as tie-breaker so
    // identical inputs always produce identical outputs.
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
    for hit in structural {
        if seen.insert(hit.item_id.clone()) {
            order.push(hit.item_id.clone());
        }
    }

    let mut results: Vec<HybridResult> = order
        .into_iter()
        .filter_map(|id| {
            let lex = lexical_meta.get(id.as_str()).copied();
            let sem = semantic_meta.get(id.as_str()).copied();
            let structural = structural_meta.get(id.as_str()).copied();
            if lex.is_none() && sem.is_none() && structural.is_none() {
                return None;
            }
            let lexical_rank = lex.map_or(usize::MAX, |(rank, _)| rank);
            let semantic_rank = sem.map_or(usize::MAX, |(rank, _)| rank);
            let structural_rank = structural.map_or(usize::MAX, |(rank, _)| rank);

            let lexical_score = rank_to_score(lexical_rank, k);
            let semantic_score = rank_to_score(semantic_rank, k);
            let structural_score = rank_to_score(structural_rank, k);
            let score = lexical_score + semantic_score + structural_score;

            let title = lex.map_or_else(|| id.clone(), |(_, hit)| hit.title.clone());
            let lexical_raw_score = lex.map_or(0, |(_, hit)| hit.score);
            let semantic_raw_score = sem.map_or(0.0, |(_, score)| score);
            let structural_raw_score = structural.map_or(0.0, |(_, score)| score);

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
            if let Some((rank, ppr)) = structural {
                reasons.push(format!(
                    "structural match (rank {rank}, score {ppr:.2})"
                ));
            }

            Some(HybridResult {
                item_id: id,
                title,
                score,
                lexical_score,
                semantic_score,
                structural_score,
                lexical_rank,
                semantic_rank,
                structural_rank,
                lexical_raw_score,
                semantic_raw_score,
                structural_raw_score,
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
    let backend = HashEmbedBackend::new();
    plan_retrieval_hybrid_with_backend(root, query, token_budget, options, &backend)
}

/// Same as [`plan_retrieval_hybrid`] but with a caller-provided backend.
///
/// Use this when you've built a [`super::semantic::SemanticBackend`] from
/// `kb.toml [semantic]` so the query embedding matches the corpus.
///
/// Equivalent to [`plan_retrieval_hybrid_with_backend_and_reranker`] with
/// `reranker = None`. Callers that want the cross-encoder rerank pass
/// (bn-1cp2) should use the `_and_reranker` variant.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded.
pub fn plan_retrieval_hybrid_with_backend(
    root: &Path,
    query: &str,
    token_budget: u32,
    options: HybridOptions,
    backend: &dyn EmbeddingBackend,
) -> Result<RetrievalPlan> {
    plan_retrieval_hybrid_with_backend_and_reranker(
        root,
        query,
        token_budget,
        options,
        backend,
        None,
    )
}

/// Full hybrid plan with optional cross-encoder rerank (bn-1cp2).
///
/// When `reranker` is `Some` and `options.rerank.enabled` is true, the
/// fused candidate list is passed to the reranker after step 4. The
/// reranker scores each `(query, candidate_text)` pair, candidates are
/// re-sorted by descending score, then truncated to `options.rerank.keep`.
/// Each reranked candidate gets a "cross-encoder rerank (score 0.42)"
/// reason appended for explainability, and `fused_score` is overwritten
/// with the cross-encoder logit so downstream consumers (eval harness,
/// CLI dry-run output) report the post-rerank order rather than the
/// fused-only order.
///
/// When either flag is missing, the function behaves identically to
/// pre-rerank: the cross-encoder pass is skipped end-to-end.
///
/// # Errors
///
/// Returns an error when the lexical index cannot be loaded. Reranker
/// failures degrade gracefully — the un-reranked fused order is returned
/// with a one-shot warning.
#[allow(clippy::too_many_lines)]
pub fn plan_retrieval_hybrid_with_backend_and_reranker(
    root: &Path,
    query: &str,
    token_budget: u32,
    options: HybridOptions,
    backend: &dyn EmbeddingBackend,
    reranker: Option<&dyn Reranker>,
) -> Result<RetrievalPlan> {
    let lexical_index = LexicalIndex::load(root).context("load lexical index")?;
    let mut plan = lexical_index.plan_retrieval(query, token_budget, root);

    let semantic_enabled = options.semantic_enabled && semantic_env_enabled();
    if semantic_enabled {
        // Treat the already-budgeted lexical candidate count as the soft
        // cap for the semantic-only tail. We never want to silently bloat
        // the candidate set far past the lexical's natural top-K.
        let semantic_limit = plan
            .candidates
            .len()
            .max(crate::lexical::DEFAULT_RETRIEVAL_TOP_K_FOR_HYBRID);

        let semantic_hits = match run_semantic(root, query, semantic_limit, backend) {
            Ok(hits) => filter_semantic(
                hits,
                plan.candidates.is_empty(),
                options.min_semantic_score,
                options.min_semantic_top_score_no_lexical,
            ),
            Err(err) => {
                if !SEMANTIC_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
                    warn!("semantic retrieval unavailable; using lexical only: {err:#}");
                }
                Vec::new()
            }
        };

        if !semantic_hits.is_empty() {
            let semantic_ranks: std::collections::HashMap<&str, (usize, f32)> = semantic_hits
                .iter()
                .enumerate()
                .map(|(i, hit)| (hit.item_id.as_str(), (i + 1, hit.score)))
                .collect();

            // Step 2: enrich existing candidates with semantic + lexical ranks.
            for (lex_rank, candidate) in plan.candidates.iter_mut().enumerate() {
                candidate.lexical_rank = Some(lex_rank + 1);
                if let Some((sem_rank, sem_score)) =
                    semantic_ranks.get(candidate.id.as_str()).copied()
                {
                    candidate.semantic_rank = Some(sem_rank);
                    candidate.semantic_score = Some(sem_score);
                    candidate.reasons.push(format!(
                        "semantic match (rank {sem_rank}, score {sem_score:.2})"
                    ));
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
                    structural_score: None,
                    structural_rank: None,
                });
            }

            // Step 4: structural tier (bn-32od). Run personalized
            // PageRank over the citation graph using the existing
            // candidate set as seeds, then enrich each candidate with
            // its structural rank/score. Folded into the same RRF
            // formula as lexical and semantic so the three signals
            // compose cleanly.
            let structural_seeds: Vec<String> =
                plan.candidates.iter().map(|c| c.id.clone()).collect();
            let structural_hits = run_structural(
                root,
                &structural_seeds,
                options.structural,
                structural_seeds.len().max(crate::lexical::DEFAULT_RETRIEVAL_TOP_K_FOR_HYBRID),
            );
            let structural_ranks: std::collections::HashMap<&str, (usize, f32)> = structural_hits
                .iter()
                .enumerate()
                .map(|(i, hit)| (hit.item_id.as_str(), (i + 1, hit.score)))
                .collect();
            for candidate in &mut plan.candidates {
                if let Some((rank, score)) =
                    structural_ranks.get(candidate.id.as_str()).copied()
                {
                    candidate.structural_rank = Some(rank);
                    candidate.structural_score = Some(score);
                    candidate.reasons.push(format!(
                        "structural match (rank {rank}, score {score:.2})"
                    ));
                }
            }

            // Step 5: compute fused RRF score per candidate from its
            // (lexical_rank, semantic_rank, structural_rank) triple,
            // then sort by it. Without this the plan would keep the
            // lexical-only order even though the candidate set has
            // been augmented with semantic-only tail hits — confusing
            // for users who see `semantic match (rank 1, score 0.59)`
            // on a candidate sitting below several lower-ranked
            // lexical-only entries.
            for candidate in &mut plan.candidates {
                let lex = candidate
                    .lexical_rank
                    .map_or(0.0, |r| rank_to_score(r, RRF_K));
                let sem = candidate
                    .semantic_rank
                    .map_or(0.0, |r| rank_to_score(r, RRF_K));
                let structural = candidate
                    .structural_rank
                    .map_or(0.0, |r| rank_to_score(r, RRF_K));
                candidate.fused_score = Some(lex + sem + structural);
            }
            plan.candidates.sort_by(|a, b| {
                b.fused_score
                    .unwrap_or(0.0)
                    .partial_cmp(&a.fused_score.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
        }
    }

    // Step 5 (bn-1cp2): cross-encoder rerank pass.
    //
    // Re-orders the top-K candidates by reading (query, candidate) jointly
    // — dramatically more precise than the bi-encoder cosine that produced
    // the fused order, but slow, so we cap at K (default 30) and keep
    // (default 8). Skipped when:
    //  - the caller didn't pass a `reranker` (no model loaded),
    //  - `options.rerank.enabled` is false (kb.toml opt-in),
    //  - the candidate list is empty,
    //  - or the rerank scoring call returns Err (degrade to fused order).
    if options.rerank.enabled
        && let Some(reranker) = reranker
        && !plan.candidates.is_empty()
    {
        apply_rerank(&mut plan, query, reranker, &lexical_index, options.rerank);
    }

    Ok(plan)
}

/// Re-rank `plan.candidates` in place using the cross-encoder.
///
/// Truncates to `settings.effective_top_k()` before scoring (so we don't
/// pay for candidates we'd discard anyway), runs `reranker.score_batch`
/// once for the whole window, then sorts by descending score and
/// truncates to `settings.effective_keep()`. Candidate text is the
/// lexical entry's summary when available, with title fallbacks.
///
/// Failures degrade silently to the un-reranked fused order plus a
/// one-shot warning. We don't propagate the error because the rerank pass
/// is opt-in: the user already has a usable answer; killing the query
/// because the cross-encoder choked is worse than returning the fused
/// list.
fn apply_rerank(
    plan: &mut RetrievalPlan,
    query: &str,
    reranker: &dyn Reranker,
    lexical_index: &LexicalIndex,
    settings: RerankSettings,
) {
    let top_k = settings.effective_top_k();
    let keep = settings.effective_keep();
    if plan.candidates.len() > top_k {
        plan.candidates.truncate(top_k);
    }

    let summaries: std::collections::HashMap<&str, &str> = lexical_index
        .entries
        .iter()
        .map(|e| (e.id.as_str(), e.summary.as_str()))
        .collect();

    let texts: Vec<String> = plan
        .candidates
        .iter()
        .map(|c| candidate_text_for_rerank(c, &summaries))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let scores = match reranker.score_batch(query, &text_refs) {
        Ok(scores) => scores,
        Err(err) => {
            if !RERANK_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
                warn!("cross-encoder rerank failed; using fused order: {err:#}");
            }
            // Don't truncate on failure — preserve the un-reranked
            // ordering rather than silently dropping candidates the
            // caller expected to see.
            return;
        }
    };

    if scores.len() != plan.candidates.len() {
        if !RERANK_DEGRADED_WARNED.swap(true, Ordering::SeqCst) {
            warn!(
                "cross-encoder rerank returned {} scores for {} candidates; using fused order",
                scores.len(),
                plan.candidates.len()
            );
        }
        return;
    }

    // Pair candidates with scores, sort by score desc + id asc, truncate.
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                plan.candidates[a.0]
                    .id
                    .cmp(&plan.candidates[b.0].id)
            })
    });

    let original = std::mem::take(&mut plan.candidates);
    plan.candidates = indexed
        .into_iter()
        .take(keep)
        .map(|(idx, score)| {
            let mut cand = original[idx].clone();
            // Overwrite fused_score with the cross-encoder logit so the
            // dry-run output and eval harness reflect the post-rerank
            // ordering. Keep the original semantic_score / lexical_rank
            // intact for explainability.
            cand.fused_score = Some(score);
            cand.reasons
                .push(format!("cross-encoder rerank (score {score:.4})"));
            cand
        })
        .collect();
}

fn candidate_text_for_rerank(
    candidate: &RetrievalCandidate,
    summaries: &std::collections::HashMap<&str, &str>,
) -> String {
    // Prefer summary when the lexical entry has one (typical case for
    // wiki/sources/* and concept pages). Fall back to title alone for
    // semantic-only or fallback-injected candidates with no lexical
    // entry. Never empty — the cross-encoder needs *some* text to score.
    let summary = summaries
        .get(candidate.id.as_str())
        .copied()
        .unwrap_or("")
        .trim();
    if summary.is_empty() {
        candidate.title.clone()
    } else if candidate.title.is_empty() {
        summary.to_string()
    } else {
        format!("{}: {}", candidate.title, summary)
    }
}

static RERANK_DEGRADED_WARNED: AtomicBool = AtomicBool::new(false);

fn semantic_env_enabled() -> bool {
    !matches!(
        std::env::var("KB_SEMANTIC").ok().as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

/// Mirror of [`semantic_env_enabled`] for the structural tier (bn-32od).
/// `KB_STRUCTURAL=0` short-circuits the citation-graph walk so the rest
/// of hybrid retrieval behaves as it did before bn-32od. Useful in
/// integration tests that want to A/B the tier.
fn structural_env_enabled() -> bool {
    !matches!(
        std::env::var("KB_STRUCTURAL").ok().as_deref(),
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

        let fused = fuse(&lexical, &semantic, &[], 4, 60);
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
        let fused = fuse(&lexical, &semantic, &[], 5, 60);
        assert_eq!(fused.len(), 2);
        let ids: Vec<&str> = fused.iter().map(|r| r.item_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn filter_semantic_drops_below_threshold() {
        // Default min is MIN_SEMANTIC_SCORE; only hits >= that survive.
        let hits = vec![sem("a", 0.7), sem("b", 0.40), sem("c", 0.10)];
        let filtered = filter_semantic(
            hits,
            false,
            MIN_SEMANTIC_SCORE,
            MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL,
        );
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
        let filtered = filter_semantic(
            hits,
            true,
            MIN_SEMANTIC_SCORE,
            MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL,
        );
        assert!(filtered.is_empty());
    }

    #[test]
    fn filter_semantic_keeps_high_top_when_lexical_empty() {
        let hits = vec![sem("a", 0.85), sem("b", 0.60), sem("c", 0.10)];
        let filtered = filter_semantic(
            hits,
            true,
            MIN_SEMANTIC_SCORE,
            MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL,
        );
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn fused_reasons_include_both_tiers() {
        let mut a = lex("a", 9);
        a.reasons.push("title contains alpha".to_string());
        let lexical = vec![a];
        let semantic = vec![sem("a", 0.62)];
        let fused = fuse(&lexical, &semantic, &[], 5, 60);
        let reasons = &fused[0].reasons;
        assert!(reasons.iter().any(|r| r.contains("lexical match")));
        assert!(reasons.iter().any(|r| r.contains("title contains alpha")));
        assert!(reasons.iter().any(|r| r.contains("semantic match")));
    }

    /// Rerank wiring tests — bn-1cp2.
    ///
    /// We exercise [`apply_rerank`] with a stub [`Reranker`] that maps
    /// candidate id → score by lookup table. This proves:
    /// 1. The hook is wired (a stub that *reverses* the fused order
    ///    actually reverses the result).
    /// 2. With rerank disabled (or no reranker passed), the candidate
    ///    list is unchanged.
    /// 3. Truncation honors `keep`.
    /// 4. Failures degrade silently rather than panicking.
    mod rerank {
        use super::*;
        use crate::lexical::{LexicalEntry, RetrievalCandidate, RetrievalPlan};

        struct StubReranker {
            // Returns the score for candidate text; default 0.0 if missing.
            scores: std::collections::HashMap<String, f32>,
            // When set, score_batch returns Err so we can exercise the
            // graceful-degrade path.
            fail: bool,
        }

        impl StubReranker {
            fn from_pairs(pairs: &[(&str, f32)]) -> Self {
                Self {
                    scores: pairs
                        .iter()
                        .map(|(t, s)| ((*t).to_string(), *s))
                        .collect(),
                    fail: false,
                }
            }

            fn always_fails() -> Self {
                Self {
                    scores: std::collections::HashMap::new(),
                    fail: true,
                }
            }
        }

        impl Reranker for StubReranker {
            fn score_batch(&self, _query: &str, candidates: &[&str]) -> Result<Vec<f32>> {
                if self.fail {
                    anyhow::bail!("stub reranker failure");
                }
                Ok(candidates
                    .iter()
                    .map(|c| self.scores.get(*c).copied().unwrap_or(0.0))
                    .collect())
            }
        }

        fn entry(id: &str, summary: &str) -> LexicalEntry {
            LexicalEntry {
                id: id.to_string(),
                title: id.to_string(),
                aliases: Vec::new(),
                headings: Vec::new(),
                summary: summary.to_string(),
            }
        }

        fn cand(id: &str, fused: f32) -> RetrievalCandidate {
            RetrievalCandidate {
                id: id.to_string(),
                title: id.to_string(),
                score: 1,
                estimated_tokens: 10,
                reasons: vec![format!("lexical match (rank ?, score 1)")],
                semantic_score: None,
                semantic_rank: None,
                lexical_rank: Some(1),
                fused_score: Some(fused),
                structural_score: None,
                structural_rank: None,
            }
        }

        fn plan_with(candidates: Vec<RetrievalCandidate>) -> RetrievalPlan {
            RetrievalPlan {
                query: "q".to_string(),
                token_budget: 1000,
                estimated_tokens: 100,
                candidates,
                fallback_reason: None,
            }
        }

        #[test]
        fn rerank_reverses_order_when_stub_assigns_inverted_scores() {
            // Pre-rerank fused order: a (0.5), b (0.4), c (0.3).
            // Stub assigns: a=0.1, b=0.5, c=0.9 → post-rerank: c, b, a.
            let mut plan = plan_with(vec![cand("a", 0.5), cand("b", 0.4), cand("c", 0.3)]);
            let index = LexicalIndex {
                entries: vec![
                    entry("a", "alpha summary"),
                    entry("b", "beta summary"),
                    entry("c", "charlie summary"),
                ],
            };
            let stub = StubReranker::from_pairs(&[
                ("a: alpha summary", 0.1),
                ("b: beta summary", 0.5),
                ("c: charlie summary", 0.9),
            ]);
            let settings = RerankSettings {
                enabled: true,
                top_k: 30,
                keep: 8,
            };
            apply_rerank(&mut plan, "q", &stub, &index, settings);

            let order: Vec<&str> = plan.candidates.iter().map(|c| c.id.as_str()).collect();
            assert_eq!(order, vec!["c", "b", "a"]);
            // Cross-encoder reason is appended.
            assert!(
                plan.candidates[0]
                    .reasons
                    .iter()
                    .any(|r| r.contains("cross-encoder rerank"))
            );
            // fused_score is overwritten with the cross-encoder logit.
            let top_score = plan.candidates[0]
                .fused_score
                .expect("rerank pass populates fused_score with the cross-encoder logit");
            assert!((top_score - 0.9).abs() < 1e-6);
        }

        #[test]
        fn rerank_truncates_to_keep() {
            let mut plan = plan_with(vec![
                cand("a", 0.5),
                cand("b", 0.4),
                cand("c", 0.3),
                cand("d", 0.2),
            ]);
            let index = LexicalIndex {
                entries: vec![
                    entry("a", ""),
                    entry("b", ""),
                    entry("c", ""),
                    entry("d", ""),
                ],
            };
            // Empty summaries → text falls back to title.
            let stub = StubReranker::from_pairs(&[
                ("a", 1.0),
                ("b", 0.5),
                ("c", 0.7),
                ("d", 0.2),
            ]);
            let settings = RerankSettings {
                enabled: true,
                top_k: 4,
                keep: 2,
            };
            apply_rerank(&mut plan, "q", &stub, &index, settings);
            assert_eq!(plan.candidates.len(), 2);
            // Top 2 by score: a (1.0), c (0.7).
            assert_eq!(plan.candidates[0].id, "a");
            assert_eq!(plan.candidates[1].id, "c");
        }

        #[test]
        fn rerank_caps_input_at_top_k_before_scoring() {
            // 5 candidates, top_k=3. Only the first 3 should reach the
            // reranker; the rest are dropped before scoring.
            let mut plan = plan_with(vec![
                cand("a", 0.5),
                cand("b", 0.4),
                cand("c", 0.3),
                cand("d", 0.2),
                cand("e", 0.1),
            ]);
            let index = LexicalIndex {
                entries: vec![
                    entry("a", ""),
                    entry("b", ""),
                    entry("c", ""),
                    entry("d", ""),
                    entry("e", ""),
                ],
            };
            let stub = StubReranker::from_pairs(&[
                ("a", 0.1),
                ("b", 0.2),
                ("c", 0.3),
                // d/e shouldn't be queried; if they were, default 0.0
                // would still leave them after a/b/c.
            ]);
            let settings = RerankSettings {
                enabled: true,
                top_k: 3,
                keep: 3,
            };
            apply_rerank(&mut plan, "q", &stub, &index, settings);
            assert_eq!(plan.candidates.len(), 3);
            // Sorted ascending of pre-rerank index by score asc → c, b, a
            // (since stub assigned 0.3, 0.2, 0.1 to a, b, c reversed)
            assert_eq!(plan.candidates[0].id, "c");
            assert_eq!(plan.candidates[1].id, "b");
            assert_eq!(plan.candidates[2].id, "a");
        }

        #[test]
        fn rerank_failure_preserves_original_order() {
            let original = vec![cand("a", 0.5), cand("b", 0.4), cand("c", 0.3)];
            let mut plan = plan_with(original);
            let index = LexicalIndex {
                entries: vec![entry("a", ""), entry("b", ""), entry("c", "")],
            };
            let stub = StubReranker::always_fails();
            let settings = RerankSettings {
                enabled: true,
                top_k: 30,
                keep: 8,
            };
            apply_rerank(&mut plan, "q", &stub, &index, settings);

            let order: Vec<&str> = plan.candidates.iter().map(|c| c.id.as_str()).collect();
            assert_eq!(order, vec!["a", "b", "c"], "fail-on-degrade preserves order");
            // No "cross-encoder rerank" reason because we degraded.
            assert!(
                !plan.candidates[0]
                    .reasons
                    .iter()
                    .any(|r| r.contains("cross-encoder rerank"))
            );
        }

        #[test]
        fn candidate_text_uses_summary_with_title_prefix() {
            let summaries: HashMap<&str, &str> =
                std::iter::once(("a", "Alpha summary text")).collect();
            let c = cand("a", 0.5);
            let text = candidate_text_for_rerank(&c, &summaries);
            assert_eq!(text, "a: Alpha summary text");
        }

        #[test]
        fn candidate_text_falls_back_to_title_when_no_summary() {
            let summaries: HashMap<&str, &str> = HashMap::new();
            let c = cand("a", 0.5);
            let text = candidate_text_for_rerank(&c, &summaries);
            assert_eq!(text, "a");
        }
    }
}
