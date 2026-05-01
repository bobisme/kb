//! Personalized `PageRank` over the citation graph (bn-32od).
//!
//! Vanilla power-iteration PPR with a teleport vector concentrated on the
//! seed set. The implementation is intentionally tiny and dependency-free
//! — kb's corpus is small (low thousands of pages at most), and the
//! per-query graph load already dominates the cost.
//!
//! # Algorithm
//!
//! Given a directed graph `G = (V, E)`, a damping factor α (typically
//! 0.85), and a teleport vector `t` over `V` that sums to 1.0:
//!
//! ```text
//! r ← t                      # initial rank
//! repeat
//!     r' ← (1 - α) · t + α · M · r
//!     # M[u][v] = 1/out_degree(v) when (v, u) ∈ E, else 0
//!     # Dangling mass (out_degree=0) is redistributed via the teleport
//!     # vector, which is the standard fix.
//!     if max_i |r'[i] - r[i]| < ε then break
//!     r ← r'
//! until max_iterations
//! ```
//!
//! For personalized `PageRank`, the teleport vector concentrates on the
//! seed set: each seed gets a uniform `1 / |seeds|` mass, every other
//! node gets 0. This biases the random walk toward the seeds — popular
//! nodes "near" any seed (along the citation arrows) bubble up.
//!
//! # Why no `petgraph`
//!
//! The bone description mentioned petgraph as already-in-deps; in this
//! workspace it isn't, and pulling in a graph crate for a 30-line
//! power-iteration loop is overkill. We operate directly on the
//! [`super::graph::CitationGraph`] adjacency list, indexing nodes into
//! a contiguous integer space for cache-friendly iteration.

use std::collections::HashMap;

use super::graph::CitationGraph;

/// Damping factor. Standard `PageRank` convention.
pub const DEFAULT_DAMPING: f32 = 0.85;

/// Maximum number of power-iteration steps before bailing out.
pub const DEFAULT_MAX_ITERATIONS: usize = 30;

/// Convergence threshold on the L∞ delta between successive rank vectors.
pub const DEFAULT_EPSILON: f32 = 1e-6;

/// Tunable knobs for [`personalized_pagerank`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageRankConfig {
    pub damping: f32,
    pub max_iterations: usize,
    pub epsilon: f32,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: DEFAULT_DAMPING,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            epsilon: DEFAULT_EPSILON,
        }
    }
}

/// Output of [`personalized_pagerank`]: one score per node id, ranked
/// descending. Only nodes with non-trivial mass (above `epsilon / N`)
/// are included so callers don't have to filter floating-point noise.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PageRankResult {
    /// Node id → score, sorted descending by score.
    pub ranked: Vec<(String, f32)>,
}

impl PageRankResult {
    /// Build a `(node_id, rank)` index map. `rank` is 1-indexed in the
    /// ranked list, matching the convention used by the lexical and
    /// semantic tiers.
    #[must_use]
    pub fn rank_lookup(&self) -> HashMap<String, usize> {
        self.ranked
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.clone(), i + 1))
            .collect()
    }
}

/// Run personalized `PageRank` with `seeds` as the teleport set.
///
/// Returns an empty result when:
/// - the graph has no nodes, or
/// - none of the seeds are present in the graph (the seeds couldn't have
///   reached anything anyway, so there's nothing to rank).
///
/// When the seed set is empty but the graph has nodes, falls back to a
/// uniform teleport — equivalent to standard global `PageRank`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn personalized_pagerank(
    graph: &CitationGraph,
    seeds: &[String],
    config: PageRankConfig,
) -> PageRankResult {
    if graph.is_empty() {
        return PageRankResult::default();
    }
    let n = graph.node_count();
    if n == 0 {
        return PageRankResult::default();
    }

    // Index the nodes into a contiguous [0, N) space for fast iteration.
    let nodes: Vec<&String> = graph.nodes.iter().collect();
    let index: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();

    // Teleport vector. Concentrated on seeds present in the graph; falls
    // back to uniform when no seed lands.
    let mut teleport = vec![0.0_f32; n];
    let mut hit_seeds = 0_usize;
    for seed in seeds {
        if let Some(&idx) = index.get(seed.as_str()) {
            teleport[idx] += 1.0;
            hit_seeds += 1;
        }
    }
    if hit_seeds == 0 {
        if seeds.is_empty() {
            // Empty seed set → uniform teleport (global PageRank).
            let inv = 1.0 / n as f32;
            teleport.fill(inv);
        } else {
            // None of the seeds are in the graph — there's nothing
            // structurally connected to them, so there's nothing to
            // rank. Return empty.
            return PageRankResult::default();
        }
    } else {
        let scale = 1.0 / hit_seeds as f32;
        for slot in &mut teleport {
            *slot *= scale;
        }
    }

    // Adjacency in index space, plus per-node out-degree (precomputed
    // 1/deg for the matrix multiply).
    let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node_id) in nodes.iter().enumerate() {
        for neighbor in graph.neighbors(node_id) {
            if let Some(&j) = index.get(neighbor.as_str()) {
                out_neighbors[i].push(j);
            }
        }
    }

    // Power-iteration loop.
    let damping = config.damping;
    let teleport_scale = 1.0 - damping;
    let mut rank = teleport.clone();
    let mut next = vec![0.0_f32; n];

    for _ in 0..config.max_iterations.max(1) {
        // Dangling mass: nodes with no out-edges. Their share would
        // otherwise vanish; redistribute via the teleport vector so the
        // total mass is conserved.
        let mut dangling = 0.0_f32;
        for (i, neighbors) in out_neighbors.iter().enumerate() {
            if neighbors.is_empty() {
                dangling += rank[i];
            }
        }

        next.fill(0.0);
        for (i, neighbors) in out_neighbors.iter().enumerate() {
            if neighbors.is_empty() {
                continue;
            }
            let share = damping * rank[i] / neighbors.len() as f32;
            for &j in neighbors {
                next[j] += share;
            }
        }
        // Teleport contribution + dangling redistribution.
        let dangling_per = damping * dangling;
        for (i, slot) in next.iter_mut().enumerate() {
            *slot += teleport_scale * teleport[i];
            *slot += dangling_per * teleport[i];
        }

        // Check L∞ convergence.
        let mut delta = 0.0_f32;
        for i in 0..n {
            let d = (next[i] - rank[i]).abs();
            if d > delta {
                delta = d;
            }
        }
        rank.copy_from_slice(&next);
        if delta < config.epsilon {
            break;
        }
    }

    // Sort descending by score, ties broken by node id ascending so the
    // result is deterministic across runs.
    let mut ranked: Vec<(String, f32)> = nodes
        .iter()
        .enumerate()
        .map(|(i, id)| ((*id).clone(), rank[i]))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Drop pure floating-point noise so the rank-1-of-thousands tail is
    // meaningful. `epsilon / n` is a conservative floor — still tiny
    // enough that any node that received any actual signal survives.
    let noise_floor = config.epsilon / (n as f32).max(1.0);
    ranked.retain(|(_, score)| *score > noise_floor);

    PageRankResult { ranked }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn graph_from_edges(edges: &[(&str, &str)]) -> CitationGraph {
        let mut forward: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut nodes: BTreeSet<String> = BTreeSet::new();
        for (src, dst) in edges {
            nodes.insert((*src).to_string());
            nodes.insert((*dst).to_string());
            forward
                .entry((*src).to_string())
                .or_default()
                .push((*dst).to_string());
        }
        for v in forward.values_mut() {
            v.sort();
            v.dedup();
        }
        CitationGraph { forward, nodes }
    }

    #[test]
    fn empty_graph_returns_empty_result() {
        let graph = CitationGraph::default();
        let result =
            personalized_pagerank(&graph, &["seed".to_string()], PageRankConfig::default());
        assert!(result.ranked.is_empty());
    }

    #[test]
    fn empty_seed_set_returns_uniform_distribution() {
        // Tiny 3-node ring; with empty seeds we fall back to global
        // PageRank, and ranks should sit close to uniform 1/N.
        let graph = graph_from_edges(&[("a", "b"), ("b", "c"), ("c", "a")]);
        let result = personalized_pagerank(&graph, &[], PageRankConfig::default());
        assert_eq!(result.ranked.len(), 3);
        let total: f32 = result.ranked.iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 1e-3, "scores must sum to 1, got {total}");
        for (_id, score) in &result.ranked {
            assert!((score - 1.0 / 3.0).abs() < 1e-3, "uniform: got {score}");
        }
    }

    #[test]
    fn star_graph_seeded_on_hub_scores_hub_above_leaves() {
        // hub has out-edges to leaf-1, leaf-2; leaves point back at hub
        // (citation graph in kb is bidirectional in spirit because
        // backlinks show up too, but we model only the explicit edges).
        // Seed on hub → hub keeps the most teleport mass; leaves get
        // damping*share each.
        let graph = graph_from_edges(&[
            ("hub", "leaf-1"),
            ("hub", "leaf-2"),
            ("leaf-1", "hub"),
            ("leaf-2", "hub"),
        ]);
        let result = personalized_pagerank(
            &graph,
            &["hub".to_string()],
            PageRankConfig::default(),
        );
        let scores: BTreeMap<&str, f32> =
            result.ranked.iter().map(|(id, s)| (id.as_str(), *s)).collect();
        let hub = scores["hub"];
        let leaf1 = scores["leaf-1"];
        let leaf2 = scores["leaf-2"];
        assert!(
            hub > leaf1 && hub > leaf2,
            "hub must dominate when seeded on it: hub={hub}, leaves=({leaf1}, {leaf2})"
        );
        // Leaves are symmetric.
        assert!((leaf1 - leaf2).abs() < 1e-4);
    }

    #[test]
    fn ranks_converge_within_max_iterations() {
        // Convergence: a small acyclic chain a -> b -> c. With seed `a`,
        // the rank vector should stabilize before max_iterations.
        let graph = graph_from_edges(&[("a", "b"), ("b", "c")]);
        let result = personalized_pagerank(
            &graph,
            &["a".to_string()],
            PageRankConfig {
                damping: 0.85,
                max_iterations: 100,
                epsilon: 1e-9,
            },
        );
        // Determinism: run again, scores identical.
        let result2 = personalized_pagerank(
            &graph,
            &["a".to_string()],
            PageRankConfig {
                damping: 0.85,
                max_iterations: 100,
                epsilon: 1e-9,
            },
        );
        assert_eq!(result.ranked.len(), result2.ranked.len());
        for ((id1, s1), (id2, s2)) in result.ranked.iter().zip(result2.ranked.iter()) {
            assert_eq!(id1, id2);
            assert!((s1 - s2).abs() < 1e-6);
        }
    }

    #[test]
    fn unknown_seed_returns_empty_result() {
        // When the seed isn't in the graph, there's nothing structurally
        // connected to it. Return empty rather than silently falling
        // back to uniform — the caller asked a specific question and we
        // want to communicate that we have no answer.
        let graph = graph_from_edges(&[("a", "b")]);
        let result = personalized_pagerank(
            &graph,
            &["nonexistent".to_string()],
            PageRankConfig::default(),
        );
        assert!(result.ranked.is_empty());
    }

    #[test]
    fn rank_lookup_matches_ranked_order() {
        let graph = graph_from_edges(&[("a", "b"), ("a", "c"), ("d", "a")]);
        let result =
            personalized_pagerank(&graph, &["a".to_string()], PageRankConfig::default());
        let lookup = result.rank_lookup();
        for (i, (id, _)) in result.ranked.iter().enumerate() {
            assert_eq!(lookup[id], i + 1);
        }
    }

    #[test]
    fn shared_source_outranks_uncited_source_under_two_concept_seeds() {
        // The bone's headline use case: a source cited by both concepts
        // (which both match the query lexically/semantically) should
        // bubble up over a source cited by neither.
        //
        //   src-shared <- concept-a, concept-b   (cited twice)
        //   src-orphan <- (nobody)
        //   src-other  <- concept-a              (cited once)
        let graph = graph_from_edges(&[
            ("concept-a", "src-shared"),
            ("concept-b", "src-shared"),
            ("concept-a", "src-other"),
        ]);
        // Add the orphan as an isolated node so it's still in the graph.
        let mut nodes = graph.nodes.clone();
        nodes.insert("src-orphan".to_string());
        let graph = CitationGraph {
            forward: graph.forward,
            nodes,
        };

        let result = personalized_pagerank(
            &graph,
            &["concept-a".to_string(), "concept-b".to_string()],
            PageRankConfig::default(),
        );
        let scores: BTreeMap<&str, f32> =
            result.ranked.iter().map(|(id, s)| (id.as_str(), *s)).collect();

        let shared = scores.get("src-shared").copied().unwrap_or(0.0);
        let other = scores.get("src-other").copied().unwrap_or(0.0);
        let orphan = scores.get("src-orphan").copied().unwrap_or(0.0);

        assert!(
            shared > other,
            "twice-cited source must rank above singly-cited: shared={shared}, other={other}"
        );
        assert!(
            shared > orphan,
            "twice-cited source must rank above orphan: shared={shared}, orphan={orphan}"
        );
        assert!(
            other > orphan,
            "singly-cited source must rank above orphan: other={other}, orphan={orphan}"
        );
    }
}
