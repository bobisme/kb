//! Structural-rank tier (bn-32od).
//!
//! Builds a citation graph at compile time and uses personalized
//! `PageRank` seeded on the lexical+semantic top-K to surface sources that
//! are *structurally relevant* — cited by many of the concepts the query
//! already matched. This is a third RRF tier after lexical and semantic.
//!
//! See [`graph`] for the build/persist/load surface and [`pagerank`] for
//! the PPR algorithm.

pub mod graph;
pub mod pagerank;

use std::path::Path;

use anyhow::Result;

pub use graph::{
    CitationGraph, GraphBuildStats, GraphEdge, GRAPH_DB_REL, build_graph, ensure_graph_schema,
    graph_db_path, load_graph, open_graph_db,
};
pub use pagerank::{
    DEFAULT_DAMPING, DEFAULT_EPSILON, DEFAULT_MAX_ITERATIONS, PageRankConfig, PageRankResult,
    personalized_pagerank,
};

/// Tunable knobs for [`StructuralScorer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralOptions {
    /// Master switch. When `false`, [`StructuralScorer::score`] returns
    /// an empty result without touching the graph database — exactly the
    /// pre-bone behavior.
    pub enabled: bool,
    /// PPR damping factor.
    pub damping: f32,
    /// PPR max power-iteration steps.
    pub max_iterations: usize,
    /// PPR convergence threshold.
    pub epsilon: f32,
}

impl Default for StructuralOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            damping: DEFAULT_DAMPING,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            epsilon: DEFAULT_EPSILON,
        }
    }
}

impl StructuralOptions {
    /// Project to the [`PageRankConfig`] subset.
    #[must_use]
    pub const fn pagerank_config(self) -> PageRankConfig {
        PageRankConfig {
            damping: self.damping,
            max_iterations: self.max_iterations,
            epsilon: self.epsilon,
        }
    }
}

/// Holds an in-memory copy of the citation graph for a single query.
///
/// Build one of these at the top of `plan_retrieval_hybrid_with_*` (or
/// the equivalent search variant) and call [`Self::score`] once with the
/// fused-tier seed set.
#[derive(Debug, Clone, Default)]
pub struct StructuralScorer {
    graph: CitationGraph,
    options: StructuralOptions,
}

impl StructuralScorer {
    /// Load the graph from `<root>/.kb/state/graph.db`. Returns an empty
    /// scorer (no nodes, no edges) when the file doesn't exist yet —
    /// callers can still call [`Self::score`], which will short-circuit
    /// to an empty result.
    ///
    /// # Errors
    ///
    /// Returns an error when the database file exists but cannot be read.
    pub fn load(root: &Path, options: StructuralOptions) -> Result<Self> {
        let graph = load_graph(root)?;
        Ok(Self { graph, options })
    }

    /// Construct from a pre-built graph (mostly for tests).
    #[must_use]
    pub const fn from_graph(graph: CitationGraph, options: StructuralOptions) -> Self {
        Self { graph, options }
    }

    /// Number of nodes in the underlying graph. Useful for the
    /// "skipped because the graph is empty" log line.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the underlying graph (deduped across edge types).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Run personalized `PageRank` with `seeds` as the teleport set, or
    /// return an empty result when:
    /// - the structural tier is disabled in config, or
    /// - the graph has no edges (fresh kb that never compiled the
    ///   structural pass), or
    /// - none of the seeds appear in the graph.
    #[must_use]
    pub fn score(&self, seeds: &[String]) -> PageRankResult {
        if !self.options.enabled || self.graph.is_empty() {
            return PageRankResult::default();
        }
        personalized_pagerank(&self.graph, seeds, self.options.pagerank_config())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn small_graph() -> CitationGraph {
        let mut forward: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut nodes: BTreeSet<String> = BTreeSet::new();
        for (s, d) in [
            ("wiki/concepts/a.md", "wiki/sources/x.md"),
            ("wiki/concepts/b.md", "wiki/sources/x.md"),
            ("wiki/concepts/a.md", "wiki/sources/y.md"),
        ] {
            nodes.insert(s.to_string());
            nodes.insert(d.to_string());
            forward
                .entry(s.to_string())
                .or_default()
                .push(d.to_string());
        }
        CitationGraph { forward, nodes }
    }

    #[test]
    fn disabled_returns_empty_regardless_of_seeds() {
        let scorer = StructuralScorer::from_graph(
            small_graph(),
            StructuralOptions {
                enabled: false,
                ..StructuralOptions::default()
            },
        );
        let seeds = vec!["wiki/concepts/a.md".to_string()];
        assert!(scorer.score(&seeds).ranked.is_empty());
    }

    #[test]
    fn scorer_ranks_shared_source_above_singly_cited_source() {
        let scorer = StructuralScorer::from_graph(small_graph(), StructuralOptions::default());
        let seeds = vec![
            "wiki/concepts/a.md".to_string(),
            "wiki/concepts/b.md".to_string(),
        ];
        let result = scorer.score(&seeds);
        let lookup = result.rank_lookup();
        let x_rank = lookup
            .get("wiki/sources/x.md")
            .copied()
            .expect("x should rank");
        let y_rank = lookup
            .get("wiki/sources/y.md")
            .copied()
            .expect("y should rank");
        assert!(x_rank < y_rank, "shared source should rank above singly-cited (x={x_rank}, y={y_rank})");
    }
}
