#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use kb_core::fs::atomic_write;
use serde::{Deserialize, Serialize};

pub type NodeId = String;

const GRAPH_PATH: [&str; 2] = ["state", "graph.json"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Graph {
    pub nodes: BTreeMap<NodeId, GraphNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GraphNode {
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInspection {
    pub id: NodeId,
    pub direct_inputs: Vec<NodeId>,
    pub direct_outputs: Vec<NodeId>,
    pub upstream: Vec<NodeId>,
    pub downstream: Vec<NodeId>,
}

impl Graph {
    #[must_use]
    pub fn graph_path(root: &Path) -> PathBuf {
        root.join(GRAPH_PATH[0]).join(GRAPH_PATH[1])
    }

    /// Load the persisted dependency graph from `state/graph.json`.
    ///
    /// Returns an empty graph when the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph file cannot be read, deserialized, or validated.
    pub fn load_from(root: &Path) -> Result<Self> {
        let path = Self::graph_path(root);
        if !path.exists() {
            return Ok(Self::default());
        }

        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read dependency graph {}", path.display()))?;
        let graph: Self =
            serde_json::from_str(&raw).context("deserialize dependency graph JSON")?;
        graph.validate()
    }

    /// Validate and persist the dependency graph to `state/graph.json`.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails, including cycle detection failures,
    /// or when the graph file cannot be written.
    pub fn persist_to(&self, root: &Path) -> Result<PathBuf> {
        self.validate()?;
        let path = Self::graph_path(root);
        let json = serde_json::to_vec_pretty(self).context("serialize dependency graph JSON")?;
        atomic_write(&path, &json)
            .with_context(|| format!("write dependency graph {}", path.display()))?;
        Ok(path)
    }

    /// Normalize, consistency-check, and cycle-check the graph.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph references missing nodes, has asymmetric edges,
    /// or contains a cycle.
    pub fn validate(&self) -> Result<Self> {
        let mut graph = self.clone();
        graph.normalize();
        graph.assert_consistent()?;
        graph.assert_acyclic()?;
        Ok(graph)
    }

    pub fn add_node(&mut self, id: impl Into<NodeId>) {
        self.nodes.entry(id.into()).or_default();
    }

    pub fn add_edge(&mut self, input: impl Into<NodeId>, output: impl Into<NodeId>) {
        let input = input.into();
        let output = output.into();
        self.add_node(input.clone());
        self.add_node(output.clone());

        if input == output {
            return;
        }

        if let Some(node) = self.nodes.get_mut(&input) {
            node.outputs.push(output.clone());
        }
        if let Some(node) = self.nodes.get_mut(&output) {
            node.inputs.push(input);
        }
    }

    pub fn record<I, O>(&mut self, inputs: I, outputs: O)
    where
        I: IntoIterator,
        I::Item: Into<NodeId>,
        O: IntoIterator,
        O::Item: Into<NodeId>,
    {
        let inputs: Vec<NodeId> = inputs.into_iter().map(Into::into).collect();
        let outputs: Vec<NodeId> = outputs.into_iter().map(Into::into).collect();

        for input in &inputs {
            self.add_node(input.clone());
        }
        for output in &outputs {
            self.add_node(output.clone());
        }

        for input in &inputs {
            for output in &outputs {
                self.add_edge(input.clone(), output.clone());
            }
        }

        self.normalize();
    }

    #[must_use]
    pub fn node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    #[must_use]
    pub fn resolve_node_id(&self, target: &str) -> Option<NodeId> {
        if self.nodes.contains_key(target) {
            return Some(target.to_string());
        }

        let matches: Vec<_> = self
            .nodes
            .keys()
            .filter(|candidate| candidate.ends_with(target))
            .cloned()
            .collect();

        match matches.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        }
    }

    /// Inspect a node in the dependency graph.
    ///
    /// Exact node IDs are preferred, but a unique suffix match is also accepted to make
    /// file-oriented inspection easier from the CLI.
    ///
    /// # Errors
    ///
    /// Returns an error when the target cannot be resolved to exactly one node.
    pub fn inspect(&self, target: &str) -> Result<GraphInspection> {
        let id = self.resolve_node_id(target).ok_or_else(|| {
            anyhow!(
                "'{target}' was not found in state/graph.json (exact match or unique suffix required)"
            )
        })?;
        let node = self
            .node(&id)
            .ok_or_else(|| anyhow!("'{id}' resolved but is missing from the graph"))?;

        Ok(GraphInspection {
            id: id.clone(),
            direct_inputs: node.inputs.clone(),
            direct_outputs: node.outputs.clone(),
            upstream: self.upstream_closure(&id),
            downstream: self.downstream_closure(&id),
        })
    }

    /// Ensure the graph has no cycles.
    ///
    /// # Errors
    ///
    /// Returns an error describing the discovered cycle when one is present.
    pub fn assert_acyclic(&self) -> Result<()> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Mark {
            Visiting,
            Visited,
        }

        fn visit(
            graph: &Graph,
            node: &str,
            marks: &mut BTreeMap<NodeId, Mark>,
            stack: &mut Vec<NodeId>,
        ) -> Result<()> {
            if let Some(mark) = marks.get(node) {
                if *mark == Mark::Visited {
                    return Ok(());
                }
                if *mark == Mark::Visiting {
                    if let Some(pos) = stack.iter().position(|entry| entry == node) {
                        let mut cycle = stack[pos..].to_vec();
                        cycle.push(node.to_string());
                        bail!("dependency graph contains a cycle: {}", cycle.join(" -> "));
                    }
                    bail!("dependency graph contains a cycle involving {node}");
                }
            }

            marks.insert(node.to_string(), Mark::Visiting);
            stack.push(node.to_string());

            let outputs = graph
                .nodes
                .get(node)
                .map_or_else(Vec::new, |entry| entry.outputs.clone());
            for next in outputs {
                visit(graph, &next, marks, stack)?;
            }

            stack.pop();
            marks.insert(node.to_string(), Mark::Visited);
            Ok(())
        }

        let mut marks = BTreeMap::new();
        let mut stack = Vec::new();
        for id in self.nodes.keys() {
            if !matches!(marks.get(id), Some(Mark::Visited)) {
                visit(self, id, &mut marks, &mut stack)?;
            }
        }
        Ok(())
    }

    fn assert_consistent(&self) -> Result<()> {
        for (id, node) in &self.nodes {
            for input in &node.inputs {
                let input_node = self.nodes.get(input).ok_or_else(|| {
                    anyhow!("node '{id}' references missing input node '{input}'")
                })?;
                if !input_node.outputs.iter().any(|output| output == id) {
                    bail!(
                        "node '{id}' lists '{input}' as an input, but the reverse output edge is missing"
                    );
                }
            }

            for output in &node.outputs {
                let output_node = self.nodes.get(output).ok_or_else(|| {
                    anyhow!("node '{id}' references missing output node '{output}'")
                })?;
                if !output_node.inputs.iter().any(|input| input == id) {
                    bail!(
                        "node '{id}' lists '{output}' as an output, but the reverse input edge is missing"
                    );
                }
            }
        }

        Ok(())
    }

    fn normalize(&mut self) {
        for node in self.nodes.values_mut() {
            node.inputs.sort_unstable();
            node.inputs.dedup();
            node.outputs.sort_unstable();
            node.outputs.dedup();
        }
    }

    fn upstream_closure(&self, start: &str) -> Vec<NodeId> {
        self.walk_closure(start, |node| &node.inputs)
    }

    fn downstream_closure(&self, start: &str) -> Vec<NodeId> {
        self.walk_closure(start, |node| &node.outputs)
    }

    fn walk_closure<F>(&self, start: &str, next: F) -> Vec<NodeId>
    where
        F: Fn(&GraphNode) -> &Vec<NodeId>,
    {
        let mut visited = BTreeSet::new();
        let mut stack = vec![start.to_string()];

        while let Some(current) = stack.pop() {
            let Some(node) = self.nodes.get(&current) else {
                continue;
            };

            for candidate in next(node) {
                if visited.insert(candidate.clone()) {
                    stack.push(candidate.clone());
                }
            }
        }

        visited.into_iter().collect()
    }
}

impl GraphInspection {
    #[must_use]
    pub fn render(&self) -> String {
        fn section(label: &str, values: &[NodeId]) -> String {
            if values.is_empty() {
                format!("{label}:\n  (none)")
            } else {
                let items = values
                    .iter()
                    .map(|value| format!("  - {value}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{label}:\n{items}")
            }
        }

        [
            format!("node: {}", self.id),
            section("direct inputs", &self.direct_inputs),
            section("direct outputs", &self.direct_outputs),
            section("all upstream dependencies", &self.upstream),
            section("all downstream dependents", &self.downstream),
        ]
        .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn graph_captures_source_to_page_to_index_chain() {
        let mut graph = Graph::default();
        graph.record(["raw/inbox/doc.md"], ["normalized/doc.json"]);
        graph.record(["normalized/doc.json"], ["wiki/sources/doc.md"]);
        graph.record(["wiki/sources/doc.md"], ["wiki/concepts/rust.md"]);
        graph.record(["wiki/concepts/rust.md"], ["wiki/index.md"]);

        let index = graph.inspect("wiki/index.md").expect("inspect index");
        assert_eq!(index.direct_inputs, vec!["wiki/concepts/rust.md"]);
        assert_eq!(
            index.upstream,
            vec![
                "normalized/doc.json",
                "raw/inbox/doc.md",
                "wiki/concepts/rust.md",
                "wiki/sources/doc.md"
            ]
        );
    }

    #[test]
    fn prompt_template_changes_propagate_to_generated_pages() {
        let mut graph = Graph::default();
        graph.record(
            ["normalized/doc.json", "prompt:compile-v1"],
            ["wiki/sources/doc.md"],
        );
        graph.record(["wiki/sources/doc.md"], ["wiki/concepts/rust.md"]);
        graph.record(["wiki/concepts/rust.md"], ["wiki/index.md"]);

        let template = graph.inspect("prompt:compile-v1").expect("inspect prompt");
        assert_eq!(template.direct_outputs, vec!["wiki/sources/doc.md"]);
        assert_eq!(
            template.downstream,
            vec![
                "wiki/concepts/rust.md",
                "wiki/index.md",
                "wiki/sources/doc.md"
            ]
        );
    }

    #[test]
    fn cycle_detection_runs_on_every_write() {
        let dir = tempdir().expect("tempdir");
        let mut graph = Graph::default();
        graph.add_edge("a", "b");
        graph.add_edge("b", "a");

        let err = graph
            .persist_to(dir.path())
            .expect_err("cycle should be rejected");
        assert!(
            err.to_string()
                .contains("dependency graph contains a cycle")
        );
        assert!(!Graph::graph_path(dir.path()).exists());
    }

    #[test]
    fn persist_and_load_round_trip() {
        let dir = tempdir().expect("tempdir");
        let mut graph = Graph::default();
        graph.record(["a", "b"], ["c"]);

        let path = graph.persist_to(dir.path()).expect("persist graph");
        assert!(path.exists());

        let loaded = Graph::load_from(dir.path()).expect("load graph");
        assert_eq!(loaded, graph.validate().expect("normalized graph"));
    }

    #[test]
    fn inspect_allows_unique_suffix_matches() {
        let mut graph = Graph::default();
        graph.record(["raw/inbox/doc.md"], ["wiki/sources/doc.md"]);

        let inspection = graph.inspect("wiki/sources/doc.md").expect("exact match");
        assert_eq!(inspection.id, "wiki/sources/doc.md");

        let suffix = graph.inspect("sources/doc.md").expect("suffix match");
        assert_eq!(suffix.id, "wiki/sources/doc.md");
    }
}
