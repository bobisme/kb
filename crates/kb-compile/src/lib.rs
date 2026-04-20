#![forbid(unsafe_code)]

pub mod backlinks;
pub mod concept_candidate;
pub mod concept_extraction;
pub mod concept_merge;
pub mod imputed_fix;
pub mod index_page;
pub mod pipeline;
pub mod progress;
pub mod promotion;
pub mod source_page;
pub mod source_summary;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use kb_core::fs::atomic_write;
use serde::{Deserialize, Serialize};

use kb_core::{BuildRecord, Hash, hash_many};

pub type NodeId = String;

const GRAPH_PATH: [&str; 2] = ["state", "graph.json"];
const STATE_HASHES_PATH: &str = "state/hashes.json";

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

    /// Remove `id` from the graph along with every edge that touches it.
    ///
    /// Returns `true` when the node was present.
    pub fn remove_node(&mut self, id: &str) -> bool {
        if self.nodes.remove(id).is_none() {
            return false;
        }
        for node in self.nodes.values_mut() {
            node.inputs.retain(|other| other != id);
            node.outputs.retain(|other| other != id);
        }
        true
    }

    /// Surgically remove every node whose id references the forgotten
    /// `src_id`, along with all edges that touch those nodes.
    ///
    /// Matches `source-document-<src>`, `wiki-page-<src>` (source pages use
    /// the raw src-id as their slug via `source_page_path_for_id`), and any
    /// node whose id contains `/<src>/` or ends with `/<src>.md` — the last
    /// two catch filesystem-path nodes that older build records may have
    /// emitted verbatim before the `graph_node_for_id` normalization existed.
    ///
    /// Returns the number of nodes removed so callers can surface the count
    /// in logs / test assertions. `bn-3f6` uses this on the `kb forget` path
    /// to avoid a full compile rebuild when only a single src went away.
    pub fn prune_for_src(&mut self, src_id: &str) -> usize {
        let src_doc_node = format!("source-document-{src_id}");
        let wiki_page_node = format!("wiki-page-{src_id}");
        let path_segment = format!("/{src_id}/");
        let md_suffix = format!("/{src_id}.md");

        let to_remove: Vec<NodeId> = self
            .nodes
            .keys()
            .filter(|id| {
                id.as_str() == src_doc_node
                    || id.as_str() == wiki_page_node
                    || id.contains(&path_segment)
                    || id.ends_with(&md_suffix)
            })
            .cloned()
            .collect();
        let removed = to_remove.len();
        for id in to_remove {
            self.remove_node(&id);
        }
        removed
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

    /// Return all nodes in topological order (dependencies before dependents).
    ///
    /// # Errors
    ///
    /// Returns an error when the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let mut in_degree: BTreeMap<&str, usize> = BTreeMap::new();
        for (id, node) in &self.nodes {
            in_degree.entry(id.as_str()).or_insert(0);
            for output in &node.outputs {
                *in_degree.entry(output.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id.to_string())
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&id) {
                for output in &node.outputs {
                    if let Some(deg) = in_degree.get_mut(output.as_str()) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(output.clone());
                        }
                    }
                }
            }
            order.push(id);
        }

        if order.len() != self.nodes.len() {
            bail!("dependency graph contains a cycle (topological sort incomplete)");
        }

        Ok(order)
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

/// Persistent snapshot of per-node fingerprints used for incremental stale detection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HashState {
    pub hashes: BTreeMap<String, String>,
}

impl HashState {
    /// Load a previously persisted hash state from disk.
    ///
    /// Missing files are treated as an empty state.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be read or parsed as JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }

        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read hash state {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse hash state {}", path.display()))
    }

    /// Load `state/hashes.json` from a KB root.
    ///
    /// # Errors
    /// Returns an error if the state file exists but cannot be read or parsed.
    pub fn load_from_root(root: impl AsRef<Path>) -> Result<Self> {
        Self::load(root.as_ref().join(STATE_HASHES_PATH))
    }

    /// Persist the hash state to disk.
    ///
    /// # Errors
    /// Returns an error if the parent directory cannot be created, the state cannot
    /// be serialized, or the file cannot be written.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create hash state directory {}", parent.display())
            })?;
        }

        let json = serde_json::to_string_pretty(self).context("failed to serialize hash state")?;
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("failed to write hash state {}", path.display()))
    }

    /// Persist to `state/hashes.json` under a KB root.
    ///
    /// # Errors
    /// Returns an error if the state cannot be serialized or written.
    pub fn save_to_root(&self, root: impl AsRef<Path>) -> Result<()> {
        self.save(root.as_ref().join(STATE_HASHES_PATH))
    }
}

/// Current fingerprint inputs for a build node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleNode {
    pub id: String,
    pub dependencies: Vec<String>,
    pub content_hash: String,
    pub prompt_template_hash: Option<String>,
    pub model_version: Option<String>,
}

impl StaleNode {
    #[must_use]
    pub fn fingerprint(&self) -> Hash {
        let prompt_template_hash = self.prompt_template_hash.as_deref().unwrap_or("");
        let model_version = self.model_version.as_deref().unwrap_or("");

        hash_many(&[
            self.id.as_bytes(),
            b"\0",
            self.content_hash.as_bytes(),
            b"\0",
            prompt_template_hash.as_bytes(),
            b"\0",
            model_version.as_bytes(),
        ])
    }
}

/// Result of stale detection for a compile graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleReport {
    pub changed_nodes: BTreeSet<String>,
    pub stale_nodes: BTreeSet<String>,
    pub current_state: HashState,
}

/// Detect changed node fingerprints by comparing the current graph against a previous state.
#[must_use]
pub fn detect_stale(previous: &HashState, nodes: &[StaleNode]) -> StaleReport {
    let current_state = HashState {
        hashes: nodes
            .iter()
            .map(|node| (node.id.clone(), node.fingerprint().to_hex()))
            .collect(),
    };

    let changed_nodes = changed_nodes(previous, &current_state);
    let stale_nodes = propagate_stale(&changed_nodes, nodes);

    StaleReport {
        changed_nodes,
        stale_nodes,
        current_state,
    }
}

/// Build a stale-detection node from a recorded build artifact.
#[must_use]
pub fn stale_node_from_build_record(record: &BuildRecord) -> StaleNode {
    let mut dependencies = record.input_ids.clone();
    dependencies.extend(record.output_ids.clone());
    dependencies.sort();
    dependencies.dedup();

    StaleNode {
        id: record.metadata.id.clone(),
        dependencies,
        content_hash: record.manifest_hash.clone(),
        prompt_template_hash: record.metadata.prompt_template_hash.clone(),
        model_version: record.metadata.model_version.clone(),
    }
}

fn changed_nodes(previous: &HashState, current: &HashState) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    ids.extend(previous.hashes.keys().cloned());
    ids.extend(current.hashes.keys().cloned());

    ids.into_iter()
        .filter(|id| previous.hashes.get(id) != current.hashes.get(id))
        .collect()
}

fn propagate_stale(changed: &BTreeSet<String>, nodes: &[StaleNode]) -> BTreeSet<String> {
    let mut reverse_edges: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in nodes {
        for dependency in &node.dependencies {
            reverse_edges
                .entry(dependency.as_str())
                .or_default()
                .push(node.id.as_str());
        }
    }

    let mut stale = changed.clone();
    let mut queue = changed.iter().cloned().collect::<VecDeque<_>>();

    while let Some(id) = queue.pop_front() {
        if let Some(dependents) = reverse_edges.get(id.as_str()) {
            for dependent in dependents {
                if stale.insert((*dependent).to_string()) {
                    queue.push_back((*dependent).to_string());
                }
            }
        }
    }

    stale
}

#[cfg(test)]
mod tests {
    use super::*;
    use kb_core::{ContentHash, EntityMetadata, Status};
    use tempfile::{TempDir, tempdir};

    // --- Graph tests ---

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

    #[test]
    fn prune_for_src_removes_prefixed_nodes_and_their_edges() {
        // bn-3f6: surgical removal of every node that references a forgotten
        // src, leaving unrelated graph topology intact.
        let mut graph = Graph::default();
        let src = "src-3f60001f";
        // Prefixed node forms emitted by `build_graph_from_state`.
        graph.record(
            [format!("source-document-{src}")],
            [format!("wiki-page-{src}")],
        );
        // Downstream concept should lose its incoming edge from the pruned
        // wiki-page node but otherwise remain.
        graph.add_edge(format!("wiki-page-{src}"), "concept-shared".to_string());
        // Unrelated src must survive.
        graph.record(["source-document-src-keepme"], ["wiki-page-src-keepme"]);

        let removed = graph.prune_for_src(src);
        assert_eq!(removed, 2, "source-document + wiki-page nodes pruned");
        assert!(!graph.nodes.contains_key(&format!("source-document-{src}")));
        assert!(!graph.nodes.contains_key(&format!("wiki-page-{src}")));
        assert!(graph.nodes.contains_key("concept-shared"));
        let concept = graph.node("concept-shared").expect("concept survives");
        assert!(
            concept.inputs.is_empty(),
            "concept must lose the dangling input edge; got {:?}",
            concept.inputs
        );
        // Unrelated src lane untouched.
        assert!(graph.nodes.contains_key("source-document-src-keepme"));
        assert!(graph.nodes.contains_key("wiki-page-src-keepme"));
        // Post-prune graph is still valid.
        graph.validate().expect("pruned graph validates");
    }

    #[test]
    fn prune_for_src_matches_filesystem_path_nodes() {
        // Older graph entries sometimes store raw filesystem paths as node
        // ids; those must also match when they carry `/<src>/` or end with
        // `/<src>.md`.
        let mut graph = Graph::default();
        let src = "src-3f600029";
        graph.add_edge(
            format!("raw/inbox/{src}/source_document.json"),
            format!("normalized/{src}/doc.json"),
        );
        graph.add_edge(
            format!("normalized/{src}/doc.json"),
            format!("wiki/sources/{src}.md"),
        );
        // Unrelated node.
        graph.add_edge("raw/inbox/other/source.json", "normalized/other/doc.json");

        let removed = graph.prune_for_src(src);
        assert_eq!(removed, 3);
        assert!(!graph.nodes.contains_key(&format!("raw/inbox/{src}/source_document.json")));
        assert!(!graph.nodes.contains_key(&format!("normalized/{src}/doc.json")));
        assert!(!graph.nodes.contains_key(&format!("wiki/sources/{src}.md")));
        assert!(graph.nodes.contains_key("raw/inbox/other/source.json"));
        assert!(graph.nodes.contains_key("normalized/other/doc.json"));
    }

    #[test]
    fn prune_for_src_is_noop_when_absent() {
        let mut graph = Graph::default();
        graph.record(["source-document-src-aaaa"], ["wiki-page-src-aaaa"]);
        let removed = graph.prune_for_src("src-notpresent");
        assert_eq!(removed, 0);
        assert!(graph.nodes.contains_key("source-document-src-aaaa"));
    }

    // --- Stale detection tests ---

    fn stale_node(
        id: &str,
        content_hash: &str,
        dependencies: &[&str],
        prompt_template_hash: Option<&str>,
        model_version: Option<&str>,
    ) -> StaleNode {
        StaleNode {
            id: id.to_string(),
            dependencies: dependencies.iter().map(|id| (*id).to_string()).collect(),
            content_hash: content_hash.to_string(),
            prompt_template_hash: prompt_template_hash.map(ToString::to_string),
            model_version: model_version.map(ToString::to_string),
        }
    }

    fn metadata(id: &str) -> EntityMetadata {
        EntityMetadata {
            id: id.to_string(),
            created_at_millis: 1,
            updated_at_millis: 1,
            source_hashes: vec![],
            model_version: None,
            tool_version: Some("kb/0.1.0".to_string()),
            prompt_template_hash: None,
            dependencies: vec![],
            output_paths: vec![],
            status: Status::Fresh,
        }
    }

    fn build_record(
        id: &str,
        input_ids: &[&str],
        output_ids: &[&str],
        manifest_hash: &str,
        prompt_template_hash: Option<ContentHash>,
        model_version: Option<&str>,
    ) -> BuildRecord {
        BuildRecord {
            metadata: EntityMetadata {
                model_version: model_version.map(ToString::to_string),
                prompt_template_hash,
                ..metadata(id)
            },
            pass_name: "test_pass".to_string(),
            input_ids: input_ids.iter().map(|id| (*id).to_string()).collect(),
            output_ids: output_ids.iter().map(|id| (*id).to_string()).collect(),
            manifest_hash: manifest_hash.to_string(),
        }
    }

    #[test]
    fn second_compile_with_no_changes_rebuilds_nothing() {
        let nodes = vec![
            stale_node("source-a", "hash-source-a", &[], None, None),
            stale_node(
                "page-a",
                "hash-page-a",
                &["source-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "concept-a",
                "hash-concept-a",
                &["page-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
        ];

        let first = detect_stale(&HashState::default(), &nodes);
        let second = detect_stale(&first.current_state, &nodes);

        assert!(second.changed_nodes.is_empty());
        assert!(second.stale_nodes.is_empty());
    }

    #[test]
    fn changing_one_source_only_rebuilds_its_downstream_dependents() {
        let previous_nodes = vec![
            stale_node("source-a", "hash-source-a-v1", &[], None, None),
            stale_node("source-b", "hash-source-b-v1", &[], None, None),
            stale_node(
                "page-a",
                "hash-page-a",
                &["source-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "page-b",
                "hash-page-b",
                &["source-b"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "concept-shared",
                "hash-concept-shared",
                &["page-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
        ];
        let previous = detect_stale(&HashState::default(), &previous_nodes).current_state;

        let current_nodes = vec![
            stale_node("source-a", "hash-source-a-v2", &[], None, None),
            stale_node("source-b", "hash-source-b-v1", &[], None, None),
            stale_node(
                "page-a",
                "hash-page-a",
                &["source-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "page-b",
                "hash-page-b",
                &["source-b"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "concept-shared",
                "hash-concept-shared",
                &["page-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
        ];

        let report = detect_stale(&previous, &current_nodes);

        assert_eq!(
            report.changed_nodes,
            BTreeSet::from(["source-a".to_string()])
        );
        assert_eq!(
            report.stale_nodes,
            BTreeSet::from([
                "source-a".to_string(),
                "page-a".to_string(),
                "concept-shared".to_string(),
            ])
        );
        assert!(!report.stale_nodes.contains("source-b"));
        assert!(!report.stale_nodes.contains("page-b"));
    }

    #[test]
    fn prompt_template_changes_only_invalidate_artifacts_that_used_it() {
        let previous_nodes = vec![
            stale_node(
                "page-a",
                "hash-page-a",
                &["source-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "page-b",
                "hash-page-b",
                &["source-b"],
                Some("template-2"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "concept-a",
                "hash-concept-a",
                &["page-a"],
                Some("template-1"),
                Some("gpt-4o-mini"),
            ),
        ];
        let previous = detect_stale(&HashState::default(), &previous_nodes).current_state;

        let current_nodes = vec![
            stale_node(
                "page-a",
                "hash-page-a",
                &["source-a"],
                Some("template-1-updated"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "page-b",
                "hash-page-b",
                &["source-b"],
                Some("template-2"),
                Some("gpt-4o-mini"),
            ),
            stale_node(
                "concept-a",
                "hash-concept-a",
                &["page-a"],
                Some("template-1-updated"),
                Some("gpt-4o-mini"),
            ),
        ];

        let report = detect_stale(&previous, &current_nodes);

        assert_eq!(
            report.stale_nodes,
            BTreeSet::from(["page-a".to_string(), "concept-a".to_string()])
        );
        assert!(!report.stale_nodes.contains("page-b"));
    }

    #[test]
    fn model_version_changes_mark_outputs_stale() {
        let previous_nodes = vec![stale_node(
            "page-a",
            "hash-page-a",
            &["source-a"],
            Some("template-1"),
            Some("gpt-4o-mini"),
        )];
        let previous = detect_stale(&HashState::default(), &previous_nodes).current_state;

        let current_nodes = vec![stale_node(
            "page-a",
            "hash-page-a",
            &["source-a"],
            Some("template-1"),
            Some("gpt-5-mini"),
        )];

        let report = detect_stale(&previous, &current_nodes);
        assert_eq!(report.stale_nodes, BTreeSet::from(["page-a".to_string()]));
    }

    #[test]
    fn hash_state_round_trips_on_disk() {
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("state/hashes.json");
        let state = HashState {
            hashes: BTreeMap::from([
                ("page-a".to_string(), "hash-a".to_string()),
                ("page-b".to_string(), "hash-b".to_string()),
            ]),
        };

        state.save(&path).expect("save hash state");
        let loaded = HashState::load(&path).expect("load hash state");

        assert_eq!(loaded, state);
    }

    #[test]
    fn build_record_conversion_uses_manifest_prompt_and_model_metadata() {
        let record = build_record(
            "build-1",
            &["source-a"],
            &["page-a"],
            "manifest-1",
            Some("template-1".to_string()),
            Some("gpt-4o-mini"),
        );

        let node = stale_node_from_build_record(&record);

        assert_eq!(node.id, "build-1");
        assert_eq!(
            node.dependencies,
            vec!["page-a".to_string(), "source-a".to_string()]
        );
        assert_eq!(node.content_hash, "manifest-1");
        assert_eq!(node.prompt_template_hash.as_deref(), Some("template-1"));
        assert_eq!(node.model_version.as_deref(), Some("gpt-4o-mini"));
    }
}
