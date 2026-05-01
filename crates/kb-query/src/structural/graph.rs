//! Citation graph build, persistence, and load (bn-32od).
//!
//! The graph captures two relationships across `wiki/`:
//!
//! - **`cites`** — concept page (or any wiki page) → wiki source page.
//!   Built from `[src-id]` / `(src-id)` citations via
//!   [`kb_core::extract_src_id_references`]; src-ids are resolved to the
//!   underlying `wiki/sources/<slug>.md` page id.
//! - **`links`** — concept→concept and concept→source wikilinks
//!   (`[[wiki/concepts/foo]]`, `[[wiki/sources/bar]]`). The same regex
//!   `[[...]]` shape used by the backlinks pass.
//!
//! Edges are persisted in `<root>/.kb/state/graph.db`. The schema is small
//! and idempotent: an `edges` table keyed on `(src_id, dst_id, edge_type)`
//! plus a singleton `graph_meta` row that records the build version and
//! the last-compile timestamp. Each compile drops every edge whose `src_id`
//! is no longer present on disk (parallel to how the embedding store
//! handles stale rows).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kb_core::{extract_src_id_references, state_dir};
use regex::Regex;
use rusqlite::{Connection, params};

/// Relative path of the graph database under `<root>/.kb/state/`.
pub const GRAPH_DB_REL: &str = "graph.db";

/// String tag for `cites` edges.
pub const EDGE_CITES: &str = "cites";

/// String tag for `links` edges (Obsidian-style `[[...]]` wikilinks).
pub const EDGE_LINKS: &str = "links";

const WIKI_DIR: &str = "wiki";
const WIKI_SOURCES_DIR: &str = "wiki/sources";
const WIKI_CONCEPTS_DIR: &str = "wiki/concepts";

const GRAPH_META_ID: i64 = 1;
const BUILD_VERSION: &str = "bn-32od/v1";

/// `[[<path>]]` wiki-link regex matching the backlinks-pass scanner. We
/// only retain links whose normalized target lives under `wiki/concepts/`
/// or `wiki/sources/` — bare `[[]]`, anchors, and external URLs are dropped.
static WIKILINK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[\[([^\]\r\n]+)\]\]").expect("valid wikilink regex")
});

/// One directed edge in the citation graph.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct GraphEdge {
    /// Page id of the citing/linking page (e.g. `wiki/concepts/auth.md`).
    pub src_id: String,
    /// Page id of the cited/linked-to page (e.g. `wiki/sources/foo.md`).
    pub dst_id: String,
    /// Either [`EDGE_CITES`] or [`EDGE_LINKS`].
    pub edge_type: String,
}

/// In-memory snapshot of the persisted citation graph.
///
/// Hold this for the lifetime of a query. Loading is cheap — a single
/// indexed table scan plus a small in-memory adjacency list build.
#[derive(Debug, Clone, Default)]
pub struct CitationGraph {
    /// Forward adjacency: `src_id -> [dst_id, ...]`. Sorted, deduped.
    pub forward: BTreeMap<String, Vec<String>>,
    /// All distinct node ids. A node appears whenever it's the source or
    /// destination of any edge.
    pub nodes: BTreeSet<String>,
}

impl CitationGraph {
    /// Total node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total edge count (sums adjacency-list lengths). Multi-typed edges
    /// between the same pair count once because we deduplicate by `(src,
    /// dst)` when building the adjacency list — `PageRank` doesn't care
    /// whether a link was a citation or a wiki-link, only that there's a
    /// connection.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.forward.values().map(Vec::len).sum()
    }

    /// True when there are no edges. Callers can short-circuit PPR.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Outgoing neighbors of `node`. Empty when the node has no out-edges.
    #[must_use]
    pub fn neighbors(&self, node: &str) -> &[String] {
        self.forward
            .get(node)
            .map_or(&[][..], Vec::as_slice)
    }
}

/// Resolve the absolute path of the graph database under `root`.
#[must_use]
pub fn graph_db_path(root: &Path) -> PathBuf {
    state_dir(root).join(GRAPH_DB_REL)
}

/// Open the graph database, creating the parent directory and the schema
/// when needed.
///
/// # Errors
///
/// Returns an error when the parent directory cannot be created, the
/// database cannot be opened, or schema creation fails.
pub fn open_graph_db(root: &Path) -> Result<Connection> {
    let path = graph_db_path(root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create graph db parent {}", parent.display()))?;
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("open graph db {}", path.display()))?;
    ensure_graph_schema(&conn)?;
    Ok(conn)
}

/// Create the `edges` and `graph_meta` tables when they are missing.
///
/// # Errors
///
/// Returns an error when the schema DDL fails.
pub fn ensure_graph_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS edges (
            src_id TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            PRIMARY KEY (src_id, dst_id, edge_type)
        );
        CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src_id);
        CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst_id);

        CREATE TABLE IF NOT EXISTS graph_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            build_version TEXT NOT NULL DEFAULT '',
            last_compile_at_millis INTEGER NOT NULL DEFAULT 0
        );

        INSERT OR IGNORE INTO graph_meta (id, build_version, last_compile_at_millis)
        VALUES (1, '', 0);",
    )
    .context("create graph schema")?;
    Ok(())
}

/// Per-pass build statistics for the structural compile pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphBuildStats {
    /// Total edges that survived in the table after the pass.
    pub edges: usize,
    /// Edges newly inserted during this pass.
    pub inserted: usize,
    /// Edges deleted because their `src_id` no longer exists on disk or
    /// they were no longer produced by the current scan.
    pub removed: usize,
}

/// Walk every wiki page under `root` and rebuild the citation graph.
///
/// The walk is deterministic and idempotent: edges that survive in the
/// fresh scan are upserted; pre-existing rows that the scan didn't
/// re-emit are deleted. The build version and last-compile timestamp on
/// `graph_meta` are refreshed unconditionally.
///
/// # Errors
///
/// Returns an error when the database cannot be opened, the wiki tree
/// cannot be walked, or any DB operation fails.
pub fn build_graph(root: &Path) -> Result<GraphBuildStats> {
    let conn = open_graph_db(root)?;
    let scanned = scan_wiki_for_edges(root)?;
    let (inserted, removed) = persist_edges(&conn, &scanned)?;

    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
        .context("count edges after build")?;

    set_last_compile_at(&conn, current_millis())?;
    Ok(GraphBuildStats {
        edges: usize::try_from(total).unwrap_or(0),
        inserted,
        removed,
    })
}

/// Load the entire graph into memory.
///
/// Returns an empty graph when the database file doesn't exist yet — fresh
/// kb installs that haven't been compiled. Callers that want to short-
/// circuit PPR on empty graphs can check [`CitationGraph::is_empty`].
///
/// # Errors
///
/// Returns an error when the DB exists but cannot be read.
pub fn load_graph(root: &Path) -> Result<CitationGraph> {
    let path = graph_db_path(root);
    if !path.exists() {
        return Ok(CitationGraph::default());
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("open graph db {}", path.display()))?;
    ensure_graph_schema(&conn)?;
    let mut stmt = conn
        .prepare("SELECT src_id, dst_id FROM edges")
        .context("prepare load edges")?;
    let rows = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
        .context("query edges")?;

    // Dedupe `(src,dst)` across `cites`/`links` — for PPR purposes we
    // care that the connection exists, not which kind it was.
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    for row in rows {
        let (src, dst) = row.context("read edge row")?;
        nodes.insert(src.clone());
        nodes.insert(dst.clone());
        adjacency.entry(src).or_default().insert(dst);
    }
    let forward: BTreeMap<String, Vec<String>> = adjacency
        .into_iter()
        .map(|(src, set)| (src, set.into_iter().collect()))
        .collect();
    Ok(CitationGraph { forward, nodes })
}

fn scan_wiki_for_edges(root: &Path) -> Result<BTreeSet<GraphEdge>> {
    // First pass: enumerate wiki/source pages so we can resolve `src-id`
    // citations to the canonical `wiki/sources/<filename>.md` page id.
    let source_id_map = build_src_id_to_page_map(root)?;
    let live_pages = enumerate_wiki_page_ids(root)?;

    let mut edges: BTreeSet<GraphEdge> = BTreeSet::new();

    let wiki_dir = root.join(WIKI_DIR);
    if !wiki_dir.exists() {
        return Ok(edges);
    }

    for path in markdown_files_recursive(&wiki_dir) {
        let rel = page_id_for(&path, root);
        // Skip auto-generated index pages — the structural signal we
        // want is "concept cites source", not "index page links to
        // every page".
        if is_index_page(&rel) {
            continue;
        }
        let body = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                tracing::warn!("structural graph: skipping {}: {err}", path.display());
                continue;
            }
        };

        // `cites` edges: every [src-id] reference resolves to a
        // wiki/sources/<file>.md page when the source still exists on
        // disk. Otherwise we drop the citation — there's no node for
        // the missing source, and the stale-citations lint already
        // surfaces it as a separate problem.
        for r in extract_src_id_references(&body) {
            if let Some(target) = source_id_map.get(&r.src_id)
                && target != &rel
            {
                edges.insert(GraphEdge {
                    src_id: rel.clone(),
                    dst_id: target.clone(),
                    edge_type: EDGE_CITES.to_string(),
                });
            }
        }

        // `links` edges: [[wiki/concepts/...]] or [[wiki/sources/...]]
        // wiki-links. We accept anchors (`#section`) and aliases (`|`)
        // by reusing the same normalization the backlinks pass uses.
        for capture in WIKILINK_RE.captures_iter(&body) {
            let Some(raw_link) = capture.get(1).map(|c| c.as_str()) else {
                continue;
            };
            let Some(target) = normalize_wiki_link(raw_link) else {
                continue;
            };
            if target == rel {
                continue;
            }
            if !live_pages.contains(&target) {
                continue;
            }
            edges.insert(GraphEdge {
                src_id: rel.clone(),
                dst_id: target,
                edge_type: EDGE_LINKS.to_string(),
            });
        }
    }

    Ok(edges)
}

fn persist_edges(conn: &Connection, scanned: &BTreeSet<GraphEdge>) -> Result<(usize, usize)> {
    // Snapshot existing edges so we can compute insert/delete deltas.
    let existing: BTreeSet<GraphEdge> = {
        let mut stmt = conn
            .prepare("SELECT src_id, dst_id, edge_type FROM edges")
            .context("prepare read edges")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(GraphEdge {
                    src_id: row.get(0)?,
                    dst_id: row.get(1)?,
                    edge_type: row.get(2)?,
                })
            })
            .context("query existing edges")?;
        let mut out = BTreeSet::new();
        for row in rows {
            out.insert(row.context("read edge row")?);
        }
        out
    };

    let to_insert: Vec<&GraphEdge> = scanned.difference(&existing).collect();
    // Delete edges that aren't in the fresh scan. This catches both
    // citations removed from a page body and edges whose source page no
    // longer exists on disk (parallel to the embedding store dropping
    // rows whose item disappeared).
    let to_delete: Vec<&GraphEdge> = existing.difference(scanned).collect();

    let tx = conn
        .unchecked_transaction()
        .context("begin graph edge transaction")?;

    {
        let mut insert_stmt = tx
            .prepare(
                "INSERT INTO edges (src_id, dst_id, edge_type) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(src_id, dst_id, edge_type) DO NOTHING",
            )
            .context("prepare insert edge")?;
        for edge in &to_insert {
            insert_stmt
                .execute(params![edge.src_id, edge.dst_id, edge.edge_type])
                .with_context(|| {
                    format!(
                        "insert edge {} -> {} ({})",
                        edge.src_id, edge.dst_id, edge.edge_type
                    )
                })?;
        }

        let mut delete_stmt = tx
            .prepare("DELETE FROM edges WHERE src_id = ?1 AND dst_id = ?2 AND edge_type = ?3")
            .context("prepare delete edge")?;
        for edge in &to_delete {
            delete_stmt
                .execute(params![edge.src_id, edge.dst_id, edge.edge_type])
                .with_context(|| {
                    format!(
                        "delete edge {} -> {} ({})",
                        edge.src_id, edge.dst_id, edge.edge_type
                    )
                })?;
        }
    }

    tx.commit().context("commit graph edge transaction")?;

    Ok((to_insert.len(), to_delete.len()))
}

/// Map `src-<token>` (parsed from filename stem) to canonical
/// `wiki/sources/<file>.md` page id.
fn build_src_id_to_page_map(root: &Path) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    let dir = root.join(WIKI_SOURCES_DIR);
    if !dir.exists() {
        return Ok(out);
    }
    for path in markdown_files(&dir)? {
        let Some(stem) = path.file_stem().and_then(OsStr::to_str) else {
            continue;
        };
        let Some(src_id) = parse_src_id_from_stem(stem) else {
            continue;
        };
        // First wins on collision. Filename naming conflicts already get
        // surfaced by the orphan/stale lints elsewhere.
        out.entry(src_id)
            .or_insert_with(|| page_id_for(&path, root));
    }
    Ok(out)
}

fn enumerate_wiki_page_ids(root: &Path) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    for relative in [WIKI_SOURCES_DIR, WIKI_CONCEPTS_DIR] {
        let dir = root.join(relative);
        if !dir.exists() {
            continue;
        }
        for path in markdown_files(&dir)? {
            out.insert(page_id_for(&path, root));
        }
    }
    Ok(out)
}

fn markdown_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry.with_context(|| format!("entry in {}", dir.display()))?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Recursive walk under `dir`. Skips IO errors quietly (matches the
/// rest of the embedding/lexical pipeline which never aborts a compile
/// because of one unreadable file).
fn markdown_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "md") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Convert an absolute wiki page path under `root` into the canonical
/// `wiki/sources/<file>.md`-style page id. Forward slashes always.
fn page_id_for(path: &Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut out = String::with_capacity(rel.as_os_str().len());
    for (i, comp) in rel.components().enumerate() {
        if i > 0 {
            out.push('/');
        }
        out.push_str(&comp.as_os_str().to_string_lossy());
    }
    out
}

/// Drop the `wiki/<dir>/index.md` auto-generated pages from the scan;
/// they wikilink to every page and would dominate the rank teleport.
fn is_index_page(page_id: &str) -> bool {
    matches!(
        page_id,
        "wiki/index.md"
            | "wiki/sources/index.md"
            | "wiki/concepts/index.md"
            | "wiki/questions/index.md"
    )
}

/// Parse `src-abc-foo` → `Some("src-abc")`. Mirrors the same routine in
/// the stale-citations lint so the two stay in lock-step.
fn parse_src_id_from_stem(stem: &str) -> Option<String> {
    let rest = stem.strip_prefix("src-")?;
    let token_end = rest
        .char_indices()
        .find(|(_, ch)| !ch.is_ascii_alphanumeric())
        .map_or(rest.len(), |(idx, _)| idx);
    if token_end == 0 {
        return None;
    }
    Some(format!("src-{}", &rest[..token_end]))
}

/// Normalize a `[[...]]` wiki-link target into a `wiki/<dir>/<file>.md`
/// page id, or `None` when the link doesn't point at the wiki tree.
///
/// Mirrors [`crate::structural`]'s tests against the same edge cases the
/// backlinks pass already handles: alias suffix `|Display`, anchor
/// `#section`, leading `./` and `/`, optional `.md` extension.
fn normalize_wiki_link(raw: &str) -> Option<String> {
    let without_alias = raw.split('|').next()?.trim();
    if without_alias.is_empty() {
        return None;
    }
    let without_anchor = without_alias
        .split('#')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    if without_anchor.starts_with("http://")
        || without_anchor.starts_with("https://")
        || without_anchor.starts_with("mailto:")
        || without_anchor.starts_with('#')
    {
        return None;
    }

    let mut target = without_anchor.trim_start_matches("./");
    target = target.trim_start_matches('/');
    let trimmed = target
        .trim_end_matches('/')
        .trim_end_matches(".md");
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("wiki/concepts/") || trimmed.starts_with("wiki/sources/") {
        Some(format!("{trimmed}.md"))
    } else {
        None
    }
}

fn set_last_compile_at(conn: &Connection, millis: i64) -> Result<()> {
    conn.execute(
        "UPDATE graph_meta SET build_version = ?1, last_compile_at_millis = ?2 WHERE id = ?3",
        params![BUILD_VERSION, millis, GRAPH_META_ID],
    )
    .context("update graph_meta")?;
    Ok(())
}

fn current_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        fs::write(&p, body).expect("write");
    }

    #[test]
    fn parse_src_id_from_stem_recognizes_prefix_only() {
        assert_eq!(parse_src_id_from_stem("src-abc"), Some("src-abc".into()));
        assert_eq!(
            parse_src_id_from_stem("src-abc-extra-slug"),
            Some("src-abc".into())
        );
        assert_eq!(parse_src_id_from_stem("source-abc"), None);
        assert_eq!(parse_src_id_from_stem("src-"), None);
    }

    #[test]
    fn normalize_wiki_link_keeps_only_wiki_targets() {
        assert_eq!(
            normalize_wiki_link("wiki/sources/abc"),
            Some("wiki/sources/abc.md".to_string())
        );
        assert_eq!(
            normalize_wiki_link("wiki/concepts/auth#section|Auth"),
            Some("wiki/concepts/auth.md".to_string())
        );
        assert_eq!(normalize_wiki_link("https://example.com"), None);
        assert_eq!(normalize_wiki_link("not-wiki/foo"), None);
        assert_eq!(normalize_wiki_link(""), None);
    }

    #[test]
    fn build_graph_emits_cites_and_links() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        // Two sources: src-abc cited, src-def cited.
        write(root, "wiki/sources/src-abc.md", "# A\nbody\n");
        write(root, "wiki/sources/src-def.md", "# D\nbody\n");
        // Two concepts: auth cites both sources and links to threats;
        // threats cites only src-abc.
        write(
            root,
            "wiki/concepts/auth.md",
            "# Auth\nSee [src-abc] and [src-def].\nRelated: [[wiki/concepts/threats]].\n",
        );
        write(
            root,
            "wiki/concepts/threats.md",
            "# Threats\nThreat model uses [src-abc]. Wikilink: [[wiki/sources/src-def]].\n",
        );

        let stats = build_graph(root).expect("build graph");
        assert!(stats.edges >= 5, "expected at least 5 edges, got {stats:?}");

        let graph = load_graph(root).expect("load graph");
        // auth -> src-abc, auth -> src-def, auth -> threats
        let auth_neighbors: BTreeSet<&str> = graph
            .neighbors("wiki/concepts/auth.md")
            .iter()
            .map(String::as_str)
            .collect();
        assert!(auth_neighbors.contains("wiki/sources/src-abc.md"));
        assert!(auth_neighbors.contains("wiki/sources/src-def.md"));
        assert!(auth_neighbors.contains("wiki/concepts/threats.md"));

        // threats -> src-abc, threats -> src-def
        let threats_neighbors: BTreeSet<&str> = graph
            .neighbors("wiki/concepts/threats.md")
            .iter()
            .map(String::as_str)
            .collect();
        assert!(threats_neighbors.contains("wiki/sources/src-abc.md"));
        assert!(threats_neighbors.contains("wiki/sources/src-def.md"));
    }

    #[test]
    fn build_graph_is_idempotent_and_drops_stale_edges() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nSee [src-abc].\n",
        );
        let first = build_graph(root).expect("build 1");
        assert_eq!(first.edges, 1);

        // Second pass with no changes — still 1 edge, none inserted.
        let second = build_graph(root).expect("build 2");
        assert_eq!(second.edges, 1);
        assert_eq!(second.inserted, 0);
        assert_eq!(second.removed, 0);

        // Remove the citation. Edge should vanish on the next build.
        write(root, "wiki/concepts/foo.md", "# Foo\nNo citation.\n");
        let third = build_graph(root).expect("build 3");
        assert_eq!(third.edges, 0);
        assert!(third.removed >= 1);
    }

    #[test]
    fn build_graph_skips_index_pages() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        // An index page that wikilinks to every concept — must not pollute.
        write(
            root,
            "wiki/concepts/index.md",
            "# Index\n[[wiki/sources/src-abc]]\n",
        );
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nSee [src-abc].\n",
        );
        let stats = build_graph(root).expect("build");
        assert_eq!(
            stats.edges, 1,
            "index pages must be skipped, got {stats:?}"
        );
    }

    #[test]
    fn load_graph_returns_empty_when_db_missing() {
        let dir = tempdir().expect("tempdir");
        let graph = load_graph(dir.path()).expect("load empty");
        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn load_graph_dedups_cites_and_links_between_same_pair() {
        // When a concept page both cites a source AND wikilinks the same
        // source, PPR should see exactly one edge (not two parallel
        // copies inflating the out-degree).
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "wiki/sources/src-abc.md", "# A\n");
        write(
            root,
            "wiki/concepts/foo.md",
            "# Foo\nSee [src-abc] and [[wiki/sources/src-abc]].\n",
        );
        build_graph(root).expect("build");
        let graph = load_graph(root).expect("load");
        let neighbors = graph.neighbors("wiki/concepts/foo.md");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], "wiki/sources/src-abc.md");
    }
}
