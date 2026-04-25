//! Embedding pipeline: walks the wiki, computes content hashes, and upserts
//! vectors into `.kb/state/embeddings.db`.
//!
//! The pipeline is intentionally idempotent. Each item is keyed by a content
//! hash (blake3 of the embed input plus the backend id) — when nothing has
//! changed, the second compile is a no-op aside from the per-item hash check.
//!
//! Schema (created on first run):
//!
//! ```sql
//! CREATE TABLE item_embeddings (
//!     item_id TEXT PRIMARY KEY,
//!     content_hash TEXT NOT NULL,
//!     embedding_json TEXT NOT NULL
//! );
//! CREATE TABLE semantic_meta (
//!     id INTEGER PRIMARY KEY CHECK (id = 1),
//!     backend_id TEXT NOT NULL,
//!     embedding_dim INTEGER NOT NULL,
//!     last_compile_at_millis INTEGER NOT NULL DEFAULT 0
//! );
//! ```
//!
//! When `backend_id` or `embedding_dim` changes, the embeddings table is
//! truncated and rebuilt — old vectors from a different backend are
//! incompatible.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use kb_core::{extract_managed_regions, frontmatter::read_frontmatter, state_dir};
use rusqlite::{Connection, OptionalExtension, params};
use serde_yaml::Value;
use tracing::{debug, warn};

use super::model::EmbeddingBackend;

/// Relative path of the embedding database from the kb root.
pub const EMBEDDING_DB_REL: &[&str] = &["embeddings.db"];

/// Maximum number of characters fed into the embedder per item. Beyond this,
/// the body is truncated. The embed input is title + body (managed-regions
/// stripped) + aliases for concepts.
pub const MAX_EMBED_CHARS: usize = 4096;

const SEMANTIC_META_ID: i64 = 1;

const WIKI_SOURCES: &str = "wiki/sources";
const WIKI_CONCEPTS: &str = "wiki/concepts";

/// Per-pass synchronization summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SyncStats {
    /// Number of items embedded (new or content-changed).
    pub embedded: usize,
    /// Number of stored embeddings deleted because their item disappeared.
    pub removed: usize,
    /// Number of items that were already up to date (hash unchanged).
    pub up_to_date: usize,
}

/// Snapshot of the semantic index, used by `kb status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SemanticIndexStats {
    /// Total embeddings stored on disk.
    pub embeddings: usize,
    /// Wiki pages that exist on disk but are missing or stale in the store.
    pub stale: usize,
}

/// Resolve the absolute path of the embeddings database.
#[must_use]
pub fn embedding_db_path(root: &Path) -> PathBuf {
    state_dir(root).join(EMBEDDING_DB_REL[0])
}

/// Translate a `wiki/...md` path (under `root`) into the canonical item id
/// stored in `item_embeddings.item_id`. Always uses forward slashes so the
/// id round-trips between Linux and Windows-built kbs.
#[must_use]
pub fn item_id_for_relpath(rel: &Path) -> String {
    let mut out = String::with_capacity(rel.as_os_str().len());
    for (idx, comp) in rel.components().enumerate() {
        if idx > 0 {
            out.push('/');
        }
        out.push_str(&comp.as_os_str().to_string_lossy());
    }
    out
}

/// Open the embedding database, creating the parent directory and schema as
/// needed. Caller is responsible for closing the connection (drop).
///
/// # Errors
///
/// Returns an error when the parent directory cannot be created, the DB
/// cannot be opened, or schema creation fails.
pub fn open_embedding_db(root: &Path) -> Result<Connection> {
    let path = embedding_db_path(root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create embedding db parent {}", parent.display()))?;
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("open embedding db {}", path.display()))?;
    ensure_embedding_schema(&conn)?;
    Ok(conn)
}

/// Create `item_embeddings` and `semantic_meta` if they don't exist.
///
/// # Errors
///
/// Returns an error when the schema DDL fails.
pub fn ensure_embedding_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS item_embeddings (
            item_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS semantic_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            backend_id TEXT NOT NULL DEFAULT '',
            embedding_dim INTEGER NOT NULL DEFAULT 0,
            last_compile_at_millis INTEGER NOT NULL DEFAULT 0
        );

        INSERT OR IGNORE INTO semantic_meta (id, backend_id, embedding_dim, last_compile_at_millis)
        VALUES (1, '', 0, 0);",
    )
    .context("create semantic index tables")?;
    Ok(())
}

/// One pass of `kb compile`'s embedding sync.
///
/// Walks `wiki/sources/` + `wiki/concepts/`, recomputes each item's content
/// hash, and re-embeds anything that has changed. Stored rows whose item id
/// no longer corresponds to a wiki page on disk are deleted. The
/// `semantic_meta` row is updated with the active backend id, dim, and the
/// current timestamp.
///
/// # Errors
///
/// Returns an error when the database operations or any embedding inference
/// step fails.
pub fn sync_embeddings<B: EmbeddingBackend>(root: &Path, backend: &B) -> Result<SyncStats> {
    let conn = open_embedding_db(root)?;
    handle_backend_change(&conn, backend.backend_id(), backend.dimensions())?;

    let pipeline = EmbeddingPipeline {
        backend,
        conn: &conn,
    };

    let items = collect_wiki_items(root)?;
    let live_ids: HashSet<String> = items.iter().map(|item| item.item_id.clone()).collect();
    let existing_hashes = load_existing_hashes(&conn)?;

    let mut stats = SyncStats::default();
    for item in &items {
        let stored = existing_hashes.get(&item.item_id);
        if stored.map(String::as_str) == Some(item.content_hash.as_str()) {
            stats.up_to_date += 1;
            continue;
        }
        let embedding = backend
            .embed(&item.embed_input)
            .with_context(|| format!("embed {}", item.item_id))?;
        if embedding.len() != backend.dimensions() {
            bail!(
                "embedding dim mismatch for {}: expected {}, got {}",
                item.item_id,
                backend.dimensions(),
                embedding.len()
            );
        }
        pipeline.upsert(&item.item_id, &item.content_hash, &embedding)?;
        stats.embedded += 1;
    }

    stats.removed = remove_stale_embeddings(&conn, &live_ids)?;

    set_last_compile_at(&conn, current_millis())?;
    Ok(stats)
}

/// Gather a snapshot of the semantic index for status reporting.
///
/// Returns zeroed counts when the embedding store does not yet exist —
/// that's a fresh kb that hasn't been compiled.
///
/// # Errors
///
/// Returns an error when the DB exists but cannot be read.
pub fn semantic_index_stats(root: &Path) -> Result<SemanticIndexStats> {
    let path = embedding_db_path(root);
    if !path.exists() {
        return Ok(SemanticIndexStats::default());
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("open embedding db {}", path.display()))?;
    ensure_embedding_schema(&conn)?;

    let embeddings: i64 = conn
        .query_row("SELECT COUNT(*) FROM item_embeddings", [], |row| row.get(0))
        .context("count item_embeddings")?;

    let existing_hashes = load_existing_hashes(&conn)?;
    let items = collect_wiki_items(root).unwrap_or_default();
    let live_ids: HashSet<String> = items.iter().map(|i| i.item_id.clone()).collect();

    let mut stale = 0_usize;
    for item in &items {
        match existing_hashes.get(&item.item_id) {
            Some(hash) if hash == &item.content_hash => {}
            _ => stale += 1,
        }
    }
    for stored in existing_hashes.keys() {
        if !live_ids.contains(stored) {
            stale += 1;
        }
    }

    Ok(SemanticIndexStats {
        embeddings: usize::try_from(embeddings).unwrap_or(0),
        stale,
    })
}

/// Lower-level helper: upsert a single item without going through the
/// content-hash dedup. Useful in `kb forget` to drop a row eagerly.
pub struct EmbeddingPipeline<'a, B: EmbeddingBackend> {
    backend: &'a B,
    conn: &'a Connection,
}

impl<B: EmbeddingBackend> EmbeddingPipeline<'_, B> {
    /// Upsert the embedding for `item_id`.
    ///
    /// # Errors
    ///
    /// Returns an error when the DB operation fails.
    pub fn upsert(&self, item_id: &str, content_hash: &str, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.backend.dimensions() {
            bail!(
                "embedding dim mismatch upserting {}: expected {}, got {}",
                item_id,
                self.backend.dimensions(),
                embedding.len()
            );
        }
        let json = encode_embedding_json(embedding);
        self.conn
            .execute(
                "INSERT INTO item_embeddings (item_id, content_hash, embedding_json)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(item_id)
                 DO UPDATE SET content_hash = excluded.content_hash,
                               embedding_json = excluded.embedding_json",
                params![item_id, content_hash, json],
            )
            .with_context(|| format!("upsert embedding for {item_id}"))?;
        Ok(())
    }
}

fn handle_backend_change(conn: &Connection, backend_id: &str, dim: usize) -> Result<()> {
    let stored = conn
        .query_row(
            "SELECT backend_id, embedding_dim FROM semantic_meta WHERE id = ?1",
            params![SEMANTIC_META_ID],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .context("read semantic_meta")?;

    let stored_dim = i64::try_from(dim).unwrap_or(i64::MAX);
    match stored {
        Some((id, d)) if !id.is_empty() && (id != backend_id || d != stored_dim) => {
            // Backend changed: wipe old vectors. They were produced by a
            // different model and would corrupt cosine similarity.
            conn.execute("DELETE FROM item_embeddings", [])
                .context("truncate item_embeddings on backend change")?;
            conn.execute(
                "UPDATE semantic_meta SET backend_id = ?1, embedding_dim = ?2 WHERE id = ?3",
                params![backend_id, stored_dim, SEMANTIC_META_ID],
            )
            .context("update semantic_meta after backend change")?;
            debug!(
                "semantic backend changed ({id}/{d} -> {backend_id}/{stored_dim}); embedding store truncated"
            );
        }
        Some(_) => {}
        None => {
            // Should be impossible thanks to the INSERT OR IGNORE above,
            // but cover the case for robustness.
            conn.execute(
                "INSERT OR REPLACE INTO semantic_meta (id, backend_id, embedding_dim, last_compile_at_millis)
                 VALUES (?1, ?2, ?3, 0)",
                params![SEMANTIC_META_ID, backend_id, stored_dim],
            ).context("seed semantic_meta")?;
        }
    }

    // Always keep backend_id + dim current (handles the empty-row first run
    // path without forcing a truncate).
    conn.execute(
        "UPDATE semantic_meta SET backend_id = ?1, embedding_dim = ?2 WHERE id = ?3",
        params![backend_id, stored_dim, SEMANTIC_META_ID],
    )
    .context("update semantic_meta backend_id/dim")?;

    Ok(())
}

fn set_last_compile_at(conn: &Connection, millis: i64) -> Result<()> {
    conn.execute(
        "UPDATE semantic_meta SET last_compile_at_millis = ?1 WHERE id = ?2",
        params![millis, SEMANTIC_META_ID],
    )
    .context("update last_compile_at_millis")?;
    Ok(())
}

fn current_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

fn load_existing_hashes(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn
        .prepare("SELECT item_id, content_hash FROM item_embeddings")
        .context("prepare hash query")?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .context("query item_embeddings")?;
    let mut out = HashMap::new();
    for row in rows {
        let (id, hash) = row.context("read item_embeddings row")?;
        out.insert(id, hash);
    }
    Ok(out)
}

fn remove_stale_embeddings(conn: &Connection, live_ids: &HashSet<String>) -> Result<usize> {
    let mut stmt = conn
        .prepare("SELECT item_id FROM item_embeddings")
        .context("prepare stale id query")?;
    let stored_ids: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .context("query stale ids")?
        .filter_map(Result::ok)
        .collect();

    let mut removed = 0_usize;
    for id in &stored_ids {
        if !live_ids.contains(id) {
            conn.execute("DELETE FROM item_embeddings WHERE item_id = ?1", params![id])
                .with_context(|| format!("delete stale embedding {id}"))?;
            removed += 1;
        }
    }
    Ok(removed)
}

#[derive(Debug, Clone)]
struct WikiItem {
    item_id: String,
    embed_input: String,
    content_hash: String,
}

fn collect_wiki_items(root: &Path) -> Result<Vec<WikiItem>> {
    let mut items = Vec::new();

    let sources_dir = root.join(WIKI_SOURCES);
    if sources_dir.is_dir() {
        for path in collect_md_files(&sources_dir)? {
            match build_wiki_item(&path, root, false) {
                Ok(item) => items.push(item),
                Err(err) => warn!(
                    "embedding sync: skipping {} (parse failed): {err}",
                    path.display()
                ),
            }
        }
    }

    let concepts_dir = root.join(WIKI_CONCEPTS);
    if concepts_dir.is_dir() {
        for path in collect_md_files(&concepts_dir)? {
            match build_wiki_item(&path, root, true) {
                Ok(item) => items.push(item),
                Err(err) => warn!(
                    "embedding sync: skipping {} (parse failed): {err}",
                    path.display()
                ),
            }
        }
    }

    Ok(items)
}

fn collect_md_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry.with_context(|| format!("entry in {}", dir.display()))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "md") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn build_wiki_item(path: &Path, root: &Path, is_concept: bool) -> Result<WikiItem> {
    let (frontmatter, body) =
        read_frontmatter(path).with_context(|| format!("read frontmatter {}", path.display()))?;

    let title_key = if is_concept { "name" } else { "title" };
    let title = frontmatter
        .get(title_key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();

    let aliases = if is_concept {
        frontmatter
            .get("aliases")
            .and_then(Value::as_sequence)
            .map(|seq| {
                seq.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let body_clean = strip_managed_regions(&body);
    let embed_input = build_embed_input(&title, &aliases, &body_clean);
    let content_hash = content_hash_hex(&embed_input, super::model::HASH_BACKEND_ID);

    let rel = path.strip_prefix(root).unwrap_or(path);
    Ok(WikiItem {
        item_id: item_id_for_relpath(rel),
        embed_input,
        content_hash,
    })
}

/// Drop the managed-region *fences* from `body`, keeping the inner
/// content. The fences are HTML-comment markers (`<!-- kb:begin id=... -->`
/// / `<!-- kb:end id=... -->`); their text carries no semantic signal and
/// would just inject noise into the hash-embed n-grams.
///
/// Stripping the fences but keeping the content is the right balance for
/// kb's wiki pages: the rendered body is almost entirely managed regions
/// (title, summary, `key_topics`, citations), so dropping them entirely
/// would leave the embedder with little more than the page heading. The
/// content inside managed regions changes whenever the LLM re-renders the
/// summary, which causes the content hash to shift and a re-embed to fire
/// — that is exactly the desired behavior: when the canonical summary
/// changes, the semantic vector should follow.
fn strip_managed_regions(body: &str) -> String {
    let regions = extract_managed_regions(body);
    if regions.is_empty() {
        return body.to_string();
    }

    let mut out = String::with_capacity(body.len());
    let mut cursor = 0_usize;
    for region in &regions {
        if region.full_start >= cursor {
            out.push_str(&body[cursor..region.full_start]);
        }
        // Keep the region's content, dropping just the fence comments.
        if region.content_start <= region.content_end {
            out.push_str(&body[region.content_start..region.content_end]);
        }
        cursor = region.full_end;
    }
    if cursor < body.len() {
        out.push_str(&body[cursor..]);
    }
    out
}

fn build_embed_input(title: &str, aliases: &[String], body: &str) -> String {
    let mut buf = String::new();
    if !title.is_empty() {
        buf.push_str(title);
        buf.push('\n');
    }
    for alias in aliases {
        if !alias.is_empty() {
            buf.push_str(alias);
            buf.push('\n');
        }
    }
    buf.push_str(body.trim());

    // Cap input length so degenerate giant pages can't blow up embedding
    // time. Truncate at a char boundary by counting chars rather than
    // bytes — &str slicing on a non-boundary panics.
    if buf.chars().count() > MAX_EMBED_CHARS {
        let truncated: String = buf.chars().take(MAX_EMBED_CHARS).collect();
        return truncated;
    }
    buf
}

fn content_hash_hex(content: &str, backend_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(backend_id.as_bytes());
    hasher.update(b":");
    hasher.update(content.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn encode_embedding_json(embedding: &[f32]) -> String {
    // Hand-rolled to match sqlite-vec's `vec_f32(json_text)` parser, which
    // accepts a plain JSON array of numbers.
    let mut encoded = String::from("[");
    for (idx, value) in embedding.iter().enumerate() {
        if idx != 0 {
            encoded.push(',');
        }
        encoded.push_str(&value.to_string());
    }
    encoded.push(']');
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::model::HashEmbedBackend;
    use std::fs;
    use tempfile::tempdir;

    fn write_source(root: &Path, slug: &str, title: &str, body: &str) {
        let dir = root.join(WIKI_SOURCES);
        fs::create_dir_all(&dir).expect("mkdir sources");
        let path = dir.join(format!("{slug}.md"));
        let frontmatter = format!("---\ntitle: \"{title}\"\n---\n");
        fs::write(&path, format!("{frontmatter}{body}")).expect("write source");
    }

    fn write_concept(root: &Path, slug: &str, name: &str, body: &str, aliases: &[&str]) {
        let dir = root.join(WIKI_CONCEPTS);
        fs::create_dir_all(&dir).expect("mkdir concepts");
        let path = dir.join(format!("{slug}.md"));
        let alias_yaml = if aliases.is_empty() {
            String::new()
        } else {
            let body: Vec<String> = aliases.iter().map(|a| format!("  - \"{a}\"")).collect();
            format!("aliases:\n{}\n", body.join("\n"))
        };
        let frontmatter = format!("---\nname: \"{name}\"\n{alias_yaml}---\n");
        fs::write(&path, format!("{frontmatter}{body}")).expect("write concept");
    }

    #[test]
    fn item_id_uses_forward_slashes() {
        let p = PathBuf::from("wiki").join("sources").join("foo.md");
        assert_eq!(item_id_for_relpath(&p), "wiki/sources/foo.md");
    }

    #[test]
    fn sync_embeds_sources_and_concepts() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        write_source(root, "alpha", "Alpha doc", "# Alpha\nbody about login\n");
        write_concept(root, "auth", "Authentication", "Auth concept body", &["authn"]);

        let backend = HashEmbedBackend::new();
        let stats = sync_embeddings(root, &backend).expect("sync");
        assert_eq!(stats.embedded, 2);
        assert_eq!(stats.removed, 0);
        assert_eq!(stats.up_to_date, 0);

        // Second pass is a no-op.
        let again = sync_embeddings(root, &backend).expect("sync2");
        assert_eq!(again.embedded, 0);
        assert_eq!(again.up_to_date, 2);
    }

    #[test]
    fn sync_drops_stale_rows() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        write_source(root, "alpha", "Alpha", "Body A\n");
        write_source(root, "beta", "Beta", "Body B\n");

        let backend = HashEmbedBackend::new();
        sync_embeddings(root, &backend).expect("first sync");

        fs::remove_file(root.join(WIKI_SOURCES).join("beta.md")).expect("rm beta");
        let stats = sync_embeddings(root, &backend).expect("second sync");
        assert_eq!(stats.removed, 1);
        assert_eq!(stats.up_to_date, 1);
    }

    #[test]
    fn semantic_index_stats_reflects_disk_state() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        write_source(root, "alpha", "Alpha", "Body A\n");
        write_source(root, "beta", "Beta", "Body B\n");

        let stats_before = semantic_index_stats(root).expect("stats before");
        assert_eq!(stats_before.embeddings, 0);
        assert_eq!(stats_before.stale, 0);

        let backend = HashEmbedBackend::new();
        sync_embeddings(root, &backend).expect("sync");
        let stats_after = semantic_index_stats(root).expect("stats after");
        assert_eq!(stats_after.embeddings, 2);
        assert_eq!(stats_after.stale, 0);
    }

    #[test]
    fn strip_managed_regions_removes_fences_and_keeps_content() {
        let body = "preamble\n<!-- kb:begin id=summary -->managed body<!-- kb:end id=summary -->\nepilogue";
        let cleaned = strip_managed_regions(body);
        // Inner content is preserved (real wiki bodies are almost entirely
        // managed regions, so dropping them entirely would leave nothing
        // semantic for the embedder to chew on).
        assert!(cleaned.contains("managed body"));
        // Fence comments are gone.
        assert!(!cleaned.contains("kb:begin"));
        assert!(!cleaned.contains("kb:end"));
        // Surrounding content survives.
        assert!(cleaned.contains("preamble"));
        assert!(cleaned.contains("epilogue"));
    }

    #[test]
    fn embed_input_truncates_overlong_bodies() {
        let huge = "a".repeat(MAX_EMBED_CHARS * 2);
        let input = build_embed_input("title", &[], &huge);
        assert!(input.chars().count() <= MAX_EMBED_CHARS);
    }
}
