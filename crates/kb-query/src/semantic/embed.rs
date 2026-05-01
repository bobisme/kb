//! Embedding pipeline: walks the wiki, chunks each page by H2/H3, computes
//! per-chunk content hashes, and upserts vectors into
//! `.kb/state/embeddings.db`.
//!
//! Per-chunk embeddings (bn-3rzz) replaced the old whole-page approach
//! because long sources averaged to nothing under MiniLM-L6-v2's 256-token
//! window. See [`crate::semantic::chunk`] for the chunking algorithm.
//!
//! The pipeline is intentionally idempotent. Each chunk is keyed by a
//! content hash (blake3 of `backend_id + ":" + chunk_idx + ":" + heading + ":" + body`)
//! — when nothing has changed, the second compile is a no-op aside from
//! the per-chunk hash check.
//!
//! Schema (created on first run):
//!
//! ```sql
//! CREATE TABLE chunk_embeddings (
//!     chunk_id TEXT PRIMARY KEY,         -- "<item_id>#<idx>"
//!     item_id TEXT NOT NULL,
//!     chunk_idx INTEGER NOT NULL,
//!     heading TEXT,
//!     content_hash TEXT NOT NULL,
//!     embedding_json TEXT NOT NULL
//! );
//! CREATE INDEX idx_chunk_item ON chunk_embeddings(item_id);
//!
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
//! incompatible. Bumping the `backend_id` (e.g. `ort-minilm-384` →
//! `ort-minilm-384-chunked` for bn-3rzz) is therefore the migration
//! mechanism: existing kb installs auto re-embed cleanly on first compile
//! after upgrade.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use kb_core::{extract_managed_regions, frontmatter::read_frontmatter, state_dir};
use rusqlite::{Connection, OptionalExtension, params};
use serde_yaml::Value;
use tracing::{debug, warn};

use super::chunk::{Chunk, chunk_markdown};
use super::model::EmbeddingBackend;

/// Relative path of the embedding database from the kb root.
pub const EMBEDDING_DB_REL: &[&str] = &["embeddings.db"];

const SEMANTIC_META_ID: i64 = 1;

const WIKI_SOURCES: &str = "wiki/sources";
const WIKI_CONCEPTS: &str = "wiki/concepts";

/// Per-pass synchronization summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SyncStats {
    /// Number of chunks embedded (new or content-changed).
    pub embedded: usize,
    /// Number of stored chunk embeddings deleted because their item disappeared
    /// or their `chunk_idx` exceeds the current chunk count for that item.
    pub removed: usize,
    /// Number of chunks that were already up to date (hash unchanged).
    pub up_to_date: usize,
}

/// Snapshot of the semantic index, used by `kb status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SemanticIndexStats {
    /// Total chunk embeddings stored on disk.
    pub embeddings: usize,
    /// Items that exist on disk but are missing or stale in the store. An
    /// item is "stale" when at least one of its chunks is missing or has a
    /// different content hash than the on-disk one.
    pub stale: usize,
}

/// Resolve the absolute path of the embeddings database.
#[must_use]
pub fn embedding_db_path(root: &Path) -> PathBuf {
    state_dir(root).join(EMBEDDING_DB_REL[0])
}

/// Translate a `wiki/...md` path (under `root`) into the canonical item id
/// stored in `chunk_embeddings.item_id`. Always uses forward slashes so the
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

/// Build the canonical chunk id stored in `chunk_embeddings.chunk_id` from
/// an item id and a chunk index. The format is `"<item_id>#<idx>"`.
#[must_use]
pub fn chunk_id_for(item_id: &str, chunk_idx: usize) -> String {
    format!("{item_id}#{chunk_idx}")
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

/// Create `chunk_embeddings` and `semantic_meta` if they don't exist.
///
/// Also drops the legacy `item_embeddings` table from earlier kb versions
/// — bn-3rzz replaced it with `chunk_embeddings`. Dropping is safe because
/// the old rows would mismatch the chunked backend ids and `handle_backend_change`
/// would force a re-embed anyway.
///
/// # Errors
///
/// Returns an error when the schema DDL fails.
pub fn ensure_embedding_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DROP TABLE IF EXISTS item_embeddings;

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT PRIMARY KEY,
            item_id TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            heading TEXT,
            content_hash TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_item ON chunk_embeddings(item_id);

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
/// Walks `wiki/sources/` + `wiki/concepts/`, chunks each item via
/// [`chunk_markdown`], recomputes each chunk's content hash, and re-embeds
/// anything that has changed. Stored chunk rows whose item id no longer
/// corresponds to a wiki page on disk — or whose `chunk_idx` exceeds the
/// current chunk count for that item — are deleted. The `semantic_meta`
/// row is updated with the active backend id, dim, and the current timestamp.
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
    let existing = load_existing_chunks(&conn)?;

    // For stale-cleanup: live (item_id, idx) pairs that survive this sync.
    let mut live_pairs: HashSet<(String, usize)> = HashSet::new();

    let mut stats = SyncStats::default();
    for item in &items {
        for (idx, item_chunk) in item.chunks.iter().enumerate() {
            live_pairs.insert((item.item_id.clone(), idx));
            let chunk_id = chunk_id_for(&item.item_id, idx);
            let stored_hash = existing.get(&chunk_id);
            if stored_hash.map(String::as_str) == Some(item_chunk.content_hash.as_str()) {
                stats.up_to_date += 1;
                continue;
            }
            let embedding = backend
                .embed(&item_chunk.embed_input)
                .with_context(|| format!("embed {chunk_id}"))?;
            if embedding.len() != backend.dimensions() {
                bail!(
                    "embedding dim mismatch for {chunk_id}: expected {}, got {}",
                    backend.dimensions(),
                    embedding.len()
                );
            }
            pipeline.upsert(
                &chunk_id,
                &item.item_id,
                idx,
                item_chunk.heading.as_deref(),
                &item_chunk.content_hash,
                &embedding,
            )?;
            stats.embedded += 1;
        }
    }

    stats.removed = remove_stale_chunks(&conn, &live_pairs)?;

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
        .query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| row.get(0))
        .context("count chunk_embeddings")?;

    let existing = load_existing_chunks(&conn)?;
    let items = collect_wiki_items(root).unwrap_or_default();
    let live_item_ids: HashSet<String> = items.iter().map(|i| i.item_id.clone()).collect();

    let mut stale_items = 0_usize;
    for item in &items {
        let mut item_is_stale = false;
        for (idx, item_chunk) in item.chunks.iter().enumerate() {
            let chunk_id = chunk_id_for(&item.item_id, idx);
            match existing.get(&chunk_id) {
                Some(hash) if hash == &item_chunk.content_hash => {}
                _ => {
                    item_is_stale = true;
                    break;
                }
            }
        }
        if item_is_stale {
            stale_items += 1;
        }
    }
    // Stored item ids whose source page is gone count as stale items too.
    let stored_item_ids: HashSet<String> = existing
        .keys()
        .filter_map(|chunk_id| split_chunk_id(chunk_id).map(|(item, _)| item.to_string()))
        .collect();
    for stored_item in &stored_item_ids {
        if !live_item_ids.contains(stored_item) {
            stale_items += 1;
        }
    }

    Ok(SemanticIndexStats {
        embeddings: usize::try_from(embeddings).unwrap_or(0),
        stale: stale_items,
    })
}

/// Lower-level helper: upsert a single chunk without going through the
/// content-hash dedup. Useful in `kb forget` to drop rows eagerly.
pub struct EmbeddingPipeline<'a, B: EmbeddingBackend> {
    backend: &'a B,
    conn: &'a Connection,
}

impl<B: EmbeddingBackend> EmbeddingPipeline<'_, B> {
    /// Upsert a single chunk's embedding row.
    ///
    /// # Errors
    ///
    /// Returns an error when the DB operation fails or the embedding
    /// vector dimensionality doesn't match the backend.
    pub fn upsert(
        &self,
        chunk_id: &str,
        item_id: &str,
        chunk_idx: usize,
        heading: Option<&str>,
        content_hash: &str,
        embedding: &[f32],
    ) -> Result<()> {
        if embedding.len() != self.backend.dimensions() {
            bail!(
                "embedding dim mismatch upserting {chunk_id}: expected {}, got {}",
                self.backend.dimensions(),
                embedding.len()
            );
        }
        let json = encode_embedding_json(embedding);
        let chunk_idx_i64 = i64::try_from(chunk_idx).unwrap_or(i64::MAX);
        self.conn
            .execute(
                "INSERT INTO chunk_embeddings (chunk_id, item_id, chunk_idx, heading, content_hash, embedding_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(chunk_id)
                 DO UPDATE SET item_id = excluded.item_id,
                               chunk_idx = excluded.chunk_idx,
                               heading = excluded.heading,
                               content_hash = excluded.content_hash,
                               embedding_json = excluded.embedding_json",
                params![chunk_id, item_id, chunk_idx_i64, heading, content_hash, json],
            )
            .with_context(|| format!("upsert chunk_embedding for {chunk_id}"))?;
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
            conn.execute("DELETE FROM chunk_embeddings", [])
                .context("truncate chunk_embeddings on backend change")?;
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

fn load_existing_chunks(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn
        .prepare("SELECT chunk_id, content_hash FROM chunk_embeddings")
        .context("prepare chunk hash query")?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .context("query chunk_embeddings")?;
    let mut out = HashMap::new();
    for row in rows {
        let (id, hash) = row.context("read chunk_embeddings row")?;
        out.insert(id, hash);
    }
    Ok(out)
}

fn remove_stale_chunks(conn: &Connection, live_pairs: &HashSet<(String, usize)>) -> Result<usize> {
    let mut stmt = conn
        .prepare("SELECT chunk_id FROM chunk_embeddings")
        .context("prepare stale chunk query")?;
    let stored_ids: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .context("query stale chunk ids")?
        .filter_map(Result::ok)
        .collect();

    let mut removed = 0_usize;
    for chunk_id in &stored_ids {
        let Some((item_id, idx)) = split_chunk_id(chunk_id) else {
            // Malformed chunk_id — drop it; it'll get rebuilt next sync.
            conn.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id = ?1",
                params![chunk_id],
            )
            .with_context(|| format!("delete malformed chunk_id {chunk_id}"))?;
            removed += 1;
            continue;
        };
        if !live_pairs.contains(&(item_id.to_string(), idx)) {
            conn.execute(
                "DELETE FROM chunk_embeddings WHERE chunk_id = ?1",
                params![chunk_id],
            )
            .with_context(|| format!("delete stale chunk {chunk_id}"))?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Parse `"<item_id>#<idx>"` back into its components. Returns `None`
/// when the id doesn't contain the `#` separator or the suffix isn't a
/// valid `usize`. Item ids with embedded `#` (`wiki/sources/foo#bar.md`)
/// are not currently produced anywhere in kb, but we use `rsplit_once` so
/// only the trailing `#<digits>` is taken as the chunk index.
fn split_chunk_id(chunk_id: &str) -> Option<(&str, usize)> {
    let (item, idx_str) = chunk_id.rsplit_once('#')?;
    let idx = idx_str.parse::<usize>().ok()?;
    Some((item, idx))
}

#[derive(Debug, Clone)]
struct WikiItem {
    item_id: String,
    chunks: Vec<ItemChunk>,
}

#[derive(Debug, Clone)]
struct ItemChunk {
    heading: Option<String>,
    /// The text actually fed into the embedder (title prefix + body when applicable).
    embed_input: String,
    /// Hash of the embed input scoped by backend id + chunk index + heading,
    /// so identical bodies under different headings dedup independently.
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
    // Build a lead-in carrying title + aliases. We prepend it to every
    // chunk's embed input so MiniLM sees the page's identity even when
    // the chunk is mid-document. The hash is per-chunk so dedup still
    // fires correctly when only a single section's body changes.
    let lead = build_title_lead(&title, &aliases);

    let chunks = chunk_markdown(&body_clean);
    let item_chunks: Vec<ItemChunk> = if chunks.is_empty() {
        // No body at all — emit a single chunk carrying just the title +
        // aliases so the page is still retrievable. Drop entirely if even
        // the title is empty (we have nothing meaningful to embed).
        let lead_trimmed = lead.trim();
        if lead_trimmed.is_empty() {
            Vec::new()
        } else {
            let chunk_idx = 0_usize;
            let heading: Option<String> = None;
            let content_hash = chunk_content_hash(
                super::model::HASH_BACKEND_ID,
                chunk_idx,
                heading.as_deref(),
                lead_trimmed,
            );
            vec![ItemChunk {
                heading,
                embed_input: lead_trimmed.to_string(),
                content_hash,
            }]
        }
    } else {
        chunks
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| build_item_chunk(idx, &lead, &chunk))
            .collect()
    };

    let rel = path.strip_prefix(root).unwrap_or(path);
    Ok(WikiItem {
        item_id: item_id_for_relpath(rel),
        chunks: item_chunks,
    })
}

fn build_item_chunk(idx: usize, lead: &str, chunk: &Chunk) -> ItemChunk {
    let mut embed_input = String::new();
    if !lead.is_empty() {
        embed_input.push_str(lead);
        if !embed_input.ends_with('\n') {
            embed_input.push('\n');
        }
    }
    if let Some(heading) = chunk.heading.as_deref()
        && !heading.is_empty()
    {
        embed_input.push_str(heading);
        embed_input.push('\n');
    }
    embed_input.push_str(&chunk.body);

    let content_hash = chunk_content_hash(
        super::model::HASH_BACKEND_ID,
        idx,
        chunk.heading.as_deref(),
        &embed_input,
    );
    ItemChunk {
        heading: chunk.heading.clone(),
        embed_input,
        content_hash,
    }
}

fn build_title_lead(title: &str, aliases: &[String]) -> String {
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
    buf
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

fn chunk_content_hash(
    backend_id: &str,
    chunk_idx: usize,
    heading: Option<&str>,
    content: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(backend_id.as_bytes());
    hasher.update(b":");
    hasher.update(chunk_idx.to_string().as_bytes());
    hasher.update(b":");
    hasher.update(heading.unwrap_or("").as_bytes());
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

    fn count_chunk_rows(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM chunk_embeddings", [], |row| row.get(0))
            .expect("count rows")
    }

    fn count_chunks_for_item(conn: &Connection, item_id: &str) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE item_id = ?1",
            params![item_id],
            |row| row.get(0),
        )
        .expect("count item rows")
    }

    #[test]
    fn item_id_uses_forward_slashes() {
        let p = PathBuf::from("wiki").join("sources").join("foo.md");
        assert_eq!(item_id_for_relpath(&p), "wiki/sources/foo.md");
    }

    #[test]
    fn chunk_id_round_trips() {
        let cid = chunk_id_for("wiki/sources/foo.md", 3);
        assert_eq!(cid, "wiki/sources/foo.md#3");
        let parsed = split_chunk_id(&cid).expect("parses");
        assert_eq!(parsed, ("wiki/sources/foo.md", 3));
    }

    #[test]
    fn sync_embeds_sources_and_concepts_per_chunk() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        // Single-section source → one chunk.
        write_source(root, "alpha", "Alpha doc", "## Topic\n\nbody about login\n");
        // No-H2 concept → single chunk.
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
    fn long_source_with_h2_produces_multiple_chunks() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        // Two H2 sections, each well above TARGET_MIN_TOKENS so they
        // survive merging. (~600 chars each ≈ 150 estimated tokens.)
        let body = format!(
            "## First topic\n\n{}\n\n## Second topic\n\n{}\n",
            "alpha ".repeat(120),
            "beta ".repeat(120),
        );
        write_source(root, "doc", "A multi-section source", &body);

        let backend = HashEmbedBackend::new();
        sync_embeddings(root, &backend).expect("sync");

        let conn = open_embedding_db(root).expect("open db");
        let total = count_chunk_rows(&conn);
        assert_eq!(total, 2, "expected 2 chunks for 2-H2 source");
        let item_id = "wiki/sources/doc.md".to_string();
        assert_eq!(count_chunks_for_item(&conn, &item_id), 2);

        // Headings are stored.
        let headings: Vec<String> = conn
            .prepare("SELECT heading FROM chunk_embeddings WHERE item_id = ?1 ORDER BY chunk_idx")
            .expect("prep")
            .query_map(params![item_id], |row| row.get::<_, Option<String>>(0))
            .expect("query")
            .filter_map(Result::ok)
            .map(Option::unwrap_or_default)
            .collect();
        assert_eq!(
            headings,
            vec!["First topic".to_string(), "Second topic".to_string()],
        );
    }

    #[test]
    fn sync_drops_stale_rows_when_item_disappears() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        write_source(root, "alpha", "Alpha", "## A\n\nBody A\n");
        write_source(root, "beta", "Beta", "## B\n\nBody B\n");

        let backend = HashEmbedBackend::new();
        sync_embeddings(root, &backend).expect("first sync");

        fs::remove_file(root.join(WIKI_SOURCES).join("beta.md")).expect("rm beta");
        let stats = sync_embeddings(root, &backend).expect("second sync");
        assert!(stats.removed >= 1, "expected stale chunks dropped, stats={stats:?}");
    }

    #[test]
    fn sync_drops_chunks_whose_idx_exceeds_current_count() {
        // Edit a multi-chunk source down to a single chunk and re-sync —
        // the dropped chunk_idx rows must be cleaned up.
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        let big_body = format!(
            "## First\n\n{}\n\n## Second\n\n{}\n",
            "alpha ".repeat(120),
            "beta ".repeat(120),
        );
        write_source(root, "doc", "Doc", &big_body);

        let backend = HashEmbedBackend::new();
        sync_embeddings(root, &backend).expect("first sync");
        let conn = open_embedding_db(root).expect("open db");
        let item_id = "wiki/sources/doc.md".to_string();
        assert_eq!(count_chunks_for_item(&conn, &item_id), 2);
        drop(conn);

        // Shrink the source to a single section.
        let small_body = format!("## Only\n\n{}\n", "single ".repeat(60));
        write_source(root, "doc", "Doc", &small_body);
        let stats = sync_embeddings(root, &backend).expect("second sync");
        // At least one chunk (the old idx=1) was dropped.
        assert!(stats.removed >= 1, "stats={stats:?}");

        let conn = open_embedding_db(root).expect("reopen db");
        assert_eq!(count_chunks_for_item(&conn, &item_id), 1);
    }

    #[test]
    fn semantic_index_stats_reflects_disk_state() {
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        write_source(root, "alpha", "Alpha", "## A\n\nBody A\n");
        write_source(root, "beta", "Beta", "## B\n\nBody B\n");

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
        assert!(cleaned.contains("managed body"));
        assert!(!cleaned.contains("kb:begin"));
        assert!(!cleaned.contains("kb:end"));
        assert!(cleaned.contains("preamble"));
        assert!(cleaned.contains("epilogue"));
    }

    #[test]
    fn empty_body_with_title_still_embeds_a_chunk() {
        // A page with no body but a title — the embed pipeline must still
        // produce a single chunk so the page is retrievable.
        let tmp = tempdir().expect("tempdir");
        let root = tmp.path();
        write_source(root, "stub", "A stub title", "");
        let backend = HashEmbedBackend::new();
        sync_embeddings(root, &backend).expect("sync");
        let conn = open_embedding_db(root).expect("open");
        assert_eq!(count_chunk_rows(&conn), 1);
    }
}
