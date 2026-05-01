//! Semantic KNN search over `chunk_embeddings`.
//!
//! Tries the sqlite-vec fast path first (`vec_distance_cosine` in pure SQL,
//! sorted in C) and falls back to a Rust-side cosine implementation when the
//! extension is unavailable — for example because the user opted out via
//! `KB_SQLITE_VEC_AUTO=0` or the extension entry point is missing on the
//! current sqlite build.
//!
//! Output scores are mapped from cosine `[-1, 1]` to `[0, 1]` so they fuse
//! cleanly with lexical ranks in the hybrid layer.
//!
//! bn-3rzz: search now operates on chunks (sections within a source). The
//! [`knn_search`] entry point returns [`SemanticChunkHit`]s and callers
//! that want source-level hits aggregate via [`aggregate_chunks_to_items`]
//! (max-chunk-score per item, attaching the highest-scoring chunk for
//! snippet UX).

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use serde::Serialize;
use tracing::debug;

/// Backwards-compatible chunk hit type. Preserved as the canonical search
/// result to keep the public API stable; aliased as
/// [`SemanticChunkHit`] for clarity at call sites that aggregate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SemanticSearchResult {
    /// `<item_id>#<chunk_idx>` — globally unique chunk identifier.
    pub chunk_id: String,
    /// Page id, e.g. `"wiki/sources/src-1abc-foo.md"`.
    pub item_id: String,
    /// 0-based chunk index within the item.
    pub chunk_idx: usize,
    /// H2/H3 heading text for this chunk; `None` for pre-heading prose.
    pub heading: Option<String>,
    /// Similarity score in `[0, 1]` (higher = more similar).
    pub score: f32,
}

/// Alias for clarity in callers that distinguish chunk-level vs item-level hits.
pub type SemanticChunkHit = SemanticSearchResult;

/// Item-level result produced by [`aggregate_chunks_to_items`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SemanticItemHit {
    /// Page id.
    pub item_id: String,
    /// Max-chunk-score across the item's chunks (in `[0, 1]`).
    pub score: f32,
    /// The best-scoring chunk (heading + index) for snippet rendering.
    pub best_chunk: Option<SemanticChunkHit>,
}

/// Run KNN over the stored chunk embeddings using cosine similarity.
///
/// Returns chunk hits sorted by descending score. Use
/// [`aggregate_chunks_to_items`] to collapse to source-level when needed.
///
/// # Errors
///
/// Returns an error when the query embedding is empty, the underlying SQL
/// fails, or stored rows cannot be parsed (malformed rows are skipped, not
/// errored).
pub fn knn_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<SemanticChunkHit>> {
    if query_embedding.is_empty() {
        bail!("query embedding must not be empty");
    }
    if limit == 0 {
        return Ok(Vec::new());
    }

    if let Some(results) = try_knn_via_sqlite_vec(conn, query_embedding, limit)? {
        return Ok(results);
    }

    knn_via_rust_cosine(conn, query_embedding, limit)
}

/// Collapse a chunk-level hit list to one item per page, taking the
/// max-chunk-score and attaching the highest-scoring chunk for snippet
/// rendering.
///
/// `hits` may be in any order; the input is consumed and one
/// [`SemanticItemHit`] per distinct `item_id` is returned, sorted by
/// descending score (then by `item_id` for determinism), capped at `limit`.
#[must_use]
pub fn aggregate_chunks_to_items(
    hits: Vec<SemanticChunkHit>,
    limit: usize,
) -> Vec<SemanticItemHit> {
    if limit == 0 {
        return Vec::new();
    }
    // First pass: group by item_id, keep best score + best chunk.
    let mut by_item: HashMap<String, (f32, SemanticChunkHit)> = HashMap::new();
    for hit in hits {
        let entry = by_item.entry(hit.item_id.clone()).or_insert_with(|| (hit.score, hit.clone()));
        if hit.score > entry.0 {
            *entry = (hit.score, hit);
        }
    }

    let mut items: Vec<SemanticItemHit> = by_item
        .into_iter()
        .map(|(item_id, (score, best_chunk))| SemanticItemHit {
            item_id,
            score,
            best_chunk: Some(best_chunk),
        })
        .collect();
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.item_id.cmp(&b.item_id))
    });
    items.truncate(limit);
    items
}

fn try_knn_via_sqlite_vec(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Option<Vec<SemanticChunkHit>>> {
    let vec_available = conn
        .query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0))
        .is_ok();
    if !vec_available {
        return Ok(None);
    }

    let query_json = encode_embedding_json(query_embedding);
    let mut stmt = match conn.prepare(
        "SELECT chunk_id, item_id, chunk_idx, heading,
                vec_distance_cosine(vec_f32(embedding_json), vec_f32(?1)) AS distance
         FROM chunk_embeddings
         ORDER BY distance ASC
         LIMIT ?2",
    ) {
        Ok(stmt) => stmt,
        Err(err) => {
            debug!("sqlite-vec KNN unavailable, falling back to Rust KNN: {err}");
            return Ok(None);
        }
    };

    let rows = match stmt.query_map(
        rusqlite::params![query_json, i64::try_from(limit).unwrap_or(i64::MAX)],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, f64>(4)?,
            ))
        },
    ) {
        Ok(rows) => rows,
        Err(err) => {
            debug!("sqlite-vec KNN query failed, falling back to Rust KNN: {err}");
            return Ok(None);
        }
    };

    let mut out = Vec::with_capacity(limit);
    for row in rows {
        let (chunk_id, item_id, chunk_idx, heading, distance) =
            row.context("read sqlite-vec KNN row")?;
        // sqlite-vec's vec_distance_cosine returns `1 - cosine_similarity`
        // for unit vectors (range [0, 2]). We want a "0 = unrelated, 1 =
        // identical" score so the threshold guards in the hybrid layer
        // (MIN_SEMANTIC_SCORE etc) work the same regardless of backend
        // quality. Clamp negative cosine to 0 instead of mapping
        // (cosine+1)/2: with the hash backend, unrelated documents have
        // cosine in `[0, 0.1]` rather than `[-1, 0]`, and the symmetric
        // mapping would put pure noise around 0.5.
        #[allow(clippy::cast_possible_truncation)]
        let cosine = (1.0 - distance as f32).clamp(-1.0, 1.0);
        let score = cosine.max(0.0);
        out.push(SemanticChunkHit {
            chunk_id,
            item_id,
            chunk_idx: usize::try_from(chunk_idx).unwrap_or(0),
            heading,
            score,
        });
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });

    Ok(Some(out))
}

fn knn_via_rust_cosine(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<SemanticChunkHit>> {
    let mut stmt = conn
        .prepare(
            "SELECT chunk_id, item_id, chunk_idx, heading, embedding_json FROM chunk_embeddings",
        )
        .context("prepare semantic KNN query (semantic index missing?)")?;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
            ))
        })
        .context("execute semantic KNN query")?;

    let mut scored = Vec::new();
    for row in rows {
        let (chunk_id, item_id, chunk_idx, heading, json) =
            row.context("read semantic KNN row")?;
        let embedding: Vec<f32> = match serde_json::from_str(&json) {
            Ok(value) => value,
            Err(err) => {
                debug!("skipping malformed semantic embedding row {chunk_id}: {err}");
                continue;
            }
        };
        let Some(cosine) = cosine_similarity(query_embedding, &embedding) else {
            continue;
        };
        // Match the sqlite-vec path: clamp negative cosine to 0 so the
        // hybrid threshold guards continue to discriminate noise from
        // signal. See the parallel comment above for rationale.
        let score = cosine.max(0.0);
        scored.push(SemanticChunkHit {
            chunk_id,
            item_id,
            chunk_idx: usize::try_from(chunk_idx).unwrap_or(0),
            heading,
            score,
        });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    scored.truncate(limit);

    Ok(scored)
}

fn encode_embedding_json(embedding: &[f32]) -> String {
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

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut dot = 0.0_f32;
    let mut left_norm_sq = 0.0_f32;
    let mut right_norm_sq = 0.0_f32;
    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm_sq += a * a;
        right_norm_sq += b * b;
    }
    let denom = left_norm_sq.sqrt() * right_norm_sq.sqrt();
    if denom <= f32::EPSILON {
        return None;
    }
    Some((dot / denom).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    fn setup_mock_db() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch(
            "CREATE TABLE chunk_embeddings (
                chunk_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                heading TEXT,
                content_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            );",
        )
        .expect("create mock table");
        conn
    }

    fn insert(
        conn: &Connection,
        item_id: &str,
        chunk_idx: usize,
        heading: Option<&str>,
        embedding: &[f32],
    ) {
        let chunk_id = format!("{item_id}#{chunk_idx}");
        conn.execute(
            "INSERT INTO chunk_embeddings (chunk_id, item_id, chunk_idx, heading, content_hash, embedding_json)
             VALUES (?1, ?2, ?3, ?4, 'h', ?5)",
            params![
                chunk_id,
                item_id,
                i64::try_from(chunk_idx).unwrap_or(0),
                heading,
                serde_json::to_string(embedding).expect("encode")
            ],
        )
        .expect("insert");
    }

    #[test]
    fn knn_rejects_empty_query() {
        let conn = setup_mock_db();
        let err = knn_search(&conn, &[], 10).expect_err("empty query");
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn knn_zero_limit_is_empty() {
        let conn = setup_mock_db();
        let q = vec![1.0_f32, 0.0_f32];
        let result = knn_search(&conn, &q, 0).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn knn_returns_ranked_chunk_hits() {
        let conn = setup_mock_db();
        let mut near = vec![0.0_f32; 8];
        near[0] = 1.0;
        let mut far = vec![0.0_f32; 8];
        far[0] = -1.0;
        insert(&conn, "near.md", 0, Some("topic"), &near);
        insert(&conn, "far.md", 0, None, &far);

        let results = knn_search(&conn, &near, 10).expect("knn");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].item_id, "near.md");
        assert_eq!(results[0].chunk_id, "near.md#0");
        assert_eq!(results[0].heading.as_deref(), Some("topic"));
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn rust_cosine_path_orders_chunks_correctly() {
        let conn = setup_mock_db();
        let mut a = vec![0.0_f32; 16];
        a[0] = 1.0;
        let mut b = vec![0.0_f32; 16];
        b[1] = 1.0;
        let mut c = vec![0.0_f32; 16];
        c[0] = 0.7;
        c[1] = 0.7;
        insert(&conn, "doc.md", 0, None, &a);
        insert(&conn, "doc.md", 1, Some("h2"), &b);
        insert(&conn, "doc.md", 2, Some("h3"), &c);

        let result = knn_search(&conn, &a, 3).expect("knn");
        assert_eq!(result[0].chunk_idx, 0);
        // c is closer to a than b is.
        assert_eq!(result[1].chunk_idx, 2);
        assert_eq!(result[2].chunk_idx, 1);
    }

    #[test]
    fn aggregate_chunks_takes_max_score_per_item() {
        let hits = vec![
            SemanticChunkHit {
                chunk_id: "doc-a.md#0".to_string(),
                item_id: "doc-a.md".to_string(),
                chunk_idx: 0,
                heading: Some("A0".to_string()),
                score: 0.5,
            },
            SemanticChunkHit {
                chunk_id: "doc-a.md#1".to_string(),
                item_id: "doc-a.md".to_string(),
                chunk_idx: 1,
                heading: Some("A1".to_string()),
                score: 0.9,
            },
            SemanticChunkHit {
                chunk_id: "doc-b.md#0".to_string(),
                item_id: "doc-b.md".to_string(),
                chunk_idx: 0,
                heading: None,
                score: 0.6,
            },
        ];
        let items = aggregate_chunks_to_items(hits, 10);
        assert_eq!(items.len(), 2);
        // Item with max chunk score wins.
        assert_eq!(items[0].item_id, "doc-a.md");
        assert!((items[0].score - 0.9).abs() < f32::EPSILON);
        let best = items[0].best_chunk.as_ref().expect("best chunk");
        assert_eq!(best.chunk_idx, 1);
        assert_eq!(best.heading.as_deref(), Some("A1"));
        assert_eq!(items[1].item_id, "doc-b.md");
        assert!((items[1].score - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn aggregate_respects_limit() {
        let hits: Vec<SemanticChunkHit> = (0..5)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let score = (i as f32).mul_add(-0.1, 1.0);
                SemanticChunkHit {
                    chunk_id: format!("doc-{i}.md#0"),
                    item_id: format!("doc-{i}.md"),
                    chunk_idx: 0,
                    heading: None,
                    score,
                }
            })
            .collect();
        let items = aggregate_chunks_to_items(hits, 3);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].item_id, "doc-0.md");
    }

    #[test]
    fn aggregate_empty_returns_empty() {
        assert!(aggregate_chunks_to_items(Vec::new(), 10).is_empty());
        let hits = vec![SemanticChunkHit {
            chunk_id: "x#0".to_string(),
            item_id: "x".to_string(),
            chunk_idx: 0,
            heading: None,
            score: 1.0,
        }];
        assert!(aggregate_chunks_to_items(hits, 0).is_empty());
    }
}
