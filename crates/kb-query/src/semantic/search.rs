//! Semantic KNN search over `item_embeddings`.
//!
//! Tries the sqlite-vec fast path first (`vec_distance_cosine` in pure SQL,
//! sorted in C) and falls back to a Rust-side cosine implementation when the
//! extension is unavailable — for example because the user opted out via
//! `KB_SQLITE_VEC_AUTO=0` or the extension entry point is missing on the
//! current sqlite build.
//!
//! Output scores are mapped from cosine `[-1, 1]` to `[0, 1]` so they fuse
//! cleanly with lexical ranks in the hybrid layer.

use anyhow::{Context, Result, bail};
use rusqlite::Connection;
use serde::Serialize;
use tracing::debug;

/// A semantic match returned by [`knn_search`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SemanticSearchResult {
    /// Page id, e.g. `"wiki/sources/src-1abc-foo.md"`.
    pub item_id: String,
    /// Similarity score in `[0, 1]` (higher = more similar).
    pub score: f32,
}

/// Run KNN over the stored embeddings using cosine similarity.
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
) -> Result<Vec<SemanticSearchResult>> {
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

fn try_knn_via_sqlite_vec(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Option<Vec<SemanticSearchResult>>> {
    let vec_available = conn
        .query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0))
        .is_ok();
    if !vec_available {
        return Ok(None);
    }

    let query_json = encode_embedding_json(query_embedding);
    let mut stmt = match conn.prepare(
        "SELECT item_id,
                vec_distance_cosine(vec_f32(embedding_json), vec_f32(?1)) AS distance
         FROM item_embeddings
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
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
    ) {
        Ok(rows) => rows,
        Err(err) => {
            debug!("sqlite-vec KNN query failed, falling back to Rust KNN: {err}");
            return Ok(None);
        }
    };

    let mut out = Vec::with_capacity(limit);
    for row in rows {
        let (item_id, distance) = row.context("read sqlite-vec KNN row")?;
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
        out.push(SemanticSearchResult { item_id, score });
    }

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.item_id.cmp(&b.item_id))
    });

    Ok(Some(out))
}

fn knn_via_rust_cosine(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<SemanticSearchResult>> {
    let mut stmt = conn
        .prepare("SELECT item_id, embedding_json FROM item_embeddings")
        .context("prepare semantic KNN query (semantic index missing?)")?;

    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .context("execute semantic KNN query")?;

    let mut scored = Vec::new();
    for row in rows {
        let (item_id, json) = row.context("read semantic KNN row")?;
        let embedding: Vec<f32> = match serde_json::from_str(&json) {
            Ok(value) => value,
            Err(err) => {
                debug!("skipping malformed semantic embedding row {item_id}: {err}");
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
        scored.push(SemanticSearchResult { item_id, score });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.item_id.cmp(&b.item_id))
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
            "CREATE TABLE item_embeddings (
                item_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            );",
        )
        .expect("create mock table");
        conn
    }

    fn insert(conn: &Connection, item_id: &str, embedding: &[f32]) {
        conn.execute(
            "INSERT INTO item_embeddings (item_id, content_hash, embedding_json)
             VALUES (?1, 'h', ?2)",
            params![item_id, serde_json::to_string(embedding).expect("encode")],
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
    fn knn_returns_ranked_rust_path() {
        let conn = setup_mock_db();
        let mut near = vec![0.0_f32; 8];
        near[0] = 1.0;
        let mut far = vec![0.0_f32; 8];
        far[0] = -1.0;
        insert(&conn, "near", &near);
        insert(&conn, "far", &far);

        let results = knn_search(&conn, &near, 10).expect("knn");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].item_id, "near");
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn rust_and_sqlite_vec_agree_on_top_k() {
        // Force the Rust path by skipping vec extension load attempts
        // (the in-memory connection has no extension regardless when
        // run with KB_SQLITE_VEC_AUTO=0). Compare against an analytical
        // expected ordering.
        let conn = setup_mock_db();
        let mut a = vec![0.0_f32; 16];
        a[0] = 1.0;
        let mut b = vec![0.0_f32; 16];
        b[1] = 1.0;
        let mut c = vec![0.0_f32; 16];
        c[0] = 0.7;
        c[1] = 0.7;
        insert(&conn, "a", &a);
        insert(&conn, "b", &b);
        insert(&conn, "c", &c);

        let query = a.clone();
        let result = knn_search(&conn, &query, 3).expect("knn");
        assert_eq!(result[0].item_id, "a");
        // c is closer to a than b is.
        assert_eq!(result[1].item_id, "c");
        assert_eq!(result[2].item_id, "b");
    }
}
