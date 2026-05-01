//! Per-stage incremental compile state — bn-2n7l.
//!
//! `kb compile` previously re-ran every per-source / per-concept stage on
//! every invocation. The embedding-sync (bn-3rzz) and caption (bn-2qda)
//! passes already had their own row-level dedup, but the LLM-heavy passes
//! (`source_summary`, `concept_extraction`) still paid the full cost on
//! repeat runs even when nothing had changed.
//!
//! This module stores a per-(stage, `entity_id`) input fingerprint in
//! `<root>/.kb/state/compile.db`. When the new fingerprint matches the
//! recorded one, the pipeline skips the stage. `--force` ignores lookups
//! but still upserts new fingerprints so the cache stays current.
//!
//! Hash-input recipes (deliberately conservative — anything missing here
//! means a needless re-run, anything extra means a stale-cache miss):
//!
//! * `source_summary`: source body hash + `raw_path` + prompt-template
//!   hash + model id.
//! * `concept_extraction`: source body hash + prompt-template hash +
//!   model id.
//!
//! Other stages (`concept_merge`, `embedding_sync`, `captions`, batch
//! passes like `backlinks` / `lexical_index` / `index_pages`) are NOT
//! gated by `stage_state` — see `pipeline.rs` for why each one was kept
//! out of the per-stage cache (existing dedup, cheap walk, etc.).
//!
//! The schema migrations are idempotent: opening the same file from
//! several kb versions in a row is safe and never re-runs LLM work just
//! because the DB was upgraded.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kb_core::{hash_many, state_dir};
use rusqlite::{Connection, OptionalExtension, params};

/// Filename of the incremental compile state DB under `.kb/state/`.
const COMPILE_DB_FILE: &str = "compile.db";

/// Stage names recorded in `stage_state.stage`. Kept as `&'static str`
/// constants so callers can't typo them; the wire format is plain ASCII.
pub const STAGE_SOURCE_SUMMARY: &str = "source_summary";
pub const STAGE_CONCEPT_EXTRACTION: &str = "concept_extraction";

/// Absolute filesystem path of `compile.db` for a KB root.
#[must_use]
pub fn compile_db_path(root: &Path) -> PathBuf {
    state_dir(root).join(COMPILE_DB_FILE)
}

/// Open `<root>/.kb/state/compile.db`, creating the parent directory and
/// schema as needed.
///
/// Idempotent — calling this from a kb older or newer than the one that
/// created the file is safe; missing tables are added, existing rows
/// preserved. Mirrors the pattern in
/// `kb_query::semantic::embed::open_embedding_db` so kb has exactly one
/// rusqlite open/migrate idiom.
///
/// # Errors
///
/// Returns an error when the parent directory cannot be created, the DB
/// cannot be opened, or schema creation fails.
pub fn open_state_db(root: &Path) -> Result<Connection> {
    let path = compile_db_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create compile state dir {}", parent.display()))?;
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("open compile state db {}", path.display()))?;
    ensure_schema(&conn)?;
    Ok(conn)
}

/// Create the per-source / per-concept / per-stage tables if they don't
/// yet exist. Safe to call repeatedly — every statement uses
/// `IF NOT EXISTS`.
fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS source_state (
            source_id TEXT PRIMARY KEY,
            raw_path TEXT,
            raw_hash TEXT,
            extracted_hash TEXT,
            body_hash TEXT,
            last_compiled_at_millis INTEGER
        );

        CREATE TABLE IF NOT EXISTS concept_state (
            concept_slug TEXT PRIMARY KEY,
            body_hash TEXT,
            cited_sources_json TEXT,
            last_compiled_at_millis INTEGER
        );

        CREATE TABLE IF NOT EXISTS stage_state (
            stage TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            last_run_at_millis INTEGER,
            PRIMARY KEY (stage, entity_id)
        );",
    )
    .context("create compile state tables")?;
    Ok(())
}

/// Snapshot of one row in `stage_state`. Returned by [`lookup_stage`] so
/// callers can compare the recorded `input_hash` against a freshly
/// computed one and decide whether to re-run or skip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageRecord {
    pub stage: String,
    pub entity_id: String,
    pub input_hash: String,
    pub last_run_at_millis: Option<i64>,
}

/// Read the recorded stage record for `(stage, entity_id)`, if any.
///
/// Returns `Ok(None)` when no row exists — typically because this is the
/// first compile that touched this entity, or `--force` was used after a
/// state-DB delete.
///
/// # Errors
///
/// Returns an error when the underlying SQL query fails.
pub fn lookup_stage(
    conn: &Connection,
    stage: &str,
    entity_id: &str,
) -> Result<Option<StageRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT input_hash, last_run_at_millis
             FROM stage_state
             WHERE stage = ?1 AND entity_id = ?2",
        )
        .context("prepare stage_state lookup")?;
    let row = stmt
        .query_row(params![stage, entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<i64>>(1)?,
            ))
        })
        .optional()
        .context("query stage_state lookup")?;
    Ok(row.map(|(input_hash, last_run_at_millis)| StageRecord {
        stage: stage.to_string(),
        entity_id: entity_id.to_string(),
        input_hash,
        last_run_at_millis,
    }))
}

/// Insert or update the stage fingerprint for `(stage, entity_id)`.
///
/// `last_run_at_millis` should be the timestamp at which the stage
/// finished running successfully — a future compile compares the fresh
/// input hash against the stored one and skips the LLM call when they
/// match.
///
/// # Errors
///
/// Returns an error when the underlying SQL execute fails.
pub fn upsert_stage(
    conn: &Connection,
    stage: &str,
    entity_id: &str,
    input_hash: &str,
    last_run_at_millis: i64,
) -> Result<()> {
    conn.execute(
        "INSERT INTO stage_state (stage, entity_id, input_hash, last_run_at_millis)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(stage, entity_id) DO UPDATE SET
            input_hash = excluded.input_hash,
            last_run_at_millis = excluded.last_run_at_millis",
        params![stage, entity_id, input_hash, last_run_at_millis],
    )
    .with_context(|| format!("upsert stage_state {stage}/{entity_id}"))?;
    Ok(())
}

/// Compute a stable input fingerprint from a list of byte slices.
///
/// Components are joined with a NUL byte before hashing so reordering
/// distinct values can't accidentally collide. Inputs are passed in a
/// fixed order at every call site (see `pipeline.rs`).
#[must_use]
pub fn fingerprint_inputs(parts: &[&[u8]]) -> String {
    let mut combined: Vec<&[u8]> = Vec::with_capacity(parts.len() * 2);
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            combined.push(b"\0");
        }
        combined.push(part);
    }
    hash_many(&combined).to_hex()
}

/// All recorded stage rows, ordered by `(stage, entity_id)`. Used by
/// `kb compile --explain` to print a plan of what would re-run vs skip.
///
/// # Errors
///
/// Returns an error when the SQL query fails.
pub fn list_stage_records(conn: &Connection) -> Result<Vec<StageRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT stage, entity_id, input_hash, last_run_at_millis
             FROM stage_state
             ORDER BY stage, entity_id",
        )
        .context("prepare stage_state list")?;
    let rows = stmt
        .query_map([], |row| {
            Ok(StageRecord {
                stage: row.get(0)?,
                entity_id: row.get(1)?,
                input_hash: row.get(2)?,
                last_run_at_millis: row.get(3)?,
            })
        })
        .context("query stage_state list")?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.context("read stage_state row")?);
    }
    Ok(out)
}

/// Snapshot of a row in `source_state`.
///
/// Currently informational only — the per-stage fingerprints in
/// `stage_state` carry the actual skip signal — but tracked here so
/// future stages (e.g. raw-extraction) can gate on `raw_hash` /
/// `extracted_hash` without bolting on another table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceStateRow {
    pub source_id: String,
    pub raw_path: Option<String>,
    pub raw_hash: Option<String>,
    pub extracted_hash: Option<String>,
    pub body_hash: Option<String>,
    pub last_compiled_at_millis: Option<i64>,
}

/// Upsert a row in `source_state`. Called from the compile pipeline once
/// a source's fresh hashes have been computed, so subsequent compiles
/// can read them without re-walking the normalized doc.
///
/// # Errors
///
/// Returns an error when the SQL execute fails.
pub fn upsert_source_state(conn: &Connection, row: &SourceStateRow) -> Result<()> {
    conn.execute(
        "INSERT INTO source_state (
            source_id, raw_path, raw_hash, extracted_hash, body_hash, last_compiled_at_millis
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(source_id) DO UPDATE SET
            raw_path = excluded.raw_path,
            raw_hash = excluded.raw_hash,
            extracted_hash = excluded.extracted_hash,
            body_hash = excluded.body_hash,
            last_compiled_at_millis = excluded.last_compiled_at_millis",
        params![
            row.source_id,
            row.raw_path,
            row.raw_hash,
            row.extracted_hash,
            row.body_hash,
            row.last_compiled_at_millis,
        ],
    )
    .with_context(|| format!("upsert source_state {}", row.source_id))?;
    Ok(())
}

/// Read back a `source_state` row, if any.
///
/// # Errors
///
/// Returns an error when the SQL query fails.
pub fn lookup_source_state(conn: &Connection, source_id: &str) -> Result<Option<SourceStateRow>> {
    let mut stmt = conn
        .prepare(
            "SELECT source_id, raw_path, raw_hash, extracted_hash, body_hash, last_compiled_at_millis
             FROM source_state WHERE source_id = ?1",
        )
        .context("prepare source_state lookup")?;
    let row = stmt
        .query_row(params![source_id], |row| {
            Ok(SourceStateRow {
                source_id: row.get(0)?,
                raw_path: row.get(1)?,
                raw_hash: row.get(2)?,
                extracted_hash: row.get(3)?,
                body_hash: row.get(4)?,
                last_compiled_at_millis: row.get(5)?,
            })
        })
        .optional()
        .context("query source_state lookup")?;
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn open_state_db_is_idempotent_across_calls() {
        let temp = tempdir().expect("tempdir");
        // First open creates the schema.
        {
            let _conn = open_state_db(temp.path()).expect("first open");
        }
        // Second open against the same path must not fail or wipe rows.
        {
            let conn = open_state_db(temp.path()).expect("second open");
            upsert_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1", "hash-v1", 100)
                .expect("seed row");
        }
        // Third open: row from second call must still be readable.
        {
            let conn = open_state_db(temp.path()).expect("third open");
            let record = lookup_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1")
                .expect("lookup")
                .expect("row should still exist");
            assert_eq!(record.input_hash, "hash-v1");
            assert_eq!(record.last_run_at_millis, Some(100));
        }
    }

    #[test]
    fn upsert_and_lookup_stage_round_trips() {
        let temp = tempdir().expect("tempdir");
        let conn = open_state_db(temp.path()).expect("open");

        // Missing key → None.
        assert!(
            lookup_stage(&conn, STAGE_SOURCE_SUMMARY, "src-missing")
                .expect("lookup")
                .is_none()
        );

        // Insert.
        upsert_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1", "hash-v1", 100)
            .expect("insert");
        let row = lookup_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(row.input_hash, "hash-v1");
        assert_eq!(row.last_run_at_millis, Some(100));

        // Update (same primary key) overwrites previous value.
        upsert_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1", "hash-v2", 200)
            .expect("update");
        let row = lookup_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(row.input_hash, "hash-v2");
        assert_eq!(row.last_run_at_millis, Some(200));

        // Different stage/entity does not collide with src-1's row.
        upsert_stage(
            &conn,
            STAGE_CONCEPT_EXTRACTION,
            "src-1",
            "hash-extract",
            300,
        )
        .expect("insert different stage");
        let row = lookup_stage(&conn, STAGE_CONCEPT_EXTRACTION, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(row.input_hash, "hash-extract");
        // Original row untouched.
        let row = lookup_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(row.input_hash, "hash-v2");
    }

    #[test]
    fn fingerprint_changes_when_any_input_changes() {
        let baseline = fingerprint_inputs(&[b"body-v1", b"template-1", b"model-x"]);
        let body_changed = fingerprint_inputs(&[b"body-v2", b"template-1", b"model-x"]);
        let template_changed = fingerprint_inputs(&[b"body-v1", b"template-2", b"model-x"]);
        let model_changed = fingerprint_inputs(&[b"body-v1", b"template-1", b"model-y"]);
        let same_again = fingerprint_inputs(&[b"body-v1", b"template-1", b"model-x"]);

        // Same inputs → same fingerprint.
        assert_eq!(baseline, same_again);
        // Any single component change → new fingerprint.
        assert_ne!(baseline, body_changed);
        assert_ne!(baseline, template_changed);
        assert_ne!(baseline, model_changed);
    }

    #[test]
    fn fingerprint_distinguishes_split_at_separator() {
        // Without the NUL separator, `(b"foo", b"bar")` would hash the
        // same as `(b"fooba", b"r")`. Verify the separator survives.
        let split_a = fingerprint_inputs(&[b"foo", b"bar"]);
        let split_b = fingerprint_inputs(&[b"fooba", b"r"]);
        assert_ne!(split_a, split_b);
    }

    #[test]
    fn skip_signal_matches_when_inputs_unchanged_and_misses_when_changed() {
        let temp = tempdir().expect("tempdir");
        let conn = open_state_db(temp.path()).expect("open");

        // Initial run records a fingerprint.
        let v1 = fingerprint_inputs(&[b"body-v1", b"template-1", b"model"]);
        upsert_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1", &v1, 100).expect("upsert");

        // Recompute: same inputs → fingerprint matches → skip.
        let v1_again = fingerprint_inputs(&[b"body-v1", b"template-1", b"model"]);
        let recorded = lookup_stage(&conn, STAGE_SOURCE_SUMMARY, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(
            recorded.input_hash, v1_again,
            "matching inputs must produce a matching fingerprint",
        );

        // Mutate the body: fingerprint changes → re-run required.
        let v2 = fingerprint_inputs(&[b"body-v2", b"template-1", b"model"]);
        assert_ne!(
            recorded.input_hash, v2,
            "changing the source body must invalidate the cached fingerprint",
        );
    }

    #[test]
    fn list_stage_records_returns_rows_in_stable_order() {
        let temp = tempdir().expect("tempdir");
        let conn = open_state_db(temp.path()).expect("open");

        upsert_stage(&conn, STAGE_CONCEPT_EXTRACTION, "src-b", "h-b", 200).expect("up1");
        upsert_stage(&conn, STAGE_SOURCE_SUMMARY, "src-a", "h-a", 100).expect("up2");
        upsert_stage(&conn, STAGE_SOURCE_SUMMARY, "src-c", "h-c", 300).expect("up3");

        let rows = list_stage_records(&conn).expect("list");
        let ids: Vec<(String, String)> = rows
            .iter()
            .map(|r| (r.stage.clone(), r.entity_id.clone()))
            .collect();
        assert_eq!(
            ids,
            vec![
                (STAGE_CONCEPT_EXTRACTION.to_string(), "src-b".to_string()),
                (STAGE_SOURCE_SUMMARY.to_string(), "src-a".to_string()),
                (STAGE_SOURCE_SUMMARY.to_string(), "src-c".to_string()),
            ],
        );
    }

    #[test]
    fn source_state_round_trips() {
        let temp = tempdir().expect("tempdir");
        let conn = open_state_db(temp.path()).expect("open");

        let row = SourceStateRow {
            source_id: "src-1".to_string(),
            raw_path: Some("/tmp/raw.md".to_string()),
            raw_hash: Some("rawh".to_string()),
            extracted_hash: Some("exth".to_string()),
            body_hash: Some("bodh".to_string()),
            last_compiled_at_millis: Some(42),
        };
        upsert_source_state(&conn, &row).expect("upsert");

        let loaded = lookup_source_state(&conn, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(loaded, row);

        // Update overwrites.
        let updated = SourceStateRow {
            body_hash: Some("bodh-v2".to_string()),
            last_compiled_at_millis: Some(99),
            ..row
        };
        upsert_source_state(&conn, &updated).expect("upsert v2");
        let loaded = lookup_source_state(&conn, "src-1")
            .expect("lookup")
            .expect("row");
        assert_eq!(loaded.body_hash.as_deref(), Some("bodh-v2"));
        assert_eq!(loaded.last_compiled_at_millis, Some(99));
    }
}
