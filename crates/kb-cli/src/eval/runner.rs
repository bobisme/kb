//! Drives golden queries through the hybrid retriever and assembles
//! per-query + aggregate metrics.
//!
//! The runner performs **no LLM call** — only retrieval. That makes
//! `kb eval run` cheap (sub-second on small kbs) and reproducible: same
//! corpus, same backend, same metrics.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use kb_query::EmbeddingBackend;
use serde::{Deserialize, Serialize};

use super::golden::{GoldenQuery, GoldenSet};
use super::scoring::{mean_reciprocal_rank, ndcg_at_k, precision_at_k};

/// Default rank cutoff for nDCG and the captured `ranking` array.
pub const DEFAULT_TOP_K: usize = 10;

/// Token budget passed to the hybrid retriever. Mirrors the `kb ask`
/// default but is independent of the user's `kb.toml` so eval results are
/// comparable across kbs that have tweaked their `[ask]` budget.
pub const EVAL_TOKEN_BUDGET: u32 = 20_000;

/// Per-query evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Stable id from the golden set.
    pub id: String,
    /// The query text that was run.
    pub query: String,
    /// Precision at 5.
    pub p_at_5: f32,
    /// Precision at 10.
    pub p_at_10: f32,
    /// Reciprocal rank of the first relevant hit (0.0 if none).
    pub mrr: f32,
    /// nDCG at 10 with binary relevance.
    pub ndcg_10: f32,
    /// Top-`DEFAULT_TOP_K` retrieved item ids, in rank order.
    pub ranking: Vec<String>,
    /// Subset of `ranking` that was scored as relevant. Useful for
    /// debugging "why is P@5 low?" without re-running the eval.
    pub relevant_hits: Vec<String>,
}

/// Aggregate metrics across all queries (simple arithmetic mean).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub p_at_5: f32,
    pub p_at_10: f32,
    pub mrr: f32,
    pub ndcg_10: f32,
}

/// Top-level eval-run JSON payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    /// UTC timestamp in ISO-8601 form (with `:` for human readability).
    /// The on-disk filename uses a colon-free variant so it works on
    /// Windows.
    pub run_id: String,
    /// BLAKE3 of `state/indexes/lexical.json` bytes — pins the run to a
    /// specific corpus snapshot. Empty string when the index is missing
    /// (eval was run without `kb compile`).
    pub corpus_hash: String,
    /// `kb_query::EmbeddingBackend::backend_id` for the backend that ran
    /// the queries. Allows side-by-side comparisons across backends.
    pub backend_id: String,
    /// Per-query metrics + ranking.
    pub queries: Vec<QueryResult>,
    /// Aggregate (mean) of the per-query metrics.
    pub aggregate: AggregateMetrics,
}

/// Inputs for a single eval run.
pub struct RunInputs<'a> {
    pub root: &'a Path,
    pub golden: &'a GoldenSet,
    pub options: kb_query::HybridOptions,
    pub backend: &'a kb_query::SemanticBackend,
    pub backend_id: &'a str,
    pub run_id: String,
    pub corpus_hash: String,
}

/// Run every golden query through the hybrid retriever and assemble an
/// [`EvalRun`].
///
/// # Errors
///
/// Returns the first retrieval error. Individual query failures abort
/// the whole run so the user gets a clear stderr message instead of a
/// silently lopsided aggregate.
pub fn run(inputs: &RunInputs<'_>) -> Result<EvalRun> {
    let mut queries = Vec::with_capacity(inputs.golden.query.len());
    for q in &inputs.golden.query {
        queries.push(score_query(q, inputs)?);
    }
    let aggregate = aggregate_metrics(&queries);
    Ok(EvalRun {
        run_id: inputs.run_id.clone(),
        corpus_hash: inputs.corpus_hash.clone(),
        backend_id: inputs.backend_id.to_string(),
        queries,
        aggregate,
    })
}

fn score_query(q: &GoldenQuery, inputs: &RunInputs<'_>) -> Result<QueryResult> {
    let plan = kb_query::plan_retrieval_hybrid_with_backend(
        inputs.root,
        &q.query,
        EVAL_TOKEN_BUDGET,
        inputs.options,
        inputs.backend,
    )
    .with_context(|| format!("retrieval failed for golden query `{}`", q.id))?;

    let ranking: Vec<String> = plan
        .candidates
        .iter()
        .take(DEFAULT_TOP_K)
        .map(|c| c.id.clone())
        .collect();

    let mut binary = Vec::with_capacity(ranking.len());
    let mut relevance = Vec::with_capacity(ranking.len());
    let mut relevant_hits = Vec::new();
    for item_id in &ranking {
        let hit = is_relevant(item_id, &q.expected_sources, &q.expected_concepts);
        binary.push(hit);
        relevance.push(u32::from(hit));
        if hit {
            relevant_hits.push(item_id.clone());
        }
    }

    Ok(QueryResult {
        id: q.id.clone(),
        query: q.query.clone(),
        p_at_5: precision_at_k(&binary, 5),
        p_at_10: precision_at_k(&binary, 10),
        mrr: mean_reciprocal_rank(&binary),
        ndcg_10: ndcg_at_k(&relevance, 10),
        ranking,
        relevant_hits,
    })
}

/// Return `true` when `item_id` matches any expected source or concept.
///
/// Matching is intentionally permissive — `expected_sources` are bare
/// ids like `src-cred`, and `item_id`s look like
/// `wiki/sources/src-cred-authn-token-flow.md`. A substring match handles
/// both forms; case-insensitive to absorb slug-vs-id casing differences.
fn is_relevant(
    item_id: &str,
    expected_sources: &[String],
    expected_concepts: &[String],
) -> bool {
    let lower = item_id.to_ascii_lowercase();
    expected_sources
        .iter()
        .chain(expected_concepts.iter())
        .any(|expected| {
            let needle = expected.trim().to_ascii_lowercase();
            !needle.is_empty() && lower.contains(&needle)
        })
}

#[allow(
    clippy::cast_precision_loss,
    reason = "queries.len() is bounded by the size of golden.toml (< 1000); precision is irrelevant"
)]
fn aggregate_metrics(queries: &[QueryResult]) -> AggregateMetrics {
    if queries.is_empty() {
        return AggregateMetrics::default();
    }
    let n = queries.len() as f32;
    let mut agg = AggregateMetrics::default();
    for q in queries {
        agg.p_at_5 += q.p_at_5;
        agg.p_at_10 += q.p_at_10;
        agg.mrr += q.mrr;
        agg.ndcg_10 += q.ndcg_10;
    }
    agg.p_at_5 /= n;
    agg.p_at_10 /= n;
    agg.mrr /= n;
    agg.ndcg_10 /= n;
    agg
}

/// Hash `state/indexes/lexical.json` to pin the run to a corpus snapshot.
/// Returns an empty string when the file does not exist (no compile yet)
/// — callers translate that into a stderr warning, not an error, so
/// `kb eval run` still works on a fresh KB.
#[must_use]
pub fn compute_corpus_hash(root: &Path) -> String {
    let path = kb_query::lexical_index_path(root);
    let Ok(bytes) = fs::read(&path) else {
        return String::new();
    };
    blake3::hash(&bytes).to_hex().to_string()
}

/// Build a stable `backend_id` for the given backend without holding a
/// long-lived borrow.
#[must_use]
pub fn backend_id_for(backend: &kb_query::SemanticBackend) -> String {
    backend.backend_id().to_string()
}

/// Filesystem layout for `evals/`.
pub struct EvalPaths {
    pub root: PathBuf,
}

impl EvalPaths {
    #[must_use]
    pub fn new(kb_root: &Path) -> Self {
        Self {
            root: kb_root.join("evals"),
        }
    }

    #[must_use]
    pub fn results_dir(&self) -> PathBuf {
        self.root.join("results")
    }

    #[must_use]
    pub fn golden_path(&self) -> PathBuf {
        self.root.join("golden.toml")
    }

    #[must_use]
    pub fn latest_md_path(&self) -> PathBuf {
        self.results_dir().join("latest.md")
    }

    #[must_use]
    pub fn result_json_path(&self, name: &str) -> PathBuf {
        self.results_dir().join(format!("{name}.json"))
    }

    /// Ensure both `evals/` and `evals/results/` exist.
    ///
    /// # Errors
    ///
    /// Returns the first I/O error encountered while creating the directory.
    pub fn ensure(&self) -> Result<()> {
        fs::create_dir_all(self.results_dir())
            .with_context(|| format!("create {}", self.results_dir().display()))?;
        Ok(())
    }
}

/// Generate the timestamp pair: a human-readable `run_id` and a
/// filesystem-safe `file_stem`.
///
/// `run_id` keeps the standard `:` separators (`2026-04-30T12:34:56Z`).
/// `file_stem` replaces them with `-` so the path works on Windows
/// without quoting (`2026-04-30T12-34-56Z`).
#[must_use]
pub fn make_run_id(now_millis: i64) -> (String, String) {
    let secs = now_millis.div_euclid(1000);
    let (year, month, day, hour, minute, second) = epoch_to_utc(secs);
    let id = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z");
    let stem = format!("{year:04}-{month:02}-{day:02}T{hour:02}-{minute:02}-{second:02}Z");
    (id, stem)
}

/// Convert a Unix epoch second count to a UTC `(year, month, day, hour,
/// minute, second)` tuple. Civil-calendar arithmetic only — no
/// time-zone or leap-second handling needed for filesystem timestamps.
///
/// Inputs are bounded by realistic system clocks (<= year 9999), so the
/// numeric range fits in `i32` and `u32` without truncation. The `as`
/// casts are flagged by clippy's pedantic group; the bounds make them
/// safe in this domain, so we silence the lint locally.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    reason = "year fits in i32 and time-of-day in u32 for any realistic system clock"
)]
fn epoch_to_utc(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    // Howard Hinnant's days_from_civil inverse, public-domain reference
    // implementation — sufficient for stamping result filenames.
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let hour = (secs_of_day / 3600) as u32;
    let minute = ((secs_of_day % 3600) / 60) as u32;
    let second = (secs_of_day % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if month <= 2 { y + 1 } else { y };
    (year as i32, month, day, hour, minute, second)
}

/// Write the run's JSON payload to `evals/results/<stem>.json` and a
/// human-readable summary to `evals/results/latest.md`.
///
/// Returns the JSON path that was written so callers can include it in
/// the user-facing summary.
///
/// # Errors
///
/// Returns the first I/O error encountered while writing.
pub fn write_run(paths: &EvalPaths, run: &EvalRun, file_stem: &str) -> Result<PathBuf> {
    paths.ensure()?;
    let json_path = paths.result_json_path(file_stem);
    let json = serde_json::to_string_pretty(run).context("serialize eval run")?;
    fs::write(&json_path, json)
        .with_context(|| format!("write eval result json {}", json_path.display()))?;
    let md = render_latest_markdown(run);
    fs::write(paths.latest_md_path(), md)
        .with_context(|| format!("write eval markdown {}", paths.latest_md_path().display()))?;
    Ok(json_path)
}

/// Render the per-run markdown table written to `evals/results/latest.md`.
#[must_use]
pub fn render_latest_markdown(run: &EvalRun) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(out, "# kb eval run {}", run.run_id).ok();
    writeln!(out).ok();
    writeln!(out, "- backend: `{}`", run.backend_id).ok();
    writeln!(
        out,
        "- corpus_hash: `{}`",
        if run.corpus_hash.is_empty() {
            "(no compile yet)"
        } else {
            &run.corpus_hash
        }
    )
    .ok();
    writeln!(out).ok();
    writeln!(out, "## Per-query metrics").ok();
    writeln!(out).ok();
    writeln!(out, "| id | P@5 | P@10 | MRR | nDCG@10 |").ok();
    writeln!(out, "|----|-----|------|-----|---------|").ok();
    for q in &run.queries {
        writeln!(
            out,
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} |",
            q.id, q.p_at_5, q.p_at_10, q.mrr, q.ndcg_10
        )
        .ok();
    }
    writeln!(out).ok();
    writeln!(out, "## Aggregate").ok();
    writeln!(out).ok();
    writeln!(out, "| metric | value |").ok();
    writeln!(out, "|--------|-------|").ok();
    writeln!(out, "| P@5 | {:.3} |", run.aggregate.p_at_5).ok();
    writeln!(out, "| P@10 | {:.3} |", run.aggregate.p_at_10).ok();
    writeln!(out, "| MRR | {:.3} |", run.aggregate.mrr).ok();
    writeln!(out, "| nDCG@10 | {:.3} |", run.aggregate.ndcg_10).ok();
    out
}

/// Render the side-by-side `--baseline` diff table.
#[must_use]
pub fn render_baseline_diff(current: &EvalRun, baseline: &EvalRun, baseline_name: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(
        out,
        "# Eval diff: {} (current) vs {} (baseline)",
        current.run_id, baseline_name
    )
    .ok();
    if current.corpus_hash != baseline.corpus_hash {
        writeln!(
            out,
            "\n> WARNING: corpus_hash differs ({} -> {}); diff is not apples-to-apples.",
            short_hash(&baseline.corpus_hash),
            short_hash(&current.corpus_hash),
        )
        .ok();
    }
    writeln!(out).ok();

    // Per-query: union of ids, baseline first then any new ones.
    let mut ids: Vec<String> = baseline.queries.iter().map(|q| q.id.clone()).collect();
    for q in &current.queries {
        if !ids.contains(&q.id) {
            ids.push(q.id.clone());
        }
    }
    writeln!(
        out,
        "| id | P@5 (cur) | P@5 (base) | dP@5 | nDCG (cur) | nDCG (base) | dnDCG | MRR (cur) | MRR (base) | dMRR |"
    )
    .ok();
    writeln!(
        out,
        "|----|-----------|------------|------|------------|-------------|-------|-----------|------------|------|"
    )
    .ok();
    for id in &ids {
        let cur = current.queries.iter().find(|q| &q.id == id);
        let base = baseline.queries.iter().find(|q| &q.id == id);
        let row = diff_row(id, cur, base);
        writeln!(out, "{row}").ok();
    }
    writeln!(out).ok();
    writeln!(out, "## Aggregate").ok();
    writeln!(out).ok();
    writeln!(out, "| metric | current | baseline | delta |").ok();
    writeln!(out, "|--------|---------|----------|-------|").ok();
    write_agg_row(&mut out, "P@5", current.aggregate.p_at_5, baseline.aggregate.p_at_5);
    write_agg_row(&mut out, "P@10", current.aggregate.p_at_10, baseline.aggregate.p_at_10);
    write_agg_row(&mut out, "MRR", current.aggregate.mrr, baseline.aggregate.mrr);
    write_agg_row(&mut out, "nDCG@10", current.aggregate.ndcg_10, baseline.aggregate.ndcg_10);
    out
}

fn diff_row(id: &str, cur: Option<&QueryResult>, base: Option<&QueryResult>) -> String {
    let cp5 = cur.map(|q| q.p_at_5);
    let bp5 = base.map(|q| q.p_at_5);
    let cnd = cur.map(|q| q.ndcg_10);
    let bnd = base.map(|q| q.ndcg_10);
    let cmr = cur.map(|q| q.mrr);
    let bmr = base.map(|q| q.mrr);
    format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        id,
        fmt_opt(cp5),
        fmt_opt(bp5),
        fmt_delta(cp5, bp5),
        fmt_opt(cnd),
        fmt_opt(bnd),
        fmt_delta(cnd, bnd),
        fmt_opt(cmr),
        fmt_opt(bmr),
        fmt_delta(cmr, bmr),
    )
}

fn write_agg_row(out: &mut String, label: &str, cur: f32, base: f32) {
    use std::fmt::Write as _;
    let delta = cur - base;
    let sign = if delta >= 0.0 { "+" } else { "" };
    writeln!(
        out,
        "| {label} | {cur:.3} | {base:.3} | {sign}{delta:.3} |"
    )
    .ok();
}

fn fmt_opt(v: Option<f32>) -> String {
    v.map_or_else(|| "—".to_string(), |x| format!("{x:.3}"))
}

fn fmt_delta(cur: Option<f32>, base: Option<f32>) -> String {
    match (cur, base) {
        (Some(c), Some(b)) => {
            let d = c - b;
            let sign = if d >= 0.0 { "+" } else { "" };
            format!("{sign}{d:.3}")
        }
        _ => "—".to_string(),
    }
}

fn short_hash(h: &str) -> String {
    if h.is_empty() {
        return "(none)".to_string();
    }
    let take = h.len().min(12);
    h[..take].to_string()
}

/// Load a previously-written run from `evals/results/<name>.json`.
///
/// `name` may be passed with or without the `.json` extension — both
/// `latest` and `latest.json` resolve to the same file.
///
/// # Errors
///
/// Returns an error when the file cannot be read or the JSON cannot be
/// parsed as an [`EvalRun`].
pub fn load_run(paths: &EvalPaths, name: &str) -> Result<EvalRun> {
    let stem = name.strip_suffix(".json").unwrap_or(name);
    let path = paths.result_json_path(stem);
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("read baseline result {}", path.display()))?;
    let run: EvalRun = serde_json::from_str(&raw)
        .with_context(|| format!("parse baseline result {}", path.display()))?;
    Ok(run)
}

/// List previously-written runs in `evals/results/` sorted by mtime
/// (newest first). Returns an empty vec if the directory does not exist.
///
/// # Errors
///
/// Returns an error when the directory exists but cannot be read.
pub fn list_runs(paths: &EvalPaths) -> Result<Vec<RunListing>> {
    let dir = paths.results_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("list {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let meta = entry.metadata()?;
        let mtime = meta.modified().ok();
        out.push(RunListing {
            name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string(),
            path,
            mtime,
        });
    }
    out.sort_by_key(|r| std::cmp::Reverse(r.mtime));
    Ok(out)
}

/// One entry in [`list_runs`]'s output.
pub struct RunListing {
    pub name: String,
    pub path: PathBuf,
    pub mtime: Option<std::time::SystemTime>,
}

/// Copy a previously-written run JSON to `evals/results/<save_as>.json`.
/// Used by `kb eval run --save-as <name>`.
///
/// # Errors
///
/// Returns an error if the source path does not exist or the copy fails.
pub fn save_as(paths: &EvalPaths, source: &Path, save_as_name: &str) -> Result<PathBuf> {
    let target = paths.result_json_path(save_as_name);
    if !source.exists() {
        bail!("source result file does not exist: {}", source.display());
    }
    fs::copy(source, &target)
        .with_context(|| format!("copy result to {}", target.display()))?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relevance_substring_match_on_source_id() {
        let item_id = "wiki/sources/src-cred-authn-token-flow.md";
        assert!(is_relevant(
            item_id,
            &["src-cred".to_string()],
            &[],
        ));
    }

    #[test]
    fn relevance_substring_match_on_concept() {
        let item_id = "wiki/concepts/authentication.md";
        assert!(is_relevant(
            item_id,
            &[],
            &["authentication".to_string()],
        ));
    }

    #[test]
    fn relevance_no_match() {
        let item_id = "wiki/sources/src-frontend-rendering.md";
        assert!(!is_relevant(
            item_id,
            &["src-cred".to_string()],
            &["authentication".to_string()],
        ));
    }

    #[test]
    fn relevance_case_insensitive() {
        let item_id = "wiki/sources/SRC-CRED-AUTHN.md";
        assert!(is_relevant(
            item_id,
            &["src-cred".to_string()],
            &[],
        ));
    }

    #[test]
    fn relevance_ignores_empty_expected_strings() {
        // An empty expected_sources entry must NOT match every item id.
        // Otherwise a typo'd golden.toml would silently mark every result
        // as relevant.
        let item_id = "wiki/sources/anything.md";
        assert!(!is_relevant(item_id, &[String::new()], &[]));
    }

    #[test]
    fn aggregate_averages_across_queries() {
        let queries = vec![
            QueryResult {
                id: "a".to_string(),
                query: "q".to_string(),
                p_at_5: 1.0,
                p_at_10: 0.5,
                mrr: 1.0,
                ndcg_10: 1.0,
                ranking: vec![],
                relevant_hits: vec![],
            },
            QueryResult {
                id: "b".to_string(),
                query: "q".to_string(),
                p_at_5: 0.0,
                p_at_10: 0.0,
                mrr: 0.0,
                ndcg_10: 0.0,
                ranking: vec![],
                relevant_hits: vec![],
            },
        ];
        let agg = aggregate_metrics(&queries);
        assert!((agg.p_at_5 - 0.5).abs() < 1e-6);
        assert!((agg.p_at_10 - 0.25).abs() < 1e-6);
        assert!((agg.mrr - 0.5).abs() < 1e-6);
        assert!((agg.ndcg_10 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn aggregate_empty_returns_zeros() {
        let agg = aggregate_metrics(&[]);
        assert!(agg.p_at_5.abs() < f32::EPSILON);
        assert!(agg.ndcg_10.abs() < f32::EPSILON);
    }

    #[test]
    fn render_markdown_includes_aggregate_and_rows() {
        let run = EvalRun {
            run_id: "2026-04-30T12:34:56Z".to_string(),
            corpus_hash: "deadbeef".to_string(),
            backend_id: "hash-embed-256".to_string(),
            queries: vec![QueryResult {
                id: "auth".to_string(),
                query: "how does authentication work?".to_string(),
                p_at_5: 0.6,
                p_at_10: 0.5,
                mrr: 0.5,
                ndcg_10: 0.78,
                ranking: vec!["wiki/sources/src-cred-x.md".to_string()],
                relevant_hits: vec!["wiki/sources/src-cred-x.md".to_string()],
            }],
            aggregate: AggregateMetrics {
                p_at_5: 0.6,
                p_at_10: 0.5,
                mrr: 0.5,
                ndcg_10: 0.78,
            },
        };
        let md = render_latest_markdown(&run);
        assert!(md.contains("auth"));
        assert!(md.contains("0.600"));
        assert!(md.contains("hash-embed-256"));
        assert!(md.contains("Aggregate"));
    }

    #[test]
    fn baseline_diff_warns_on_corpus_change() {
        let cur = EvalRun {
            run_id: "now".to_string(),
            corpus_hash: "aaaa".to_string(),
            backend_id: "hash-embed-256".to_string(),
            queries: vec![],
            aggregate: AggregateMetrics::default(),
        };
        let mut base = cur.clone();
        base.corpus_hash = "bbbb".to_string();
        base.run_id = "before".to_string();
        let md = render_baseline_diff(&cur, &base, "before");
        assert!(md.contains("WARNING"), "got: {md}");
        assert!(md.contains("corpus_hash"), "got: {md}");
    }

    #[test]
    fn run_id_round_trip_is_iso8601_like() {
        // 2024-01-15T10:20:30Z -> epoch 1_705_314_030 s.
        let secs: i64 = 1_705_314_030;
        let (id, stem) = make_run_id(secs * 1_000);
        assert_eq!(id, "2024-01-15T10:20:30Z");
        assert_eq!(stem, "2024-01-15T10-20-30Z");
    }

    #[test]
    fn run_id_handles_epoch_zero() {
        // Sanity: 1970-01-01T00:00:00Z is a fixed point for epoch math.
        let (id, stem) = make_run_id(0);
        assert_eq!(id, "1970-01-01T00:00:00Z");
        assert_eq!(stem, "1970-01-01T00-00-00Z");
    }
}
