//! Golden Q/A evaluation harness for kb's hybrid retriever (bn-3sco).
//!
//! `kb eval run` runs every query in `<kb-root>/evals/golden.toml` against
//! `kb_query::plan_retrieval_hybrid_with_backend` (no LLM call) and scores
//! the top-10 ranking against the expected sources/concepts using P@K,
//! MRR, and nDCG@10. Results are written to
//! `<kb-root>/evals/results/<UTC-timestamp>.json` plus a `latest.md`
//! summary.
//!
//! The harness is read-only — it never touches the wiki, never calls an
//! LLM, and never re-builds the index. Re-run after `kb compile` whenever
//! you tune retrieval knobs (`[retrieval]` in `kb.toml`, embedding
//! backend, RRF k, etc.) and compare against a saved baseline.

pub mod golden;
pub mod runner;
pub mod scoring;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::config::Config;
use golden::GoldenSet;
use runner::{
    EvalPaths, RunInputs, backend_id_for, compute_corpus_hash, list_runs, load_run, make_run_id,
    render_baseline_diff, run as run_inner, save_as as save_as_copy, write_run,
};

/// Flags wired from the clap subcommand for `kb eval run`.
#[derive(Debug, Default)]
pub struct RunFlags {
    pub baseline: Option<String>,
    pub save_as: Option<String>,
    pub json: bool,
}

/// Entry point for `kb eval run`.
///
/// Loads `evals/golden.toml`, runs every query through the hybrid
/// retriever, writes a JSON+markdown report, and prints a one-line
/// summary. With `--baseline <name>` also prints the diff table.
///
/// # Errors
///
/// Returns an error when the golden set is missing/malformed, when
/// retrieval fails, or when the result files cannot be written.
pub fn cmd_run(root: &Path, flags: &RunFlags) -> Result<()> {
    let cfg = Config::load_from_root(root, None)?;
    let backend_config = cfg.semantic.to_backend_config();
    let backend = kb_query::SemanticBackend::from_config(&backend_config)?;
    // bn-1cp2: fold rerank settings into the eval options so opt-in
    // benchmarks vs. baseline are a one-flag flip in kb.toml.
    let options = cfg.to_hybrid_options(backend_config.kind);
    let reranker = crate::load_optional_reranker(&cfg)?;

    let paths = EvalPaths::new(root);
    let golden_path = paths.golden_path();
    let golden = GoldenSet::load(&golden_path).with_context(|| {
        format!(
            "load golden set from {}\n\
             hint: copy `crates/kb-cli/src/eval/golden.toml.example` from the kb repo to {} to bootstrap",
            golden_path.display(),
            golden_path.display()
        )
    })?;

    let now_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0u128, |d| d.as_millis());
    // Realistic system clocks (post-2025) live well within `i64::MAX` ms,
    // so the saturating cast keeps the timestamp meaningful without
    // panicking on weird clocks.
    let now: i64 = i64::try_from(now_millis).unwrap_or(i64::MAX);
    let (run_id, file_stem) = make_run_id(now);

    let corpus_hash = compute_corpus_hash(root);
    if corpus_hash.is_empty() {
        eprintln!(
            "warning: no lexical index at {} — run `kb compile` first for meaningful results",
            kb_query::lexical_index_path(root).display()
        );
    }

    let backend_id = backend_id_for(&backend);
    let inputs = RunInputs {
        root,
        golden: &golden,
        options,
        backend: &backend,
        backend_id: &backend_id,
        run_id,
        corpus_hash,
        reranker: reranker.as_deref(),
    };
    let result = run_inner(&inputs)?;

    let json_path = write_run(&paths, &result, &file_stem)?;

    if let Some(name) = &flags.save_as {
        let saved = save_as_copy(&paths, &json_path, name)?;
        eprintln!("saved as {}", saved.display());
    }

    if let Some(baseline_name) = &flags.baseline {
        match load_run(&paths, baseline_name) {
            Ok(baseline) => {
                let diff = render_baseline_diff(&result, &baseline, baseline_name);
                println!("{diff}");
            }
            Err(err) => {
                eprintln!("warning: could not load baseline `{baseline_name}`: {err:#}");
            }
        }
    }

    if flags.json {
        crate::emit_json("eval-run", &result)?;
    } else {
        println!(
            "kb eval run: {} queries — P@5 {:.3}, P@10 {:.3}, MRR {:.3}, nDCG@10 {:.3}",
            result.queries.len(),
            result.aggregate.p_at_5,
            result.aggregate.p_at_10,
            result.aggregate.mrr,
            result.aggregate.ndcg_10
        );
        println!("  result: {}", json_path.display());
        println!("  latest.md: {}", paths.latest_md_path().display());
    }
    Ok(())
}

/// Entry point for `kb eval list`.
///
/// # Errors
///
/// Returns an error when `evals/results/` exists but cannot be enumerated.
pub fn cmd_list(root: &Path, json: bool) -> Result<()> {
    let paths = EvalPaths::new(root);
    let runs = list_runs(&paths)?;
    if json {
        let payload: Vec<serde_json::Value> = runs
            .iter()
            .map(|r| {
                serde_json::json!({
                    "name": r.name,
                    "path": r.path,
                })
            })
            .collect();
        crate::emit_json("eval-list", &payload)?;
        return Ok(());
    }
    if runs.is_empty() {
        println!(
            "no eval runs found under {} (run `kb eval run` first)",
            paths.results_dir().display()
        );
        return Ok(());
    }
    println!("eval runs under {}:", paths.results_dir().display());
    for (i, r) in runs.iter().enumerate() {
        let marker = if i == 0 { " (latest)" } else { "" };
        println!("  {}{}", r.name, marker);
    }
    Ok(())
}
