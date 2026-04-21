//! `kb jobs` subcommand handlers: list + prune over `state/jobs/*.json`.
//!
//! `kb status` surfaces interrupted/failed job runs forever (bn-2cr capped
//! the display, bn-3qn caps interrupted the same way). This module gives
//! users an explicit, auditable way to *clean up* that accumulated state.
//!
//! ## Safety: prune is a `mv`, not an `rm`
//!
//! `prune` never deletes — it moves manifests (and their `.log` sidecars)
//! into `trash/jobs-<timestamp>/` at the KB root. Logs can still be
//! needed for forensics (e.g. reviewing why a batch of ingests kept
//! getting SIGINT'd), so a user who prunes too eagerly can always
//! `mv trash/jobs-<ts>/* state/jobs/` to undo.
//!
//! ## Locking
//!
//! `run_prune` is invoked under `execute_mutating_command`, which holds
//! the root lock for the duration. That serializes prune against any
//! concurrent `kb compile` / `kb ingest` that might be writing new job
//! manifests. Readers (`kb status`, `kb doctor`) don't need the lock —
//! they tolerate a partially-empty `state/jobs/` directory (manifest
//! either loads or doesn't; `recent_jobs` returns whatever's on disk).
//!
//! ## Filter semantics
//!
//! `list` and `prune` both require at least one status selector
//! (`--interrupted` / `--failed` / `--all`). The zero-selector case is
//! a user error — "prune nothing" is a no-op worth flagging, not a
//! silent success.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use kb_core::{JobRun, JobRunStatus, trash_dir};
use serde::Serialize;

use crate::emit_json;
use crate::jobs::{self, jobs_dir};

/// Hard cap on `recent_jobs` scan depth. Matches the existing
/// `check_interrupted_jobs`/status bounds — 1000 is well beyond any real
/// workflow and keeps pathological `state/jobs/` directories from
/// bogging down list/prune.
const MAX_SCAN: usize = 10_000;

/// One line in `kb jobs list` output (text + JSON). Separate from
/// `JobRun` so we can include the resolved on-disk log path without
/// leaking `EntityMetadata` internals.
#[derive(Debug, Serialize)]
struct JobListEntry {
    id: String,
    command: String,
    status: JobRunStatus,
    started_at_millis: u64,
    ended_at_millis: Option<u64>,
    duration_ms: Option<u64>,
    exit_code: Option<i32>,
    /// Best-effort: `state/jobs/<id>.log` if the sidecar exists on disk.
    /// `None` for manifests whose log was already cleaned up or whose
    /// job never wrote a log (e.g. early-abort dry runs).
    log_path: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct ListPayload {
    page: usize,
    page_size: usize,
    total: usize,
    jobs: Vec<JobListEntry>,
}

/// `kb jobs list --interrupted|--failed`.
pub fn run_list(
    root: &Path,
    interrupted: bool,
    failed: bool,
    page: usize,
    page_size: usize,
    json: bool,
) -> Result<()> {
    if !interrupted && !failed {
        bail!(
            "kb jobs list: specify at least one of --interrupted, --failed"
        );
    }
    if page == 0 || page_size == 0 {
        bail!("kb jobs list: --page and --page-size must be >= 1");
    }

    let statuses = selected_statuses(interrupted, failed, false);
    let all = jobs::recent_jobs(root, MAX_SCAN)?
        .into_iter()
        .filter(|j| statuses.contains(&j.status))
        .collect::<Vec<_>>();
    let total = all.len();

    let start = page.saturating_sub(1).saturating_mul(page_size);
    let slice: Vec<JobListEntry> = all
        .into_iter()
        .skip(start)
        .take(page_size)
        .map(|job| to_list_entry(root, &job))
        .collect();

    if json {
        emit_json(
            "jobs.list",
            ListPayload {
                page,
                page_size,
                total,
                jobs: slice,
            },
        )?;
        return Ok(());
    }

    if total == 0 {
        println!("No matching job runs.");
        return Ok(());
    }
    println!(
        "job runs ({} total, page {}/{}, page_size {}):",
        total,
        page,
        total.div_ceil(page_size).max(1),
        page_size,
    );
    for entry in &slice {
        let duration_str = entry
            .duration_ms
            .map_or_else(|| "running".to_string(), |ms| format!("{ms}ms"));
        let log = entry
            .log_path
            .as_ref()
            .map_or_else(|| "-".to_string(), |p| p.display().to_string());
        println!(
            "  {} | {:<11} | {} [{}]  log={}",
            entry.id,
            format!("{:?}", entry.status),
            entry.command,
            duration_str,
            log,
        );
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct PruneReport {
    pruned: usize,
    trash_dir: Option<PathBuf>,
    /// Manifests we considered but left in place (newer than `older_than`).
    kept: usize,
}

/// `kb jobs prune --interrupted|--failed|--all [--older-than DAYS]`.
///
/// The caller holds the root lock. See module docs for safety notes.
#[allow(clippy::fn_params_excessive_bools)]
pub fn run_prune(
    root: &Path,
    interrupted: bool,
    failed: bool,
    all: bool,
    older_than_days: u64,
    json: bool,
) -> Result<()> {
    let interrupted = interrupted || all;
    let failed = failed || all;
    if !interrupted && !failed {
        bail!(
            "kb jobs prune: specify at least one of --interrupted, --failed, --all"
        );
    }

    let statuses = selected_statuses(interrupted, failed, false);
    let cutoff_millis = cutoff_millis(older_than_days);

    // `recent_jobs` already sorts newest-first and handles a missing
    // `state/jobs/` gracefully (returns empty).
    let candidates: Vec<JobRun> = jobs::recent_jobs(root, MAX_SCAN)?
        .into_iter()
        .filter(|j| statuses.contains(&j.status))
        .collect();

    if candidates.is_empty() {
        if json {
            emit_json(
                "jobs.prune",
                PruneReport {
                    pruned: 0,
                    trash_dir: None,
                    kept: 0,
                },
            )?;
        } else {
            println!("No matching job runs to prune.");
        }
        return Ok(());
    }

    let jobs_root = jobs_dir(root);
    let mut to_prune: Vec<&JobRun> = Vec::new();
    let mut kept = 0usize;
    for job in &candidates {
        if job.started_at_millis <= cutoff_millis {
            to_prune.push(job);
        } else {
            kept += 1;
        }
    }

    if to_prune.is_empty() {
        if json {
            emit_json(
                "jobs.prune",
                PruneReport {
                    pruned: 0,
                    trash_dir: None,
                    kept,
                },
            )?;
        } else {
            println!(
                "No job runs older than {older_than_days} day(s) to prune ({kept} newer kept)."
            );
        }
        return Ok(());
    }

    // Create a timestamped trash bucket. Using millis guarantees no
    // collision even if two prune calls land in the same second (the
    // root lock would serialize them anyway, but the naming is
    // independently defensive).
    let trash_bucket = trash_dir(root).join(format!("jobs-{}", now_millis()));
    fs::create_dir_all(&trash_bucket)
        .with_context(|| format!("create trash dir {}", trash_bucket.display()))?;

    let mut pruned = 0usize;
    for job in &to_prune {
        let id = &job.metadata.id;
        let manifest = jobs_root.join(format!("{id}.json"));
        let log = jobs_root.join(format!("{id}.log"));

        // Move manifest. If it's already gone (concurrent cleanup,
        // partial prior run) skip and count as pruned so the user isn't
        // confused by a phantom "kept" entry.
        if manifest.exists() {
            let dest = trash_bucket.join(format!("{id}.json"));
            move_file(&manifest, &dest).with_context(|| {
                format!("move {} -> {}", manifest.display(), dest.display())
            })?;
        }
        // Move sidecar log if present. Missing logs are fine — early
        // aborts (e.g. lock-timeout jobs) never wrote one.
        if log.exists() {
            let dest = trash_bucket.join(format!("{id}.log"));
            move_file(&log, &dest).with_context(|| {
                format!("move {} -> {}", log.display(), dest.display())
            })?;
        }
        pruned += 1;
    }

    if json {
        emit_json(
            "jobs.prune",
            PruneReport {
                pruned,
                trash_dir: Some(trash_bucket),
                kept,
            },
        )?;
        return Ok(());
    }
    println!(
        "Pruned {pruned} job run(s) to {}{}",
        trash_bucket.display(),
        if kept > 0 {
            format!(" ({kept} newer than {older_than_days} day(s) kept)")
        } else {
            String::new()
        }
    );
    Ok(())
}

fn selected_statuses(
    interrupted: bool,
    failed: bool,
    include_other: bool,
) -> Vec<JobRunStatus> {
    let mut v = Vec::new();
    if interrupted {
        v.push(JobRunStatus::Interrupted);
    }
    if failed {
        v.push(JobRunStatus::Failed);
    }
    if include_other {
        v.push(JobRunStatus::Succeeded);
        v.push(JobRunStatus::Running);
    }
    v
}

fn to_list_entry(root: &Path, job: &JobRun) -> JobListEntry {
    let duration_ms = job
        .ended_at_millis
        .map(|e| e.saturating_sub(job.started_at_millis));
    // Prefer the manifest's own `log_path` field when present (absolute
    // or repo-relative as the job wrote it), falling back to probing
    // `state/jobs/<id>.log` on disk. A `None` result signals "no log
    // available" cleanly for the JSON consumer.
    let log_path = job.log_path.clone().or_else(|| {
        let probe = jobs_dir(root).join(format!("{}.log", job.metadata.id));
        probe.exists().then_some(probe)
    });
    JobListEntry {
        id: job.metadata.id.clone(),
        command: job.command.clone(),
        status: job.status,
        started_at_millis: job.started_at_millis,
        ended_at_millis: job.ended_at_millis,
        duration_ms,
        exit_code: job.exit_code,
        log_path,
    }
}

/// Compute the unix-millis cutoff: manifests started at-or-before this
/// instant are prune candidates. `days == 0` means "everything", which
/// the test acceptance exercises and which is also the obvious
/// workflow after a developer wants a total reset.
fn cutoff_millis(days: u64) -> u64 {
    if days == 0 {
        return u64::MAX;
    }
    let now = now_millis();
    let window_ms = days.saturating_mul(24).saturating_mul(3_600_000);
    now.saturating_sub(window_ms)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| {
            d.as_secs()
                .saturating_mul(1_000)
                .saturating_add(u64::from(d.subsec_millis()))
        })
}

/// Best-effort atomic rename, falling back to copy+remove for cross-
/// filesystem moves (e.g. `/tmp` tmpdirs vs. the KB root on a
/// different mount — common in integration tests).
fn move_file(src: &Path, dst: &Path) -> Result<()> {
    if let Err(rename_err) = fs::rename(src, dst) {
        // `rename` across mount points returns EXDEV; fall back.
        fs::copy(src, dst).with_context(|| {
            format!(
                "copy {} -> {} after rename failed ({rename_err})",
                src.display(),
                dst.display()
            )
        })?;
        fs::remove_file(src)
            .with_context(|| format!("remove {} after copy", src.display()))?;
    }
    Ok(())
}
