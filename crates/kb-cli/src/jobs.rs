use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use fs2::FileExt;
use kb_compile::pipeline::LogSink;
use kb_core::EntityMetadata;
use kb_core::fs::atomic_write;
use kb_core::{JobRun, JobRunStatus, Status, state_dir};
use serde::{Deserialize, Serialize};

/// `LogSink` backed by a job's on-disk log file. Clones share the same path,
/// so cheap `Arc` sharing across the compile pipeline still writes to the
/// single canonical log. Used by `execute_mutating_command` to plumb the
/// active `JobRun`'s log through `CompileOptions::log_sink`.
#[derive(Debug, Clone)]
pub struct JobLogSink {
    log_path: PathBuf,
}

impl JobLogSink {
    #[must_use]
    pub const fn new(log_path: PathBuf) -> Self {
        Self { log_path }
    }
}

impl LogSink for JobLogSink {
    fn append_log(&self, message: &str) {
        // Best effort: a failed append must never kill the compile. Errors
        // here mean `state/jobs/<id>.log` loses a line, not that a pass
        // couldn't run.
        let _ = append_log_line(&self.log_path, message);
    }

    fn log_path(&self) -> Option<&Path> {
        Some(&self.log_path)
    }
}

impl JobHandle {
    /// Build an `Arc<dyn LogSink>` that writes to this job's log file.
    /// Pass the result into `CompileOptions::log_sink` to have the compile
    /// pipeline stream per-pass events into `state/jobs/<id>.log`.
    #[must_use]
    pub fn log_sink(&self) -> Arc<dyn LogSink> {
        Arc::new(JobLogSink::new(self.log_path.clone()))
    }
}

#[derive(Debug)]
pub struct JobHandle {
    manifest_path: PathBuf,
    log_path: PathBuf,
    pub run: JobRun,
}

#[derive(Debug)]
pub struct KbLock {
    file: File,
    metadata_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockMetadata {
    pub command: String,
    pub pid: u32,
    pub started_at_millis: u64,
}

pub fn jobs_dir(root: &Path) -> PathBuf {
    state_dir(root).join("jobs")
}

fn locks_dir(root: &Path) -> PathBuf {
    state_dir(root).join("locks")
}

fn root_lock_path(root: &Path) -> PathBuf {
    locks_dir(root).join("root.lock")
}

fn root_lock_metadata_path(root: &Path) -> PathBuf {
    locks_dir(root).join("root.lock.json")
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since_epoch| {
            since_epoch
                .as_secs()
                .saturating_mul(1_000)
                .saturating_add(u64::from(since_epoch.subsec_millis()))
        })
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since_epoch| since_epoch.as_nanos())
}

/// Normalize a command label for embedding inside a job id. We want the
/// command token to stay visible in `ls state/jobs/` (bn-221i Option B:
/// `job-{command}-{terseid}`), but the raw command string can contain
/// characters that would confuse filename scanning or terseid parsing —
/// spaces (none today, but cheap to guard), dots (`review.approve`,
/// `jobs.prune`), or anything else non-alphanumeric. Collapse those to
/// single dashes and lowercase so the resulting id is a clean
/// `[a-z0-9-]+` token.
fn sanitize_command_token(command: &str) -> String {
    let mut out = String::with_capacity(command.len());
    let mut last_was_dash = false;
    for ch in command.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            out.push('-');
            last_was_dash = true;
        }
    }
    // Trim leading/trailing dashes so we don't emit `job--foo` or `job-foo-`.
    out.trim_matches('-').to_string()
}

/// Scan `state/jobs/` for existing manifest ids (file stems of `*.json`).
/// Used as the `exists` input to [`terseid::IdGenerator::generate`] so a
/// new job id never collides with one already on disk. Missing directory
/// is treated as "no existing ids".
fn existing_job_ids(jobs_root: &Path) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::new();
    let Ok(read) = fs::read_dir(jobs_root) else {
        return ids;
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            ids.insert(stem.to_string());
        }
    }
    ids
}

/// Generate a fresh job id of the shape `job-{command}-{hash}`
/// (Option B from bn-221i), e.g. `job-ingest-a7x`.
///
/// The terseid seed mixes now-nanos, pid, and the raw command so two
/// concurrent invocations of the same subcommand produce distinct hashes
/// deterministically. The command token is also embedded in the prefix
/// so `ls state/jobs/` is grep-friendly at a glance; the actual process
/// id still lives on the manifest (see `JobRun::pid`), so the stale-job
/// reaper continues to work without parsing the id.
fn next_job_id(jobs_root: &Path, command: &str) -> String {
    let pid = process::id();
    let now = now_nanos();
    let token = sanitize_command_token(command);
    let prefix = if token.is_empty() {
        "job".to_string()
    } else {
        format!("job-{token}")
    };

    let existing = existing_job_ids(jobs_root);
    let generator = terseid::IdGenerator::new(terseid::IdConfig::new(prefix));
    generator.generate(
        |nonce| format!("{now}|{pid}|{command}|{nonce}").into_bytes(),
        existing.len(),
        |candidate| existing.contains(candidate),
    )
}

fn manifest_path_for(root: &Path, command: &str) -> (String, PathBuf, PathBuf) {
    let jobs_root = jobs_dir(root);
    let id = next_job_id(&jobs_root, command);
    (
        id.clone(),
        jobs_root.join(format!("{id}.json")),
        jobs_root.join(format!("{id}.log")),
    )
}

fn load_job_run(path: &Path) -> Result<JobRun> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read job run manifest {}", path.display()))?;
    serde_json::from_str(&raw).context("deserialize job run manifest")
}

fn write_job_run(path: &Path, job: &JobRun) -> Result<()> {
    let json = serde_json::to_vec_pretty(job).context("serialize job run manifest")?;
    atomic_write(path, &json).with_context(|| format!("write job run manifest {}", path.display()))
}

fn append_log_line(path: &Path, message: &str) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{message}")
}

fn load_lock_metadata(path: &Path) -> Result<LockMetadata> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read root lock metadata {}", path.display()))?;
    serde_json::from_str(&raw).context("deserialize root lock metadata")
}

fn write_lock_metadata(path: &Path, metadata: &LockMetadata) -> Result<()> {
    let json = serde_json::to_vec_pretty(metadata).context("serialize root lock metadata")?;
    atomic_write(path, &json)
        .with_context(|| format!("write root lock metadata {}", path.display()))
}

fn format_lock_holder(path: &Path) -> String {
    match load_lock_metadata(path) {
        Ok(metadata) => {
            let alive_note = if is_pid_alive(metadata.pid) {
                ""
            } else {
                " (holder pid is dead; lock metadata is stale)"
            };
            format!(
                "command={} pid={} started_at_millis={}{}",
                metadata.command, metadata.pid, metadata.started_at_millis, alive_note
            )
        }
        Err(_) => "metadata unavailable".to_string(),
    }
}

/// Build the full wait-progress line printed while blocked on the root
/// lock. Reads the sidecar metadata file every call so we never cache a
/// stale holder name across iterations of the acquire loop.
///
/// When the metadata parses, we render "held by pid N running `cmd`" so
/// operators can see exactly which process and invocation is blocking
/// them. When the file is missing or unparseable, we render
/// "holder metadata missing" without the "held by" prefix — the previous
/// fallback said "held by unknown", which looked like a parse bug and
/// gave operators no next action. "holder metadata missing" says what
/// actually went wrong (the sidecar file isn't there or isn't parseable).
///
/// Extracted so tests can exercise formatting without racing the acquire
/// loop.
fn format_wait_message(metadata_path: &Path, elapsed_secs: u64) -> String {
    match load_lock_metadata(metadata_path) {
        Ok(metadata) => format!(
            "waiting for KB root lock (held by pid {} running `{}`, {}s elapsed)",
            metadata.pid, metadata.command, elapsed_secs
        ),
        Err(_) => format!(
            "waiting for KB root lock (holder metadata missing, {elapsed_secs}s elapsed)"
        ),
    }
}

/// If the sidecar metadata records a dead pid, remove it so the next acquire
/// attempt does not report a ghost holder. Returns true when stale metadata
/// was cleaned up.
fn reap_stale_metadata(path: &Path) -> bool {
    match load_lock_metadata(path) {
        Ok(metadata) => {
            if is_pid_alive(metadata.pid) {
                false
            } else {
                fs::remove_file(path).is_ok()
            }
        }
        Err(_) => false,
    }
}

fn is_pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }

    #[cfg(unix)]
    {
        Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stderr(Stdio::null())
            .stdout(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

/// How often to emit a "still waiting" progress line while blocked on the
/// root lock. The interval is intentionally short enough that users and
/// supervising agents see the lock isn't dead, but long enough to avoid
/// drowning stderr.
const WAIT_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

impl KbLock {
    pub fn acquire(root: &Path, command: &str, timeout: Duration) -> Result<Self> {
        Self::acquire_with_progress(root, command, timeout, &mut io::stderr())
    }

    /// Like [`acquire`], but emits wait-progress lines to the provided
    /// `progress` sink instead of process stderr. Primarily for tests.
    pub fn acquire_with_progress<W: Write>(
        root: &Path,
        command: &str,
        timeout: Duration,
        progress: &mut W,
    ) -> Result<Self> {
        let locks_root = locks_dir(root);
        fs::create_dir_all(&locks_root)
            .with_context(|| format!("create locks dir {}", locks_root.display()))?;

        let lock_path = root_lock_path(root);
        let metadata_path = root_lock_metadata_path(root);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("open root lock file {}", lock_path.display()))?;

        let wait_start = Instant::now();
        let deadline = wait_start + timeout;
        let mut last_progress_at: Option<Instant> = None;
        loop {
            match file.try_lock_exclusive() {
                Ok(()) => {
                    let metadata = LockMetadata {
                        command: command.to_string(),
                        pid: process::id(),
                        started_at_millis: now_millis(),
                    };
                    write_lock_metadata(&metadata_path, &metadata)?;
                    return Ok(Self {
                        file,
                        metadata_path,
                    });
                }
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    // The OS releases the advisory lock when a holder dies, but
                    // the sidecar metadata file persists. If the recorded pid is
                    // not alive, reap the stale metadata and retry acquiring
                    // immediately so we never report a ghost holder.
                    if reap_stale_metadata(&metadata_path) {
                        continue;
                    }

                    let now = Instant::now();
                    // Emit the first progress line on the very first WouldBlock
                    // so users immediately know the command is blocked, then
                    // re-emit every WAIT_PROGRESS_INTERVAL to show it's still
                    // waiting (and not hung).
                    let should_emit = last_progress_at
                        .is_none_or(|prev| now.duration_since(prev) >= WAIT_PROGRESS_INTERVAL);
                    if should_emit {
                        let elapsed_secs = now.duration_since(wait_start).as_secs();
                        let line = format_wait_message(&metadata_path, elapsed_secs);
                        let _ = writeln!(progress, "{line}");
                        let _ = progress.flush();
                        last_progress_at = Some(now);
                    }

                    if now >= deadline {
                        // Re-check one last time before building the error so
                        // the message reflects whether the holder is actually
                        // alive right now.
                        if reap_stale_metadata(&metadata_path) {
                            continue;
                        }
                        let holder = format_lock_holder(&metadata_path);
                        return Err(anyhow!(
                            "timed out waiting for KB root lock after {}ms ({holder})",
                            timeout.as_millis()
                        ));
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("acquire root lock {}", lock_path.display()));
                }
            }
        }
    }
}

impl Drop for KbLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.metadata_path);
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
pub fn read_lock_metadata(root: &Path) -> Result<LockMetadata> {
    load_lock_metadata(&root_lock_metadata_path(root))
}

/// Non-acquiring look at the root lock.
///
/// Returns `Some(metadata)` when the root lock appears to be actively held by
/// a live process (the sidecar metadata file exists, parses, and names a pid
/// that responds to `kill -0`). Returns `None` when no metadata is present,
/// when it can't be parsed, or when the recorded pid is dead (stale).
///
/// This intentionally does NOT contest or acquire the advisory file lock — it
/// is a read-only probe used by read-only commands (like `kb lint`) to detect
/// that another mutating command (like `kb compile`) is in flight so they can
/// warn that their view of the tree may be mid-rewrite.
///
/// Callers must not treat a `Some` return as exclusive access; it is purely
/// informational. Two peeks racing with a real acquire can disagree, which is
/// fine — the warning is advisory.
pub fn peek_root_lock(root: &Path) -> Option<LockMetadata> {
    let metadata_path = root_lock_metadata_path(root);
    let metadata = load_lock_metadata(&metadata_path).ok()?;
    if is_pid_alive(metadata.pid) {
        Some(metadata)
    } else {
        None
    }
}

pub fn check_stale_jobs(root: &Path) -> Result<()> {
    let jobs_root = jobs_dir(root);
    if !jobs_root.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(&jobs_root).context("scan jobs directory")? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let mut run = load_job_run(&path)?;
            if run.status == JobRunStatus::Running {
                let should_mark_interrupted = run.pid.is_none_or(|pid| !is_pid_alive(pid));

                if should_mark_interrupted {
                    let now = now_millis();
                    run.status = JobRunStatus::Interrupted;
                    run.ended_at_millis = Some(now);
                    run.metadata.updated_at_millis = now;
                    run.exit_code = Some(1);

                    write_job_run(&path, &run)?;

                    if let Some(log_path) = run.log_path.clone() {
                        let _ =
                            append_log_line(&log_path, "interrupted: process no longer running");
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn start_job(root: &Path, command: &str) -> Result<JobHandle> {
    let jobs_root = jobs_dir(root);
    fs::create_dir_all(&jobs_root)
        .with_context(|| format!("create jobs dir {}", jobs_root.display()))?;

    let (job_id, manifest_path, log_path) = manifest_path_for(root, command);

    let now = now_millis();
    let metadata = EntityMetadata {
        id: job_id,
        created_at_millis: now,
        updated_at_millis: now,
        source_hashes: Vec::new(),
        model_version: None,
        tool_version: Some("kb-cli/0.1.0".to_string()),
        prompt_template_hash: None,
        dependencies: Vec::new(),
        output_paths: Vec::new(),
        status: Status::Fresh,
    };

    let job = JobRun {
        metadata,
        command: command.to_string(),
        root_path: root.to_path_buf(),
        started_at_millis: now,
        ended_at_millis: None,
        status: JobRunStatus::Running,
        log_path: Some(log_path.clone()),
        affected_outputs: Vec::new(),
        pid: Some(process::id()),
        exit_code: None,
    };

    write_job_run(&manifest_path, &job)?;
    append_log_line(&log_path, "job started")?;

    Ok(JobHandle {
        manifest_path,
        log_path,
        run: job,
    })
}

impl JobHandle {
    /// Append a line to this job's log file. IO errors are swallowed so a
    /// transient write failure cannot tear down the job whose progress we
    /// were trying to record — the log is strictly supplementary.
    ///
    /// Used by the compile pipeline (via `JobLogSink`) to stream per-pass
    /// events (`[run]`/`[ok]`/`[err]` lines) into `state/jobs/<id>.log` so
    /// a hung or failing compile is debuggable after the fact. Without
    /// this, the log only ever contained "job started".
    ///
    /// Callers that own an `Arc<dyn LogSink>` should go through
    /// `log_sink()` instead — this method is kept public for ad-hoc
    /// single-line notes from the CLI itself and for tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn append_log(&self, message: &str) {
        let _ = append_log_line(&self.log_path, message);
    }

    /// Absolute path to the log file this job appends to. Exposed so
    /// integration tests can assert on streamed content.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Tear down a job without recording it in `state/jobs/`.
    ///
    /// Used when a command rejects user input *before* doing any real
    /// work (empty query, unknown publish target, nonexistent path):
    /// the rejection is not a system failure, so we delete the
    /// optimistically-written manifest + log rather than mark it
    /// `status: failed`. Pairs with `ValidationError` detection in
    /// `execute_mutating_command_with_handle`. See bn-1jx.
    ///
    /// Missing files are ignored — this is best-effort cleanup and the
    /// caller already has a real error to propagate.
    pub fn discard(self) {
        // Log path first: the manifest is the "is this a recorded job?"
        // marker, so it goes last to keep the on-disk invariant that a
        // surviving manifest implies a surviving log.
        if self.log_path.exists() {
            let _ = fs::remove_file(&self.log_path);
        }
        if self.manifest_path.exists() {
            let _ = fs::remove_file(&self.manifest_path);
        }
    }

    pub fn finish(
        mut self,
        status: JobRunStatus,
        affected_outputs: Vec<PathBuf>,
    ) -> Result<JobRun> {
        let now = now_millis();
        let code = match status {
            JobRunStatus::Succeeded => Some(0),
            JobRunStatus::Failed => Some(1),
            JobRunStatus::Interrupted => Some(130),
            JobRunStatus::Running => None,
        };

        self.run.ended_at_millis = Some(now);
        self.run.status = status;
        self.run.affected_outputs = affected_outputs;
        self.run.exit_code = code;
        self.run.metadata.updated_at_millis = now;

        write_job_run(&self.manifest_path, &self.run)?;
        append_log_line(
            &self.log_path,
            &format!("job finished with status {status:?}"),
        )?;

        Ok(self.run)
    }
}

pub fn recent_jobs(root: &Path, limit: usize) -> Result<Vec<JobRun>> {
    let jobs_root = jobs_dir(root);
    if !jobs_root.exists() {
        return Ok(Vec::new());
    }

    let mut jobs = Vec::new();
    for entry in fs::read_dir(&jobs_root).context("scan jobs directory")? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let job = load_job_run(&path)?;
            jobs.push(job);
        }
    }

    jobs.sort_by_key(|job| std::cmp::Reverse(job.started_at_millis));
    jobs.truncate(limit);

    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// bn-221i: job ids must follow the `job-{command}-{hash}` shape.
    /// The command token stays visible so `ls state/jobs/` is greppable,
    /// and the trailing hash is short (no more 32-char nanos-pid slugs).
    #[test]
    fn next_job_id_uses_job_command_terseid_shape() {
        let dir = tempdir().expect("tempdir");
        let jobs_root = jobs_dir(dir.path());
        fs::create_dir_all(&jobs_root).expect("create jobs dir");

        let id = next_job_id(&jobs_root, "ingest");
        assert!(id.starts_with("job-ingest-"), "got: {id}");
        // Hash after `job-ingest-` must be short (terseid defaults are
        // 3..=8 base36 chars) — definitely shorter than the old nanos+pid
        // slug which ran to 32 chars.
        let hash = id.strip_prefix("job-ingest-").expect("prefix");
        assert!(!hash.is_empty());
        assert!(hash.len() <= 8, "hash too long: {hash}");
        assert!(
            hash.chars().all(|c| c.is_ascii_digit() || c.is_ascii_lowercase()),
            "hash must be base36: {hash}"
        );
    }

    /// Commands with dots (`review.approve`, `jobs.prune`) must not leak
    /// dots into the id — that would collide with terseid's child-path
    /// syntax and confuse grepping.
    #[test]
    fn next_job_id_sanitizes_dotted_commands() {
        let dir = tempdir().expect("tempdir");
        let jobs_root = jobs_dir(dir.path());
        fs::create_dir_all(&jobs_root).expect("create jobs dir");

        let id = next_job_id(&jobs_root, "review.approve");
        assert!(id.starts_with("job-review-approve-"), "got: {id}");
        assert!(!id.contains('.'), "id must not contain dots: {id}");
    }

    /// Two back-to-back invocations at the same pid for the same command
    /// still produce distinct ids (nanos differ, and if they don't the
    /// generator escalates nonces via the `exists` closure).
    #[test]
    fn next_job_id_is_unique_across_calls() {
        let dir = tempdir().expect("tempdir");
        let jobs_root = jobs_dir(dir.path());
        fs::create_dir_all(&jobs_root).expect("create jobs dir");

        let a = next_job_id(&jobs_root, "ingest");
        // Seed the directory with `a` so the next call sees it as taken.
        fs::write(jobs_root.join(format!("{a}.json")), b"{}").expect("seed manifest");
        let b = next_job_id(&jobs_root, "ingest");
        assert_ne!(a, b, "ids must differ even with the seed file present");
    }

    #[test]
    fn lock_metadata_is_visible_while_held() {
        let dir = tempdir().expect("tempdir");
        let lock =
            KbLock::acquire(dir.path(), "compile", Duration::from_secs(1)).expect("acquire lock");

        let metadata = read_lock_metadata(dir.path()).expect("read lock metadata");
        assert_eq!(metadata.command, "compile");
        assert_eq!(metadata.pid, process::id());
        assert!(metadata.started_at_millis > 0);

        drop(lock);
        assert!(!root_lock_metadata_path(dir.path()).exists());
    }

    #[test]
    fn contended_lock_reports_current_holder() {
        let dir = tempdir().expect("tempdir");
        let _first =
            KbLock::acquire(dir.path(), "compile", Duration::from_secs(1)).expect("first lock");

        let err = KbLock::acquire(dir.path(), "lint", Duration::from_millis(100))
            .expect_err("second lock should time out");
        let message = err.to_string();
        assert!(message.contains("timed out waiting for KB root lock"));
        assert!(message.contains("command=compile"));
    }

    /// Simulates the SIGKILL scenario: the previous holder's process died, so
    /// the OS released the advisory lock, but the sidecar metadata still
    /// references a pid that is no longer alive. The next acquire must succeed
    /// and overwrite the stale metadata with its own.
    #[test]
    fn acquire_reaps_stale_metadata_from_dead_holder() {
        let dir = tempdir().expect("tempdir");
        let locks_root = locks_dir(dir.path());
        fs::create_dir_all(&locks_root).expect("create locks dir");

        // Pick a pid that is virtually guaranteed to be dead. pid 0 is treated
        // as "not alive" by is_pid_alive, but we want something that actually
        // round-trips through kill -0 as "no such process".
        let dead_pid: u32 = u32::MAX / 2;
        assert!(
            !is_pid_alive(dead_pid),
            "test prerequisite: pid {dead_pid} must not be running",
        );

        // Seed a sidecar metadata file as if a killed process had written it.
        let stale = LockMetadata {
            command: "compile".to_string(),
            pid: dead_pid,
            started_at_millis: 1,
        };
        write_lock_metadata(&root_lock_metadata_path(dir.path()), &stale)
            .expect("write stale metadata");

        // With the file advisory lock unheld, acquire should succeed quickly,
        // not time out printing the ghost pid.
        let lock = KbLock::acquire(dir.path(), "lint", Duration::from_millis(500))
            .expect("acquire should succeed despite stale metadata");

        // The sidecar file should now reflect the current holder, not the
        // dead one.
        let fresh = read_lock_metadata(dir.path()).expect("read fresh metadata");
        assert_eq!(fresh.command, "lint");
        assert_eq!(fresh.pid, process::id());
        assert_ne!(fresh.pid, dead_pid);

        drop(lock);
    }

    /// `start_job` seeds the log with a single "job started" line, and
    /// `append_log` / the `JobLogSink` wrapper must add additional lines
    /// that readers can observe by re-reading the log file. This is the
    /// core bn-3ny guarantee: a caller plumbed through `log_sink` gets
    /// structured events into `state/jobs/<id>.log`.
    #[test]
    fn job_handle_append_log_writes_lines_to_log_file() {
        let dir = tempdir().expect("tempdir");
        let handle = start_job(dir.path(), "compile").expect("start job");

        handle.append_log("  [run] source_summary: doc-x...");
        handle.append_log("  [ok]  source_summary: doc-x (0.4s)");

        // Exercise the Arc<dyn LogSink> path that the compile pipeline uses.
        let sink = handle.log_sink();
        sink.append_log("  [err] concept_merge — boom");

        let contents = fs::read_to_string(handle.log_path()).expect("read log");
        assert!(contents.starts_with("job started\n"), "got:\n{contents}");
        assert!(contents.contains("[run] source_summary: doc-x"));
        assert!(contents.contains("[ok]  source_summary: doc-x"));
        assert!(contents.contains("[err] concept_merge — boom"));
        // Every line should be newline-terminated.
        assert!(contents.ends_with('\n'));
    }

    /// While blocked on the lock, `acquire` must print at least one
    /// "waiting for KB root lock" line to its progress sink. This regression
    /// guards bn-2pk: before the fix, users got an instant 5s-timeout error
    /// with no indication the command was blocked.
    #[test]
    fn acquire_emits_wait_progress_message_while_blocked() {
        let dir = tempdir().expect("tempdir");
        let _holder =
            KbLock::acquire(dir.path(), "compile", Duration::from_secs(1)).expect("first lock");

        let mut captured: Vec<u8> = Vec::new();
        let err = KbLock::acquire_with_progress(
            dir.path(),
            "lint",
            Duration::from_millis(100),
            &mut captured,
        )
        .expect_err("should time out");
        assert!(err.to_string().contains("timed out waiting for KB root lock"));

        let out = String::from_utf8(captured).expect("utf8");
        assert!(
            out.contains("waiting for KB root lock"),
            "expected wait progress line, got: {out:?}"
        );
        assert!(
            out.contains("compile"),
            "progress line should name holder command, got: {out:?}"
        );
    }

    /// The wait-progress line must name the holder's pid and command, read
    /// from the sidecar metadata file on disk. Regression for bn-90d: the
    /// previous formatting fell back to the literal string "unknown" when
    /// rendering the holder, even though the metadata file was right there.
    #[test]
    fn format_wait_message_names_pid_and_command_from_metadata() {
        let dir = tempdir().expect("tempdir");
        let locks_root = locks_dir(dir.path());
        fs::create_dir_all(&locks_root).expect("create locks dir");

        let metadata_path = root_lock_metadata_path(dir.path());
        let seeded = LockMetadata {
            command: "kb compile".to_string(),
            pid: 12345,
            started_at_millis: 1,
        };
        write_lock_metadata(&metadata_path, &seeded).expect("seed metadata");

        let rendered = format_wait_message(&metadata_path, 3);
        assert_eq!(
            rendered,
            "waiting for KB root lock (held by pid 12345 running `kb compile`, 3s elapsed)"
        );
        // Must never render the literal "unknown" when metadata exists.
        assert!(!rendered.contains("unknown"), "got: {rendered}");
    }

    /// When the sidecar metadata file is absent, the wait-progress line must
    /// say so explicitly — not "held by unknown", which looked indistinguish-
    /// able from a parse bug.
    #[test]
    fn format_wait_message_reports_missing_metadata() {
        let dir = tempdir().expect("tempdir");
        let locks_root = locks_dir(dir.path());
        fs::create_dir_all(&locks_root).expect("create locks dir");

        // Do NOT write a metadata file — exercise the missing-file path.
        let metadata_path = root_lock_metadata_path(dir.path());
        assert!(!metadata_path.exists());

        let rendered = format_wait_message(&metadata_path, 0);
        assert_eq!(
            rendered,
            "waiting for KB root lock (holder metadata missing, 0s elapsed)"
        );
        assert!(!rendered.contains("unknown"), "got: {rendered}");
    }

    /// If the metadata file exists but can't be parsed (truncated, corrupt,
    /// wrong schema), we still must not print "unknown" — the waiter should
    /// surface that the holder record is unreadable.
    #[test]
    fn format_wait_message_reports_unparseable_metadata() {
        let dir = tempdir().expect("tempdir");
        let locks_root = locks_dir(dir.path());
        fs::create_dir_all(&locks_root).expect("create locks dir");

        let metadata_path = root_lock_metadata_path(dir.path());
        fs::write(&metadata_path, b"not json at all").expect("write junk");

        let rendered = format_wait_message(&metadata_path, 7);
        assert_eq!(
            rendered,
            "waiting for KB root lock (holder metadata missing, 7s elapsed)"
        );
    }

    /// Integration-style check: run the real acquire loop against a live
    /// holder and confirm the emitted progress line names the holder's pid
    /// and command, not "unknown". This is the user-visible symptom the
    /// bone was filed against.
    #[test]
    fn acquire_progress_line_names_live_holder_pid_and_command() {
        let dir = tempdir().expect("tempdir");
        let _holder = KbLock::acquire(dir.path(), "kb compile", Duration::from_secs(1))
            .expect("first lock");
        let holder_pid = process::id();

        let mut captured: Vec<u8> = Vec::new();
        let _ = KbLock::acquire_with_progress(
            dir.path(),
            "kb ingest",
            Duration::from_millis(100),
            &mut captured,
        )
        .expect_err("should time out");

        let out = String::from_utf8(captured).expect("utf8");
        let expected_fragment = format!("pid {holder_pid} running `kb compile`");
        assert!(
            out.contains(&expected_fragment),
            "progress should name holder pid+command, got: {out:?}",
        );
        assert!(
            !out.contains("held by unknown"),
            "progress must never render the legacy \"held by unknown\" literal, got: {out:?}",
        );
    }

    /// If the advisory lock is held by a live process AND the recorded pid is
    /// dead (shouldn't normally happen, but guards against a metadata file
    /// that was swapped in underneath us), the timeout error should mark the
    /// holder as dead rather than reporting it as a live ghost.
    #[test]
    fn timeout_error_notes_dead_pid_in_metadata() {
        let dir = tempdir().expect("tempdir");
        let locks_root = locks_dir(dir.path());
        fs::create_dir_all(&locks_root).expect("create locks dir");

        // Hold the OS lock with a live process (ourselves).
        let lock_path = root_lock_path(dir.path());
        let holder_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .expect("open lock file");
        holder_file.try_lock_exclusive().expect("hold OS lock");

        // Overwrite the metadata with a dead pid after reaping would normally
        // delete it. reap_stale_metadata will pick this up and loop, so we
        // instead verify the format_lock_holder annotation when acquire can't
        // escape the WouldBlock branch. To exercise that, we keep rewriting
        // the metadata inside a very short timeout window: we simulate it
        // statically by calling format_lock_holder directly.
        let dead_pid: u32 = u32::MAX / 2;
        let stale = LockMetadata {
            command: "compile".to_string(),
            pid: dead_pid,
            started_at_millis: 42,
        };
        let metadata_path = root_lock_metadata_path(dir.path());
        write_lock_metadata(&metadata_path, &stale).expect("write metadata");

        let rendered = format_lock_holder(&metadata_path);
        assert!(rendered.contains("holder pid is dead"), "got: {rendered}");

        FileExt::unlock(&holder_file).expect("release OS lock");
    }
}
