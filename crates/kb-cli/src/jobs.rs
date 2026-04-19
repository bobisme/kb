use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use fs2::FileExt;
use kb_core::EntityMetadata;
use kb_core::fs::atomic_write;
use kb_core::{JobRun, JobRunStatus, Status};
use serde::{Deserialize, Serialize};

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

fn jobs_dir(root: &Path) -> PathBuf {
    root.join("state").join("jobs")
}

fn locks_dir(root: &Path) -> PathBuf {
    root.join("state").join("locks")
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

fn next_job_id(command: &str) -> String {
    let pid = process::id();
    let now = now_nanos();
    let command_part = command.replace(' ', "-");
    format!("{now}-{pid}-{command_part}")
}

fn manifest_path_for(root: &Path, command: &str) -> (String, PathBuf, PathBuf) {
    let jobs_root = jobs_dir(root);
    let id = next_job_id(command);
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

impl KbLock {
    pub fn acquire(root: &Path, command: &str, timeout: Duration) -> Result<Self> {
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

        let deadline = Instant::now() + timeout;
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
                    if Instant::now() >= deadline {
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

    jobs.sort_by(|a, b| b.started_at_millis.cmp(&a.started_at_millis));
    jobs.truncate(limit);

    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
