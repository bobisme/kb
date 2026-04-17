use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use kb_core::{JobRun, JobRunStatus, Status};
use kb_core::fs::atomic_write;
use kb_core::EntityMetadata;

#[derive(Debug)]
pub struct JobHandle {
    manifest_path: PathBuf,
    log_path: PathBuf,
    pub run: JobRun,
}

fn jobs_dir(root: &Path) -> PathBuf {
    root.join("state").join("jobs")
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
    let raw = fs::read_to_string(path).with_context(|| format!("read job run manifest {}", path.display()))?;
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
                let should_mark_interrupted = run
                    .pid
                    .is_none_or(|pid| !is_pid_alive(pid));

                if should_mark_interrupted {
                    let now = now_millis();
                    run.status = JobRunStatus::Interrupted;
                    run.ended_at_millis = Some(now);
                    run.metadata.updated_at_millis = now;
                    run.exit_code = Some(1);

                    write_job_run(&path, &run)?;

                    if let Some(log_path) = run.log_path.clone() {
                        let _ = append_log_line(&log_path, "interrupted: process no longer running");
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn start_job(root: &Path, command: &str) -> Result<JobHandle> {
    let jobs_root = jobs_dir(root);
    fs::create_dir_all(&jobs_root).with_context(|| format!("create jobs dir {}", jobs_root.display()))?;

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
    pub fn finish(mut self, status: JobRunStatus, affected_outputs: Vec<PathBuf>) -> Result<JobRun> {
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
        append_log_line(&self.log_path, &format!("job finished with status {status:?}"))?;

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
