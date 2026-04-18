use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use thiserror::Error;

/// Captured subprocess result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubprocessOutput {
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// Process exit code, or `None` if the process exited due to a signal.
    pub exit_code: Option<i32>,
}

/// Subprocess execution failure.
#[derive(Debug, Error)]
pub enum SubprocessError {
    /// The subprocess exceeded its allotted timeout.
    #[error("command timed out after {timeout:?}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}")]
    TimedOut {
        /// Human-readable command string.
        command: String,
        /// Configured timeout.
        timeout: Duration,
        /// Partial standard output captured before termination.
        stdout: String,
        /// Partial standard error captured before termination.
        stderr: String,
    },

    /// The subprocess could not be spawned or monitored.
    #[error("subprocess failed: {0}")]
    Other(#[from] anyhow::Error),
}

/// Run a shell command with timeout enforcement.
///
/// On Unix this uses `setsid` so the child becomes the leader of a fresh process group;
/// if the timeout expires, the whole group is terminated with `SIGKILL`.
/// Stdout and stderr are drained concurrently so long-running writers cannot deadlock.
///
/// # Errors
/// Returns [`SubprocessError::TimedOut`] when the process exceeds `timeout`, including
/// whatever stdout/stderr was captured before termination.
pub fn run_shell_command(
    command: &str,
    timeout: Duration,
) -> Result<SubprocessOutput, SubprocessError> {
    let mut child = spawn_shell_command(command)?;

    let stdout_handle = child.stdout.take().map(spawn_reader_thread);
    let stderr_handle = child.stderr.take().map(spawn_reader_thread);

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() >= timeout {
                    terminate_child(&child);
                    let _ = child.wait();

                    let stdout = join_reader(stdout_handle)?;
                    let stderr = join_reader(stderr_handle)?;

                    return Err(SubprocessError::TimedOut {
                        command: command.to_string(),
                        timeout,
                        stdout,
                        stderr,
                    });
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                return Err(SubprocessError::Other(
                    anyhow!(err).context("poll subprocess exit status"),
                ));
            }
        }
    };

    let stdout = join_reader(stdout_handle)?;
    let stderr = join_reader(stderr_handle)?;

    Ok(SubprocessOutput {
        stdout,
        stderr,
        exit_code: status.code(),
    })
}

fn spawn_shell_command(command: &str) -> Result<std::process::Child, SubprocessError> {
    let mut cmd = shell_command(command);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    cmd.spawn().map_err(|err| {
        SubprocessError::Other(anyhow!(err).context(format!("spawn shell command `{command}`")))
    })
}

fn shell_command(command: &str) -> Command {
    #[cfg(unix)]
    {
        let mut cmd = Command::new("setsid");
        cmd.arg("sh").arg("-c").arg(command);
        cmd
    }

    #[cfg(not(unix))]
    {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd
    }
}

fn spawn_reader_thread<T>(mut stream: T) -> thread::JoinHandle<Result<Vec<u8>>>
where
    T: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        stream
            .read_to_end(&mut bytes)
            .context("read subprocess stream")?;
        Ok(bytes)
    })
}

fn join_reader(
    handle: Option<thread::JoinHandle<Result<Vec<u8>>>>,
) -> Result<String, SubprocessError> {
    let Some(handle) = handle else {
        return Ok(String::new());
    };

    match handle.join() {
        Ok(Ok(bytes)) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Ok(Err(err)) => Err(SubprocessError::Other(
            err.context("collect subprocess output"),
        )),
        Err(_) => Err(SubprocessError::Other(anyhow!(
            "subprocess reader thread panicked"
        ))),
    }
}

fn terminate_child(child: &std::process::Child) {
    #[cfg(unix)]
    {
        let child_pid = child.id().to_string();
        let process_group = format!("-{}", child.id());
        let _ = Command::new("kill")
            .args(["-KILL", &process_group])
            .status();
        let _ = Command::new("kill").args(["-KILL", &child_pid]).status();
    }

    #[cfg(not(unix))]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/F", "/T"])
            .status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn pid_is_alive(pid: u32) -> bool {
        #[cfg(unix)]
        {
            Command::new("kill")
                .args(["-0", &pid.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        }

        #[cfg(not(unix))]
        {
            let _ = pid;
            false
        }
    }

    fn wait_for_process_exit(pid: u32, timeout: Duration) -> bool {
        let started = Instant::now();
        while started.elapsed() < timeout {
            if !pid_is_alive(pid) {
                return true;
            }
            thread::sleep(Duration::from_millis(50));
        }
        !pid_is_alive(pid)
    }

    fn read_pid(path: &Path) -> u32 {
        fs::read_to_string(path)
            .expect("read pid file")
            .trim()
            .parse::<u32>()
            .expect("parse pid")
    }

    fn unique_test_dir() -> PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after unix epoch")
            .as_millis();
        std::env::temp_dir().join(format!(
            "kb-llm-subprocess-tests-{}-{millis}",
            std::process::id()
        ))
    }

    #[test]
    fn kills_long_running_stub_with_sigkill_timeout() {
        let started = Instant::now();
        let err = run_shell_command("printf 'tick\\n'; sleep 100", Duration::from_secs(2))
            .expect_err("command should time out");

        assert!(started.elapsed() < Duration::from_secs(5));
        match err {
            SubprocessError::TimedOut { stdout, stderr, .. } => {
                assert!(stdout.contains("tick"));
                assert!(stderr.is_empty());
            }
            other @ SubprocessError::Other(_) => panic!("expected timeout, got {other:?}"),
        }
    }

    #[test]
    fn timeout_error_includes_partial_stdout_and_stderr() {
        let err = run_shell_command(
            "printf 'stdout-before-timeout\\n'; printf 'stderr-before-timeout\\n' >&2; sleep 100",
            Duration::from_secs(2),
        )
        .expect_err("command should time out");

        match err {
            SubprocessError::TimedOut { stdout, stderr, .. } => {
                assert!(stdout.contains("stdout-before-timeout"));
                assert!(stderr.contains("stderr-before-timeout"));
            }
            other @ SubprocessError::Other(_) => panic!("expected timeout, got {other:?}"),
        }
    }

    #[test]
    fn timeout_kills_background_descendants_without_zombies() {
        let dir = unique_test_dir();
        fs::create_dir_all(&dir).expect("create test dir");
        let pid_path = dir.join("child.pid");
        let command = format!(
            "sleep 100 & echo $! > '{}' ; printf 'spawned\\n'; wait",
            pid_path.display()
        );

        let err = run_shell_command(&command, Duration::from_secs(2))
            .expect_err("command should time out");
        let pid = read_pid(&pid_path);

        match err {
            SubprocessError::TimedOut { stdout, .. } => {
                assert!(stdout.contains("spawned"));
            }
            other @ SubprocessError::Other(_) => panic!("expected timeout, got {other:?}"),
        }

        assert!(wait_for_process_exit(pid, Duration::from_secs(3)));
        fs::remove_dir_all(&dir).expect("remove test dir");
    }
}
