use std::io::{Read, Write};
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

/// Run a command directly (no shell) with stdin delivery and timeout enforcement.
///
/// `argv[0]` is the executable; `argv[1..]` are the literal arguments. No
/// shell interpolation is performed, so callers don't need to quote anything.
/// `stdin_bytes` is written to the child's standard input on a dedicated
/// thread; stdout and stderr are drained concurrently on two more threads —
/// this three-thread layout is required so a child that fills any one pipe
/// (notably stderr while we're still feeding stdin) can't deadlock us.
///
/// Extra environment variables in `env` are layered onto the inherited
/// environment via [`Command::env`], so callers can pass things like
/// `OPENCODE_CONFIG` without inlining them into a shell command line.
///
/// On Unix this uses `setsid` so the child becomes the leader of a fresh
/// process group; if the timeout expires, the whole group is terminated with
/// `SIGKILL` (mirrors [`run_shell_command`]).
///
/// # Errors
/// Returns [`SubprocessError::TimedOut`] when the process exceeds `timeout`,
/// including whatever stdout/stderr was captured before termination.
/// Returns [`SubprocessError::Other`] for spawn / I/O / join failures.
pub fn run_command_with_stdin(
    argv: &[String],
    stdin_bytes: &[u8],
    env: &[(String, String)],
    timeout: Duration,
) -> Result<SubprocessOutput, SubprocessError> {
    if argv.is_empty() {
        return Err(SubprocessError::Other(anyhow!(
            "run_command_with_stdin called with empty argv"
        )));
    }

    let mut child = spawn_argv_command(argv, env)?;

    // Hand stdin to a dedicated writer thread so a child that fills its
    // stderr (or stdout) buffer while we're still feeding stdin can't
    // deadlock us. We *must* close the child's stdin handle when we're
    // done writing — opencode/claude wait for EOF before producing any
    // response.
    let stdin_handle = child.stdin.take().map(|stdin| spawn_writer_thread(stdin, stdin_bytes));
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
                    // Drop the stdin writer's result on timeout — the child
                    // is dead, an EPIPE there is expected and uninteresting.
                    join_writer_quiet(stdin_handle);

                    return Err(SubprocessError::TimedOut {
                        command: argv.join(" "),
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
    // The writer thread also needs to be joined to surface I/O errors —
    // but a child that read all of stdin and exited normally produces an
    // Ok(()) here, so this is cheap on the happy path.
    join_writer_quiet(stdin_handle);

    Ok(SubprocessOutput {
        stdout,
        stderr,
        exit_code: status.code(),
    })
}

fn spawn_argv_command(
    argv: &[String],
    env: &[(String, String)],
) -> Result<std::process::Child, SubprocessError> {
    let mut cmd = argv_command(argv);
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    cmd.spawn().map_err(|err| {
        SubprocessError::Other(
            anyhow!(err).context(format!("spawn command `{}`", argv.join(" "))),
        )
    })
}

fn argv_command(argv: &[String]) -> Command {
    // Caller already checked argv is non-empty.
    let program = &argv[0];
    let rest = &argv[1..];

    #[cfg(unix)]
    {
        let mut cmd = Command::new("setsid");
        cmd.arg(program);
        cmd.args(rest);
        cmd
    }

    #[cfg(not(unix))]
    {
        let mut cmd = Command::new(program);
        cmd.args(rest);
        cmd
    }
}

fn spawn_writer_thread<W>(mut stream: W, payload: &[u8]) -> thread::JoinHandle<Result<()>>
where
    W: Write + Send + 'static,
{
    let bytes = payload.to_vec();
    thread::spawn(move || {
        // EPIPE here means the child closed stdin before reading everything
        // we wanted to send (or we were killed mid-write on timeout). That's
        // not a hard error — the caller surfaces the real failure via the
        // child's exit status / stderr — so the writer thread reports it but
        // doesn't try to interpret it.
        stream.write_all(&bytes).context("write subprocess stdin")?;
        // Drop closes the handle, which sends EOF — required because
        // opencode/claude block on stdin until EOF before responding.
        drop(stream);
        Ok(())
    })
}

fn join_writer_quiet(handle: Option<thread::JoinHandle<Result<()>>>) {
    if let Some(handle) = handle {
        // Best-effort: a panicked writer thread is logged-and-forgotten;
        // a write error (typically EPIPE on a child that already exited
        // with the answer it had) is also non-fatal — the caller will see
        // the real failure via the exit code / stderr path.
        let _ = handle.join();
    }
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
    fn run_command_with_stdin_pipes_payload_to_child_and_captures_output() {
        // Use `cat` so whatever we send on stdin echoes back on stdout.
        // Verifies argv-direct mode works AND that stdin delivery doesn't
        // deadlock for payloads larger than a typical pipe buffer (64 KB on
        // Linux). 1 MB exercises the third-thread-writer / concurrent-reader
        // contract.
        let payload = "x".repeat(1024 * 1024);
        let argv = vec!["cat".to_string()];
        let env: Vec<(String, String)> = Vec::new();
        let output = run_command_with_stdin(
            &argv,
            payload.as_bytes(),
            &env,
            Duration::from_secs(10),
        )
        .expect("run cat with 1 MB stdin");

        assert_eq!(output.exit_code, Some(0));
        assert_eq!(output.stdout.len(), payload.len(), "cat should echo all bytes");
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn run_command_with_stdin_passes_extra_env_vars() {
        // Use a shell builtin via /bin/sh -c just to read $FOO back on stdout.
        // The command itself doesn't use the shell to assemble a prompt — we
        // pass argv directly, which is what the new opencode/claude paths do.
        let argv = vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "printf '%s' \"$KB_TEST_VAR\"".to_string(),
        ];
        let env = vec![("KB_TEST_VAR".to_string(), "kb-stdin-env-ok".to_string())];
        let output =
            run_command_with_stdin(&argv, b"", &env, Duration::from_secs(5)).expect("run sh -c");

        assert_eq!(output.exit_code, Some(0));
        assert_eq!(output.stdout, "kb-stdin-env-ok");
    }

    #[test]
    fn run_command_with_stdin_surfaces_nonzero_exit() {
        let argv = vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "printf 'oops' >&2; exit 7".to_string(),
        ];
        let output = run_command_with_stdin(&argv, b"", &[], Duration::from_secs(5))
            .expect("spawn must succeed");
        assert_eq!(output.exit_code, Some(7));
        assert_eq!(output.stderr, "oops");
    }

    #[test]
    fn run_command_with_stdin_times_out_on_slow_child() {
        let argv = vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            "printf 'started\\n'; sleep 100".to_string(),
        ];
        let err = run_command_with_stdin(&argv, b"", &[], Duration::from_millis(500))
            .expect_err("should time out");
        match err {
            SubprocessError::TimedOut { stdout, .. } => {
                assert!(stdout.contains("started"), "partial stdout missing: {stdout}");
            }
            other @ SubprocessError::Other(_) => panic!("expected TimedOut, got {other:?}"),
        }
    }

    #[test]
    fn run_command_with_stdin_rejects_empty_argv() {
        let err = run_command_with_stdin(&[], b"", &[], Duration::from_secs(1))
            .expect_err("empty argv must error");
        assert!(matches!(err, SubprocessError::Other(_)));
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
