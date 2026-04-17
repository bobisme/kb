#![forbid(unsafe_code)]

mod config;
mod jobs;
mod root;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use config::Config;
use kb_core::JobRunStatus;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "kb", version, about = "Personal knowledge base compiler")]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    /// Root directory of the knowledge base
    #[arg(global = true, long)]
    root: Option<PathBuf>,

    /// Output format
    #[arg(global = true, long, value_parser = ["md", "marp", "json", "png"])]
    format: Option<String>,

    /// LLM model to use
    #[arg(global = true, long)]
    model: Option<String>,

    /// Filter results since a given time
    #[arg(global = true, long)]
    since: Option<String>,

    /// Perform a dry-run without making changes
    #[arg(global = true, long)]
    dry_run: bool,

    /// Output structured JSON
    #[arg(global = true, long)]
    json: bool,

    /// Force operation (overwrite, skip confirmations, etc.)
    #[arg(global = true, long)]
    force: bool,

    /// Enable review mode
    #[arg(global = true, long)]
    review: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Initialize a new knowledge base
    Init {
        /// Path to initialize at
        path: Option<PathBuf>,
    },
    /// Ingest documents into the knowledge base
    Ingest {
        /// Files or directories to ingest
        #[arg(required = true)]
        sources: Vec<PathBuf>,
    },
    /// Compile the knowledge base
    Compile,
    /// Query the knowledge base with natural language
    Ask {
        /// Question to ask
        #[arg(required = true)]
        query: String,
    },
    /// Lint knowledge base for issues
    Lint {
        /// Check specific rules
        #[arg(long)]
        rule: Option<String>,
    },
    /// Run health checks on the knowledge base
    Doctor,
    /// Show status of the knowledge base
    Status,
    /// Publish the knowledge base
    Publish {
        /// Destination for publishing
        #[arg(long)]
        dest: Option<String>,
    },
    /// Search the knowledge base
    Search {
        /// Search query
        #[arg(required = true)]
        query: String,
    },
    /// Inspect a document or entity
    Inspect {
        /// Document or entity to inspect
        #[arg(required = true)]
        target: String,
    },
    /// Request or manage reviews
    Review {
        /// Review operation (list, approve, reject, etc.)
        #[arg(required = true)]
        operation: String,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if let Err(err) = run(cli) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn run(cli: Cli) -> Result<()> {
    let root = if matches!(cli.command, Some(Command::Init { .. }) | None) {
        cli.root
    } else {
        Some(root::discover_root(cli.root.as_deref())?.path)
    };

    match cli.command {
        Some(Command::Compile) => execute_mutating_command(root.as_deref(), "compile", || {
            println!("compile is not implemented yet");
            Ok(())
        }),
        Some(Command::Doctor) => execute_mutating_command(root.as_deref(), "doctor", || {
            println!("doctor is not implemented yet");
            Ok(())
        }),
        Some(Command::Ask { query }) => {
            execute_mutating_command(root.as_deref(), "ask", move || {
                println!("ask is not implemented yet: {query}");
                Ok(())
            })
        }
        Some(Command::Ingest { sources }) => {
            execute_mutating_command(root.as_deref(), "ingest", move || {
                println!(
                    "ingest is not implemented yet for {} sources",
                    sources.len()
                );
                Ok(())
            })
        }
        Some(Command::Lint { rule }) => {
            execute_mutating_command(root.as_deref(), "lint", move || {
                if let Some(rule) = rule {
                    println!("lint is not implemented yet (rule: {rule})");
                } else {
                    println!("lint is not implemented yet");
                }
                Ok(())
            })
        }
        Some(Command::Publish { dest }) => {
            execute_mutating_command(root.as_deref(), "publish", move || {
                if let Some(dest) = dest {
                    println!("publish is not implemented yet (dest: {dest})");
                } else {
                    println!("publish is not implemented yet");
                }
                Ok(())
            })
        }
        Some(Command::Status) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let jobs = jobs::recent_jobs(root, 20)?;
            if cli.json {
                let payload = serde_json::to_string_pretty(&jobs)?;
                println!("{payload}");
                return Ok(());
            }

            if jobs.is_empty() {
                println!("No job runs yet in {}", root.display());
            } else {
                println!("Recent job runs ({})", jobs.len());
                for job in jobs {
                    let ended = job
                        .ended_at_millis
                        .map_or_else(|| "running".to_string(), |ended| format!("{ended}"));
                    println!(
                        "{} | {:<11} | {} | started={} | ended={ended}",
                        job.metadata.id,
                        format!("{:?}", job.status),
                        job.command,
                        job.started_at_millis
                    );
                }
            }
            Ok(())
        }
        Some(Command::Init { path }) => {
            println!(
                "init is not implemented yet{}",
                path.map_or(String::new(), |p| format!(" at {}", p.display()))
            );
            Ok(())
        }
        Some(Command::Search { query }) => {
            println!("search is not implemented yet: {query}");
            Ok(())
        }
        Some(Command::Inspect { target }) => {
            println!("inspect is not implemented yet: {target}");
            Ok(())
        }
        Some(Command::Review { operation }) => {
            println!("review is not implemented yet: {operation}");
            Ok(())
        }
        None => {
            println!("kb: a personal knowledge base compiler");
            println!("Run 'kb --help' for more information");
            Ok(())
        }
    }
}

fn execute_mutating_command(
    root: Option<&Path>,
    command: &str,
    action: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let root = root.expect("root resolved for mutating commands");
    let cfg = Config::load_from_root(root, None)?;
    let _lock = jobs::KbLock::acquire(root, command, Duration::from_millis(cfg.lock.timeout_ms))?;
    jobs::check_stale_jobs(root)?;
    let handle = jobs::start_job(root, command)?;

    let result = action();
    match result {
        Ok(()) => {
            handle.finish(JobRunStatus::Succeeded, Vec::new())?;
            Ok(())
        }
        Err(err) => {
            handle.finish(JobRunStatus::Failed, Vec::new())?;
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    fn write_kb_config(root: &Path) {
        fs::create_dir_all(root).expect("create kb root");
        fs::write(root.join(Config::FILE_NAME), "\n").expect("write kb config");
    }

    #[test]
    fn status_command_does_not_block_on_root_lock() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        let lock =
            jobs::KbLock::acquire(&root, "compile", Duration::from_secs(1)).expect("acquire lock");
        let started = Instant::now();
        run(Cli {
            root: Some(root),
            format: None,
            model: None,
            since: None,
            dry_run: false,
            json: false,
            force: false,
            review: false,
            command: Some(Command::Status),
        })
        .expect("status command succeeds");
        assert!(started.elapsed() < Duration::from_millis(250));
        drop(lock);
    }

    #[test]
    fn mutating_commands_serialize_through_root_lock() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        let first_lock =
            jobs::KbLock::acquire(&root, "compile", Duration::from_secs(1)).expect("first lock");
        let (tx, rx) = mpsc::channel();
        let root_for_thread = root;
        let worker = thread::spawn(move || {
            execute_mutating_command(Some(root_for_thread.as_path()), "lint", || {
                tx.send(Instant::now()).expect("send acquisition time");
                Ok(())
            })
            .expect("mutating command succeeds");
        });

        thread::sleep(Duration::from_millis(200));
        assert!(
            rx.try_recv().is_err(),
            "second command should still be waiting for the root lock"
        );
        drop(first_lock);
        let acquired_at = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("second command acquires lock");
        assert!(acquired_at.elapsed() < Duration::from_secs(1));
        worker.join().expect("join worker");
    }
}
