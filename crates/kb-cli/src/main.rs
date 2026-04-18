#![forbid(unsafe_code)]

mod config;
mod init;
mod jobs;
mod root;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use config::Config;
use kb_core::JobRunStatus;
use serde::Serialize;
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
        /// Files, directories, or URLs to ingest
        #[arg(required = true)]
        sources: Vec<String>,
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
            let ingest_root = root.clone();
            let action = move || {
                let root = ingest_root
                    .as_deref()
                    .expect("root resolved for non-init commands");
                run_ingest(root, &sources, cli.json, cli.dry_run)
            };

            if cli.dry_run {
                action()
            } else {
                execute_mutating_command(root.as_deref(), "ingest", action)
            }
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
        Some(Command::Init { path }) => init::init(root, path, cli.force),
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

#[derive(Debug, Serialize)]
struct IngestPayload {
    dry_run: bool,
    results: Vec<IngestResult>,
    summary: IngestSummary,
}

#[derive(Debug, Serialize)]
struct IngestResult {
    input: String,
    source_kind: &'static str,
    outcome: kb_ingest::IngestOutcome,
    source_document_id: String,
    source_revision_id: String,
    content_path: PathBuf,
    metadata_path: PathBuf,
}

#[derive(Debug, Default, Serialize)]
struct IngestSummary {
    total: usize,
    new_sources: usize,
    new_revisions: usize,
    skipped: usize,
}

fn run_ingest(root: &Path, sources: &[String], json: bool, dry_run: bool) -> Result<()> {
    let mut urls = Vec::new();
    let mut local_paths = Vec::new();
    for source in sources {
        if source.starts_with("http://") || source.starts_with("https://") {
            urls.push(source.as_str());
        } else {
            local_paths.push(PathBuf::from(source));
        }
    }

    let mut results = Vec::new();
    for report in kb_ingest::ingest_paths_with_options(root, &local_paths, dry_run)? {
        results.push(IngestResult {
            input: report.source_path.display().to_string(),
            source_kind: "file",
            outcome: report.outcome,
            source_document_id: report.ingested.document.metadata.id,
            source_revision_id: report.ingested.revision.metadata.id,
            content_path: report.ingested.copied_path,
            metadata_path: report.ingested.metadata_sidecar_path,
        });
    }

    if !urls.is_empty() {
        let rt = tokio::runtime::Runtime::new()?;
        let url_results = rt.block_on(async {
            let mut collected = Vec::with_capacity(urls.len());
            for url in &urls {
                collected.push(kb_ingest::ingest_url_with_options(root, url, dry_run).await?);
            }
            Ok::<Vec<kb_ingest::UrlIngestReport>, anyhow::Error>(collected)
        })?;

        for report in url_results {
            results.push(IngestResult {
                input: report.source_url,
                source_kind: "url",
                outcome: report.outcome,
                source_document_id: report.source_document_id,
                source_revision_id: report.source_revision_id,
                content_path: report.normalized_path,
                metadata_path: report.metadata_path,
            });
        }
    }

    let summary = summarize_ingest(&results);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&IngestPayload {
                dry_run,
                results,
                summary,
            })?
        );
        return Ok(());
    }

    if results.is_empty() {
        println!("No sources ingested");
        return Ok(());
    }

    for item in &results {
        let prefix = if dry_run { "would" } else { "status" };
        println!(
            "{prefix} {} {} {} {} {} {}",
            outcome_label(item.outcome),
            item.source_kind,
            item.input,
            item.source_document_id,
            item.source_revision_id,
            item.content_path.display()
        );
    }
    println!(
        "Summary: {} total | {} new sources | {} new revisions | {} skipped",
        summary.total, summary.new_sources, summary.new_revisions, summary.skipped
    );

    Ok(())
}

fn summarize_ingest(results: &[IngestResult]) -> IngestSummary {
    let mut summary = IngestSummary {
        total: results.len(),
        ..IngestSummary::default()
    };
    for item in results {
        match item.outcome {
            kb_ingest::IngestOutcome::NewSource => summary.new_sources += 1,
            kb_ingest::IngestOutcome::NewRevision => summary.new_revisions += 1,
            kb_ingest::IngestOutcome::Skipped => summary.skipped += 1,
        }
    }
    summary
}

const fn outcome_label(outcome: kb_ingest::IngestOutcome) -> &'static str {
    match outcome {
        kb_ingest::IngestOutcome::NewSource => "new_source",
        kb_ingest::IngestOutcome::NewRevision => "new_revision",
        kb_ingest::IngestOutcome::Skipped => "skipped",
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
