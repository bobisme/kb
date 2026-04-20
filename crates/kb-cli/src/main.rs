#![forbid(unsafe_code)]

mod config;
mod forget;
mod init;
mod jobs;
mod jobs_cmd;
mod publish;
mod review;
mod root;

use std::env;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use config::{Config, LlmRunnerConfig};
use kb_compile::Graph;
use kb_core::{
    Artifact, ArtifactKind, EntityMetadata, JobRun, JobRunStatus, Question, QuestionContext,
    ReviewItem, ReviewKind, ReviewStatus, Status, hash_many, slug_from_title,
};
use kb_llm::{
    ClaudeCliAdapter, ClaudeCliConfig, LlmAdapter, OpencodeAdapter, OpencodeConfig,
    RunHealthCheckRequest, Template,
};
use serde::Serialize;

/// Top-level envelope wrapping all `--json` output (`schema_version`: 1).
#[derive(Serialize)]
struct JsonEnvelope {
    schema_version: u32,
    command: String,
    data: serde_json::Value,
    warnings: Vec<String>,
    errors: Vec<String>,
}

pub(crate) fn emit_json<T: Serialize>(command: &str, data: T) -> Result<()> {
    let envelope = JsonEnvelope {
        schema_version: 1,
        command: command.to_string(),
        data: serde_json::to_value(data)?,
        warnings: Vec::new(),
        errors: Vec::new(),
    };
    println!("{}", serde_json::to_string_pretty(&envelope)?);
    Ok(())
}

#[derive(Parser)]
#[command(name = "kb", version, about = "Personal knowledge base compiler")]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    /// Root directory of the knowledge base
    #[arg(global = true, long)]
    root: Option<PathBuf>,

    /// LLM model to use
    #[arg(global = true, long)]
    model: Option<String>,

    /// Perform a dry-run without making changes
    #[arg(global = true, long)]
    dry_run: bool,

    /// Output structured JSON
    #[arg(global = true, long)]
    json: bool,

    /// Force operation (overwrite, skip confirmations, etc.)
    #[arg(global = true, long)]
    force: bool,

    /// Suppress non-essential output (e.g. post-command hints)
    #[arg(global = true, long)]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Initialize a new knowledge base
    Init {
        /// Path to initialize at
        path: Option<PathBuf>,
        /// Regenerate kb.toml even if an existing, parseable config is present.
        /// Without this flag, `--force` preserves user config (publish targets,
        /// runner overrides, etc.) and only rebuilds the directory scaffold
        /// and state files.
        #[arg(long)]
        reset_config: bool,
    },
    /// Ingest documents into the knowledge base
    Ingest {
        /// Files, directories, or URLs to ingest
        #[arg(required = true)]
        sources: Vec<String>,
        /// Ingest files even if they are empty or contain only YAML frontmatter
        #[arg(long)]
        allow_empty: bool,
    },
    /// Compile the knowledge base
    Compile,
    /// Query the knowledge base with natural language
    Ask {
        /// Question to ask (reads from stdin if omitted or "-")
        query: Option<String>,

        /// Artifact format for the answer (md, marp, json, png)
        #[arg(long, value_parser = ["md", "marp", "json", "png"])]
        format: Option<String>,

        /// Propose promoting the answer into the wiki
        #[arg(long)]
        promote: bool,
    },
    /// Lint knowledge base for issues
    Lint {
        /// Check a single lint check
        #[arg(long, alias = "rule")]
        check: Option<String>,
        /// Treat warnings as errors (exit 1 on warnings-only)
        #[arg(long)]
        strict: bool,
    },
    /// Run health checks on the knowledge base
    Doctor,
    /// Show status of the knowledge base
    Status,
    /// Publish selected artifacts to a project notes folder
    Publish {
        /// Named target from [publish.targets] in kb.toml
        target: String,
    },
    /// Search the knowledge base
    Search {
        /// Search query
        #[arg(required = true)]
        query: String,
        /// Limit number of results (default 10)
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Inspect a document or entity
    Inspect {
        /// Walk the full provenance chain recursively
        #[arg(long)]
        trace: bool,

        /// Document or entity to inspect
        #[arg(required = true)]
        target: String,
    },
    /// Remove an ingested source (moves files into `trash/` for safety)
    Forget {
        /// `src-<hex>` id or a path to the source file that was ingested.
        /// The path is canonicalized and matched against each
        /// `raw/inbox/<src>/source_document.json::stable_location`.
        #[arg(required = true)]
        target: String,
        /// Opt out of the cascade. Preserves the pre-bn-did behavior: only
        /// the source's own normalized/, raw/inbox/, and wiki/sources/
        /// entries are touched. Orphaned concept pages, cited question
        /// pages, and stale build records are left in place — `kb lint`
        /// will surface them afterwards.
        #[arg(long)]
        no_cascade: bool,
    },
    /// Inspect and manage review queue
    Review {
        #[command(subcommand)]
        action: ReviewAction,
    },
    /// Inspect and prune job-run manifests under `state/jobs/`
    Jobs {
        #[command(subcommand)]
        action: Option<JobsAction>,
    },
}

#[derive(clap::Subcommand)]
enum JobsAction {
    /// List job runs filtered by status, with log paths.
    List {
        /// Show interrupted job runs.
        #[arg(long)]
        interrupted: bool,
        /// Show failed job runs.
        #[arg(long)]
        failed: bool,
        /// Page size (default: 50).
        #[arg(long, default_value_t = 50)]
        page_size: usize,
        /// 1-based page index (default: 1).
        #[arg(long, default_value_t = 1)]
        page: usize,
    },
    /// Move old job-run manifests (and their logs) into `trash/jobs-<ts>/`.
    ///
    /// The prune is explicit, non-recursive, and requires at least one of
    /// `--interrupted`, `--failed`, or `--all`. A root lock is held for the
    /// duration — status/doctor readers stay consistent.
    Prune {
        /// Prune interrupted job runs.
        #[arg(long)]
        interrupted: bool,
        /// Prune failed job runs.
        #[arg(long)]
        failed: bool,
        /// Prune both interrupted AND failed job runs.
        #[arg(long)]
        all: bool,
        /// Only prune manifests whose `started_at` is older than this many
        /// days. Default: 7. Pass `0` to prune everything matching the
        /// selected status (useful for tests / one-off cleanup).
        #[arg(long, default_value_t = 7)]
        older_than: u64,
    },
}

#[derive(clap::Subcommand)]
enum ReviewAction {
    /// List pending review items with counts by kind
    List,
    /// Show details of a review item
    Show {
        /// Review item ID
        #[arg(required = true)]
        id: String,
    },
    /// Approve a pending review item and execute the change
    Approve {
        /// Review item ID
        #[arg(required = true)]
        id: String,
    },
    /// Reject a pending review item
    Reject {
        /// Review item ID
        #[arg(required = true)]
        id: String,
        /// Rejection reason
        #[arg(long)]
        reason: Option<String>,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if let Err(err) = run(cli) {
        let exit_code = err
            .downcast_ref::<ExitCodeError>()
            .map(|err| err.exit_code)
            .or_else(|| {
                err.downcast_ref::<ValidationError>()
                    .map(|err| err.exit_code)
            })
            .unwrap_or(1);
        eprintln!("error: {err:#}");
        std::process::exit(exit_code);
    }
}

#[derive(Debug)]
struct ExitCodeError {
    exit_code: i32,
    message: String,
}

impl std::fmt::Display for ExitCodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExitCodeError {}

/// User-input validation error.
///
/// Returned by command handlers when a request is rejected *before* any
/// mutating work begins (empty query, unsupported format, nonexistent
/// ingest path, unknown publish target, unknown review id). The outer
/// CLI exits 1 with the message — same as [`ExitCodeError`] — but
/// [`execute_mutating_command_with_handle`] treats these specially:
/// instead of recording a `status: failed` job manifest, it deletes
/// the manifest entirely so the rejection never pollutes
/// `kb status` / `kb doctor` failed-job counts.
///
/// Rule of thumb: if the check fires before the job has actually opened
/// any downstream locks or touched the KB state, it's a
/// `ValidationError`. If the job has started mutating work (LLM call,
/// compile pass, file write), stick with `bail!`/`ExitCodeError` so the
/// failure is recorded. See bn-1jx.
#[derive(Debug)]
struct ValidationError {
    exit_code: i32,
    message: String,
}

impl ValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            exit_code: 1,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}


#[allow(clippy::too_many_lines)]
fn run(cli: Cli) -> Result<()> {
    let root = if matches!(cli.command, Some(Command::Init { .. }) | None) {
        cli.root
    } else {
        Some(root::discover_root(cli.root.as_deref())?.path)
    };

    // Validate kb.toml upfront for every command except `kb init`. A broken
    // config is always a bug worth surfacing immediately — otherwise
    // read-only commands like `status`, `search`, and `inspect` would run
    // silently against defaults and the user wouldn't learn their config
    // is broken until they try `ask` or `compile`. `kb init --force` is
    // allowed to overwrite a broken config, so init skips this check.
    if !matches!(cli.command, Some(Command::Init { .. }) | None)
        && let Some(root_path) = root.as_deref()
    {
        Config::load_from_root(root_path, None).with_context(|| {
            format!(
                "kb.toml is invalid at {}/kb.toml — fix it manually or run `kb init --force` to regenerate",
                root_path.display()
            )
        })?;
    }

    match cli.command {
        Some(Command::Compile) => {
            let compile_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let force = cli.force;
            let dry_run = cli.dry_run;
            let json = cli.json;
            let cli_model = cli.model.clone();
            if dry_run {
                // Dry-run reads graph/hashes and prints what would happen; it
                // writes nothing. Bypass `execute_mutating_command_with_handle`
                // so we neither block on the root lock (held by an in-flight
                // real compile) nor leave a job manifest behind if the caller
                // SIGPIPEs us (e.g. `kb compile --dry-run | head`).
                run_compile_action(compile_root, force, true, json, cli_model.as_deref(), None)
            } else {
                execute_mutating_command_with_handle(Some(compile_root), "compile", move |handle| {
                    // Stream per-pass events into `state/jobs/<id>.log` so a
                    // hung or failing compile leaves a useful trail.
                    run_compile_action(
                        compile_root,
                        force,
                        false,
                        json,
                        cli_model.as_deref(),
                        Some(handle.log_sink()),
                    )
                })
            }
        }
        Some(Command::Doctor) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_doctor_command(root, cli.json, cli.model.as_deref())
        }
        Some(Command::Ask { query, format, promote }) => {
            let ask_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let query = resolve_query(query)?;
            let model = cli.model.clone();
            let dry_run = cli.dry_run;
            let json = cli.json;
            let action = move || {
                run_ask(
                    ask_root,
                    &query,
                    format.as_deref(),
                    model.as_deref(),
                    json,
                    dry_run,
                    promote,
                )
            };
            if dry_run {
                // Dry-run doesn't write anything; no lock needed.
                action()
            } else {
                execute_mutating_command(Some(ask_root), "ask", action)
            }
        }
        Some(Command::Ingest { sources, allow_empty }) => {
            // bn-1jx: validate local source paths exist before we acquire
            // the root lock and start a job manifest. `kb_ingest::collect_files`
            // bails with the same message, but doing it here means a bad
            // path never leaves a "failed" job behind in `state/jobs/`.
            for source in &sources {
                if kb_ingest::is_url(source) {
                    continue;
                }
                let path = Path::new(source);
                if !path.exists() {
                    return Err(ValidationError::new(format!(
                        "source path does not exist or is not a regular file/directory: {source}"
                    ))
                    .into());
                }
            }
            let ingest_root = root.clone();
            let action = move || {
                let root = ingest_root
                    .as_deref()
                    .expect("root resolved for non-init commands");
                run_ingest(root, &sources, cli.json, cli.dry_run, allow_empty)
            };

            if cli.dry_run {
                action()
            } else {
                execute_mutating_command(root.as_deref(), "ingest", action)
            }
        }
        Some(Command::Lint { check, strict }) => {
            // Lint is read-only: it walks generated artifacts and surfaces findings.
            // It must not take the root lock, so users can `kb lint` during a compile.
            let lint_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_lint(lint_root, cli.json, check.as_deref(), strict)
        }
        Some(Command::Publish { target }) => {
            let publish_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let dry_run = cli.dry_run;
            let json = cli.json;
            // bn-1jx: pre-validate the target exists in kb.toml before we
            // enter `execute_mutating_command`. A bad target name is user
            // error, not a system failure — no job should be recorded.
            let cfg = config::Config::load_from_root(publish_root, None)?;
            let mut available: Vec<String> =
                cfg.publish.targets.keys().cloned().collect();
            available.sort();
            let Some(target_cfg) = cfg.publish.targets.get(&target).cloned() else {
                return Err(ValidationError::new(
                    publish::target_not_found_message(&target, &available),
                )
                .into());
            };
            execute_mutating_command(Some(publish_root), "publish", move || {
                publish::run_publish(publish_root, &target, &target_cfg, dry_run, json)
            })
        }
        Some(Command::Status) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            jobs::check_stale_jobs(root)?;
            let status = gather_status(root)?;
            if cli.json {
                emit_json("status", &status)?;
                return Ok(());
            }
            print_status(&status);
            Ok(())
        }
        Some(Command::Init { path, reset_config }) => {
            init::init(root, path, cli.force, reset_config, cli.quiet)
        }
        Some(Command::Search { query, limit }) => {
            if query.trim().is_empty() {
                bail!("search: query cannot be empty");
            }
            let search_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let index = kb_query::LexicalIndex::load(search_root)?;
            let limit = limit.unwrap_or(10);

            // Detect the "query reduced entirely to stopwords" case before
            // scoring so we can surface a helpful message instead of a silent
            // empty result set. Exit 0 — this is a user-facing hint, not an
            // error.
            if kb_query::query_reduced_to_stopwords(&query) {
                if cli.json {
                    let empty: Vec<kb_query::SearchResult> = Vec::new();
                    emit_json("search", &empty)?;
                } else {
                    println!(
                        "No results for '{query}': query reduced to stopwords; try more specific terms."
                    );
                }
                return Ok(());
            }

            let results = index.search(&query, limit);
            if cli.json {
                emit_json("search", &results)?;
            } else if results.is_empty() {
                println!("No results for '{query}'");
                if kb_query::lexical_index_path(search_root).exists() {
                    println!(
                        "Tip: try a broader query or `kb search --json` for ranking details."
                    );
                } else {
                    println!("Tip: run 'kb compile' to build the search index.");
                }
            } else {
                for result in &results {
                    println!("{} [score: {}]", result.title, result.score);
                    println!("  {}", result.id);
                    for reason in &result.reasons {
                        println!("    reason: {reason}");
                    }
                }
            }
            Ok(())
        }
        Some(Command::Inspect { trace, target }) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_inspect(root, &target, cli.json, trace)
        }
        Some(Command::Forget { target, no_cascade }) => {
            let forget_root = root
                .as_deref()
                .expect("root resolved for non-init commands")
                .to_path_buf();
            let flags = ForgetFlags {
                dry_run: cli.dry_run,
                json: cli.json,
                force: cli.force,
                quiet: cli.quiet,
                no_cascade,
            };

            // Dry-run is a pure read: no lock, no job manifest, no writes.
            // Matches `ingest`'s dry-run handling above.
            if flags.dry_run {
                return run_forget(&forget_root, &target, flags);
            }

            let forget_root_for_lock = forget_root.clone();
            execute_mutating_command(Some(&forget_root_for_lock), "forget", move || {
                run_forget(&forget_root, &target, flags)
            })
        }
        Some(Command::Review { action }) => {
            let review_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let json = cli.json;
            let json_emitter = |cmd: &str, data: serde_json::Value| -> Result<()> {
                emit_json(cmd, data)
            };
            match action {
                ReviewAction::List => {
                    review::run_review_list(review_root, json, &json_emitter)
                }
                ReviewAction::Show { id } => {
                    review::run_review_show(review_root, &id, json, &json_emitter)
                }
                ReviewAction::Approve { id } => {
                    // bn-1jx: an unknown review id is user error, not a system
                    // failure. Check for it before entering the mutating-
                    // command wrapper so no failed-job manifest is left behind.
                    if kb_core::load_review_item(review_root, &id)?.is_none() {
                        return Err(ValidationError::new(format!(
                            "review item '{id}' not found"
                        ))
                        .into());
                    }
                    execute_mutating_command(Some(review_root), "review.approve", move || {
                        review::run_review_approve(review_root, &id, json, &json_emitter)
                    })
                }
                ReviewAction::Reject { id, reason } => {
                    // Same treatment as Approve above — see bn-1jx.
                    if kb_core::load_review_item(review_root, &id)?.is_none() {
                        return Err(ValidationError::new(format!(
                            "review item '{id}' not found"
                        ))
                        .into());
                    }
                    execute_mutating_command(Some(review_root), "review.reject", move || {
                        review::run_review_reject(review_root, &id, reason.as_deref(), json, &json_emitter)
                    })
                }
            }
        }
        Some(Command::Jobs { action }) => {
            let jobs_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let json = cli.json;
            match action {
                None => {
                    // No default action: print help text and bail out. Echoes
                    // clap's convention for subcommand groups without a
                    // sensible "top-level verb" (cf. `kb review` with no
                    // action also prints help).
                    println!("kb jobs: inspect and prune job-run manifests");
                    println!();
                    println!("USAGE:");
                    println!("  kb jobs list --interrupted|--failed [--page N --page-size N]");
                    println!("  kb jobs prune --interrupted|--failed|--all [--older-than DAYS]");
                    println!();
                    println!("Run 'kb jobs --help' for details.");
                    Ok(())
                }
                Some(JobsAction::List {
                    interrupted,
                    failed,
                    page_size,
                    page,
                }) => jobs_cmd::run_list(jobs_root, interrupted, failed, page, page_size, json),
                Some(JobsAction::Prune {
                    interrupted,
                    failed,
                    all,
                    older_than,
                }) => {
                    execute_mutating_command(Some(jobs_root), "jobs.prune", move || {
                        jobs_cmd::run_prune(jobs_root, interrupted, failed, all, older_than, json)
                    })
                }
            }
        }
        None => {
            println!("kb: a personal knowledge base compiler");
            println!("Run 'kb --help' for more information");
            Ok(())
        }
    }
}

#[derive(Debug, Serialize)]
struct DoctorPayload {
    checks: Vec<DoctorCheck>,
    status: &'static str,
    warning_count: usize,
    error_count: usize,
    exit_code: i32,
}

#[derive(Debug, Clone, Serialize)]
struct DoctorCheck {
    name: String,
    status: &'static str,
    summary: String,
    remediation: Option<String>,
    details: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DoctorSeverity {
    Ok,
    Warn,
    Error,
}

impl DoctorSeverity {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }

    const fn exit_code(self) -> i32 {
        match self {
            Self::Ok => 0,
            Self::Warn => 1,
            Self::Error => 2,
        }
    }
}

fn run_doctor_command(root: &Path, json: bool, cli_model: Option<&str>) -> Result<()> {
    // Doctor is a read-only diagnostic — it must not take the root lock, so that users
    // can run `kb doctor` while a long compile is in flight to investigate.
    jobs::check_stale_jobs(root)?;
    run_doctor(root, json, cli_model)
}

#[allow(clippy::too_many_lines)]
fn run_doctor(root: &Path, json: bool, cli_model: Option<&str>) -> Result<()> {
    let mut checks = Vec::new();

    let config_result = Config::load_from_root(root, cli_model);
    let cfg = match config_result {
        Ok(cfg) => {
            checks.push(DoctorCheck {
                name: "config".to_string(),
                status: DoctorSeverity::Ok.as_str(),
                summary: format!("Parsed {} successfully.", Config::FILE_NAME),
                remediation: None,
                details: None,
            });
            Some(cfg)
        }
        Err(err) => {
            checks.push(DoctorCheck {
                name: "config".to_string(),
                status: DoctorSeverity::Error.as_str(),
                summary: format!("Failed to parse {}.", Config::FILE_NAME),
                remediation: Some(
                    "Fix kb.toml syntax or unknown fields, then rerun `kb doctor`.".to_string(),
                ),
                details: Some(err.to_string()),
            });
            None
        }
    };

    checks.push(check_root_writable(root));

    if let Some(cfg) = cfg.as_ref() {
        checks.push(check_prompt_template_directory(root, cfg));
        checks.extend(check_prompt_templates_load(root));
        checks.extend(check_runner_health(root, cfg));
    } else {
        checks.push(DoctorCheck {
            name: "prompt_templates".to_string(),
            status: DoctorSeverity::Warn.as_str(),
            summary: "Skipped prompt template checks because config did not parse.".to_string(),
            remediation: Some("Fix kb.toml so doctor can validate prompt settings.".to_string()),
            details: None,
        });
        checks.push(DoctorCheck {
            name: "llm_runners".to_string(),
            status: DoctorSeverity::Warn.as_str(),
            summary: "Skipped model access and harness checks because config did not parse."
                .to_string(),
            remediation: Some(
                "Fix kb.toml so doctor can validate configured backends.".to_string(),
            ),
            details: None,
        });
    }

    checks.push(check_interrupted_jobs(root)?);

    let error_count = checks
        .iter()
        .filter(|check| check.status == "error")
        .count();
    let warning_count = checks.iter().filter(|check| check.status == "warn").count();
    let severity = if error_count > 0 {
        DoctorSeverity::Error
    } else if warning_count > 0 {
        DoctorSeverity::Warn
    } else {
        DoctorSeverity::Ok
    };

    if json {
        emit_json("doctor", DoctorPayload {
            checks: checks.clone(),
            status: severity.as_str(),
            warning_count,
            error_count,
            exit_code: severity.exit_code(),
        })?;
    } else {
        for check in &checks {
            println!(
                "[{}] {}: {}",
                check.status.to_uppercase(),
                check.name,
                check.summary
            );
            if let Some(details) = &check.details {
                println!("  details: {details}");
            }
            if let Some(remediation) = &check.remediation {
                println!("  remediation: {remediation}");
            }
        }
        println!(
            "kb doctor: {} (warnings: {warning_count}, errors: {error_count})",
            severity.as_str().to_uppercase()
        );
    }

    if severity == DoctorSeverity::Ok {
        Ok(())
    } else {
        Err(ExitCodeError {
            exit_code: severity.exit_code(),
            message: format!(
                "doctor completed with {warning_count} warning(s) and {error_count} error(s)"
            ),
        }
        .into())
    }
}

fn check_root_writable(root: &Path) -> DoctorCheck {
    let probe = root.join("state").join(".kb-doctor-write-test");
    match fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = fs::remove_file(&probe);
            DoctorCheck {
                name: "root_writable".to_string(),
                status: DoctorSeverity::Ok.as_str(),
                summary: format!("KB root is writable at {}.", root.display()),
                remediation: None,
                details: None,
            }
        }
        Err(err) => DoctorCheck {
            name: "root_writable".to_string(),
            status: DoctorSeverity::Error.as_str(),
            summary: format!("KB root is not writable at {}.", root.display()),
            remediation: Some(
                "Fix filesystem permissions for the KB root and its state directory.".to_string(),
            ),
            details: Some(err.to_string()),
        },
    }
}

fn check_prompt_template_directory(root: &Path, cfg: &Config) -> DoctorCheck {
    let prompt_dir = root.join(&cfg.data.prompt_templates);
    if prompt_dir.is_dir() {
        DoctorCheck {
            name: "prompt_template_dir".to_string(),
            status: DoctorSeverity::Ok.as_str(),
            summary: format!(
                "Prompt template directory exists at {}.",
                prompt_dir.display()
            ),
            remediation: None,
            details: None,
        }
    } else {
        DoctorCheck {
            name: "prompt_template_dir".to_string(),
            status: DoctorSeverity::Error.as_str(),
            summary: format!(
                "Prompt template directory is missing at {}.",
                prompt_dir.display()
            ),
            remediation: Some(
                "Create the prompt template directory or re-run `kb init`.".to_string(),
            ),
            details: None,
        }
    }
}

fn check_prompt_templates_load(root: &Path) -> Vec<DoctorCheck> {
    [
        "summarize_document.md",
        "extract_concepts.md",
        "merge_concept_candidates.md",
    ]
    .into_iter()
    .map(
        |template_name| match Template::load(template_name, Some(root)) {
            Ok(_) => DoctorCheck {
                name: format!("prompt_template:{template_name}"),
                status: DoctorSeverity::Ok.as_str(),
                summary: format!("Loaded prompt template {template_name}."),
                remediation: None,
                details: None,
            },
            Err(err) => DoctorCheck {
                name: format!("prompt_template:{template_name}"),
                status: DoctorSeverity::Error.as_str(),
                summary: format!("Failed to load prompt template {template_name}."),
                remediation: Some(
                    "Restore the missing template file or rely on the bundled default template."
                        .to_string(),
                ),
                details: Some(err.to_string()),
            },
        },
    )
    .collect()
}

fn check_runner_health(root: &Path, cfg: &Config) -> Vec<DoctorCheck> {
    let default_model = cfg.llm.default_model.clone();
    let mut runner_names: Vec<_> = cfg.llm.runners.keys().cloned().collect();
    runner_names.sort_unstable();

    let mut checks = Vec::new();
    for name in runner_names {
        let runner = cfg
            .llm
            .runners
            .get(&name)
            .expect("runner key from keys() must exist");
        let model = runner
            .model
            .clone()
            .unwrap_or_else(|| default_model.clone());
        let binary = normalize_binary_command(&runner.command);

        if let Some(path) = command_on_path(&binary) {
            checks.push(DoctorCheck {
                name: format!("harness:{name}"),
                status: DoctorSeverity::Ok.as_str(),
                summary: format!("Found harness binary `{binary}` for runner `{name}`."),
                remediation: None,
                details: Some(format!(
                    "command=`{}` path={}",
                    runner.command,
                    path.display()
                )),
            });
        } else {
            checks.push(DoctorCheck {
                name: format!("harness:{name}"),
                status: DoctorSeverity::Error.as_str(),
                summary: format!("Harness binary `{binary}` is missing for runner `{name}`."),
                remediation: Some(format!(
                    "Install `{binary}` or update llm.runners.{name}.command in kb.toml."
                )),
                details: Some(format!("command=`{}` model={model}", runner.command)),
            });
            continue;
        }

        checks.push(run_runner_health_check(&name, runner, root, &default_model));
    }

    checks
}

fn run_runner_health_check(
    runner_name: &str,
    runner: &LlmRunnerConfig,
    root: &Path,
    default_model: &str,
) -> DoctorCheck {
    match runner_name {
        "opencode" => run_opencode_health_check(runner_name, runner, root, default_model),
        "claude" => run_claude_health_check(runner_name, runner, default_model),
        _ => DoctorCheck {
            name: format!("model_access:{runner_name}"),
            status: DoctorSeverity::Warn.as_str(),
            summary: format!(
                "Runner `{runner_name}` is configured but doctor has no backend health check for it."
            ),
            remediation: Some(
                "Add doctor support for this runner or validate it manually.".to_string(),
            ),
            details: Some(format!("command=`{}`", runner.command)),
        },
    }
}

fn run_opencode_health_check(
    runner_name: &str,
    runner: &LlmRunnerConfig,
    root: &Path,
    default_model: &str,
) -> DoctorCheck {
    let model = runner
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string());

    let command = normalize_binary_command(&runner.command);
    let adapter = OpencodeAdapter::new(OpencodeConfig {
        command,
        model: model.clone(),
        tools_read: runner.tools_read,
        tools_write: runner.tools_write,
        tools_edit: runner.tools_edit,
        tools_bash: runner.tools_bash,
        timeout: Duration::from_secs(runner.timeout_seconds.unwrap_or(900)),
        project_root: Some(root.to_path_buf()),
        ..Default::default()
    });

    match adapter.run_health_check(RunHealthCheckRequest {
        check_details: None,
    }) {
        Ok((response, provenance)) => {
            let severity = match response.status.as_str() {
                "healthy" => DoctorSeverity::Ok,
                "degraded" => DoctorSeverity::Warn,
                _ => DoctorSeverity::Error,
            };
            DoctorCheck {
                name: format!("model_access:{runner_name}"),
                status: severity.as_str(),
                summary: format!(
                    "Runner `{runner_name}` reported {} for model `{}`.",
                    response.status, provenance.model
                ),
                remediation: (severity != DoctorSeverity::Ok).then_some(
                    "Check authentication, selected model access, and harness configuration."
                        .to_string(),
                ),
                details: Some(format!(
                    "harness={} version={:?} latency_ms={} command=`{}`{}",
                    provenance.harness,
                    provenance.harness_version,
                    provenance.latency_ms,
                    runner.command,
                    response
                        .details
                        .as_ref()
                        .map_or(String::new(), |details| format!(" details={details}"))
                )),
            }
        }
        Err(err) => DoctorCheck {
            name: format!("model_access:{runner_name}"),
            status: DoctorSeverity::Error.as_str(),
            summary: format!("Runner `{runner_name}` health check failed for model `{model}`."),
            remediation: Some(
                "Verify the harness can authenticate and that the configured model is available."
                    .to_string(),
            ),
            details: Some(format!("command=`{}` error={err}", runner.command)),
        },
    }
}

fn run_claude_health_check(
    runner_name: &str,
    runner: &LlmRunnerConfig,
    default_model: &str,
) -> DoctorCheck {
    let model = runner
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string());

    let command = normalize_binary_command(&runner.command);
    let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
        command,
        model: runner.model.clone(),
        permission_mode: runner.permission_mode.clone(),
        timeout: Duration::from_secs(runner.timeout_seconds.unwrap_or(900)),
        project_root: None,
    });

    match adapter.run_health_check(RunHealthCheckRequest {
        check_details: None,
    }) {
        Ok((response, provenance)) => {
            let severity = match response.status.as_str() {
                "healthy" => DoctorSeverity::Ok,
                "degraded" => DoctorSeverity::Warn,
                _ => DoctorSeverity::Error,
            };
            DoctorCheck {
                name: format!("model_access:{runner_name}"),
                status: severity.as_str(),
                summary: format!(
                    "Runner `{runner_name}` reported {} for model `{}`.",
                    response.status, provenance.model
                ),
                remediation: (severity != DoctorSeverity::Ok).then_some(
                    "Check authentication, selected model access, and harness configuration."
                        .to_string(),
                ),
                details: Some(format!(
                    "harness={} version={:?} latency_ms={} command=`{}`{}",
                    provenance.harness,
                    provenance.harness_version,
                    provenance.latency_ms,
                    runner.command,
                    response
                        .details
                        .as_ref()
                        .map_or(String::new(), |details| format!(" details={details}"))
                )),
            }
        }
        Err(err) => DoctorCheck {
            name: format!("model_access:{runner_name}"),
            status: DoctorSeverity::Error.as_str(),
            summary: format!("Runner `{runner_name}` health check failed for model `{model}`."),
            remediation: Some(
                "Verify the harness can authenticate and that the configured model is available."
                    .to_string(),
            ),
            details: Some(format!("command=`{}` error={err}", runner.command)),
        },
    }
}

fn check_interrupted_jobs(root: &Path) -> Result<DoctorCheck> {
    let interrupted_jobs = jobs::recent_jobs(root, 100)?
        .into_iter()
        .filter(|job| job.status == JobRunStatus::Interrupted)
        .collect::<Vec<_>>();

    if interrupted_jobs.is_empty() {
        Ok(DoctorCheck {
            name: "interrupted_jobs".to_string(),
            status: DoctorSeverity::Ok.as_str(),
            summary: "No interrupted jobs detected.".to_string(),
            remediation: None,
            details: None,
        })
    } else {
        let ids = interrupted_jobs
            .iter()
            .map(|job| job.metadata.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Ok(DoctorCheck {
            name: "interrupted_jobs".to_string(),
            status: DoctorSeverity::Warn.as_str(),
            summary: format!("Detected {} interrupted job(s).", interrupted_jobs.len()),
            remediation: Some(
                "Inspect `kb status` and the corresponding state/jobs logs before rerunning the failed work."
                    .to_string(),
            ),
            details: Some(format!("job_ids={ids}")),
        })
    }
}

fn command_on_path(command: &str) -> Option<PathBuf> {
    let candidate = Path::new(command);
    if candidate.components().count() > 1 {
        return candidate.is_file().then(|| candidate.to_path_buf());
    }

    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|path| path.join(command))
            .find(|path| path.is_file())
    })
}

fn normalize_binary_command(command: &str) -> String {
    command
        .split_whitespace()
        .next()
        .unwrap_or(command)
        .to_string()
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

/// Flags extracted from `Cli` that `run_forget` cares about.
///
/// Packed into a struct (instead of passed as separate `bool`s) to keep the
/// handler signature under clippy's `fn_params_excessive_bools` threshold.
/// The first four flags come from global top-level CLI flags (`--dry-run`,
/// `--json`, `--force`, `--quiet`). `no_cascade` is forget-specific (see
/// the `Forget` subcommand definition).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy)]
struct ForgetFlags {
    dry_run: bool,
    json: bool,
    force: bool,
    quiet: bool,
    no_cascade: bool,
}

/// Implement `kb forget <target>`. See `forget.rs` for the helpers.
///
/// `flags.dry_run` prints the plan without touching disk. The confirmation
/// prompt is skipped in three cases: `--force` (explicit opt-out), `--quiet`
/// (non-interactive contexts like scripts), and `--json` (callers parsing
/// output can't answer `y/N`; a typo-protection prompt would just deadlock
/// them).
fn run_forget(root: &Path, target: &str, flags: ForgetFlags) -> Result<()> {
    let (src_id, origin) = forget::resolve_target(root, target)?;
    let plan = forget::plan(root, &src_id, origin, !flags.no_cascade)?;

    // Dry-run and JSON paths emit the plan and return without touching disk.
    if flags.dry_run {
        // Preview the bn-i5r post-trash refreshes so users see what live
        // execution would do beyond the trash moves.
        let preview = forget::preview_refresh(root, &plan.src_id).unwrap_or_default();
        if flags.json {
            emit_json(
                "forget",
                forget::ForgetOutcome {
                    plan,
                    dry_run: true,
                    backlinks_refreshed: false,
                    cascade_refresh: preview,
                },
            )?;
        } else {
            print!("{}", forget::render_plan(&plan, true));
            print!("{}", forget::render_refresh_footer(&plan, &preview, true));
        }
        return Ok(());
    }

    if plan.moves.is_empty() && plan.cascade.is_empty() {
        // Nothing to remove. Report so users don't wonder whether the op
        // silently failed. No job manifest needed — we already opened one
        // around this handler, and "nothing to do" is a legitimate success.
        if flags.json {
            emit_json(
                "forget",
                forget::ForgetOutcome {
                    plan,
                    dry_run: false,
                    backlinks_refreshed: false,
                    cascade_refresh: forget::CascadeRefresh::default(),
                },
            )?;
        } else {
            println!(
                "forget {src_id}: nothing to remove (no normalized/, raw/inbox/, or wiki/sources/ entry)"
            );
        }
        return Ok(());
    }

    if !flags.force
        && !flags.quiet
        && !flags.json
        && !forget::confirm_on_stderr(&plan)?
    {
        // Declining the confirmation is the "user changed their mind" happy
        // path — return Ok so exit code is 0 and the job manifest records
        // Succeeded (nothing was done, nothing failed).
        println!("forget cancelled");
        return Ok(());
    }

    let outcome = forget::execute(root, &plan)?;

    if flags.json {
        emit_json(
            "forget",
            forget::ForgetOutcome {
                plan,
                dry_run: false,
                backlinks_refreshed: outcome.backlinks_refreshed,
                cascade_refresh: outcome.cascade_refresh,
            },
        )?;
    } else {
        print!("{}", forget::render_plan(&plan, false));
        if outcome.backlinks_refreshed {
            println!("  backlinks refreshed");
        }
        print!(
            "{}",
            forget::render_refresh_footer(&plan, &outcome.cascade_refresh, false)
        );
    }
    Ok(())
}

fn run_ingest(
    root: &Path,
    sources: &[String],
    json: bool,
    dry_run: bool,
    allow_empty: bool,
) -> Result<()> {
    let mut urls = Vec::new();
    let mut local_paths = Vec::new();
    for source in sources {
        if kb_ingest::is_url(source) {
            urls.push(source.as_str());
        } else {
            local_paths.push(PathBuf::from(source));
        }
    }

    let mut results = Vec::new();
    for report in
        kb_ingest::ingest_paths_with_flags(root, &local_paths, dry_run, allow_empty)?
    {
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
        emit_json("ingest", IngestPayload { dry_run, results, summary })?;
        return Ok(());
    }

    if results.is_empty() {
        println!("No sources ingested");
        return Ok(());
    }

    for item in &results {
        let tag = if dry_run {
            format!("[would-{}]", outcome_tag(item.outcome))
        } else {
            format!("[{}]", outcome_tag(item.outcome))
        };
        println!("{tag}  {}", item.input);
        println!(
            "  src:        {}  rev: {}",
            item.source_document_id, item.source_revision_id
        );
        if !matches!(item.outcome, kb_ingest::IngestOutcome::Skipped) {
            println!("  saved:      {}", item.content_path.display());
            println!(
                "  normalized: {}",
                normalized_dir_for(item).display()
            );
        }
    }
    println!(
        "Summary: {} total | {} new sources | {} new revisions | {} skipped",
        summary.total, summary.new_sources, summary.new_revisions, summary.skipped
    );

    Ok(())
}

fn normalized_dir_for(item: &IngestResult) -> PathBuf {
    if item.source_kind == "url" {
        // For URL ingest, content_path already points into normalized/<id>/...
        // Report the directory containing it.
        item.content_path
            .parent()
            .map_or_else(|| item.content_path.clone(), Path::to_path_buf)
    } else {
        PathBuf::from("normalized").join(&item.source_document_id)
    }
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

const fn outcome_tag(outcome: kb_ingest::IngestOutcome) -> &'static str {
    match outcome {
        kb_ingest::IngestOutcome::NewSource => "new-source",
        kb_ingest::IngestOutcome::NewRevision => "new-revision",
        kb_ingest::IngestOutcome::Skipped => "skipped",
    }
}

#[derive(Serialize)]
struct AskOutput<'a> {
    question_id: &'a str,
    question_path: &'a str,
    artifact_path: &'a str,
    requested_format: &'a str,
}

#[allow(clippy::too_many_lines, clippy::fn_params_excessive_bools)]
fn run_ask(
    root: &Path,
    query: &str,
    requested_format: Option<&str>,
    cli_model: Option<&str>,
    json: bool,
    dry_run: bool,
    promote: bool,
) -> Result<()> {
    if query.trim().is_empty() {
        // bn-1jx: validation rejection — never reaches the LLM, never writes
        // a question page. Return `ValidationError` so the outer mutating-
        // command wrapper discards the optimistic job manifest.
        return Err(ValidationError::new("ask: question cannot be empty").into());
    }
    let cfg = Config::load_from_root(root, cli_model)?;
    let requested_format =
        normalize_ask_format(requested_format.unwrap_or(cfg.ask.artifact_default_format.as_str()))?;

    // `png` is accepted by clap (so `--help` keeps advertising it as a
    // placeholder for future support) but we refuse it cleanly rather than
    // silently writing markdown under a `.png` label. See bn-iiq.
    //
    // bn-1jx: this rejection runs before any retrieval/LLM work — mark it
    // as validation so `kb status` doesn't count it as a failed run.
    if requested_format == "png" {
        return Err(ValidationError::new(
            "--format png is not yet supported; supported formats: md, marp, json",
        )
        .into());
    }

    let retrieval_plan =
        kb_query::LexicalIndex::load(root)?.plan_retrieval(query, cfg.ask.token_budget);

    if dry_run {
        if json {
            emit_json("ask", &retrieval_plan)?;
        } else {
            println!("Retrieval plan for: {query}");
            println!(
                "Token budget: {} | Estimated tokens: {}",
                retrieval_plan.token_budget, retrieval_plan.estimated_tokens
            );
            println!("Candidates ({}):", retrieval_plan.candidates.len());
            for candidate in &retrieval_plan.candidates {
                println!(
                    "  {} [score: {}, ~{} tokens]",
                    candidate.id, candidate.score, candidate.estimated_tokens
                );
                for reason in &candidate.reasons {
                    println!("    reason: {reason}");
                }
            }
        }
        return Ok(());
    }

    let timestamp = now_millis()?;
    let question_id = format!("question-{}", unique_question_suffix(timestamp, query));

    // Filename tracks the requested format. JSON gets `.json`; everything
    // else (md/marp) stays as `.md`. Keep in sync with `kb_query::write_artifact`.
    let answer_file_name = match requested_format {
        "json" => "answer.json",
        _ => "answer.md",
    };
    let answer_rel = PathBuf::from("outputs/questions")
        .join(&question_id)
        .join(answer_file_name);
    let question_rel = PathBuf::from("outputs/questions")
        .join(&question_id)
        .join("question.json");
    let plan_rel = PathBuf::from("outputs/questions")
        .join(&question_id)
        .join("retrieval_plan.json");

    let assembled = kb_query::assemble_context(root, &retrieval_plan)?;
    let citation_manifest = kb_query::build_citation_manifest(&assembled);
    let manifest_text = kb_query::render_manifest_for_prompt(&citation_manifest);

    let llm_outcome = try_generate_answer(&cfg, root, query, &assembled, &manifest_text);

    let (model_version, template_hash, artifact_status, artifact_body, llm_info) =
        match llm_outcome {
            Ok((result, provenance)) => (
                Some(provenance.model.clone()),
                Some(provenance.prompt_template_hash.to_hex()),
                if result.invalid_citations.is_empty() {
                    Status::Fresh
                } else {
                    Status::NeedsReview
                },
                result.body.clone(),
                Some((result, provenance)),
            ),
            Err(err) => (
                None,
                None,
                Status::Failed,
                format!(
                    "> **LLM unavailable:** {err}\n\n\
                     Question recorded. Re-run `kb ask` when a backend is available.\n"
                ),
                None,
            ),
        };

    let question = Question {
        metadata: EntityMetadata {
            id: question_id.clone(),
            created_at_millis: timestamp,
            updated_at_millis: now_millis()?,
            source_hashes: Vec::new(),
            model_version: model_version.clone(),
            tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: template_hash,
            dependencies: Vec::new(),
            output_paths: vec![question_rel, answer_rel.clone(), plan_rel],
            status: artifact_status,
        },
        raw_query: query.to_string(),
        requested_format: requested_format.to_string(),
        requesting_context: QuestionContext::ProjectKb,
        retrieval_plan: format!(
            "outputs/questions/{question_id}/retrieval_plan.json"
        ),
        token_budget: Some(cfg.ask.token_budget),
    };

    let artifact = Artifact {
        metadata: EntityMetadata {
            id: format!("artifact-{question_id}"),
            created_at_millis: timestamp,
            updated_at_millis: now_millis()?,
            source_hashes: Vec::new(),
            model_version,
            tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: None,
            dependencies: vec![question_id.clone()],
            output_paths: vec![answer_rel],
            status: artifact_status,
        },
        question_id: question_id.clone(),
        artifact_kind: match requested_format {
            "png" => ArtifactKind::Figure,
            "marp" => ArtifactKind::SlideDeck,
            "json" => ArtifactKind::JsonSpec,
            _ => ArtifactKind::AnswerNote,
        },
        format: requested_format.to_string(),
        output_path: PathBuf::from(format!(
            "outputs/questions/{question_id}/{answer_file_name}"
        )),
    };

    // Emit a BuildRecord for this ask, when the LLM actually produced something.
    // Placeholder artifacts (backend unavailable) get no BuildRecord — there's no
    // real input-to-output mapping to record.
    let build_record_id: Option<String> = if let Some((_, provenance)) = llm_info.as_ref() {
        let record_id = format!("build:ask:{question_id}");
        let source_ids: Vec<String> = retrieval_plan
            .candidates
            .iter()
            .map(|c| c.id.clone())
            .collect();
        let manifest_hash = hash_many(&[
            question_id.as_bytes(),
            b"\0",
            artifact_body.as_bytes(),
            b"\0",
            provenance.prompt_render_hash.to_hex().as_bytes(),
        ])
        .to_hex();
        let record = kb_core::BuildRecord {
            pass_name: "ask".to_string(),
            metadata: EntityMetadata {
                id: record_id.clone(),
                created_at_millis: timestamp,
                updated_at_millis: now_millis()?,
                source_hashes: vec![provenance.prompt_render_hash.to_hex()],
                model_version: Some(provenance.model.clone()),
                tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
                prompt_template_hash: Some(provenance.prompt_template_hash.to_hex()),
                dependencies: source_ids.clone(),
                output_paths: vec![artifact.output_path.clone()],
                status: artifact_status,
            },
            input_ids: source_ids,
            output_ids: vec![artifact.output_path.to_string_lossy().into_owned()],
            manifest_hash,
        };
        kb_core::save_build_record(root, &record)?;
        Some(record_id)
    } else {
        None
    };

    // Derive the wiki-page paths actually referenced by valid [N] citations in
    // the answer body. These feed `source_document_ids` so the frontmatter
    // only lists sources that ground the answer — not the whole retrieval scope.
    let cited_source_paths: Vec<String> = llm_info
        .as_ref()
        .map(|(result, _)| {
            let mut seen: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for key in &result.valid_citations {
                if let Some(entry) = citation_manifest.entries.get(key) {
                    seen.insert(entry.source_id.clone());
                }
            }
            seen.into_iter().collect()
        })
        .unwrap_or_default();

    let write_output = kb_query::write_artifact(&kb_query::WriteArtifactInput {
        root,
        question: &question,
        artifact: &artifact,
        retrieval_plan: &retrieval_plan,
        artifact_result: llm_info.as_ref().map(|(r, _)| r),
        provenance: llm_info.as_ref().map(|(_, p)| p),
        artifact_body: &artifact_body,
        cited_source_paths: &cited_source_paths,
        build_record_id: build_record_id.as_deref(),
    })?;

    if promote {
        let proposed_destination = PathBuf::from("wiki/questions")
            .join(format!("{}.md", slug_from_title(query)));
        let citations = llm_info
            .as_ref()
            .map(|(result, _)| {
                result
                    .valid_citations
                    .iter()
                    .filter_map(|key| citation_manifest.entries.get(key))
                    .map(|entry| entry.label.clone())
                    .collect()
            })
            .unwrap_or_default();
        let review_input_hash = hash_many(&[
            query.as_bytes(),
            b"\0",
            artifact_body.as_bytes(),
            b"\0",
            artifact.output_path.to_string_lossy().as_bytes(),
        ])
        .to_hex();
        let review_item = ReviewItem {
            metadata: EntityMetadata {
                id: format!("review-{question_id}"),
                created_at_millis: now_millis()?,
                updated_at_millis: now_millis()?,
                source_hashes: vec![review_input_hash],
                model_version: None,
                tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
                prompt_template_hash: None,
                dependencies: vec![question_id.clone(), artifact.metadata.id.clone()],
                output_paths: vec![artifact.output_path.clone(), proposed_destination.clone()],
                status: Status::NeedsReview,
            },
            kind: ReviewKind::Promotion,
            target_entity_id: artifact.metadata.id,
            proposed_destination: Some(proposed_destination.clone()),
            citations,
            affected_pages: vec![proposed_destination],
            created_at_millis: now_millis()?,
            status: ReviewStatus::Pending,
            comment: format!("Promote answer for: {query}"),
        };
        kb_core::save_review_item(root, &review_item)?;
    }

    let question_path = write_output.question_path.to_string_lossy().into_owned();
    let artifact_path = write_output.answer_path.to_string_lossy().into_owned();
    if json {
        emit_json("ask", AskOutput {
            question_id: &question_id,
            question_path: &question_path,
            artifact_path: &artifact_path,
            requested_format,
        })?;
    } else if let Some((result, provenance)) = &llm_info {
        println!("Artifact written: {artifact_path}");
        println!(
            "Citations: {} valid, {} unresolved",
            result.valid_citations.len(),
            result.invalid_citations.len()
        );
        if result.has_uncertainty_banner {
            println!("Note: low source coverage — uncertainty banner added.");
        }
        println!("Model: {} ({}ms)", provenance.model, provenance.latency_ms);
        if promote {
            println!("ReviewItem created for promotion.");
        }
    } else {
        println!("Artifact written: {artifact_path}");
        println!("LLM backend unavailable — placeholder artifact created.");
        if promote {
            println!("ReviewItem created for promotion.");
        }
    }

    Ok(())
}

fn try_generate_answer(
    cfg: &Config,
    root: &Path,
    query: &str,
    assembled: &kb_query::AssembledContext,
    manifest_text: &str,
) -> Result<(kb_query::ArtifactResult, kb_llm::ProvenanceRecord)> {
    let adapter = create_ask_adapter(cfg, root)?;

    let llm_request = kb_llm::AnswerQuestionRequest {
        question: query.to_string(),
        context: vec![assembled.text.clone()],
        format: Some(manifest_text.to_string()),
    };

    let (llm_response, provenance) = adapter
        .answer_question(llm_request)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let citation_manifest = kb_query::build_citation_manifest(assembled);
    let result =
        kb_query::postprocess_answer(&llm_response.answer, &citation_manifest, assembled);

    Ok((result, provenance))
}

/// Execute one `kb compile` invocation (dry-run or real) against `root`.
///
/// Split out of the `Command::Compile` dispatch arm so the dry-run path can
/// skip `execute_mutating_command_with_handle` entirely — dry-run never
/// writes, so it must not acquire the root lock or emit a job manifest.
///
/// `log_sink` is `Some` only for real compiles (streamed into the job's
/// on-disk log); dry-run passes `None`.
fn run_compile_action(
    compile_root: &Path,
    force: bool,
    dry_run: bool,
    json: bool,
    cli_model: Option<&str>,
    log_sink: Option<std::sync::Arc<dyn kb_compile::pipeline::LogSink>>,
) -> Result<()> {
    let options = kb_compile::pipeline::CompileOptions {
        force,
        dry_run,
        // Progress lines go to stderr so `--json` stdout stays clean.
        // Suppress entirely under --json to avoid log noise.
        progress: !json,
        log_sink,
    };

    // Dry-run does not call the LLM; skip adapter construction so users can
    // preview the stale set without needing a configured backend.
    let report = if dry_run {
        kb_compile::pipeline::run_compile(compile_root, &options)?
    } else {
        match build_compile_adapter(compile_root, cli_model) {
            Ok(adapter) => kb_compile::pipeline::run_compile_with_llm(
                compile_root,
                &options,
                Some(adapter.as_ref()),
            )?,
            Err(err) => {
                tracing::warn!(
                    "LLM adapter unavailable — running compile without per-document passes: {err}"
                );
                kb_compile::pipeline::run_compile(compile_root, &options)?
            }
        }
    };

    // Near-duplicate concept detection as a safety net after merge:
    // emits review items for concept pairs that slipped past the LLM merge.
    // Dry-run skips this (no writes).
    let duplicate_review_items = if dry_run {
        0
    } else {
        let review_items = kb_lint::check_duplicate_concepts(
            compile_root,
            &kb_lint::DuplicateConceptsConfig::default(),
        )?;
        for item in &review_items {
            kb_core::save_review_item(compile_root, item)?;
        }
        review_items.len()
    };

    if json {
        emit_json(
            "compile",
            serde_json::json!({
                "total_sources": report.total_sources,
                "stale_sources": report.stale_sources,
                "build_records_emitted": report.build_records_emitted,
                "duplicate_review_items": duplicate_review_items,
                "dry_run": dry_run,
            }),
        )?;
    } else {
        println!("{}", report.render());
        if duplicate_review_items > 0 {
            println!(
                "  [ok] duplicate_concepts ({duplicate_review_items} review item(s) queued)"
            );
        }
    }

    Ok(())
}

/// Build an adapter for use by `kb compile`. Shares construction logic with `ask`
/// (so routing, timeouts, and config resolution match), but loads the config for
/// the given root with an optional CLI `--model` override.
fn build_compile_adapter(
    root: &Path,
    cli_model: Option<&str>,
) -> Result<Box<dyn kb_llm::LlmAdapter>> {
    let cfg = Config::load_from_root(root, cli_model)?;
    create_ask_adapter(&cfg, root)
}

fn create_ask_adapter(
    cfg: &Config,
    root: &Path,
) -> Result<Box<dyn kb_llm::LlmAdapter>> {
    let runner_name = &cfg.llm.default_runner;
    let runner = cfg
        .llm
        .runners
        .get(runner_name)
        .ok_or_else(|| anyhow::anyhow!("configured runner '{runner_name}' not found in kb.toml"))?;

    let model = runner
        .model
        .clone()
        .unwrap_or_else(|| cfg.llm.default_model.clone());

    // Build a router that reflects every configured runner, so claude models route
    // to ClaudeCode even when `default_runner = "opencode"`.
    let mut configured = Vec::new();
    for name in cfg.llm.runners.keys() {
        match name.as_str() {
            "claude" => configured.push(kb_llm::Backend::ClaudeCode),
            "pi" => configured.push(kb_llm::Backend::Pi),
            _ => configured.push(kb_llm::Backend::Opencode),
        }
    }
    let default_backend = match runner_name.as_str() {
        "claude" => kb_llm::Backend::ClaudeCode,
        "pi" => kb_llm::Backend::Pi,
        _ => kb_llm::Backend::Opencode,
    };
    let router = kb_llm::Router::with_backends(default_backend, configured);

    let backend = router
        .route_model(&model)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    match backend {
        kb_llm::Backend::ClaudeCode => {
            let command = normalize_binary_command(&runner.command);
            Ok(Box::new(ClaudeCliAdapter::new(ClaudeCliConfig {
                command,
                model: Some(model),
                permission_mode: runner.permission_mode.clone(),
                timeout: Duration::from_secs(runner.timeout_seconds.unwrap_or(900)),
                project_root: Some(root.to_path_buf()),
            })))
        }
        kb_llm::Backend::Opencode | kb_llm::Backend::Pi => {
            let command = normalize_binary_command(&runner.command);
            Ok(Box::new(OpencodeAdapter::new(OpencodeConfig {
                command,
                model,
                tools_read: runner.tools_read,
                tools_write: runner.tools_write,
                tools_edit: runner.tools_edit,
                tools_bash: runner.tools_bash,
                timeout: Duration::from_secs(runner.timeout_seconds.unwrap_or(900)),
                project_root: Some(root.to_path_buf()),
                ..Default::default()
            })))
        }
    }
}

fn resolve_query(query: Option<String>) -> Result<String> {
    match query {
        Some(q) if q != "-" => Ok(q),
        _ => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)?;
            let trimmed = buf.trim().to_string();
            if trimmed.is_empty() {
                bail!("no question provided (pass as argument or pipe via stdin)");
            }
            Ok(trimmed)
        }
    }
}

fn normalize_ask_format(format: &str) -> Result<&str> {
    match format {
        "markdown" => Ok("md"),
        "md" | "marp" | "json" | "png" => Ok(format),
        other => bail!("unsupported ask format: {other}"),
    }
}

fn now_millis() -> Result<u64> {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(u64::try_from(millis)?)
}

fn unique_question_suffix(timestamp: u64, query: &str) -> String {
    let hash = blake3::hash(query.as_bytes()).to_hex();
    format!("{timestamp}-{}", &hash[..10])
}


#[derive(Debug, Serialize)]
struct InspectReport {
    target: String,
    resolved_id: String,
    kind: String,
    metadata: InspectMetadata,
    freshness: String,
    graph: Option<InspectGraph>,
    citations: Vec<String>,
    build_records: Vec<InspectBuildRecord>,
    generating_jobs: Vec<InspectJob>,
    trace: Option<Vec<InspectTraceNode>>,
}

#[derive(Debug, Serialize)]
struct InspectMetadata {
    file_path: Option<String>,
    exists_on_disk: bool,
    size_bytes: Option<u64>,
    modified_at_millis: Option<u64>,
}

#[derive(Debug, Serialize)]
struct InspectGraph {
    direct_inputs: Vec<String>,
    direct_outputs: Vec<String>,
    upstream: Vec<String>,
    downstream: Vec<String>,
}

#[derive(Debug, Serialize)]
struct InspectBuildRecord {
    id: String,
    pass_name: String,
    model: Option<String>,
    prompt_template_hash: Option<String>,
    started_at_millis: u64,
    ended_at_millis: u64,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct InspectJob {
    id: String,
    command: String,
    status: String,
    started_at_millis: u64,
    ended_at_millis: Option<u64>,
    affected_outputs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct InspectTraceNode {
    id: String,
    kind: String,
    inputs: Vec<Self>,
}

fn run_inspect(root: &Path, target: &str, json: bool, trace: bool) -> Result<()> {
    if target.trim().is_empty() {
        bail!("inspect: target cannot be empty");
    }

    // I2: reject absolute paths that resolve outside the KB root. `root.join(abs)`
    // silently replaces the base on Unix, so without this check `/etc/hosts`
    // would be treated as a KB entity. Relative targets are still joined to
    // root below and may escape via `..`; we also reject those.
    if let Some(reason) = path_outside_root(root, target) {
        bail!("'{target}' is outside the KB root ({reason}); inspect only accepts ids or paths under {}", root.display());
    }

    let graph = Graph::load_from(root)?;
    let hash_state = kb_compile::HashState::load_from_root(root)?;
    let changed_inputs = find_changed_inputs(root, &hash_state)?;
    let jobs = jobs::recent_jobs(root, 1_000)?;

    let mut report = if let Some(path) = resolve_source_id(root, target) {
        // I1: bare `src-<hex>` identifiers resolve to their wiki/sources page
        // (preferred) or their normalized/<id>/source.md.
        build_file_report(
            root,
            target,
            &path,
            &jobs,
            &graph,
            &changed_inputs,
            &hash_state,
        )?
    } else if let Some(id) = graph.resolve_node_id(target) {
        build_graph_inspect_report(root, &graph, target, &id, &changed_inputs, &jobs)?
    } else if let Some(record) = kb_core::load_build_record(root, target)? {
        build_build_record_report(root, target, &record, &jobs)?
    } else if let Some(job) = jobs.iter().find(|job| job.metadata.id == target) {
        build_job_report(target, job)
    } else {
        let candidate = root.join(target);
        let resolved = if candidate.exists() {
            Some(candidate)
        } else {
            resolve_wiki_missing_md(root, target)
        };
        if let Some(path) = resolved {
            build_file_report(
                root,
                target,
                &path,
                &jobs,
                &graph,
                &changed_inputs,
                &hash_state,
            )?
        } else {
            let frontmatter_matches = find_by_frontmatter_id(root, target);
            match frontmatter_matches.len() {
                1 => {
                    let path = &frontmatter_matches[0];
                    build_file_report(
                        root,
                        target,
                        path,
                        &jobs,
                        &graph,
                        &changed_inputs,
                        &hash_state,
                    )?
                }
                0 => bail!(
                    "'{target}' was not found. Try an exact ID, a unique graph suffix, a build record ID, a job ID, a frontmatter id, or a path under the KB root. Run 'kb compile' first if the dependency graph has not been created yet."
                ),
                _ => {
                    let paths: Vec<String> = frontmatter_matches
                        .iter()
                        .map(|path| {
                            path.strip_prefix(root)
                                .unwrap_or(path)
                                .to_string_lossy()
                                .into_owned()
                        })
                        .collect();
                    bail!(
                        "'{target}' is ambiguous - matches multiple files: [{}]. Pass the full path to disambiguate.",
                        paths.join(", ")
                    );
                }
            }
        }
    };

    if trace {
        if let Some(graph_data) = &report.graph {
            let _ = graph_data;
            report.trace = Some(build_trace(&graph, &report.resolved_id, &mut std::collections::BTreeSet::new()));
        } else {
            report.trace = Some(Vec::new());
        }
    }

    if json {
        emit_json("inspect", &report)?;
    } else {
        println!("{}", render_inspect_report(&report));
    }

    Ok(())
}

fn build_graph_inspect_report(
    root: &Path,
    graph: &Graph,
    target: &str,
    id: &str,
    changed_inputs: &[ChangedInput],
    jobs: &[JobRun],
) -> Result<InspectReport> {
    let inspection = graph.inspect(id)?;
    let path = root.join(id);
    let metadata = file_metadata(root, path.exists().then_some(path.as_path()))?;
    let citations = if metadata.exists_on_disk {
        extract_citations(&fs::read_to_string(&path).unwrap_or_default())
    } else {
        Vec::new()
    };
    let records = kb_core::find_build_records_for_output(root, id)?;
    let generating_jobs = find_jobs_for_output(jobs, id);
    let freshness = inspect_freshness(graph, id, changed_inputs, metadata.exists_on_disk);

    Ok(InspectReport {
        target: target.to_string(),
        resolved_id: id.to_string(),
        kind: inspect_kind_for_path(id, path.exists().then_some(path.as_path())),
        metadata,
        freshness,
        graph: Some(InspectGraph {
            direct_inputs: inspection.direct_inputs,
            direct_outputs: inspection.direct_outputs,
            upstream: inspection.upstream,
            downstream: inspection.downstream,
        }),
        citations,
        build_records: records.into_iter().map(summarize_build_record).collect(),
        generating_jobs,
        trace: None,
    })
}

fn build_build_record_report(
    root: &Path,
    target: &str,
    record: &kb_core::BuildRecord,
    jobs: &[JobRun],
) -> Result<InspectReport> {
    Ok(InspectReport {
        target: target.to_string(),
        resolved_id: record.metadata.id.clone(),
        kind: "build_record".to_string(),
        metadata: file_metadata(
            root,
            Some(&kb_core::build_records_dir(root).join(format!("{}.json", record.metadata.id))),
        )?,
        freshness: format!("{:?}", record.metadata.status).to_lowercase(),
        graph: None,
        citations: Vec::new(),
        build_records: vec![summarize_build_record(record.clone())],
        generating_jobs: jobs
            .iter()
            .filter(|job| job.command == "compile")
            .map(summarize_job)
            .collect(),
        trace: None,
    })
}

fn build_job_report(target: &str, job: &JobRun) -> InspectReport {
    InspectReport {
        target: target.to_string(),
        resolved_id: job.metadata.id.clone(),
        kind: "job_run".to_string(),
        metadata: InspectMetadata {
            file_path: job.log_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            exists_on_disk: true,
            size_bytes: None,
            modified_at_millis: Some(job.metadata.updated_at_millis),
        },
        freshness: format!("{:?}", job.metadata.status).to_lowercase(),
        graph: None,
        citations: Vec::new(),
        build_records: Vec::new(),
        generating_jobs: vec![summarize_job(job)],
        trace: None,
    }
}

/// Recursively scan `wiki/` and `outputs/` for `.md` files whose frontmatter `id:`
/// field matches `target`. Returns absolute paths of all matches.
///
/// Used as a fallback resolver in `run_inspect` so users can inspect entities
/// by their frontmatter id (e.g. `wiki-source-manual`, `artifact-question-abcd`)
/// without having to hunt for the file path.
fn find_by_frontmatter_id(root: &Path, target: &str) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    for subdir in ["wiki", "outputs"] {
        let base = root.join(subdir);
        if base.exists() {
            collect_frontmatter_id_matches(&base, target, &mut matches);
        }
    }
    matches
}

fn collect_frontmatter_id_matches(dir: &Path, target: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_frontmatter_id_matches(&path, target, out);
        } else if file_type.is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("md")
        {
            if let Ok((frontmatter, _body)) = kb_core::frontmatter::read_frontmatter(&path) {
                if let Some(id) = frontmatter
                    .get(serde_yaml::Value::String("id".to_string()))
                    .and_then(serde_yaml::Value::as_str)
                {
                    if id == target {
                        out.push(path);
                    }
                }
            }
        }
    }
}

fn build_file_report(
    root: &Path,
    target: &str,
    path: &Path,
    jobs: &[JobRun],
    graph: &Graph,
    changed_inputs: &[ChangedInput],
    hash_state: &kb_compile::HashState,
) -> Result<InspectReport> {
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();
    let body = fs::read_to_string(path).unwrap_or_default();
    let exists = path.exists();
    let freshness = file_freshness(root, path, &rel, graph, changed_inputs, hash_state, exists);
    Ok(InspectReport {
        target: target.to_string(),
        resolved_id: rel.clone(),
        kind: inspect_kind_for_path(&rel, Some(path)),
        metadata: file_metadata(root, Some(path))?,
        freshness,
        graph: None,
        citations: extract_citations(&body),
        build_records: find_build_records_for_path(root, &rel)?
            .into_iter()
            .map(summarize_build_record)
            .collect(),
        generating_jobs: find_jobs_for_output(jobs, &rel),
        trace: None,
    })
}

fn file_metadata(root: &Path, path: Option<&Path>) -> Result<InspectMetadata> {
    let Some(path) = path else {
        return Ok(InspectMetadata {
            file_path: None,
            exists_on_disk: false,
            size_bytes: None,
            modified_at_millis: None,
        });
    };

    let stat = fs::metadata(path)?;
    let modified_at_millis = stat
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .and_then(|duration| u64::try_from(duration.as_millis()).ok());

    Ok(InspectMetadata {
        file_path: Some(
            path.strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned(),
        ),
        exists_on_disk: true,
        size_bytes: Some(stat.len()),
        modified_at_millis,
    })
}

fn extract_citations(body: &str) -> Vec<String> {
    let mut citations = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "- _None yet._" {
            continue;
        }
        if trimmed.contains("[[") || (trimmed.contains('[') && trimmed.contains("](")) {
            // I5: strip any leading markdown list prefix ("- ", "* ", "+ ") so
            // the inspect renderer does not emit a "- - [[...]]" double-bullet.
            let cleaned = strip_list_prefix(trimmed).to_string();
            if !cleaned.is_empty() {
                citations.push(cleaned);
            }
        }
    }
    citations.sort();
    citations.dedup();
    citations
}

fn strip_list_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return rest.trim_start();
        }
    }
    trimmed
}

/// Return `Some(reason)` if `target` resolves outside the KB root.
///
/// - An absolute path outside `root` returns a reason. `root.join(abs)`
///   discards `root` on Unix and would otherwise let `/etc/hosts` succeed.
/// - A relative path that escapes via `..` returns a reason.
/// - Anything else (ids, relative paths under root) returns `None`.
fn path_outside_root(root: &Path, target: &str) -> Option<String> {
    let candidate = Path::new(target);
    if candidate.is_absolute() {
        let canon_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let canon_target = candidate.canonicalize().unwrap_or_else(|_| candidate.to_path_buf());
        if !canon_target.starts_with(&canon_root) {
            return Some(format!("absolute path {} does not start with root", candidate.display()));
        }
        return None;
    }

    // Relative target: walk the candidate's components and track depth. Any
    // `..` that would take us above the join point means the target escapes
    // root. Lexical (not filesystem) so non-existent targets still work.
    let mut depth: i32 = 0;
    for comp in candidate.components() {
        match comp {
            std::path::Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Some(format!("relative target {target} escapes root via '..'"));
                }
            }
            std::path::Component::Normal(_) => depth += 1,
            _ => {}
        }
    }
    None
}

/// L1: accept wiki page paths without the `.md` suffix, matching the
/// Obsidian/wiki convention (`wiki/sources/src-0e2e3f8b` ->
/// `wiki/sources/src-0e2e3f8b.md`). Returns the on-disk file path when the
/// `.md` form exists and the bare form does not resolve to a file.
fn resolve_wiki_missing_md(root: &Path, target: &str) -> Option<PathBuf> {
    if !target.starts_with("wiki/") {
        return None;
    }
    let with_md = root.join(format!("{target}.md"));
    if with_md.is_file() { Some(with_md) } else { None }
}

/// I1: resolve a bare `src-<hex>` identifier to the wiki/sources page (if any)
/// or to `normalized/<id>/source.md` (if any). Returns the file path for use
/// in the standard file report.
fn resolve_source_id(root: &Path, target: &str) -> Option<PathBuf> {
    if !is_source_id(target) {
        return None;
    }
    let wiki_page = root.join("wiki/sources").join(format!("{target}.md"));
    if wiki_page.exists() {
        return Some(wiki_page);
    }
    let normalized = root.join("normalized").join(target).join("source.md");
    if normalized.exists() {
        return Some(normalized);
    }
    None
}

fn is_source_id(s: &str) -> bool {
    let Some(hex) = s.strip_prefix("src-") else {
        return false;
    };
    !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit())
}

/// Extend `kb_core::find_build_records_for_output` with a fallback match on
/// `metadata.output_paths` so build records that only list the output as a
/// `PathBuf` (e.g. concept-merge and promotion passes) are still surfaced by
/// inspect. Results are deduplicated by record id and sorted newest-first.
fn find_build_records_for_path(root: &Path, rel: &str) -> Result<Vec<kb_core::BuildRecord>> {
    let dir = kb_core::build_records_dir(root);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = kb_core::find_build_records_for_output(root, rel)?;
    let mut seen: std::collections::BTreeSet<String> =
        out.iter().map(|r| r.metadata.id.clone()).collect();
    for entry in fs::read_dir(&dir).with_context(|| format!("scan {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("read build record {}", path.display()))?;
            let record: kb_core::BuildRecord = serde_json::from_str(&raw)
                .with_context(|| format!("deserialize build record {}", path.display()))?;
            if seen.contains(&record.metadata.id) {
                continue;
            }
            let matches_path = record
                .metadata
                .output_paths
                .iter()
                .any(|p| p.to_string_lossy() == rel);
            if matches_path {
                seen.insert(record.metadata.id.clone());
                out.push(record);
            }
        }
    }
    out.sort_by(|a, b| {
        b.metadata
            .created_at_millis
            .cmp(&a.metadata.created_at_millis)
    });
    Ok(out)
}

/// I3: compute freshness for a file-based inspect report.
///
/// - If the file is a graph node, delegate to `inspect_freshness` so concept
///   and wiki pages get "fresh"/"stale" from the HashState-backed logic.
/// - Otherwise, for a `wiki/sources/*.md` page, compare the page's frontmatter
///   `source_revision_id` to the normalized document's current
///   `source_revision_id`. Equal → fresh, different → stale.
/// - Fall back to `"unknown"` when no signal is available.
fn file_freshness(
    root: &Path,
    abs_path: &Path,
    rel: &str,
    graph: &Graph,
    changed_inputs: &[ChangedInput],
    _hash_state: &kb_compile::HashState,
    exists: bool,
) -> String {
    if !exists {
        return "missing".to_string();
    }
    if graph.node(rel).is_some() {
        return inspect_freshness(graph, rel, changed_inputs, true);
    }
    if rel.starts_with("wiki/sources/") {
        if let Some(status) = source_page_freshness(root, abs_path) {
            return status;
        }
    }
    "unknown".to_string()
}

/// Compare a source wiki page's frontmatter `source_revision_id` against the
/// current one in `normalized/<source_document_id>/metadata.json`. Returns
/// `None` when either side is missing so the caller can fall back to
/// `"unknown"` rather than lying about freshness.
fn source_page_freshness(root: &Path, page_path: &Path) -> Option<String> {
    let (frontmatter, _) = kb_core::frontmatter::read_frontmatter(page_path).ok()?;
    let source_doc_id = frontmatter
        .get(serde_yaml::Value::String("source_document_id".into()))
        .and_then(serde_yaml::Value::as_str)?;
    let page_revision = frontmatter
        .get(serde_yaml::Value::String("source_revision_id".into()))
        .and_then(serde_yaml::Value::as_str)?;

    let metadata_path = root
        .join("normalized")
        .join(source_doc_id)
        .join("metadata.json");
    let raw = fs::read_to_string(&metadata_path).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let current_revision = meta.get("source_revision_id").and_then(|v| v.as_str())?;

    Some(if current_revision == page_revision {
        "fresh".to_string()
    } else {
        "stale".to_string()
    })
}

fn inspect_freshness(
    graph: &Graph,
    id: &str,
    changed_inputs: &[ChangedInput],
    exists_on_disk: bool,
) -> String {
    if !exists_on_disk {
        return "missing".to_string();
    }

    let changed: std::collections::BTreeSet<_> = changed_inputs
        .iter()
        .map(|input| input.normalized_path.to_string_lossy().into_owned())
        .collect();
    if changed.contains(id) {
        return "stale".to_string();
    }

    if let Ok(inspection) = graph.inspect(id) {
        if inspection.upstream.iter().any(|upstream| changed.contains(upstream)) {
            return "stale".to_string();
        }
    }

    "fresh".to_string()
}

fn inspect_kind(id: &str) -> String {
    if id.starts_with("wiki/") {
        "wiki_page".to_string()
    } else if id.starts_with("normalized/") {
        "normalized_document".to_string()
    } else if id.starts_with("raw/") {
        "source_revision".to_string()
    } else if id.starts_with("outputs/questions/") {
        "artifact".to_string()
    } else if id.starts_with("question-") {
        "question".to_string()
    } else if id.starts_with("artifact-") {
        "artifact".to_string()
    } else {
        "entity".to_string()
    }
}

/// L2: classify the resolved entity using filesystem metadata when available.
///
/// `inspect_kind` does pure prefix classification, which mislabels
/// intermediate directories (e.g. `raw/inbox`) as `source_revision`. When the
/// resolved path is a directory, override to `directory`; otherwise fall
/// through to the id-based heuristic.
fn inspect_kind_for_path(id: &str, path: Option<&Path>) -> String {
    if let Some(p) = path {
        if p.is_dir() {
            return "directory".to_string();
        }
    }
    inspect_kind(id)
}

fn summarize_build_record(record: kb_core::BuildRecord) -> InspectBuildRecord {
    InspectBuildRecord {
        id: record.metadata.id,
        pass_name: record.pass_name,
        model: record.metadata.model_version,
        prompt_template_hash: record.metadata.prompt_template_hash,
        started_at_millis: record.metadata.created_at_millis,
        ended_at_millis: record.metadata.updated_at_millis,
        inputs: record.input_ids,
        outputs: record.output_ids,
    }
}

fn summarize_job(job: &JobRun) -> InspectJob {
    InspectJob {
        id: job.metadata.id.clone(),
        command: job.command.clone(),
        status: format!("{:?}", job.status).to_lowercase(),
        started_at_millis: job.started_at_millis,
        ended_at_millis: job.ended_at_millis,
        affected_outputs: job
            .affected_outputs
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
    }
}

fn find_jobs_for_output(jobs: &[JobRun], output_id: &str) -> Vec<InspectJob> {
    jobs.iter()
        .filter(|job| {
            job.affected_outputs.iter().any(|path| {
                let output = path.to_string_lossy();
                output == output_id || output.ends_with(output_id)
            })
        })
        .map(summarize_job)
        .collect()
}

fn build_trace(
    graph: &Graph,
    id: &str,
    visited: &mut std::collections::BTreeSet<String>,
) -> Vec<InspectTraceNode> {
    let Some(node) = graph.node(id) else {
        return Vec::new();
    };

    node.inputs
        .iter()
        .map(|input| {
            let inputs = if visited.insert(input.clone()) {
                build_trace(graph, input, visited)
            } else {
                Vec::new()
            };
            InspectTraceNode {
                id: input.clone(),
                kind: inspect_kind(input),
                inputs,
            }
        })
        .collect()
}

fn render_inspect_report(report: &InspectReport) -> String {
    fn render_list(label: &str, values: &[String]) -> String {
        if values.is_empty() {
            format!("{label}:\n  (none)")
        } else {
            format!(
                "{label}:\n{}",
                values
                    .iter()
                    .map(|value| format!("  - {value}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    }

    fn render_trace(nodes: &[InspectTraceNode], depth: usize, out: &mut Vec<String>) {
        for node in nodes {
            out.push(format!("{}- {} [{}]", "  ".repeat(depth), node.id, node.kind));
            render_trace(&node.inputs, depth + 1, out);
        }
    }

    let mut sections = vec![
        format!("target: {}", report.target),
        format!("resolved_id: {}", report.resolved_id),
        format!("kind: {}", report.kind),
        format!("freshness: {}", report.freshness),
        format!(
            "metadata:\n  file_path: {}\n  exists_on_disk: {}\n  size_bytes: {}\n  modified_at_millis: {}",
            report.metadata.file_path.as_deref().unwrap_or("(none)"),
            report.metadata.exists_on_disk,
            report.metadata
                .size_bytes
                .map_or_else(|| "(none)".to_string(), |value| value.to_string()),
            report.metadata
                .modified_at_millis
                .map_or_else(|| "(none)".to_string(), |value| value.to_string())
        ),
    ];

    if let Some(graph) = &report.graph {
        sections.push(render_list("direct inputs", &graph.direct_inputs));
        sections.push(render_list("direct outputs", &graph.direct_outputs));
        sections.push(render_list("all upstream dependencies", &graph.upstream));
        sections.push(render_list("all downstream dependents", &graph.downstream));
    }

    sections.push(render_list("citations", &report.citations));

    if report.build_records.is_empty() {
        sections.push("build records:\n  (none)".to_string());
    } else {
        let records = report
            .build_records
            .iter()
            .map(|record| {
                format!(
                    "  - id: {}\n    pass: {}\n    model: {}\n    prompt_template_hash: {}\n    started_at_millis: {}\n    ended_at_millis: {}",
                    record.id,
                    record.pass_name,
                    record.model.as_deref().unwrap_or("(none)"),
                    record.prompt_template_hash.as_deref().unwrap_or("(none)"),
                    record.started_at_millis,
                    record.ended_at_millis,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("build records:\n{records}"));
    }

    if report.generating_jobs.is_empty() {
        sections.push("generating jobs:\n  (none)".to_string());
    } else {
        let jobs = report
            .generating_jobs
            .iter()
            .map(|job| {
                format!(
                    "  - id: {}\n    command: {}\n    status: {}\n    started_at_millis: {}\n    ended_at_millis: {}",
                    job.id,
                    job.command,
                    job.status,
                    job.started_at_millis,
                    job.ended_at_millis
                        .map_or_else(|| "(none)".to_string(), |value| value.to_string())
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("generating jobs:\n{jobs}"));
    }

    if let Some(trace) = &report.trace {
        let mut lines = Vec::new();
        render_trace(trace, 1, &mut lines);
        sections.push(if lines.is_empty() {
            "trace:\n  (none)".to_string()
        } else {
            format!("trace:\n{}", lines.join("\n"))
        });
    }

    sections.join("\n\n")
}

fn run_lint(root: &Path, json: bool, check: Option<&str>, strict: bool) -> Result<()> {
    // Peek at the root lock before we walk the tree. If another process is
    // mid-compile the on-disk state is a moving target (half-written source
    // pages, not-yet-rendered concept pages), and running lint against that
    // snapshot produces scary false positives (missing citations, orphans,
    // broken links) that resolve themselves silently once compile finishes.
    //
    // Default mode: warn on stderr and keep going — lint is still useful
    // advice and the operator may want to see it anyway.
    //
    // --strict mode: refuse to run. A strict run that fires off stale
    // warnings defeats the point of --strict (it's used in CI to gate merges
    // on a clean tree).
    //
    // The peek deliberately does not acquire the lock; lint is read-only and
    // must remain runnable concurrently when no one is writing.
    if let Some(holder) = jobs::peek_root_lock(root) {
        if holder.command.contains("compile") {
            if strict {
                return Err(ExitCodeError {
                    exit_code: 1,
                    message: format!(
                        "refusing to run --strict while kb compile is in flight (pid {}, command=`{}`); re-run after compile completes",
                        holder.pid, holder.command
                    ),
                }
                .into());
            }
            eprintln!(
                "warning: kb compile is in flight (pid {}, command=`{}`); lint output may be stale",
                holder.pid, holder.command
            );
        }
    }

    let cfg = Config::load_from_root(root, None)?;
    let check = kb_lint::LintRule::parse(check)?;
    let missing_citations_level = match cfg.lint.missing_citations_level.as_str() {
        "warn" | "warning" => kb_lint::MissingCitationsLevel::Warn,
        "error" => kb_lint::MissingCitationsLevel::Error,
        other => bail!(
            "invalid lint.missing_citations_level in kb.toml: {other} (expected warn or error)"
        ),
    };
    let options = kb_lint::LintOptions {
        require_citations: cfg.lint.require_citations,
        missing_citations_level,
    };

    let rules = if matches!(check, kb_lint::LintRule::All) {
        lint_rules_for_root(cfg.lint.require_citations)
    } else {
        vec![check]
    };

    let mut total_warnings = 0;
    let mut total_errors = 0;
    let mut reports = Vec::new();

    for rule in &rules {
        let rule = *rule;
        let report = kb_lint::run_lint_with_options(root, rule, &options)?;
        let mut warning_count = 0;
        let mut error_count = 0;
        for issue in &report.issues {
            if matches!(issue.severity, kb_lint::IssueSeverity::Warning) {
                warning_count += 1;
            } else {
                error_count += 1;
            }
        }
        total_warnings += warning_count;
        total_errors += error_count;

        print_lint_check_report(&report, rule, warning_count, error_count, json);
        reports.push(LintCheckReport {
            check: report.rule,
            issue_count: report.issue_count,
            warning_count,
            error_count,
            issues: report.issues,
        });
    }
    if json {
        emit_json("lint", LintReportPayload {
            checks: reports,
            checks_ran: rules.len(),
            total_issue_count: total_warnings + total_errors,
            warning_count: total_warnings,
            error_count: total_errors,
        })?;
    }

    if total_errors > 0 {
        Err(ExitCodeError {
            exit_code: 1,
            message: format!(
                "lint failed with {total_errors} error(s), {total_warnings} warning(s)"
            ),
        }
        .into())
    } else if total_warnings > 0 && strict {
        Err(ExitCodeError {
            exit_code: 1,
            message: format!(
                "lint failed with 0 error(s), {total_warnings} warning(s) (--strict)"
            ),
        }
        .into())
    } else {
        if total_warnings > 0 && !json {
            println!(
                "lint: {total_warnings} warning(s) (use --strict to fail on warnings)"
            );
        }
        Ok(())
    }
}
fn lint_rules_for_root(require_citations: bool) -> Vec<kb_lint::LintRule> {
    let mut rules = vec![
        kb_lint::LintRule::BrokenLinks,
        kb_lint::LintRule::Orphans,
        kb_lint::LintRule::StaleRevision,
        kb_lint::LintRule::StaleArtifacts,
    ];
    if require_citations {
        rules.push(kb_lint::LintRule::MissingCitations);
    }
    rules
}

fn print_lint_check_report(
    report: &kb_lint::LintReport,
    rule: kb_lint::LintRule,
    warning_count: usize,
    error_count: usize,
    json_mode: bool,
) {
    if json_mode {
        return;
    }

    if report.is_clean() {
        println!("[ok] {}: no issues found", report.rule);
        return;
    }

    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    for issue in &report.issues {
        if matches!(issue.severity, kb_lint::IssueSeverity::Warning) {
            warnings.push(issue);
        } else {
            errors.push(issue);
        }
    }

    if !warnings.is_empty() {
        println!("[warn] {}: {} issue(s)", rule.as_str(), warning_count);
        print_lint_issues(&warnings);
    }

    if !errors.is_empty() {
        println!("[fail] {}: {} issue(s)", rule.as_str(), error_count);
        print_lint_issues(&errors);
    }
}

fn print_lint_issues(issues: &[&kb_lint::LintIssue]) {
    for issue in issues {
        println!(
            "- [{:?}] {}:{} {}",
            issue.severity, issue.referring_page, issue.line, issue.message
        );
        if let Some(suggested_fix) = &issue.suggested_fix {
            println!("  suggested fix: {suggested_fix}");
        }
    }
}

#[derive(Debug, Serialize)]
struct LintReportPayload {
    checks: Vec<LintCheckReport>,
    checks_ran: usize,
    total_issue_count: usize,
    warning_count: usize,
    error_count: usize,
}

#[derive(Debug, Serialize)]
struct LintCheckReport {
    check: String,
    issue_count: usize,
    warning_count: usize,
    error_count: usize,
    issues: Vec<kb_lint::LintIssue>,
}
#[derive(Debug, Serialize)]
struct StatusPayload {
    /// Count of ingested sources discovered under `normalized/<id>/`.
    normalized_source_count: usize,
    /// Wiki source pages (`wiki-page-*` nodes referencing a source document),
    /// and a by-kind breakdown carried over from the pre-compile graph.
    sources: SourceCounts,
    wiki_pages: usize,
    concepts: usize,
    stale_count: usize,
    recent_jobs: Vec<JobRun>,
    /// Failed job runs, capped at `FAILED_JOBS_DISPLAY_LIMIT` most recent.
    failed_jobs: Vec<JobRun>,
    /// Total count of failed jobs, including those truncated from `failed_jobs`.
    failed_jobs_total: usize,
    /// Interrupted job runs, capped at `INTERRUPTED_JOBS_DISPLAY_LIMIT` most recent.
    interrupted_jobs: Vec<JobRun>,
    /// Total count of interrupted jobs, including those truncated from `interrupted_jobs`.
    interrupted_jobs_total: usize,
    /// Normalized source directories whose contents haven't been compiled yet,
    /// each annotated (when available) with the original filename/URL the
    /// source was ingested from so users can grep by filename — not just the
    /// opaque `src-<id>` hash.
    changed_inputs_not_compiled: Vec<ChangedInput>,
    /// Ingested sources whose original file no longer exists on disk.
    /// URL-backed sources are excluded — they don't live on local disk.
    /// Users can run `kb forget <src-id>` to retire these cleanly.
    sources_with_missing_origin: Vec<MissingOrigin>,
}

/// One ingested source whose `stable_location` path no longer exists on disk.
#[derive(Debug, Serialize)]
struct MissingOrigin {
    /// Source-document id (e.g. `src-0639ebb0`).
    src_id: String,
    /// The original filesystem path as recorded in
    /// `raw/inbox/<src>/source_document.json::stable_location`.
    origin: String,
}

#[derive(Debug, Serialize)]
struct SourceCounts {
    total: usize,
    by_kind: std::collections::BTreeMap<String, usize>,
}

/// One uncompiled normalized source, carrying both the bare on-disk path
/// (backward-compat grep target) and — when the raw-ingest record is
/// readable — the original filename or URL the source came from.
#[derive(Debug, Serialize)]
struct ChangedInput {
    /// Full normalized path like `<root>/normalized/src-0639ebb0`.
    normalized_path: PathBuf,
    /// Source document id (e.g. `src-0639ebb0`) parsed from the directory name.
    src_id: String,
    /// Best-available human-readable origin: local filesystem path or URL.
    /// `None` when `raw/inbox/<src>/source_document.json` is missing or
    /// unparseable (e.g. a legacy KB, manually-placed normalized dir, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    original_path: Option<String>,
}

/// Result of scanning `normalized/*/` for inputs needing recompile.
///
/// `entries` is the full list of uncompiled sources (both never-compiled and
/// revision-mismatched), and `revision_mismatched` is the subset count where
/// a previously-compiled source has been re-ingested with a new revision —
/// the primary signal surfaced by `kb status` as "stale artifacts".
struct ChangedInputsScan {
    entries: Vec<ChangedInput>,
    /// Number of `entries` that represent a wiki page out of date relative
    /// to a newer normalized revision (bn-2m2). Never-compiled sources are
    /// NOT counted here — those are "new", not "stale".
    revision_mismatched: usize,
}

/// Maximum number of failed jobs shown in text `kb status` output. Additional
/// failures are summarized with a "... and N more" hint. Kept small because
/// typo/bad-path failures accrue indefinitely and would otherwise drown the
/// signal in every `kb status` invocation.
const FAILED_JOBS_DISPLAY_LIMIT: usize = 10;

/// Maximum number of interrupted jobs shown in text `kb status` output.
/// Kept smaller than the failed-jobs cap (10) because interrupted runs
/// are ALSO flagged with a per-run "→ Inspect the logs and rerun" footer
/// that would otherwise turn `kb status` into a wall of remediation text
/// on any machine that's been SIGINT'd a few times. Cap 5 keeps the
/// status page compact; the `... and N more` hint points at `kb jobs
/// prune --interrupted` for cleanup.
const INTERRUPTED_JOBS_DISPLAY_LIMIT: usize = 5;

fn gather_status(root: &Path) -> Result<StatusPayload> {
    let graph = Graph::load_from(root)?;
    let manifest = kb_core::Manifest::load(root)?;
    let hash_state = kb_compile::HashState::load_from_root(root)?;
    let all_jobs = jobs::recent_jobs(root, 1000)?;

    let normalized_source_count = count_normalized_sources(root)?;

    // Counts come from a direct disk walk of `wiki/sources/` and
    // `wiki/concepts/`, NOT from `graph.nodes`. The persisted graph is
    // written atomically at the *end* of compile, so any mid-compile or
    // partial-failure state would under-report (often as 0) while
    // `wiki/sources/*.md` files are already visible on disk. A disk walk
    // is O(<few hundred files) for realistic KBs and gives users an
    // always-truthful count. See bn-1iw / l-status-stale.
    let source_counts = count_wiki_source_pages(root)?;
    // `wiki_pages` in the JSON payload is the same set — one wiki page per
    // source — so we keep it in lockstep with the disk-walked total.
    let wiki_pages = source_counts.total;
    let concepts = count_wiki_concept_pages(root)?;

    // bn-2m2: `stale_count` now reflects the primary user-visible staleness
    // signal — ingested sources whose wiki page is out of date relative to
    // the normalized `source_revision_id`. Previously this counter walked
    // the persisted graph looking for leaf outputs with non-raw inputs,
    // which was a graph-topology heuristic that never reacted to re-ingest
    // and always reported 0 in the common "edit file, re-ingest, forget to
    // compile" flow. See `/tmp/kb-pass9/findings/n-stale-count.md`.
    //
    // The scan is shared with `changed_inputs_not_compiled` below so we
    // walk `normalized/*/` exactly once per `kb status` invocation.
    //
    // TODO(bn-2m2-followup): extend `stale_count` with (#2) orphan concept
    // pages whose `source_document_ids` reference a missing `normalized/`
    // source-id, and (#3) build records naming output paths that no longer
    // exist on disk. The shape (a u64 count) already accommodates both; we
    // just need additional one-shot passes and de-duplication against the
    // revision-mismatch set so the same src-id isn't counted twice.
    let changed_scan = scan_changed_inputs(root, &hash_state)?;
    let mut stale_count = changed_scan.revision_mismatched;

    // Preserve the original graph-topology heuristic as a non-overlapping
    // additive signal. It fires on graph-level inconsistencies (missing
    // intermediate outputs, etc.) that aren't captured by the per-source
    // revision check — e.g. a concept page whose inputs were all deleted.
    // De-duplication is unnecessary because this branch inspects `graph`
    // outputs keyed by `Manifest::artifacts`, which never includes the
    // normalized-source entities counted above.
    for artifact_record in manifest.artifacts.values() {
        for output_id in &artifact_record.output_ids {
            if let Ok(inspection) = graph.inspect(output_id) {
                if let Some(entity) = graph.node(&inspection.id) {
                    if entity.outputs.is_empty() && !entity.inputs.is_empty() {
                        if let Some(node) = graph.nodes.get(&inspection.id) {
                            if node.inputs.iter().any(|input| {
                                graph.nodes.get(input).is_some_and(|n| {
                                    n.inputs.is_empty()
                                })
                            }) {
                                continue;
                            }
                        }
                        stale_count += 1;
                    }
                }
            }
        }
    }

    // Interrupted jobs: mirror the failed-jobs treatment below. Every
    // SIGINT/SIGPIPE/OS-crash leaves a manifest behind, so this list grows
    // without bound. Cap the rendered slice at
    // `INTERRUPTED_JOBS_DISPLAY_LIMIT` and keep the honest total on
    // `interrupted_jobs_total` for the header and JSON consumers.
    let interrupted_jobs_total = all_jobs
        .iter()
        .filter(|j| j.status == JobRunStatus::Interrupted)
        .count();
    let interrupted_jobs: Vec<_> = all_jobs
        .iter()
        .filter(|j| j.status == JobRunStatus::Interrupted)
        .take(INTERRUPTED_JOBS_DISPLAY_LIMIT)
        .cloned()
        .collect();

    let recent_jobs: Vec<_> = all_jobs
        .iter()
        .filter(|j| j.status != JobRunStatus::Failed && j.status != JobRunStatus::Interrupted)
        .take(20)
        .cloned()
        .collect();

    // Failed jobs: the full total is used by `print_status` to emit a
    // "... and M more" hint when more than `FAILED_JOBS_DISPLAY_LIMIT`
    // failures exist. Typo/bad-path failures accumulate forever, so only
    // the cap is rendered while the count stays honest.
    let failed_jobs_total = all_jobs
        .iter()
        .filter(|j| j.status == JobRunStatus::Failed)
        .count();
    let failed_jobs: Vec<_> = all_jobs
        .iter()
        .filter(|j| j.status == JobRunStatus::Failed)
        .take(FAILED_JOBS_DISPLAY_LIMIT)
        .cloned()
        .collect();

    let changed_inputs_not_compiled = changed_scan.entries;
    let sources_with_missing_origin = find_missing_origins(root)?;

    Ok(StatusPayload {
        normalized_source_count,
        sources: source_counts,
        wiki_pages,
        concepts,
        stale_count,
        recent_jobs,
        failed_jobs,
        failed_jobs_total,
        interrupted_jobs,
        interrupted_jobs_total,
        changed_inputs_not_compiled,
        sources_with_missing_origin,
    })
}

/// Walk `normalized/*/` and flag every source whose recorded
/// `stable_location` is a local filesystem path that no longer exists.
///
/// URL-backed sources (and any non-local stable-location scheme) are
/// skipped — we don't try to HEAD the URL in `kb status`. A missing
/// `raw/inbox/<src>/source_document.json` is treated as "can't tell": we
/// don't flag it, because a legacy / hand-placed normalized dir has no
/// origin to miss.
fn find_missing_origins(root: &Path) -> Result<Vec<MissingOrigin>> {
    let normalized_dir = root.join("normalized");
    if !normalized_dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&normalized_dir)
        .with_context(|| format!("read {}", normalized_dir.display()))?
    {
        let entry = entry
            .with_context(|| format!("read entry in {}", normalized_dir.display()))?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(origin) = lookup_source_origin(root, &name) else {
            continue;
        };
        if !forget::origin_is_local_path(&origin) {
            continue;
        }
        if !Path::new(&origin).exists() {
            out.push(MissingOrigin {
                src_id: name,
                origin,
            });
        }
    }
    out.sort_by(|a, b| a.src_id.cmp(&b.src_id));
    Ok(out)
}

/// Count ingested sources by listing subdirectories of `normalized/`.
///
/// Each ingested source lives at `normalized/<id>/` (written by
/// `kb_ingest::write_normalized_for_file` and url ingest). The directory is
/// missing on a freshly initialized KB that hasn't been ingested into yet —
/// that's not an error, we just report zero.
fn count_normalized_sources(root: &Path) -> Result<usize> {
    let dir = root.join("normalized");
    if !dir.exists() {
        return Ok(0);
    }
    let mut count = 0;
    for entry in std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("reading {}", dir.display()))?;
        if entry.file_type()?.is_dir() {
            count += 1;
        }
    }
    Ok(count)
}

/// Count wiki source pages by walking `wiki/sources/*.md` directly.
///
/// Returns a fresh count plus a by-kind breakdown. The total is the single
/// source of truth users see on the `wiki source pages: N` line in `kb
/// status`; deriving it from the persisted graph is unsafe because the graph
/// is only written atomically at end-of-compile and would report 0 mid-run
/// (or stay under-reported after a killed compile until the next successful
/// one). See bn-1iw / l-status-stale.
///
/// `wiki/sources/index.md` is excluded — it's the auto-generated index of
/// source pages, not itself a source page. A missing `wiki/sources/` dir is
/// treated as zero: freshly-initialized KBs have no wiki tree yet.
///
/// The per-kind breakdown is best-effort: it reuses `extract_source_kind` on
/// the filename stem. For today's source IDs (e.g. `src-0639ebb0`) no kind
/// token is present in the filename, so everything falls into the `other`
/// bucket, which `print_status` then collapses into the header line. When
/// ingest later plumbs kind info into the filename the breakdown will
/// light up automatically.
fn count_wiki_source_pages(root: &Path) -> Result<SourceCounts> {
    let mut counts = SourceCounts {
        total: 0,
        by_kind: std::collections::BTreeMap::new(),
    };
    let dir = root.join("wiki/sources");
    if !dir.exists() {
        return Ok(counts);
    }
    for entry in
        std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("reading {}", dir.display()))?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem == "index" {
            continue;
        }
        let kind = extract_source_kind(stem);
        *counts.by_kind.entry(kind.to_string()).or_insert(0) += 1;
        counts.total += 1;
    }
    Ok(counts)
}

/// Count wiki concept pages by walking `wiki/concepts/*.md` directly.
///
/// Same rationale as `count_wiki_source_pages`: avoid relying on the
/// end-of-compile graph snapshot so the number reflects what's actually on
/// disk right now. `wiki/concepts/index.md` is excluded. A missing dir is
/// zero.
fn count_wiki_concept_pages(root: &Path) -> Result<usize> {
    let dir = root.join("wiki/concepts");
    if !dir.exists() {
        return Ok(0);
    }
    let mut count = 0;
    for entry in
        std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("reading {}", dir.display()))?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem == "index" {
            continue;
        }
        count += 1;
    }
    Ok(count)
}

fn extract_source_kind(node_id: &str) -> &str {
    if node_id.contains("url") {
        "url"
    } else if node_id.contains("file") {
        "file"
    } else if node_id.contains("repo") {
        "repo"
    } else if node_id.contains("image") {
        "image"
    } else if node_id.contains("dataset") {
        "dataset"
    } else {
        "other"
    }
}

/// Return the set of normalized source directories whose contents do not
/// appear in the compile hash state — i.e. inputs that exist on disk but
/// have not been compiled yet (or have been changed since the last compile).
///
/// Walks `normalized/*/` under the KB root and flags any directory that is
/// either
///
/// 1. absent from the loaded `HashState` (never compiled), or
/// 2. whose `normalized/<id>/metadata.json::source_revision_id` differs from
///    the `source_revision_id` recorded in the corresponding
///    `wiki/sources/<slug>.md` frontmatter (ingested a new revision since
///    last compile).
///
/// Case (2) catches the common re-ingest flow: a user edits a source file,
/// re-ingests it (producing a new `source_revision_id`), and expects `kb
/// status` to tell them the wiki is stale. The `HashState` check alone
/// misses this because compile *did* run once and recorded a fingerprint —
/// it just recorded an out-of-date one.
///
/// For each flagged source we also try to recover the original filename or
/// URL from `raw/inbox/<src>/source_document.json` (written by ingest).
/// This lets `kb status` render `<filename> (src-<id>)` instead of an opaque
/// `normalized/src-0639ebb0`, which users can't meaningfully grep. When the
/// raw record is missing or unreadable (legacy KB, hand-placed normalized
/// dir, partial deletion), we fall back to just the src-id.
///
/// Missing `state/hashes.json` is treated as an empty state, so every
/// normalized source will be reported until `kb compile` runs.
fn find_changed_inputs(
    root: &Path,
    hash_state: &kb_compile::HashState,
) -> Result<Vec<ChangedInput>> {
    Ok(scan_changed_inputs(root, hash_state)?.entries)
}

/// Single-pass walk of `normalized/*/` that classifies each source as
/// "never compiled" or "revision-mismatched" (or neither — the clean case).
///
/// Shared by [`find_changed_inputs`] (which drops the classification and
/// returns just the entries) and [`gather_status`] (which needs the
/// revision-mismatched count to populate `StatusPayload::stale_count`).
/// Factoring this out avoids duplicating the `read_dir` walk between the
/// "changed inputs not yet compiled" list and the stale-artifact count.
fn scan_changed_inputs(
    root: &Path,
    hash_state: &kb_compile::HashState,
) -> Result<ChangedInputsScan> {
    let normalized_dir = root.join("normalized");
    if !normalized_dir.exists() {
        return Ok(ChangedInputsScan {
            entries: Vec::new(),
            revision_mismatched: 0,
        });
    }

    let mut entries = Vec::new();
    let mut revision_mismatched = 0;
    for entry in std::fs::read_dir(&normalized_dir)
        .with_context(|| format!("read normalized dir {}", normalized_dir.display()))?
    {
        let entry = entry
            .with_context(|| format!("read normalized entry in {}", normalized_dir.display()))?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let dir_name = entry.file_name();
        let Some(name) = dir_name.to_str() else {
            continue;
        };
        // Match the graph node id convention used by compile: `normalized/<id>`.
        let node_id = format!("normalized/{name}");
        if !hash_state.hashes.contains_key(&node_id) {
            let original_path = lookup_source_origin(root, name);
            entries.push(ChangedInput {
                normalized_path: entry.path(),
                src_id: name.to_string(),
                original_path,
            });
            continue;
        }
        if source_revision_mismatched(root, name)? {
            let original_path = lookup_source_origin(root, name);
            entries.push(ChangedInput {
                normalized_path: entry.path(),
                src_id: name.to_string(),
                original_path,
            });
            revision_mismatched += 1;
        }
    }
    entries.sort_by(|a, b| a.normalized_path.cmp(&b.normalized_path));
    Ok(ChangedInputsScan {
        entries,
        revision_mismatched,
    })
}

/// Return `true` if `normalized/<id>/metadata.json::source_revision_id`
/// differs from the `source_revision_id` in the matching
/// `wiki/sources/<slug>.md` frontmatter.
///
/// Returns `false` (not mismatched) when:
/// - the wiki source page doesn't exist yet (caller already handled the
///   "never compiled" case via the hash-state check),
/// - the wiki page has no frontmatter or no `source_revision_id` field
///   (treated as "can't tell" — don't spuriously flag),
/// - the normalized metadata is unreadable (propagated as an error).
///
/// A mismatch here means the user ran `kb ingest` after a successful
/// `kb compile` and the wiki page is now out of date.
fn source_revision_mismatched(root: &Path, normalized_id: &str) -> Result<bool> {
    let metadata_path = root
        .join("normalized")
        .join(normalized_id)
        .join("metadata.json");
    let Ok(metadata_bytes) = std::fs::read(&metadata_path) else {
        // Missing or unreadable metadata: don't flag — the hash-state check
        // above already handles the "never ingested" and "corrupt" cases.
        return Ok(false);
    };
    let metadata: serde_json::Value = serde_json::from_slice(&metadata_bytes)
        .with_context(|| format!("parse normalized metadata {}", metadata_path.display()))?;
    let Some(normalized_rev) = metadata
        .get("source_revision_id")
        .and_then(|v| v.as_str())
    else {
        return Ok(false);
    };

    let wiki_page = root.join(kb_compile::source_page::source_page_path_for_id(
        normalized_id,
    ));
    let Ok(markdown) = std::fs::read_to_string(&wiki_page) else {
        // No wiki page yet — not a "mismatch"; already caught by hash state
        // in the common never-compiled case. If the wiki page was deleted
        // out from under us, the next compile will regenerate it.
        return Ok(false);
    };
    let Some(frontmatter) = extract_frontmatter(&markdown) else {
        return Ok(false);
    };
    let Ok(parsed) = serde_yaml::from_str::<serde_yaml::Value>(&frontmatter) else {
        return Ok(false);
    };
    let Some(page_rev) = parsed
        .get("source_revision_id")
        .and_then(|v| v.as_str())
    else {
        return Ok(false);
    };

    Ok(page_rev != normalized_rev)
}

/// Extract the YAML frontmatter block from a markdown document.
///
/// Returns the frontmatter body (without the fence lines) if the document
/// starts with a `---` fence and has a closing `---` fence. Otherwise returns
/// `None`. Mirrors the small parser in `kb_compile::backlinks` but lives here
/// so `kb status` does not have to depend on compile-internal helpers.
fn extract_frontmatter(markdown: &str) -> Option<String> {
    let mut lines = markdown.split_inclusive('\n');
    let first = lines.next()?;
    if first != "---\n" && first != "---\r\n" && first != "---" {
        return None;
    }
    let mut fm = String::new();
    for line in lines {
        if line == "---\n" || line == "---\r\n" || line == "---" {
            return Some(fm);
        }
        fm.push_str(line);
    }
    None
}

/// Best-effort lookup of the human-readable origin for a source-document id.
///
/// Reads `raw/inbox/<src_id>/source_document.json` and returns its
/// `stable_location` field (the canonical filesystem path for local files,
/// or the normalized URL for web sources). Any failure — missing file,
/// unparseable JSON, missing field — yields `None` rather than erroring;
/// this is display-only metadata and must never block status rendering.
fn lookup_source_origin(root: &Path, src_id: &str) -> Option<String> {
    let record_path = root
        .join("raw")
        .join("inbox")
        .join(src_id)
        .join("source_document.json");
    let bytes = std::fs::read(&record_path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("stable_location")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}


#[allow(clippy::too_many_lines)]
fn print_status(status: &StatusPayload) {
    println!("kb status");
    println!();

    println!("ingested sources: {}", status.normalized_source_count);
    if status.sources.total == 0 {
        println!("wiki source pages: 0    (run 'kb compile' to generate)");
    } else {
        println!("wiki source pages: {}", status.sources.total);
        // ST2: a single-kind breakdown is always just `<kind>: <total>`,
        // which carries no information beyond the header line. Only render
        // the breakdown when there's actual variety (≥2 distinct kinds).
        // The default "other" bucket — dominant until ingest starts plumbing
        // raw-subdir categories through — collapses into the header instead
        // of showing `other: N` after every count.
        if status.sources.by_kind.len() > 1 {
            for (kind, count) in &status.sources.by_kind {
                println!("  {kind}: {count}");
            }
        }
    }
    println!("wiki concept pages: {}", status.concepts);
    println!("stale artifacts: {}", status.stale_count);
    println!();

    if !status.changed_inputs_not_compiled.is_empty() {
        println!("changed inputs not yet compiled:");
        for input in &status.changed_inputs_not_compiled {
            // ST1: users can't meaningfully grep or recognize a bare
            // `/tmp/kb/normalized/src-0639ebb0`. Prefer the original
            // filename from the raw-ingest record, falling back to the
            // normalized path when the record is absent. The src-id is
            // always included so operators can still match on it in logs.
            match &input.original_path {
                Some(origin) => {
                    let display = origin_display_name(origin);
                    println!("  - {display}  ({})", input.src_id);
                }
                None => {
                    println!("  - {}  ({})", input.normalized_path.display(), input.src_id);
                }
            }
        }
        println!();
    }

    if !status.sources_with_missing_origin.is_empty() {
        println!("sources with missing origin (run `kb forget <src-id>` to remove):");
        for missing in &status.sources_with_missing_origin {
            let display = origin_display_name(&missing.origin);
            println!(
                "  - {display}  ({}, origin {} deleted)",
                missing.src_id, missing.origin
            );
        }
        println!();
    }

    if !status.interrupted_jobs.is_empty() {
        // Header reflects the TRUE total so users see the real backlog;
        // the detail list is capped to keep status output compact (see
        // `INTERRUPTED_JOBS_DISPLAY_LIMIT`). A "... and N more" hint
        // points users at `kb jobs prune --interrupted` to clean up.
        println!("⚠ interrupted job runs ({}):", status.interrupted_jobs_total);
        for job in &status.interrupted_jobs {
            let duration = job
                .ended_at_millis
                .map(|ended| ended - job.started_at_millis);
            let duration_str =
                duration.map_or_else(|| "running".to_string(), |ms| format!("{ms}ms"));
            println!(
                "  {} | {:<11} | {} [{}]",
                job.metadata.id,
                format!("{:?}", job.status),
                job.command,
                duration_str
            );
        }
        if status.interrupted_jobs_total > status.interrupted_jobs.len() {
            let hidden = status.interrupted_jobs_total - status.interrupted_jobs.len();
            println!(
                "  ... and {hidden} more (run 'kb jobs prune --interrupted' to clean up)"
            );
        }
        println!("  → Inspect the logs and rerun: kb {}",
            status.interrupted_jobs.first().map_or("compile", |j| j.command.as_str()));
        println!();
    }

    println!("recent job runs ({}):", status.recent_jobs.len());
    for job in &status.recent_jobs {
        let duration = job
            .ended_at_millis
            .map(|ended| ended - job.started_at_millis);
        let duration_str = duration.map_or_else(|| "running".to_string(), |ms| format!("{ms}ms"));
        println!(
            "  {} | {:<11} | {} [{}]",
            job.metadata.id,
            format!("{:?}", job.status),
            job.command,
            duration_str
        );
    }
    println!();

    if !status.failed_jobs.is_empty() {
        // ST3: header shows the true total (e.g. "failed job runs (47):")
        // so users see the real scope of accumulated failures, while the
        // body is capped at `FAILED_JOBS_DISPLAY_LIMIT` most recent to
        // avoid drowning `kb status` output in stale typos.
        println!("failed job runs ({}):", status.failed_jobs_total);
        for job in &status.failed_jobs {
            let duration = job
                .ended_at_millis
                .map(|ended| ended - job.started_at_millis);
            let duration_str =
                duration.map_or_else(|| "running".to_string(), |ms| format!("{ms}ms"));
            println!(
                "  {} | {:<11} | {} [{}]",
                job.metadata.id,
                format!("{:?}", job.status),
                job.command,
                duration_str
            );
        }
        if status.failed_jobs_total > status.failed_jobs.len() {
            let hidden = status.failed_jobs_total - status.failed_jobs.len();
            // `kb jobs --failed` is not yet implemented (see bn-2cr scope
            // note); the hint points at the intended future entry point so
            // users know where to look once it lands.
            println!("  ... and {hidden} more (run 'kb jobs --failed' to inspect)");
        }
    }
}

/// Render the filename portion of a source origin for the status display.
///
/// For local filesystem paths we strip the directory prefix to keep lines
/// short — the src-id suffix already disambiguates files with the same
/// basename. URLs are passed through untouched because their "filename"
/// alone (e.g. the trailing path segment) is rarely meaningful.
fn origin_display_name(origin: &str) -> String {
    if origin.starts_with("http://") || origin.starts_with("https://") {
        return origin.to_string();
    }
    Path::new(origin)
        .file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| origin.to_string(), ToOwned::to_owned)
}
fn execute_mutating_command(
    root: Option<&Path>,
    command: &str,
    action: impl FnOnce() -> Result<()>,
) -> Result<()> {
    execute_mutating_command_with_handle(root, command, |_handle| action())
}

/// Like [`execute_mutating_command`] but hands the active `JobHandle` to
/// the closure so commands can stream per-pass progress into the `JobRun`
/// log (see `JobHandle::log_sink`). Kept internal so callers who don't
/// need the handle stay on the simpler form above.
fn execute_mutating_command_with_handle(
    root: Option<&Path>,
    command: &str,
    action: impl FnOnce(&jobs::JobHandle) -> Result<()>,
) -> Result<()> {
    let root = root.expect("root resolved for mutating commands");
    let cfg = Config::load_from_root(root, None)?;
    let _lock = jobs::KbLock::acquire(root, command, Duration::from_millis(cfg.lock.timeout_ms))?;
    jobs::check_stale_jobs(root)?;
    let handle = jobs::start_job(root, command)?;

    let result = action(&handle);
    match result {
        Ok(()) => {
            handle.finish(JobRunStatus::Succeeded, Vec::new())?;
            Ok(())
        }
        Err(err) => {
            // bn-1jx: user-input validation errors must not pollute the
            // failed-jobs list. If the action rejected early (empty query,
            // unknown publish target, nonexistent review id, etc.), delete
            // the manifest we optimistically created in `start_job` so
            // `kb status` / `kb doctor` stay clean. Real system failures
            // (LLM timeout, I/O error, compile pass crash) still finish as
            // Failed and show up as expected.
            if err.is::<ValidationError>() {
                handle.discard();
            } else {
                handle.finish(JobRunStatus::Failed, Vec::new())?;
            }
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
            model: None,
            dry_run: false,
            json: false,
            force: false,
            quiet: false,
            command: Some(Command::Status),
        })
        .expect("status command succeeds");
        assert!(started.elapsed() < Duration::from_millis(250));
        drop(lock);
    }

    #[test]
    fn count_normalized_sources_counts_subdirs() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        // Missing `normalized/` is not an error.
        assert_eq!(count_normalized_sources(root).expect("no normalized dir"), 0);

        let normalized = root.join("normalized");
        fs::create_dir_all(normalized.join("src-a")).expect("create src-a");
        fs::create_dir_all(normalized.join("src-b")).expect("create src-b");
        fs::create_dir_all(normalized.join("src-c")).expect("create src-c");
        // Stray file at the top level should be ignored.
        fs::write(normalized.join("README"), "ignored").expect("write stray file");

        assert_eq!(
            count_normalized_sources(root).expect("count after ingest"),
            3
        );
    }

    #[test]
    fn gather_status_reports_normalized_source_count() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);
        fs::create_dir_all(root.join("normalized").join("src-42"))
            .expect("create normalized source dir");

        let status = gather_status(&root).expect("gather status");
        assert_eq!(status.normalized_source_count, 1);
    }

    /// Regression test for bn-1iw: `kb status` must count wiki source pages
    /// by walking `wiki/sources/*.md` on disk, not by reading the end-of-
    /// compile graph snapshot. Drop 5 wiki source pages onto disk with no
    /// state/graph.json or state/manifest.json present (simulating mid-
    /// compile or a killed finalize), and verify status reports 5 — not 0.
    #[test]
    fn gather_status_counts_wiki_source_pages_from_disk() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        let sources_dir = root.join("wiki/sources");
        fs::create_dir_all(&sources_dir).expect("create wiki/sources");
        for i in 0..5 {
            fs::write(
                sources_dir.join(format!("src-{i:04x}.md")),
                "---\nid: wiki-source-x\n---\n\n# body\n",
            )
            .expect("write wiki source page");
        }
        // `index.md` must NOT be counted — it's the auto-generated index.
        fs::write(sources_dir.join("index.md"), "# index\n").expect("write index");
        // Non-markdown files must be ignored.
        fs::write(sources_dir.join("notes.txt"), "scratch").expect("write stray txt");

        let status = gather_status(&root).expect("gather status");
        assert_eq!(
            status.sources.total, 5,
            "should count the 5 .md pages, excluding index.md and notes.txt",
        );
        assert_eq!(
            status.wiki_pages, 5,
            "wiki_pages field must stay in lockstep with sources.total",
        );
    }

    /// `wiki/concepts/*.md` is counted the same way and `index.md` is
    /// similarly excluded.
    #[test]
    fn gather_status_counts_wiki_concept_pages_from_disk() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        let concepts_dir = root.join("wiki/concepts");
        fs::create_dir_all(&concepts_dir).expect("create wiki/concepts");
        fs::write(concepts_dir.join("rust.md"), "# rust\n").expect("rust");
        fs::write(concepts_dir.join("lifetime.md"), "# lifetime\n").expect("lifetime");
        fs::write(concepts_dir.join("index.md"), "# index\n").expect("index");

        let status = gather_status(&root).expect("gather status");
        assert_eq!(status.concepts, 2);
    }

    /// A fresh KB with no `wiki/` tree at all must not error — counts are 0.
    #[test]
    fn gather_status_reports_zero_when_wiki_tree_missing() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        let status = gather_status(&root).expect("gather status");
        assert_eq!(status.sources.total, 0);
        assert_eq!(status.wiki_pages, 0);
        assert_eq!(status.concepts, 0);
    }

    /// Helper for the re-ingest mismatch tests: fabricate
    /// `normalized/<id>/metadata.json` with the given revision.
    fn write_normalized_metadata(root: &Path, id: &str, revision: &str) {
        let dir = root.join("normalized").join(id);
        fs::create_dir_all(&dir).expect("create normalized dir");
        let metadata = serde_json::json!({
            "metadata": {
                "id": id,
                "entity_type": "source-document",
                "display_name": id,
                "canonical_path": "inbox/fake.md",
                "content_hashes": [],
                "output_paths": [],
                "status": "Fresh",
            },
            "source_revision_id": revision,
            "normalized_assets": [],
            "heading_ids": [],
        });
        fs::write(dir.join("metadata.json"), metadata.to_string())
            .expect("write metadata.json");
        fs::write(dir.join("source.md"), "# body\n").expect("write source.md");
    }

    /// Helper for the re-ingest mismatch tests: fabricate a minimal
    /// `wiki/sources/<slug>.md` page with the given frontmatter revision.
    fn write_wiki_source_page(root: &Path, source_id: &str, revision: &str) {
        let page_path =
            root.join(kb_compile::source_page::source_page_path_for_id(source_id));
        fs::create_dir_all(page_path.parent().expect("parent"))
            .expect("create wiki sources dir");
        let markdown = format!(
            "---\nid: wiki-source-{source_id}\ntype: source\ntitle: {source_id}\n\
source_document_id: {source_id}\nsource_revision_id: {revision}\n\
generated_at: 0\nbuild_record_id: build-1\n---\n\n# Source\n",
        );
        fs::write(&page_path, markdown).expect("write wiki page");
    }

    /// Regression test for bn-2gn: after a re-ingest changes the normalized
    /// `source_revision_id`, `kb status` must flag the source as a changed
    /// input even though a prior compile wrote its hash into `state/hashes.json`.
    #[test]
    fn find_changed_inputs_flags_reingested_source_with_stale_wiki_page() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();

        // Simulate a previously compiled source: hash state has the node,
        // wiki page exists with rev-A, normalized metadata still claims rev-A.
        write_normalized_metadata(root, "src-1", "rev-A");
        write_wiki_source_page(root, "src-1", "rev-A");
        let mut hash_state = kb_compile::HashState::default();
        hash_state
            .hashes
            .insert("normalized/src-1".to_string(), "fingerprint-A".to_string());

        // Clean state: not flagged.
        let changed = find_changed_inputs(root, &hash_state).expect("find changed (clean)");
        assert!(
            changed.is_empty(),
            "clean compile state should not flag any changed inputs: {changed:?}",
        );

        // Now simulate `kb ingest` of a new revision: metadata.json bumps
        // to rev-B, wiki page still reflects rev-A. Status must flag it.
        write_normalized_metadata(root, "src-1", "rev-B");
        let changed = find_changed_inputs(root, &hash_state).expect("find changed (reingested)");
        let changed_paths: Vec<PathBuf> =
            changed.into_iter().map(|c| c.normalized_path).collect();
        assert_eq!(
            changed_paths,
            vec![root.join("normalized").join("src-1")],
            "re-ingested source with stale wiki page must be reported as changed",
        );
    }

    /// First-compile behavior is preserved: normalized dirs without an
    /// entry in `HashState` are flagged, regardless of wiki page state.
    #[test]
    fn find_changed_inputs_flags_never_compiled_source() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        write_normalized_metadata(root, "src-fresh", "rev-A");

        let hash_state = kb_compile::HashState::default();
        let changed =
            find_changed_inputs(root, &hash_state).expect("find changed (never compiled)");
        let changed_paths: Vec<PathBuf> =
            changed.into_iter().map(|c| c.normalized_path).collect();
        assert_eq!(changed_paths, vec![root.join("normalized").join("src-fresh")]);
    }

    /// Regression test for bn-2m2: `gather_status` must count sources whose
    /// normalized `source_revision_id` has diverged from the corresponding
    /// wiki page as "stale artifacts". Pre-fix, `stale_count` was computed
    /// from a graph-topology heuristic that always yielded 0 in the common
    /// "edit file, re-ingest, forget to compile" flow.
    #[test]
    fn gather_status_counts_revision_mismatched_sources_as_stale() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        // Simulate a previously compiled source: hashes.json records the
        // node, wiki page pins rev-A, normalized metadata now claims rev-B
        // (i.e. the user re-ingested after compile). Status must flag it.
        write_normalized_metadata(&root, "src-1", "rev-B");
        write_wiki_source_page(&root, "src-1", "rev-A");
        let mut hash_state = kb_compile::HashState::default();
        hash_state
            .hashes
            .insert("normalized/src-1".to_string(), "fingerprint-A".to_string());
        hash_state
            .save_to_root(&root)
            .expect("persist hash state");

        let status = gather_status(&root).expect("gather status");
        assert_eq!(
            status.stale_count, 1,
            "re-ingested source with stale wiki page must be counted in stale_count",
        );
        // The "changed inputs" list must stay in sync with the count.
        assert_eq!(status.changed_inputs_not_compiled.len(), 1);
    }

    /// After compile reconciles the revisions, `stale_count` drops back to 0.
    #[test]
    fn gather_status_stale_count_clears_when_revisions_match() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        write_normalized_metadata(&root, "src-1", "rev-A");
        write_wiki_source_page(&root, "src-1", "rev-A");
        let mut hash_state = kb_compile::HashState::default();
        hash_state
            .hashes
            .insert("normalized/src-1".to_string(), "fingerprint-A".to_string());
        hash_state
            .save_to_root(&root)
            .expect("persist hash state");

        let status = gather_status(&root).expect("gather status");
        assert_eq!(status.stale_count, 0);
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

    /// Regression test for bn-2vm: `kb compile --dry-run` is read-only and
    /// must not block on the root lock held by an in-flight real compile,
    /// nor leave a job manifest behind under `state/jobs/`.
    #[test]
    fn compile_dry_run_does_not_block_on_root_lock_or_emit_job_manifest() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("kb");
        write_kb_config(&root);

        // Simulate a real `kb compile` running in another shell by holding
        // the root lock for the duration of the dry-run invocation.
        let lock =
            jobs::KbLock::acquire(&root, "compile", Duration::from_secs(1)).expect("acquire lock");

        let started = Instant::now();
        run(Cli {
            root: Some(root.clone()),
            model: None,
            dry_run: true,
            json: true,
            force: false,
            quiet: false,
            command: Some(Command::Compile),
        })
        .expect("compile --dry-run succeeds");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "dry-run should not wait for the root lock (took {:?})",
            started.elapsed()
        );

        // No job manifest should have been written — dry-run skips the
        // job lifecycle entirely so SIGPIPE / Ctrl-C can't orphan a run.
        let jobs_dir = root.join("state").join("jobs");
        if jobs_dir.exists() {
            let manifests: Vec<_> = fs::read_dir(&jobs_dir)
                .expect("read jobs dir")
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    (path.extension().and_then(|ext| ext.to_str()) == Some("json"))
                        .then_some(path)
                })
                .collect();
            assert!(
                manifests.is_empty(),
                "dry-run should not emit a job manifest, found: {manifests:?}"
            );
        }

        drop(lock);
    }
}
