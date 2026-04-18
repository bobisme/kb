#![forbid(unsafe_code)]

mod config;
mod init;
mod jobs;
mod publish;
mod review;
mod root;

use std::env;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
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

fn emit_json<T: Serialize>(command: &str, data: T) -> Result<()> {
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
        /// Question to ask (reads from stdin if omitted or "-")
        query: Option<String>,

        /// Propose promoting the answer into the wiki
        #[arg(long)]
        promote: bool,
    },
    /// Lint knowledge base for issues
    Lint {
        /// Check a single lint check
        #[arg(long, alias = "rule")]
        check: Option<String>,
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
    /// Inspect and manage review queue
    Review {
        #[command(subcommand)]
        action: ReviewAction,
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
            .map_or(1, |err| err.exit_code);
        eprintln!("error: {err}");
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


#[allow(clippy::too_many_lines)]
fn run(cli: Cli) -> Result<()> {
    let root = if matches!(cli.command, Some(Command::Init { .. }) | None) {
        cli.root
    } else {
        Some(root::discover_root(cli.root.as_deref())?.path)
    };

    match cli.command {
        Some(Command::Compile) => {
            let compile_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let force = cli.force;
            let dry_run = cli.dry_run;
            let json = cli.json;
            execute_mutating_command(Some(compile_root), "compile", move || {
                let options = kb_compile::pipeline::CompileOptions { force, dry_run };
                let report = kb_compile::pipeline::run_compile(compile_root, &options)?;

                if json {
                    emit_json("compile", serde_json::json!({
                        "total_sources": report.total_sources,
                        "stale_sources": report.stale_sources,
                        "build_records_emitted": report.build_records_emitted,
                        "dry_run": dry_run,
                    }))?;
                } else {
                    println!("{}", report.render());
                }

                Ok(())
            })
        }
        Some(Command::Doctor) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_doctor_command(root, cli.json, cli.model.as_deref())
        }
        Some(Command::Ask { query, promote }) => {
            let ask_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let query = resolve_query(query)?;
            run_ask(
                ask_root,
                &query,
                cli.format.as_deref(),
                cli.model.as_deref(),
                cli.json,
                cli.dry_run,
                promote,
            )
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
        Some(Command::Lint { check }) => {
            let lint_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let should_json = cli.json;
            execute_mutating_command(Some(lint_root), "lint", move || {
                run_lint(lint_root, should_json, check.as_deref())
            })
        }
        Some(Command::Publish { target }) => {
            let publish_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let dry_run = cli.dry_run;
            let json = cli.json;
            execute_mutating_command(Some(publish_root), "publish", move || {
                let cfg = config::Config::load_from_root(publish_root, None)?;
                let mut available: Vec<String> =
                    cfg.publish.targets.keys().cloned().collect();
                available.sort();
                let target_cfg = cfg
                    .publish
                    .targets
                    .get(&target)
                    .ok_or_else(|| {
                        publish::target_not_found_error(&target, &available)
                    })?
                    .clone();
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
        Some(Command::Init { path }) => init::init(root, path, cli.force),
        Some(Command::Search { query, limit }) => {
            let search_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let index = kb_query::LexicalIndex::load(search_root)?;
            let limit = limit.unwrap_or(10);
            let results = index.search(&query, limit);
            if cli.json {
                emit_json("search", &results)?;
            } else if results.is_empty() {
                println!("No results for '{query}'");
                println!("Tip: run 'kb compile' to build the search index.");
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
                    execute_mutating_command(Some(review_root), "review.approve", move || {
                        review::run_review_approve(review_root, &id, json, &json_emitter)
                    })
                }
                ReviewAction::Reject { id, reason } => {
                    execute_mutating_command(Some(review_root), "review.reject", move || {
                        review::run_review_reject(review_root, &id, reason.as_deref(), json, &json_emitter)
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
    let timeout_ms = Config::load_from_root(root, None).map_or_else(
        |_| Config::default().lock.timeout_ms,
        |cfg| cfg.lock.timeout_ms,
    );
    let _lock = jobs::KbLock::acquire(root, "doctor", Duration::from_millis(timeout_ms))?;
    jobs::check_stale_jobs(root)?;
    let handle = jobs::start_job(root, "doctor")?;

    let result = run_doctor(root, json, cli_model);
    match &result {
        Ok(()) => {
            handle.finish(JobRunStatus::Succeeded, Vec::new())?;
        }
        Err(_) => {
            handle.finish(JobRunStatus::Failed, Vec::new())?;
        }
    }

    result
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
        emit_json("ingest", IngestPayload { dry_run, results, summary })?;
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
    let cfg = Config::load_from_root(root, cli_model)?;
    let requested_format =
        normalize_ask_format(requested_format.unwrap_or(cfg.ask.artifact_default_format.as_str()))?;

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

    let answer_rel = PathBuf::from("outputs/questions")
        .join(&question_id)
        .join("answer.md");
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
            "outputs/questions/{question_id}/answer.md"
        )),
    };

    let write_output = kb_query::write_artifact(&kb_query::WriteArtifactInput {
        root,
        question: &question,
        artifact: &artifact,
        retrieval_plan: &retrieval_plan,
        artifact_result: llm_info.as_ref().map(|(r, _)| r),
        provenance: llm_info.as_ref().map(|(_, p)| p),
        artifact_body: &artifact_body,
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

    let router = kb_llm::Router::new(match runner_name.as_str() {
        "claude" => kb_llm::Backend::ClaudeCode,
        _ => kb_llm::Backend::Opencode,
    });

    let backend = router.route_model(&model);

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
    let graph = Graph::load_from(root)?;
    let hashes = kb_core::Hashes::load(root)?;
    let changed_inputs = find_changed_inputs(root, &hashes)?;
    let jobs = jobs::recent_jobs(root, 1_000)?;

    let mut report = if let Some(id) = graph.resolve_node_id(target) {
        build_graph_inspect_report(root, &graph, target, &id, &changed_inputs, &jobs)?
    } else if let Some(record) = kb_core::load_build_record(root, target)? {
        build_build_record_report(root, target, &record, &jobs)?
    } else if let Some(job) = jobs.iter().find(|job| job.metadata.id == target) {
        build_job_report(target, job)
    } else {
        let candidate = root.join(target);
        if candidate.exists() {
            build_file_report(root, target, &candidate, &jobs)?
        } else {
            bail!(
                "'{target}' was not found. Try an exact ID, a unique graph suffix, a build record ID, a job ID, or a path under the KB root. Run 'kb compile' first if the dependency graph has not been created yet."
            );
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
    changed_inputs: &[PathBuf],
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
        kind: inspect_kind(id),
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

fn build_file_report(root: &Path, target: &str, path: &Path, jobs: &[JobRun]) -> Result<InspectReport> {
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();
    let body = fs::read_to_string(path).unwrap_or_default();
    Ok(InspectReport {
        target: target.to_string(),
        resolved_id: rel.clone(),
        kind: inspect_kind(&rel),
        metadata: file_metadata(root, Some(path))?,
        freshness: "unknown".to_string(),
        graph: None,
        citations: extract_citations(&body),
        build_records: kb_core::find_build_records_for_output(root, &rel)?
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
            citations.push(trimmed.to_string());
        }
    }
    citations.sort();
    citations.dedup();
    citations
}

fn inspect_freshness(graph: &Graph, id: &str, changed_inputs: &[PathBuf], exists_on_disk: bool) -> String {
    if !exists_on_disk {
        return "missing".to_string();
    }

    let changed: std::collections::BTreeSet<_> = changed_inputs
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
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

fn run_lint(root: &Path, json: bool, check: Option<&str>) -> Result<()> {
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
            exit_code: 2,
            message: format!("lint failed: {total_errors} error(s) and {total_warnings} warning(s)"),
        }.into())
    } else if total_warnings > 0 {
        Err(ExitCodeError {
            exit_code: 1,
            message: format!("lint succeeded with {total_warnings} warning(s)"),
        }.into())
    } else {
        Ok(())
    }
}
fn lint_rules_for_root(require_citations: bool) -> Vec<kb_lint::LintRule> {
    let mut rules = vec![
        kb_lint::LintRule::BrokenLinks,
        kb_lint::LintRule::Orphans,
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
            "- [{:?}] {}:{} {} ({})",
            issue.severity, issue.referring_page, issue.line, issue.message, issue.target
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
    sources: SourceCounts,
    wiki_pages: usize,
    concepts: usize,
    stale_count: usize,
    recent_jobs: Vec<JobRun>,
    failed_jobs: Vec<JobRun>,
    interrupted_jobs: Vec<JobRun>,
    changed_inputs_not_compiled: Vec<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SourceCounts {
    total: usize,
    by_kind: std::collections::BTreeMap<String, usize>,
}

fn gather_status(root: &Path) -> Result<StatusPayload> {
    let graph = Graph::load_from(root)?;
    let manifest = kb_core::Manifest::load(root)?;
    let hashes = kb_core::Hashes::load(root)?;
    let all_jobs = jobs::recent_jobs(root, 1000)?;

    let mut source_counts = SourceCounts {
        total: 0,
        by_kind: std::collections::BTreeMap::new(),
    };
    let mut wiki_pages = 0;
    let mut concepts = 0;
    let mut stale_count = 0;

    for node_id in graph.nodes.keys() {
        if node_id.starts_with("source-document-") {
            let kind = extract_source_kind(node_id);
            *source_counts.by_kind.entry(kind.to_string()).or_insert(0) += 1;
            source_counts.total += 1;
        } else if node_id.starts_with("wiki-page-") {
            wiki_pages += 1;
        } else if node_id.starts_with("concept-") {
            concepts += 1;
        }
    }

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

    let interrupted_jobs: Vec<_> = all_jobs
        .iter()
        .filter(|j| j.status == JobRunStatus::Interrupted)
        .take(20)
        .cloned()
        .collect();

    let recent_jobs: Vec<_> = all_jobs
        .iter()
        .filter(|j| j.status != JobRunStatus::Failed && j.status != JobRunStatus::Interrupted)
        .take(20)
        .cloned()
        .collect();

    let failed_jobs: Vec<_> = all_jobs
        .iter()
        .filter(|j| j.status == JobRunStatus::Failed)
        .take(20)
        .cloned()
        .collect();

    let changed_inputs_not_compiled = find_changed_inputs(root, &hashes)?;

    Ok(StatusPayload {
        sources: source_counts,
        wiki_pages,
        concepts,
        stale_count,
        recent_jobs,
        failed_jobs,
        interrupted_jobs,
        changed_inputs_not_compiled,
    })
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

fn find_changed_inputs(_root: &Path, hashes: &kb_core::Hashes) -> Result<Vec<PathBuf>> {
    let mut changed = Vec::new();
    for path in hashes.inputs.keys() {
        if path.exists() {
            let current_hash = kb_core::hash_file(path)?;
            if hashes.inputs.get(path) != Some(&current_hash) {
                changed.push(path.clone());
            }
        } else {
            changed.push(path.clone());
        }
    }
    Ok(changed)
}

fn print_status(status: &StatusPayload) {
    println!("kb status");
    println!();

    println!("sources: {} total", status.sources.total);
    for (kind, count) in &status.sources.by_kind {
        println!("  {kind}: {count}");
    }
    println!();

    println!("wiki pages: {}", status.wiki_pages);
    println!("concepts: {}", status.concepts);
    println!("stale artifacts: {}", status.stale_count);
    println!();

    if !status.changed_inputs_not_compiled.is_empty() {
        println!("changed inputs not yet compiled:");
        for path in &status.changed_inputs_not_compiled {
            println!("  - {}", path.display());
        }
        println!();
    }

    if !status.interrupted_jobs.is_empty() {
        println!("⚠ interrupted job runs ({}):", status.interrupted_jobs.len());
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
        println!("failed job runs ({}):", status.failed_jobs.len());
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
