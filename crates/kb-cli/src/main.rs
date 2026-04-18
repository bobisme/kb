#![forbid(unsafe_code)]

mod config;
mod init;
mod jobs;
mod root;

use std::error::Error as StdError;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow, bail};
use clap::Parser;
use config::{Config, LlmRunnerConfig};
use kb_compile::Graph;
use kb_core::{Artifact, ArtifactKind, EntityMetadata, JobRunStatus, Question, QuestionContext, Status};
use kb_llm::{
    ClaudeCliAdapter, ClaudeCliConfig, LlmAdapter, OpencodeAdapter, OpencodeConfig,
    RunHealthCheckRequest,
};
use serde::Serialize;
use serde_yaml::{Mapping, Value};

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
        /// Check a single lint check
        #[arg(long, alias = "rule")]
        check: Option<String>,
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
        std::process::exit(exit_code_from_error(&err));
    }
}

fn exit_code_from_error(err: &anyhow::Error) -> i32 {
    for current in err.chain() {
        if let Some(exit_error) = current.downcast_ref::<LintExitCode>() {
            return exit_error.code;
        }
    }

    1
}

#[derive(Debug)]
struct LintExitCode {
    code: i32,
    message: String,
}

impl std::fmt::Display for LintExitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl StdError for LintExitCode {}

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
                    println!(
                        "{}",
                        serde_json::json!({
                            "total_sources": report.total_sources,
                            "stale_sources": report.stale_sources,
                            "build_records_emitted": report.build_records_emitted,
                            "dry_run": dry_run,
                        })
                    );
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
            let check_root = root;
            let should_json = cli.json;
            let cli_model = cli.model.clone();
            execute_mutating_command(Some(check_root), "doctor", move || {
                run_doctor(check_root, should_json, cli_model.as_deref())
            })
        }
        Some(Command::Ask { query }) => {
            let ask_root = root.clone();
            let requested_format = cli.format.clone();
            let should_json = cli.json;
            execute_mutating_command(root.as_deref(), "ask", move || {
                let root = ask_root
                    .as_deref()
                    .expect("root resolved for non-init commands");
                run_ask(root, &query, requested_format.as_deref(), should_json)
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
        Some(Command::Lint { check }) => {
            let lint_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let should_json = cli.json;
            execute_mutating_command(Some(lint_root), "lint", move || {
                run_lint(lint_root, should_json, check.as_deref())
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
            let search_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let index = kb_query::LexicalIndex::load(search_root)?;
            let results = index.search(&query, 10);
            if results.is_empty() {
                println!("No results for '{query}'");
                println!("Tip: run 'kb compile' to build the search index.");
            } else {
                for result in &results {
                    println!("{} [score: {}]", result.title, result.score);
                    println!("  {}", result.id);
                }
            }
            Ok(())
        }
        Some(Command::Inspect { target }) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let graph = Graph::load_from(root)?;
            let inspection = graph.inspect(&target)?;
            println!("{}", inspection.render());

            let records = kb_core::find_build_records_for_output(root, &inspection.id)?;
            if !records.is_empty() {
                println!("\nbuild records:");
                for record in &records {
                    println!("  id: {}", record.metadata.id);
                    println!("  pass: {}", record.pass_name);
                    if let Some(model) = &record.metadata.model_version {
                        println!("  model: {model}");
                    }
                    if let Some(tmpl) = &record.metadata.prompt_template_hash {
                        println!("  prompt_template_hash: {tmpl}");
                    }
                    println!("  started_at_millis: {}", record.metadata.created_at_millis);
                    if record.metadata.updated_at_millis != record.metadata.created_at_millis {
                        println!("  ended_at_millis: {}", record.metadata.updated_at_millis);
                    }
                    if !record.input_ids.is_empty() {
                        println!("  inputs:");
                        for input in &record.input_ids {
                            println!("    - {input}");
                        }
                    }
                    if !record.output_ids.is_empty() {
                        println!("  outputs:");
                        for output in &record.output_ids {
                            println!("    - {output}");
                        }
                    }
                }
            }
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
struct DoctorPayload {
    checks: Vec<DoctorCheck>,
    all_healthy: bool,
    failed_checks: usize,
}

#[derive(Debug, Clone, Serialize)]
struct DoctorCheck {
    runner: String,
    harness: String,
    status: String,
    model: String,
    harness_version: Option<String>,
    latency_ms: u64,
    command: String,
    details: Option<String>,
    error: Option<String>,
}

fn run_doctor(root: &Path, json: bool, cli_model: Option<&str>) -> Result<()> {
    let cfg = Config::load_from_root(root, cli_model)?;
    let mut checks = Vec::new();

    let default_model = cfg.llm.default_model.clone();
    let mut runner_names: Vec<_> = cfg.llm.runners.keys().cloned().collect();
    runner_names.sort_unstable();

    for name in runner_names {
        let runner = cfg
            .llm
            .runners
            .get(&name)
            .expect("runner key from keys() must exist");

        match name.as_str() {
            "opencode" => {
                checks.push(run_opencode_health_check(
                    &name,
                    runner,
                    root,
                    &default_model,
                ));
            }
            "claude" => {
                checks.push(run_claude_health_check(&name, runner, &default_model));
            }
            _ => {
                checks.push(DoctorCheck {
                    runner: name,
                    harness: "unsupported".to_string(),
                    status: "skipped".to_string(),
                    model: runner
                        .model
                        .clone()
                        .unwrap_or_else(|| default_model.clone()),
                    harness_version: None,
                    latency_ms: 0,
                    command: runner.command.clone(),
                    details: Some("runner not configured for doctor checks".to_string()),
                    error: None,
                });
            }
        }
    }

    let failed_checks = checks.iter().filter(|c| c.status == "failed").count();
    let failed_runners = checks
        .iter()
        .filter(|c| c.status == "failed")
        .map(|c| c.runner.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    if json {
        let payload = DoctorPayload {
            checks,
            all_healthy: failed_checks == 0,
            failed_checks,
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        for check in &checks {
            println!(
                "{}: {} (model: {}, latency: {}ms, harness_version: {:?})",
                check.runner, check.status, check.model, check.latency_ms, check.harness_version
            );
            if let Some(details) = &check.details {
                println!("  details: {details}");
            }
            if let Some(error) = &check.error {
                println!("  error: {error}");
            }
            println!("  command: {}", check.command);
        }
        if failed_checks == 0 {
            println!("kb doctor: all health checks passed");
        } else {
            println!("kb doctor: {failed_checks} health check(s) failed");
        }
    }

    if failed_runners.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "health check failed for runner(s): {failed_runners}"
        ))
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
        Ok((response, provenance)) => DoctorCheck {
            runner: runner_name.to_string(),
            harness: provenance.harness,
            status: response.status,
            model: provenance.model,
            harness_version: provenance.harness_version,
            latency_ms: provenance.latency_ms,
            command: runner.command.clone(),
            details: response.details,
            error: None,
        },
        Err(err) => DoctorCheck {
            runner: runner_name.to_string(),
            harness: "opencode".to_string(),
            status: "failed".to_string(),
            model,
            harness_version: None,
            latency_ms: 0,
            command: runner.command.clone(),
            details: None,
            error: Some(err.to_string()),
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
        Ok((response, provenance)) => DoctorCheck {
            runner: runner_name.to_string(),
            harness: provenance.harness,
            status: response.status,
            model: provenance.model,
            harness_version: provenance.harness_version,
            latency_ms: provenance.latency_ms,
            command: runner.command.clone(),
            details: response.details,
            error: None,
        },
        Err(err) => DoctorCheck {
            runner: runner_name.to_string(),
            harness: "claude".to_string(),
            status: "failed".to_string(),
            model,
            harness_version: None,
            latency_ms: 0,
            command: runner.command.clone(),
            details: None,
            error: Some(err.to_string()),
        },
    }
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

#[derive(Serialize)]
struct AskOutput<'a> {
    question_id: &'a str,
    question_path: &'a str,
    artifact_path: &'a str,
    requested_format: &'a str,
}

fn run_ask(root: &Path, query: &str, requested_format: Option<&str>, json: bool) -> Result<()> {
    let cfg = Config::load_from_root(root, None)?;
    let requested_format = normalize_ask_format(
        requested_format.unwrap_or(cfg.ask.artifact_default_format.as_str()),
    )?;
    let timestamp = now_millis()?;
    let question_id = format!("question-{}", unique_question_suffix(timestamp, query));
    let question_dir = root.join("outputs/questions").join(&question_id);
    fs::create_dir_all(&question_dir)?;

    let artifact_rel = PathBuf::from("outputs/questions")
        .join(&question_id)
        .join("artifact.md");
    let question_rel = PathBuf::from("outputs/questions")
        .join(&question_id)
        .join("question.json");

    let question = Question {
        metadata: EntityMetadata {
            id: question_id.clone(),
            created_at_millis: timestamp,
            updated_at_millis: timestamp,
            source_hashes: Vec::new(),
            model_version: None,
            tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: None,
            dependencies: Vec::new(),
            output_paths: vec![question_rel.clone(), artifact_rel.clone()],
            status: Status::Fresh,
        },
        raw_query: query.to_string(),
        requested_format: requested_format.to_string(),
        requesting_context: QuestionContext::ProjectKb,
        retrieval_plan: String::new(),
        token_budget: Some(cfg.ask.token_budget),
    };

    let artifact = Artifact {
        metadata: EntityMetadata {
            id: format!("artifact-{question_id}"),
            created_at_millis: timestamp,
            updated_at_millis: timestamp,
            source_hashes: Vec::new(),
            model_version: None,
            tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: None,
            dependencies: vec![question_id.clone()],
            output_paths: vec![artifact_rel.clone()],
            status: Status::Fresh,
        },
        question_id: question_id.clone(),
        artifact_kind: match requested_format {
            "png" => ArtifactKind::Figure,
            "marp" => ArtifactKind::SlideDeck,
            "json" => ArtifactKind::JsonSpec,
            _ => ArtifactKind::AnswerNote,
        },
        format: requested_format.to_string(),
        output_path: artifact_rel.clone(),
    };

    fs::write(
        root.join(&question_rel),
        serde_json::to_string_pretty(&question)?,
    )?;
    fs::write(
        root.join(&artifact_rel),
        render_ask_artifact(&artifact, &question, &question_rel)?,
    )?;

    let question_path = question_rel.to_string_lossy().into_owned();
    let artifact_path = artifact_rel.to_string_lossy().into_owned();
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&AskOutput {
                question_id: &question_id,
                question_path: &question_path,
                artifact_path: &artifact_path,
                requested_format,
            })?
        );
    } else {
        println!("Created question record: {question_path}");
        println!("Created artifact placeholder: {artifact_path}");
        println!("Requested format: {requested_format}");
        println!("Answer generation is not implemented yet.");
    }

    Ok(())
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

fn render_ask_artifact(artifact: &Artifact, question: &Question, question_rel: &Path) -> Result<String> {
    let mut frontmatter = Mapping::new();
    frontmatter.insert(Value::String("id".into()), Value::String(artifact.metadata.id.clone()));
    frontmatter.insert(
        Value::String("type".into()),
        Value::String("answer_artifact".into()),
    );
    frontmatter.insert(
        Value::String("question_id".into()),
        Value::String(question.metadata.id.clone()),
    );
    frontmatter.insert(
        Value::String("question_record".into()),
        Value::String(question_rel.to_string_lossy().into_owned()),
    );
    frontmatter.insert(
        Value::String("requested_format".into()),
        Value::String(question.requested_format.clone()),
    );
    if question.requested_format == "marp" {
        frontmatter.insert(Value::String("marp".into()), Value::Bool(true));
    }

    let body = format!(
        "# Question queued\n\nAnswer generation is not implemented yet.\n\n- Query: {}\n- Requested format: {}\n- Question record: {}\n",
        question.raw_query,
        question.requested_format,
        question_rel.display()
    );

    let yaml = serde_yaml::to_string(&frontmatter)?;
    Ok(format!("---\n{yaml}---\n\n{body}"))
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
        let payload = LintReportPayload {
            checks: reports,
            checks_ran: rules.len(),
            total_issue_count: total_warnings + total_errors,
            warning_count: total_warnings,
            error_count: total_errors,
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
    }

    if total_errors > 0 {
        Err(anyhow!(LintExitCode {
            code: 2,
            message: format!("lint failed: {total_errors} error(s) and {total_warnings} warning(s)"),
        }))
    } else if total_warnings > 0 {
        Err(anyhow!(LintExitCode {
            code: 1,
            message: format!("lint succeeded with {total_warnings} warning(s)"),
        }))
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
