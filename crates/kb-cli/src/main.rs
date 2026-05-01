#![forbid(unsafe_code)]

mod chat;
mod config;
mod eval;
mod forget;
mod id_resolve;
mod init;
mod jobs;
mod jobs_cmd;
mod ls;
mod migrate;
mod publish;
mod review;
mod root;
mod session;

use std::env;
use std::fs;
use std::io::{IsTerminal, Read as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use config::{Config, LlmRunnerConfig};
use kb_compile::Graph;
use kb_core::{
    Artifact, ArtifactKind, EntityMetadata, JobRun, JobRunStatus, Question, QuestionContext,
    ReviewItem, ReviewKind, ReviewStatus, Status, hash_file, hash_many, normalized_dir, normalized_rel,
    slug_from_title, state_dir,
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

/// Return true when `host` resolves to loopback (IPv4 `127.0.0.0/8`, IPv6
/// `::1`, or the string "localhost"). Used to gate `kb serve`'s network
/// exposure.
fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    false
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
        /// Files, directories, URLs, or git repo URLs to ingest.
        /// Audio files (.m4a/.mp3/.mp4/.wav/.flac) are auto-detected
        /// and routed through whisper + pyannote transcription. At
        /// least one source is required unless `--audio` is given.
        sources: Vec<String>,
        /// Ingest files even if they are empty or contain only YAML frontmatter
        #[arg(long)]
        allow_empty: bool,
        /// Git repo only: override the default doc-walk filter with these
        /// glob patterns. Can be specified multiple times.
        #[arg(long = "include", value_name = "GLOB")]
        include: Vec<String>,
        /// Git repo only: exclude paths matching this glob from the walk.
        /// Can be specified multiple times.
        #[arg(long = "exclude", value_name = "GLOB")]
        exclude: Vec<String>,
        /// Git repo only: check out this branch after cloning (default: the
        /// remote's default branch).
        #[arg(long = "branch", value_name = "NAME")]
        branch: Option<String>,
        /// Git repo only: pin to this commit SHA after cloning.
        #[arg(long = "commit", value_name = "SHA")]
        commit: Option<String>,
        /// Audio file to transcribe + diarize. Usually unnecessary —
        /// any positional source ending in .m4a/.mp3/.mp4/.wav/.flac
        /// is auto-detected as audio. First run downloads ~1.2 GB of
        /// models into `~/.kb/models/`.
        #[arg(long = "audio", value_name = "PATH")]
        audio: Option<PathBuf>,
        /// Title for the transcript's kbtx frontmatter. Defaults to
        /// the audio filename stem.
        #[arg(long = "audio-title", value_name = "TITLE")]
        audio_title: Option<String>,
        /// Recording date (YYYY-MM-DD) for the transcript's kbtx
        /// frontmatter. Defaults to today.
        #[arg(long = "audio-recording-date", value_name = "YYYY-MM-DD")]
        audio_recording_date: Option<String>,
        /// Override path for the rendered `.kbtx.md`. Default:
        /// `<audio-stem>.kbtx.md` adjacent to the audio file.
        #[arg(long = "audio-out", value_name = "PATH")]
        audio_out: Option<PathBuf>,
    },
    /// Compile the knowledge base
    Compile,
    /// Query the knowledge base with natural language
    Ask {
        /// Question to ask (reads from stdin if omitted or "-")
        query: Option<String>,

        /// Artifact format for the answer.
        /// Default `auto` lets the model pick supporting artifacts (Excalidraw,
        /// chart) per question. Pass an explicit format to pin the output.
        #[arg(long, value_parser = [
            "auto", "md", "marp", "json", "chart", "figure",
            "excalidraw", "diagram", "png",
        ])]
        format: Option<String>,

        /// Propose promoting the answer into the wiki
        #[arg(long)]
        promote: bool,

        /// Open $VISUAL/$EDITOR (falling back to `vi`) to compose the
        /// question on a tempfile. Lines starting with `#` are ignored
        /// (like `git commit`); empty content cancels the ask.
        #[arg(short = 'e', long = "editor")]
        editor: bool,

        /// Don't render the answer body to stdout; show only the artifact path
        #[arg(long)]
        no_render: bool,

        /// bn-o6wv: continue a multi-turn session.
        ///
        /// Loads `<root>/.kb/sessions/<id>.json` (creating it on first use),
        /// rewrites the new question against prior turns, runs hybrid
        /// retrieval against the rewritten query, builds an answer prompt
        /// that includes the conversation history, and appends both the
        /// user and assistant turns back to the session file. Pass with no
        /// query to drop into an interactive REPL on the same session.
        #[arg(long, value_name = "ID")]
        session: Option<String>,
    },
    /// Manage multi-turn ask sessions (bn-o6wv).
    ///
    /// Sessions are conversational `kb ask` transcripts stored under
    /// `<root>/.kb/sessions/<id>.json`. Use `kb ask --session <id>` to
    /// add new turns; the subcommands here let you list, inspect, and
    /// create empty sessions.
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// Lint knowledge base for issues
    Lint {
        /// Check a single lint check
        #[arg(long, alias = "rule")]
        check: Option<String>,
        /// Treat warnings as errors (exit 1 on warnings-only)
        #[arg(long)]
        strict: bool,
        /// bn-xt4o: after lint finishes, call a web-search-capable LLM to
        /// draft fill-in content for missing concepts and thin concept
        /// bodies, then queue each draft as a review item. Never applies
        /// changes directly — every draft lands in `kb review` for human
        /// approval. Off by default (expensive + external network).
        #[arg(long)]
        impute: bool,
    },
    /// Run health checks on the knowledge base
    Doctor,
    /// Show status of the knowledge base
    Status,
    /// Tree-list non-hidden files known to this KB.
    ///
    /// Walks the discovered KB root (the same one resolved by every other
    /// command) and prints a `tree`-style view of every entry whose name
    /// does not start with `.`. The `.kb/` internals dir, `.git/`, and any
    /// other dotfiles are skipped at every level. Symlinks and special
    /// files are skipped too — the KB tree is plain files and directories.
    Ls,
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
    /// Resolve a kb:// artifact reference to its current KB location
    Resolve {
        /// Artifact reference, e.g. <kb://wiki/concepts/runway.md>
        #[arg(required = true)]
        uri: String,
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
    /// Open an interactive chat session about the knowledge base
    ///
    /// Launches opencode's TUI with a read-only KB agent injected at
    /// startup. Requires an `[llm.runners.opencode]` block in kb.toml.
    Chat {
        /// LLM model to use (overrides kb.toml default)
        #[arg(long)]
        model: Option<String>,
    },
    /// Serve the KB as a local read-only web UI
    ///
    /// Starts a lightweight HTTP server that renders the `wiki/` tree as
    /// HTML and exposes `/search` + `/ask` endpoints. Bound to 127.0.0.1
    /// by default — there is no auth, so don't expose it publicly.
    Serve {
        /// Host interface to bind. Default: 127.0.0.1 (localhost-only).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// TCP port to listen on.
        #[arg(long, default_value_t = 8484)]
        port: u16,
    },
    /// Migrate a pre-`.kb/` layout into the current layout.
    ///
    /// Detects legacy `cache/`, `logs/`, `state/`, `trash/`, `normalized/`,
    /// and `prompts/` directories at the vault root, creates `.kb/`, and
    /// moves each one into place via `std::fs::rename`. Idempotent — a
    /// second run on an already-migrated vault is a no-op.
    Migrate,
    /// Run the golden Q/A retrieval-eval harness (bn-3sco)
    ///
    /// Loads `<kb-root>/evals/golden.toml`, runs every query through the
    /// hybrid retriever (no LLM call), and scores the top-10 ranking
    /// against the expected sources/concepts using P@K, MRR, and nDCG@10.
    /// Results land in `evals/results/<UTC-timestamp>.json` plus a
    /// `latest.md` summary.
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },
}

#[derive(clap::Subcommand)]
enum SessionAction {
    /// List sessions under `.kb/sessions/`, newest-first by last update.
    List,
    /// Print the full transcript for a session.
    Show {
        /// Session id (the `<id>` part of `<id>.json`).
        #[arg(required = true)]
        id: String,
    },
    /// Create a fresh empty session. If `<id>` is omitted, generates a terseid.
    New {
        /// Optional session id; defaults to a generated `s-...` terseid.
        id: Option<String>,
    },
}

#[derive(clap::Subcommand)]
enum EvalAction {
    /// Run every golden query through hybrid retrieval and write a result
    Run {
        /// Compare the new run against a previously-saved result
        /// (`evals/results/<name>.json`). When set, prints a side-by-side
        /// diff table after the standard summary.
        #[arg(long, value_name = "NAME")]
        baseline: Option<String>,
        /// Additionally copy the new result JSON to
        /// `evals/results/<name>.json` so it can be referenced as a
        /// baseline in later runs.
        #[arg(long, value_name = "NAME")]
        save_as: Option<String>,
    },
    /// List previously-written results in `evals/results/`, newest first
    List,
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

    // Best-effort: register sqlite-vec as a process-wide auto-extension
    // before any DB connection is opened so semantic search can use the
    // indexed KNN fast path. A failure here (extension disabled via
    // KB_SQLITE_VEC_AUTO=0, or auto-extension registration rejected) is
    // not fatal — semantic search falls back to a pure-Rust cosine path.
    if let Err(err) = kb_sqlite_vec::register_auto_extension() {
        tracing::debug!("sqlite-vec auto-extension unavailable: {err}");
    }

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
pub(crate) struct ValidationError {
    exit_code: i32,
    message: String,
}

impl ValidationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
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

    // Legacy-layout sentinel: bail fast for every command that would touch
    // the internal tree (state, cache, normalized, logs). `kb init` is
    // exempted because it may be overwriting an in-progress install, and
    // `kb migrate` is the way out. Read-only commands that only look at
    // wiki/ + outputs/ (e.g. `kb search`) still want the guard because a
    // legacy vault's state/indexes would be stale and confusing.
    if let Some(root_path) = root.as_deref() {
        let skip_sentinel = matches!(
            cli.command,
            Some(Command::Init { .. } | Command::Migrate) | None,
        );
        if !skip_sentinel {
            migrate::bail_if_legacy_layout(root_path)?;
        }
    }

    match cli.command {
        Some(Command::Compile) => {
            let compile_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let force = cli.force;
            let dry_run = cli.dry_run;
            let json = cli.json;
            let quiet = cli.quiet;
            let cli_model = cli.model.clone();
            if dry_run {
                // Dry-run reads graph/hashes and prints what would happen; it
                // writes nothing. Bypass `execute_mutating_command_with_handle`
                // so we neither block on the root lock (held by an in-flight
                // real compile) nor leave a job manifest behind if the caller
                // SIGPIPEs us (e.g. `kb compile --dry-run | head`).
                run_compile_action(
                    compile_root,
                    force,
                    true,
                    json,
                    quiet,
                    cli_model.as_deref(),
                    None,
                )
            } else {
                execute_mutating_command_with_handle(Some(compile_root), "compile", move |handle| {
                    // Stream per-pass events into `state/jobs/<id>.log` so a
                    // hung or failing compile leaves a useful trail.
                    run_compile_action(
                        compile_root,
                        force,
                        false,
                        json,
                        quiet,
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
        Some(Command::Ask { query, format, promote, editor, no_render, session }) => {
            let ask_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            let model = cli.model.clone();
            let dry_run = cli.dry_run;
            let json = cli.json;
            let quiet = cli.quiet;
            // bn-o6wv: with `--session`, branch into the multi-turn flow.
            // Without `--session`, keep the historical single-shot path
            // exactly as it was — including the resolve_query semantics
            // (positional > stdin > reedline > editor).
            if let Some(session_id) = session {
                session::validate_session_id(&session_id)
                    .map_err(|err| ValidationError::new(format!("{err}")))?;
                let action = move || {
                    run_ask_session(
                        ask_root,
                        &session_id,
                        query.as_deref(),
                        editor,
                        model.as_deref(),
                        json,
                        dry_run,
                        quiet,
                        no_render,
                        promote,
                        format.as_deref(),
                    )
                };
                if dry_run {
                    action()
                } else {
                    execute_mutating_command(Some(ask_root), "ask", action)
                }
            } else {
                let query = resolve_query(query, editor)?;
                let action = move || {
                    run_ask(
                        ask_root,
                        &query,
                        format.as_deref(),
                        model.as_deref(),
                        json,
                        dry_run,
                        promote,
                        quiet,
                        no_render,
                    )
                };
                if dry_run {
                    // Dry-run doesn't write anything; no lock needed.
                    action()
                } else {
                    execute_mutating_command(Some(ask_root), "ask", action)
                }
            }
        }
        Some(Command::Session { action }) => {
            let session_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_session_subcommand(session_root, action, cli.json)
        }
        Some(Command::Ingest {
            sources,
            allow_empty,
            include,
            exclude,
            branch,
            commit,
            audio,
            audio_title,
            audio_recording_date,
            audio_out,
        }) => {
            if sources.is_empty() && audio.is_none() {
                return Err(ValidationError::new(
                    "kb ingest requires at least one source path/URL, or --audio <PATH>",
                )
                .into());
            }
            if let Some(audio_path) = &audio {
                if !audio_path.exists() {
                    return Err(ValidationError::new(format!(
                        "audio file does not exist: {}",
                        audio_path.display()
                    ))
                    .into());
                }
            }

            // bn-1jx: validate local source paths exist before we acquire
            // the root lock and start a job manifest. `kb_ingest::collect_files`
            // bails with the same message, but doing it here means a bad
            // path never leaves a "failed" job behind in `state/jobs/`.
            for source in &sources {
                if kb_ingest::is_git_url(source) || kb_ingest::is_url(source) {
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

            // Repo-only flags only make sense when at least one source is a
            // git URL. Fail fast otherwise so users don't silently pass them
            // to local/URL ingest.
            let has_git = sources.iter().any(|s| kb_ingest::is_git_url(s));
            if !has_git
                && (!include.is_empty()
                    || !exclude.is_empty()
                    || branch.is_some()
                    || commit.is_some())
            {
                return Err(ValidationError::new(
                    "--include/--exclude/--branch/--commit require a git URL source",
                )
                .into());
            }

            let ingest_root = root.clone();
            let action = move || {
                let root = ingest_root
                    .as_deref()
                    .expect("root resolved for non-init commands");
                run_ingest(
                    root,
                    &sources,
                    cli.json,
                    cli.dry_run,
                    allow_empty,
                    &include,
                    &exclude,
                    branch.as_deref(),
                    commit.as_deref(),
                    audio.as_deref(),
                    audio_title.as_deref(),
                    audio_recording_date.as_deref(),
                    audio_out.as_deref(),
                )
            };

            if cli.dry_run {
                action()
            } else {
                execute_mutating_command(root.as_deref(), "ingest", action)
            }
        }
        Some(Command::Lint { check, strict, impute }) => {
            // Lint is read-only: it walks generated artifacts and surfaces findings.
            // It must not take the root lock, so users can `kb lint` during a compile.
            let lint_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_lint(lint_root, cli.json, check.as_deref(), strict, impute)
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
        Some(Command::Ls) => {
            let ls_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            ls::run_ls(ls_root, cli.json)
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
            let cfg = config::Config::load_from_root(search_root, None)?;
            let limit = limit.unwrap_or(10);

            // Detect the "query reduced entirely to stopwords" case before
            // scoring so we can surface a helpful message instead of a silent
            // empty result set. Exit 0 — this is a user-facing hint, not an
            // error.
            if kb_query::query_reduced_to_stopwords(&query) {
                if cli.json {
                    let empty: Vec<kb_query::HybridResult> = Vec::new();
                    emit_json("search", &empty)?;
                } else {
                    println!(
                        "No results for '{query}': query reduced to stopwords; try more specific terms."
                    );
                }
                return Ok(());
            }

            let backend_config = cfg.semantic.to_backend_config();
            let backend = kb_query::SemanticBackend::from_config(&backend_config)?;
            let results = kb_query::hybrid_search_with_backend(
                search_root,
                &query,
                limit,
                cfg.retrieval.to_hybrid_options(backend_config.kind),
                &backend,
            )?;
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
                    println!(
                        "{} [score: {:.4}]",
                        result.title, result.score
                    );
                    println!("  {}", result.item_id);
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
        Some(Command::Resolve { uri }) => {
            let root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            run_resolve(root, &uri, cli.json)
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
                        // bn-lw06: lazy adapter factory — only constructed
                        // when the approve path for the review's kind needs
                        // an LLM (currently ConceptCandidate). Keeps
                        // promotion/merge approvals from paying adapter
                        // init cost.
                        let adapter_factory = || -> Result<Box<dyn kb_llm::LlmAdapter>> {
                            build_compile_adapter(review_root, None)
                        };
                        review::run_review_approve(
                            review_root,
                            &id,
                            json,
                            &json_emitter,
                            &adapter_factory,
                        )
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
        Some(Command::Chat { model }) => {
            let chat_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            // Chat is an interactive TUI session — no job manifest, no
            // root lock. The underlying agent is read-only (tools_write
            // and tools_edit are forced false in the generated config),
            // so it can't mutate KB state even if the user asks it to.
            // --model CLI flag wins over --model subcommand flag; either
            // threads through to the opencode agent config.
            let model_override = model.as_deref().or(cli.model.as_deref());
            chat::run_chat(chat_root, model_override)
        }
        Some(Command::Serve { host, port }) => {
            let serve_root = root
                .as_deref()
                .expect("root resolved for non-init commands")
                .to_path_buf();
            // Defense-in-depth against copy/paste mistakes: v1 is explicitly
            // unauthenticated, so we refuse to bind non-loopback addresses
            // unless the operator force-opts-in. If you really need remote
            // access, set up a reverse proxy with auth — don't punch
            // `--host 0.0.0.0` straight into a dev tool.
            if !is_loopback_host(&host) && !cli.force {
                return Err(ValidationError::new(format!(
                    "refusing to bind {host}: kb serve has no authentication. \
                     Bind 127.0.0.1 (default), put a reverse proxy in front, \
                     or pass --force to override at your own risk."
                ))
                .into());
            }
            // Serve is read-only and long-running: no job manifest, no root
            // lock. The request handlers read the lexical index and wiki
            // tree directly; concurrent `kb compile` runs will not be
            // reflected until restart.
            let semantic_backend = Config::load_from_root(&serve_root, None)
                .map(|cfg| cfg.semantic.to_backend_config())
                .unwrap_or_default();
            let state = kb_web::WebState::with_backend_config(serve_root, &semantic_backend)?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("build tokio runtime for kb serve")?;
            rt.block_on(kb_web::serve(&host, port, state))
        }
        Some(Command::Migrate) => {
            let migrate_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            // Migrate is a mutating fs operation but it writes to a brand-
            // new `.kb/` tree that no other command owns yet, so it does
            // not take the root lock. Running `kb compile` concurrently
            // would already have bailed out via the legacy-layout sentinel.
            migrate::run_migrate(migrate_root, cli.json)
        }
        Some(Command::Eval { action }) => {
            let eval_root = root
                .as_deref()
                .expect("root resolved for non-init commands");
            match action {
                EvalAction::Run { baseline, save_as } => {
                    let flags = eval::RunFlags {
                        baseline,
                        save_as,
                        json: cli.json,
                    };
                    eval::cmd_run(eval_root, &flags)
                }
                EvalAction::List => eval::cmd_list(eval_root, cli.json),
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
    let probe = state_dir(root).join(".kb-doctor-write-test");
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

fn check_prompt_template_directory(root: &Path, _cfg: &Config) -> DoctorCheck {
    let prompt_dir = kb_core::prompts_dir(root);
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

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_ingest(
    root: &Path,
    sources: &[String],
    json: bool,
    dry_run: bool,
    allow_empty: bool,
    include: &[String],
    exclude: &[String],
    branch: Option<&str>,
    commit: Option<&str>,
    audio: Option<&Path>,
    audio_title: Option<&str>,
    audio_recording_date: Option<&str>,
    audio_out: Option<&Path>,
) -> Result<()> {
    let mut git_urls = Vec::new();
    let mut urls = Vec::new();
    let mut local_paths = Vec::new();

    if let Some(audio_path) = audio {
        let kbtx_path =
            transcribe_audio_to_kbtx(audio_path, audio_title, audio_recording_date, audio_out)?;
        eprintln!("[transcribed] {}", kbtx_path.display());
        local_paths.push(kbtx_path);
    }

    for source in sources {
        if kb_ingest::is_git_url(source) {
            // Test git URLs ahead of plain URLs: `https://github.com/foo/bar`
            // matches both predicates but should route to repo ingest.
            git_urls.push(source.as_str());
        } else if kb_ingest::is_url(source) {
            urls.push(source.as_str());
        } else if is_audio_path(source) {
            // Auto-route audio files through the transcription pipeline.
            // Saves the user from having to remember `--audio` for the
            // common case; explicit `--audio` is still supported above
            // for scripted use.
            let audio_path = PathBuf::from(source);
            let kbtx_path = transcribe_audio_to_kbtx(
                &audio_path,
                audio_title,
                audio_recording_date,
                audio_out,
            )?;
            eprintln!("[transcribed] {}", kbtx_path.display());
            local_paths.push(kbtx_path);
        } else {
            local_paths.push(PathBuf::from(source));
        }
    }

    // Load kb.toml so the ingest pipeline picks up `[ingest.markitdown]`
    // and `[ingest.ocr]` settings. The config loader tolerates missing
    // sections by falling back to defaults (markitdown on, OCR on for
    // scan-only PDFs), which means `kb ingest` keeps working on KBs that
    // predate bn-23am / bn-2hyr without any config edits.
    let cfg = Config::load_from_root(root, None).unwrap_or_default();
    let ingest_options = kb_ingest::IngestOptions {
        dry_run,
        allow_empty,
        markitdown: cfg.ingest.markitdown.to_options(),
        ocr: cfg.ingest.ocr.to_options(),
    };

    let mut results = Vec::new();
    for report in
        kb_ingest::ingest_paths_with_config(root, &local_paths, &ingest_options)?
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

    if !git_urls.is_empty() {
        let options = kb_ingest::RepoIngestOptions {
            includes: include.to_vec(),
            excludes: exclude.to_vec(),
            branch: branch.map(ToOwned::to_owned),
            commit: commit.map(ToOwned::to_owned),
            dry_run,
            allow_empty,
        };
        for url in &git_urls {
            let report = kb_ingest::ingest_repo(root, url, &options)?;
            // One IngestResult per ingested file, so the JSON summary and
            // per-item output stay structurally identical to local/URL ingest.
            for file in report.files {
                results.push(IngestResult {
                    input: format!("{}#{}", report.normalized_url, file.repo_path),
                    source_kind: "repo",
                    outcome: file.outcome,
                    source_document_id: file.source_document_id,
                    source_revision_id: file.source_revision_id,
                    content_path: file.content_path,
                    metadata_path: file.metadata_path,
                });
            }
        }
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

/// Recognize audio source paths by extension. Used to auto-route
/// audio files through the transcription pipeline without requiring
/// the user to pass `--audio` explicitly. Matches the container
/// formats we can decode via symphonia + the pyannote/whisper stack.
fn is_audio_path(source: &str) -> bool {
    const AUDIO_EXTENSIONS: &[&str] = &[
        ".m4a", ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".opus",
    ];
    let lower = source.to_lowercase();
    AUDIO_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Transcribe `audio_path` via kb-ingest-audio (whisper + pyannote), write
/// the rendered kbtx to `audio_out` (or `<stem>.kbtx.md` next to the audio
/// when not given), and return the kbtx path so it can be ingested as a
/// regular markdown source.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn transcribe_audio_to_kbtx(
    audio_path: &Path,
    audio_title: Option<&str>,
    audio_recording_date: Option<&str>,
    audio_out: Option<&Path>,
) -> Result<PathBuf> {
    let stem = audio_path
        .file_stem()
        .map_or_else(|| "transcript".to_string(), |s| s.to_string_lossy().to_string());
    let title = audio_title.map_or_else(|| stem.clone(), str::to_string);
    let recording_date =
        audio_recording_date.map_or_else(today_yyyy_mm_dd, str::to_string);
    let out_path = audio_out.map_or_else(
        || {
            let mut p = audio_path.to_path_buf();
            // Replace the extension with `.kbtx.md` next to the audio file.
            p.set_extension("kbtx.md");
            p
        },
        Path::to_path_buf,
    );

    let cfg = kb_ingest_audio::TranscribeConfig::new(audio_path, &title, &recording_date);
    eprintln!(
        "[transcribing] {} -> {} (title={title:?}, date={recording_date})",
        audio_path.display(),
        out_path.display()
    );
    let kbtx = kb_ingest_audio::transcribe(&cfg)
        .map_err(|e| anyhow::anyhow!("audio transcription failed: {e}"))?;

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating parent dir for {}", out_path.display())
            })?;
        }
    }
    std::fs::write(&out_path, kbtx)
        .with_context(|| format!("writing kbtx to {}", out_path.display()))?;
    Ok(out_path)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn today_yyyy_mm_dd() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Avoid pulling in chrono just for today's date — derive from epoch.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    // Days since epoch.
    let days = secs.div_euclid(86_400);
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert days since 1970-01-01 to (year, month, day). Adapted from
/// Howard Hinnant's `civil_from_days`.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
const fn days_to_ymd(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

fn normalized_dir_for(item: &IngestResult) -> PathBuf {
    if item.source_kind == "url" {
        // For URL ingest, content_path already points into normalized/<id>/...
        // Report the directory containing it.
        item.content_path
            .parent()
            .map_or_else(|| item.content_path.clone(), Path::to_path_buf)
    } else {
        normalized_rel(&item.source_document_id)
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

/// Persist `answer.md` + `metadata.json` for a chart ask that failed to
/// produce a PNG.
///
/// bn-1hqh: when `--format=chart` bails (LLM errored, printed an explicit
/// `ERROR:`, or ran to completion without writing `chart.png`), we still want
/// the outputs/questions/<q-id>/ directory to exist with the raw LLM output
/// so the user has something to diagnose. The raw body is written verbatim
/// — do **not** pass it through `strip_tool_narration`; narration is the
/// interesting part on a failure path.
///
/// `raw_llm_output` is `None` when the LLM call itself errored before
/// producing any text.
fn write_chart_failure_artifacts(
    dir_abs: &Path,
    reason: &str,
    raw_llm_output: Option<&str>,
) -> Result<()> {
    use kb_core::fs::atomic_write;

    std::fs::create_dir_all(dir_abs)
        .with_context(|| format!("create chart failure dir {}", dir_abs.display()))?;

    let raw_section = match raw_llm_output {
        Some(body) if !body.trim().is_empty() => body.to_string(),
        Some(_) => "_(LLM produced no output.)_".to_string(),
        None => "_(LLM call failed before producing any output.)_".to_string(),
    };
    let answer_body = format!(
        "## Chart generation failed\n\n\
         Reason: {reason}\n\n\
         ## LLM output\n\n\
         {raw_section}\n"
    );
    atomic_write(dir_abs.join("answer.md"), answer_body.as_bytes()).with_context(|| {
        format!("write chart failure answer.md in {}", dir_abs.display())
    })?;

    let metadata = serde_json::json!({
        "requested_format": "chart",
        "success": false,
        "error": reason,
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .context("serialize chart failure metadata.json")?;
    atomic_write(dir_abs.join("metadata.json"), metadata_json.as_bytes()).with_context(|| {
        format!("write chart failure metadata.json in {}", dir_abs.display())
    })?;

    Ok(())
}

/// bn-28ob: counterpart to [`write_chart_failure_artifacts`] for the
/// `excalidraw` format. Persists `answer.md` + `metadata.json` describing
/// why no diagrams landed in the q-dir, so the user can inspect what went
/// wrong without re-running the LLM.
fn write_excalidraw_failure_artifacts(
    dir_abs: &Path,
    reason: &str,
    raw_llm_output: Option<&str>,
) -> Result<()> {
    use kb_core::fs::atomic_write;

    std::fs::create_dir_all(dir_abs)
        .with_context(|| format!("create excalidraw failure dir {}", dir_abs.display()))?;

    let raw_section = match raw_llm_output {
        Some(body) if !body.trim().is_empty() => body.to_string(),
        Some(_) => "_(LLM produced no output.)_".to_string(),
        None => "_(LLM call failed before producing any output.)_".to_string(),
    };
    let answer_body = format!(
        "## Excalidraw generation failed\n\n\
         Reason: {reason}\n\n\
         ## LLM output\n\n\
         {raw_section}\n"
    );
    atomic_write(dir_abs.join("answer.md"), answer_body.as_bytes()).with_context(|| {
        format!("write excalidraw failure answer.md in {}", dir_abs.display())
    })?;

    let metadata = serde_json::json!({
        "requested_format": "excalidraw",
        "success": false,
        "error": reason,
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .context("serialize excalidraw failure metadata.json")?;
    atomic_write(dir_abs.join("metadata.json"), metadata_json.as_bytes()).with_context(|| {
        format!("write excalidraw failure metadata.json in {}", dir_abs.display())
    })?;

    Ok(())
}

/// bn-2cs2: classification of files promoted out of the auto-format
/// sandbox. The two buckets line up with the artifact kinds the auto prompt
/// teaches the model to produce: Excalidraw diagrams (`.excalidraw[.md]`)
/// and chart PNGs (`.png`). Anything outside these extensions is left in
/// the sandbox so the post-cleanup `remove_dir_all` sweeps it away.
#[derive(Default)]
struct AutoPromoted {
    diagrams: Vec<PathBuf>,
    charts: Vec<PathBuf>,
}

/// bn-2cs2: counterpart to [`promote_excalidraw_files`] for the auto-format
/// sandbox. Walks the sandbox once, classifies each file by extension, and
/// moves accepted ones up alongside `answer.md`. Returns both lists so the
/// caller can record them in `output_paths` and embed them in the body.
fn promote_auto_files(sandbox: &Path, dest_dir: &Path) -> Result<AutoPromoted> {
    let mut out = AutoPromoted::default();
    let entries = std::fs::read_dir(sandbox)
        .with_context(|| format!("read auto sandbox {}", sandbox.display()))?;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let kind = if name_str.ends_with(".excalidraw")
            || name_str.ends_with(".excalidraw.md")
        {
            Some("diagram")
        } else if name_str.ends_with(".png") {
            Some("chart")
        } else {
            None
        };
        let Some(kind) = kind else {
            continue;
        };
        let src = entry.path();
        let dest = dest_dir.join(&name);
        if let Err(rename_err) = std::fs::rename(&src, &dest) {
            std::fs::copy(&src, &dest).with_context(|| {
                format!(
                    "copy {} to {} (rename failed: {rename_err})",
                    src.display(),
                    dest.display(),
                )
            })?;
            std::fs::remove_file(&src).with_context(|| {
                format!("remove sandbox file {} after copy", src.display())
            })?;
        }
        match kind {
            "diagram" => out.diagrams.push(dest),
            "chart" => out.charts.push(dest),
            _ => unreachable!("kind is one of diagram|chart"),
        }
    }
    Ok(out)
}

/// bn-28ob: move accepted Excalidraw files out of the per-question sandbox
/// dir (`<q-dir>/.diagrams/`) into the q-dir itself, alongside `answer.md`.
///
/// Anything that doesn't match `*.excalidraw` or `*.excalidraw.md` is left
/// in the sandbox so the caller can decide to delete or inspect it. We
/// rename when possible and fall back to copy+remove for cross-device
/// edge cases. Returns the absolute paths of the promoted files in the
/// destination dir.
fn promote_excalidraw_files(sandbox: &Path, dest_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut moved = Vec::new();
    let entries = std::fs::read_dir(sandbox)
        .with_context(|| format!("read excalidraw sandbox {}", sandbox.display()))?;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let accepted = name_str.ends_with(".excalidraw")
            || name_str.ends_with(".excalidraw.md");
        if !accepted {
            continue;
        }
        let src = entry.path();
        let dest = dest_dir.join(&name);
        if let Err(rename_err) = std::fs::rename(&src, &dest) {
            std::fs::copy(&src, &dest).with_context(|| {
                format!(
                    "copy {} to {} (rename failed: {rename_err})",
                    src.display(),
                    dest.display(),
                )
            })?;
            std::fs::remove_file(&src).with_context(|| {
                format!("remove sandbox file {} after copy", src.display())
            })?;
        }
        moved.push(dest);
    }
    Ok(moved)
}

#[allow(clippy::too_many_lines, clippy::fn_params_excessive_bools)]
#[allow(clippy::too_many_arguments)]
fn run_ask(
    root: &Path,
    query: &str,
    requested_format: Option<&str>,
    cli_model: Option<&str>,
    json: bool,
    dry_run: bool,
    promote: bool,
    quiet: bool,
    no_render: bool,
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
            "--format png is not yet supported; supported formats: auto, md, marp, json, chart, excalidraw",
        )
        .into());
    }

    let backend_config = cfg.semantic.to_backend_config();
    let backend = kb_query::SemanticBackend::from_config(&backend_config)?;
    let retrieval_plan = kb_query::plan_retrieval_hybrid_with_backend(
        root,
        query,
        cfg.ask.token_budget,
        cfg.retrieval.to_hybrid_options(backend_config.kind),
        &backend,
    )?;

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
                if let Some(fused) = candidate.fused_score {
                    println!(
                        "  {} [score: {fused:.4} (fused), ~{} tokens]",
                        candidate.id, candidate.estimated_tokens
                    );
                } else {
                    println!(
                        "  {} [score: {}, ~{} tokens]",
                        candidate.id, candidate.score, candidate.estimated_tokens
                    );
                }
                for reason in &candidate.reasons {
                    println!("    reason: {reason}");
                }
            }
        }
        return Ok(());
    }

    let timestamp = now_millis()?;
    let question_id = generate_question_id(root, timestamp, query);

    // bn-nlw9: dir name is `q-<id>-<slug>` when the question text slugs to
    // something non-empty, else `q-<id>`. `question_id` is still the stable
    // id prefix used by lookups (`kb inspect q-<id>` resolves through
    // `resolve_question_dir`). The slug is derived from the question text
    // with the shared filename slugifier.
    let question_slug =
        kb_core::slug_for_filename(query, kb_core::DEFAULT_FILENAME_SLUG_MAX_CHARS);
    // bn-6puc guard: when the question text slugifies to `q-<id>` (or
    // `q-<id>-...`) the suffix would just duplicate the id in the dir name.
    // Fall back to the id-only form rather than emit `q-1xk-q-1xk-intro`.
    let question_dir_name = if question_slug.is_empty()
        || kb_core::slug_redundant_with_id(&question_slug, &question_id)
    {
        question_id.clone()
    } else {
        format!("{question_id}-{question_slug}")
    };
    let base_dir = PathBuf::from("outputs/questions").join(&question_dir_name);

    // Filename tracks the requested format. JSON gets `.json`; everything
    // else (md/marp/chart) stays as `.md`. Keep in sync with `kb_query::write_artifact`.
    let answer_file_name = match requested_format {
        "json" => "answer.json",
        _ => "answer.md",
    };
    let answer_rel = base_dir.join(answer_file_name);
    let question_rel = base_dir.join("question.json");
    let plan_rel = base_dir.join("retrieval_plan.json");

    // Chart-format plumbing: resolve the PNG path the LLM will be told to
    // write to. The directory has to exist before the LLM runs so its bash
    // tool can drop the file without `mkdir -p`.
    let (chart_rel, chart_abs) = if requested_format == "chart" {
        let rel = base_dir.join("chart.png");
        let abs = root.join(&rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("create chart output directory {}", parent.display())
            })?;
        }
        (Some(rel), Some(abs))
    } else {
        (None, None)
    };

    // bn-28ob: Excalidraw plumbing. The LLM is given a sandbox dir under
    // base_dir and asked to drop one or more `.excalidraw` JSON files there.
    // Once the run finishes, accepted files are moved up alongside answer.md
    // and the sandbox dir is removed — see below. Pre-create both dirs so
    // the model's `write` tool can drop files without `mkdir -p`.
    let excalidraw_sandbox_abs = if requested_format == "excalidraw" {
        let abs = root.join(&base_dir).join(".diagrams");
        fs::create_dir_all(&abs).with_context(|| {
            format!("create excalidraw sandbox dir {}", abs.display())
        })?;
        Some(abs)
    } else {
        None
    };

    // bn-2cs2: auto-format plumbing. The LLM gets a single scratch dir and
    // can drop a mix of supporting artifacts (Excalidraw diagrams, chart
    // PNGs) — or none at all if plain text answers the question. After the
    // run, accepted files are classified by extension and promoted to the
    // q-dir alongside `answer.md`; the scratch dir is removed.
    let auto_sandbox_abs = if requested_format == "auto" {
        let abs = root.join(&base_dir).join(".scratch");
        fs::create_dir_all(&abs).with_context(|| {
            format!("create auto-format sandbox dir {}", abs.display())
        })?;
        Some(abs)
    } else {
        None
    };

    let assembled = kb_query::assemble_context(root, &retrieval_plan)?;
    let citation_manifest = kb_query::build_citation_manifest(&assembled);
    let manifest_text = kb_query::render_manifest_for_prompt(&citation_manifest);

    // bn-3dkw: multimodal retrieval. For each candidate, pull out any local
    // image refs in its body and hand the absolute paths to the adapter so
    // the LLM actually sees the images. Capped at MAX_IMAGES_PER_QUERY to
    // stop context from blowing up on image-heavy corpora, and gated by a
    // cheap "does any candidate mention an image?" check so the common
    // text-only path pays nothing.
    let image_paths = if kb_query::plan_mentions_images(root, &retrieval_plan) {
        kb_query::resolve_candidate_image_paths(root, &retrieval_plan)
    } else {
        Vec::new()
    };

    let (template_override, output_path_override) = match requested_format {
        "chart" => (
            Some("ask_chart.md"),
            chart_abs
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        ),
        "excalidraw" => (
            Some("ask_excalidraw.md"),
            excalidraw_sandbox_abs
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        ),
        "auto" => (
            Some("ask_auto.md"),
            auto_sandbox_abs
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
        ),
        _ => (None, None),
    };

    // bn-1ikn: tool-driving formats (chart, excalidraw, auto) ask the
    // adapter for a structured event stream so we can pull just the final
    // assistant message and skip the per-tool narration opencode emits
    // ahead of it. Other formats keep plain-text output — their caller
    // doesn't drive the LLM through tools.
    let structured_output =
        matches!(requested_format, "chart" | "excalidraw" | "auto");

    let llm_outcome = try_generate_answer(
        &cfg,
        root,
        query,
        &assembled,
        &manifest_text,
        template_override,
        output_path_override.as_deref(),
        &image_paths,
        structured_output,
    );

    // Chart format rejects silent fallbacks. If the LLM call failed OR the
    // expected PNG is missing, we bail with a clean error pointing at the
    // outputs dir.
    //
    // bn-1hqh: on any chart failure, persist `answer.md` (raw LLM output
    // verbatim — no narration stripping) and `metadata.json` (`success:
    // false`) under outputs/questions/<q-id>/ before bailing. The user asked
    // for a chart; when they don't get one, leave enough artifacts behind
    // that they can work out *why*. The output dir's parent was already
    // created above so the LLM's bash tool could drop the PNG; here we just
    // make sure the dir itself (and the failure artifacts) exist.
    //
    // The LLM's own reply is also checked for an explicit `ERROR:` prefix,
    // which the prompt asks for when the sources don't support a chart.
    if requested_format == "chart" {
        let chart_dir_abs = chart_abs
            .as_ref()
            .and_then(|p| p.parent())
            .expect("chart_abs has a parent for chart format")
            .to_path_buf();
        let answer_md_rel_display = base_dir.join("answer.md").to_string_lossy().into_owned();
        match &llm_outcome {
            Err(err) => {
                let reason = format!("LLM call failed: {err}");
                write_chart_failure_artifacts(&chart_dir_abs, &reason, None)?;
                bail!(
                    "--format chart failed: {reason}. See {answer_md_rel_display} for details."
                );
            }
            Ok((result, _)) => {
                let trimmed = result.body.trim_start();
                if let Some(rest) = trimmed.strip_prefix("ERROR:") {
                    let reason = format!("LLM declined to produce a chart: {}", rest.trim());
                    write_chart_failure_artifacts(
                        &chart_dir_abs,
                        &reason,
                        Some(&result.body),
                    )?;
                    bail!(
                        "--format chart failed: {reason}. See {answer_md_rel_display} for LLM output."
                    );
                }
                let expected = chart_abs.as_ref().expect("chart_abs set for chart format");
                if !expected.exists() {
                    let reason = format!(
                        "expected {} was not produced",
                        expected.display()
                    );
                    write_chart_failure_artifacts(
                        &chart_dir_abs,
                        &reason,
                        Some(&result.body),
                    )?;
                    bail!(
                        "--format chart failed: {reason}. \
                         See {answer_md_rel_display} for LLM output."
                    );
                }
            }
        }
    }

    // bn-28ob: Excalidraw success criteria — at least one accepted
    // `.excalidraw[.md]` file must land in the q-dir alongside `answer.md`.
    // Mirrors the chart code path: on any failure (LLM error, ERROR: reply,
    // empty sandbox), persist `answer.md` + `metadata.json` so the user can
    // inspect *why*, then bail. On success, move accepted files up out of
    // the sandbox dir and remove the sandbox so it doesn't litter the q-dir.
    let excalidraw_promoted: Vec<PathBuf> = if requested_format == "excalidraw" {
        let sandbox = excalidraw_sandbox_abs
            .as_ref()
            .expect("sandbox set for excalidraw format")
            .clone();
        let q_dir_abs = root.join(&base_dir);
        let answer_md_rel_display = base_dir.join("answer.md").to_string_lossy().into_owned();
        match &llm_outcome {
            Err(err) => {
                let reason = format!("LLM call failed: {err}");
                let _ = std::fs::remove_dir_all(&sandbox);
                write_excalidraw_failure_artifacts(&q_dir_abs, &reason, None)?;
                bail!(
                    "--format excalidraw failed: {reason}. See {answer_md_rel_display} for details."
                );
            }
            Ok((result, _)) => {
                let trimmed = result.body.trim_start();
                if let Some(rest) = trimmed.strip_prefix("ERROR:") {
                    let reason =
                        format!("LLM declined to produce a diagram: {}", rest.trim());
                    let _ = std::fs::remove_dir_all(&sandbox);
                    write_excalidraw_failure_artifacts(
                        &q_dir_abs,
                        &reason,
                        Some(&result.body),
                    )?;
                    bail!(
                        "--format excalidraw failed: {reason}. See {answer_md_rel_display} for LLM output."
                    );
                }
                let promoted = promote_excalidraw_files(&sandbox, &q_dir_abs)
                    .with_context(|| {
                        format!(
                            "move excalidraw files from {} into {}",
                            sandbox.display(),
                            q_dir_abs.display()
                        )
                    })?;
                let _ = std::fs::remove_dir_all(&sandbox);
                if promoted.is_empty() {
                    let reason = "no .excalidraw files were produced".to_string();
                    write_excalidraw_failure_artifacts(
                        &q_dir_abs,
                        &reason,
                        Some(&result.body),
                    )?;
                    bail!(
                        "--format excalidraw failed: {reason}. \
                         See {answer_md_rel_display} for LLM output."
                    );
                }
                promoted
            }
        }
    } else {
        Vec::new()
    };

    // bn-2cs2: auto format processes its sandbox after the LLM run. Unlike
    // chart and excalidraw, auto has *no* required artifacts — a plain
    // markdown answer is a valid outcome. So no `bail!` on empty sandbox;
    // the run is treated as successful regardless. Files are classified by
    // extension and promoted into the q-dir; anything unrecognized stays in
    // the sandbox so the post-cleanup `remove_dir_all` sweeps it away. The
    // sandbox is removed on both success and failure paths so partial state
    // never leaks into the q-dir.
    let auto_promoted: AutoPromoted = if requested_format == "auto" {
        let sandbox = auto_sandbox_abs
            .as_ref()
            .expect("sandbox set for auto format")
            .clone();
        let q_dir_abs = root.join(&base_dir);
        let promoted = if llm_outcome.is_ok() {
            promote_auto_files(&sandbox, &q_dir_abs).with_context(|| {
                format!(
                    "promote auto artifacts from {} into {}",
                    sandbox.display(),
                    q_dir_abs.display(),
                )
            })?
        } else {
            // LLM failed entirely. Drop the sandbox and fall through to the
            // placeholder body — auto doesn't need its own failure artifact
            // path because the placeholder already says "LLM unavailable".
            AutoPromoted::default()
        };
        let _ = std::fs::remove_dir_all(&sandbox);
        promoted
    } else {
        AutoPromoted::default()
    };

    let (model_version, template_hash, artifact_status, mut artifact_body, llm_info) =
        match llm_outcome {
            Ok((result, provenance)) => {
                // For chart format, the artifact body is the LLM's caption
                // followed by a markdown image reference to the PNG. The PNG
                // itself was already verified to exist above. All other
                // formats render the raw post-processed body as-is.
                //
                // bn-31uk: chart runs drive the LLM through the write/bash
                // tool cycle, so opencode streams the model's per-tool-call
                // "what I'm about to do" commentary into stdout ahead of the
                // real caption. Pass the post-processed body through
                // `strip_tool_narration` to cut that preamble cleanly when
                // the body has a markdown heading or a blank-line paragraph
                // break. Markdown / JSON asks don't invoke tools and are
                // intentionally untouched.
                let body = match requested_format {
                    "chart" => {
                        let caption = kb_query::strip_tool_narration(&result.body)
                            .trim()
                            .to_string();
                        let image_line = "![chart](chart.png)";
                        if caption.is_empty() {
                            format!("{image_line}\n")
                        } else {
                            format!("{caption}\n\n{image_line}\n")
                        }
                    }
                    "excalidraw" => {
                        // bn-28ob: strip opencode's tool-call narration the
                        // same way chart does, then ensure every promoted
                        // diagram is referenced — if the model forgot to
                        // embed one, append the wikilink so the answer never
                        // ships orphaned files.
                        let caption = kb_query::strip_tool_narration(&result.body)
                            .trim()
                            .to_string();
                        let mut body = String::new();
                        if !caption.is_empty() {
                            body.push_str(&caption);
                            body.push('\n');
                        }
                        for path in &excalidraw_promoted {
                            if let Some(name) =
                                path.file_name().and_then(|s| s.to_str())
                            {
                                let embed = format!("![[{name}]]");
                                if !body.contains(&embed) {
                                    body.push('\n');
                                    body.push_str(&embed);
                                    body.push('\n');
                                }
                            }
                        }
                        body
                    }
                    "auto" => {
                        // bn-2cs2: strip tool narration, then append safety-net
                        // embeds for any promoted artifact the model forgot to
                        // reference. Unlike chart/excalidraw the model may
                        // produce zero artifacts; in that case the body is
                        // just the cleaned text.
                        let caption = kb_query::strip_tool_narration(&result.body)
                            .trim()
                            .to_string();
                        let mut body = String::new();
                        if !caption.is_empty() {
                            body.push_str(&caption);
                            body.push('\n');
                        }
                        for path in &auto_promoted.diagrams {
                            if let Some(name) =
                                path.file_name().and_then(|s| s.to_str())
                            {
                                let embed = format!("![[{name}]]");
                                if !body.contains(&embed) {
                                    body.push('\n');
                                    body.push_str(&embed);
                                    body.push('\n');
                                }
                            }
                        }
                        for path in &auto_promoted.charts {
                            if let Some(name) =
                                path.file_name().and_then(|s| s.to_str())
                            {
                                let embed = format!("![chart]({name})");
                                if !body.contains(&embed) && !body.contains(&format!("({name})"))
                                {
                                    body.push('\n');
                                    body.push_str(&embed);
                                    body.push('\n');
                                }
                            }
                        }
                        body
                    }
                    _ => result.body.clone(),
                };
                (
                    Some(provenance.model.clone()),
                    Some(provenance.prompt_template_hash.to_hex()),
                    if result.invalid_citations.is_empty() {
                        Status::Fresh
                    } else {
                        Status::NeedsReview
                    },
                    body,
                    Some((result, provenance)),
                )
            }
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

    let mut question_output_paths = vec![question_rel, answer_rel.clone(), plan_rel];
    if let Some(rel) = chart_rel.as_ref() {
        question_output_paths.push(rel.clone());
    }
    // bn-28ob: record promoted excalidraw files as outputs of this ask so
    // tooling that walks dependency graphs (compile, lint orphans) sees them.
    let excalidraw_rels: Vec<PathBuf> = excalidraw_promoted
        .iter()
        .filter_map(|abs| {
            abs.strip_prefix(root)
                .ok()
                .map(std::path::Path::to_path_buf)
        })
        .collect();
    for rel in &excalidraw_rels {
        question_output_paths.push(rel.clone());
    }

    // bn-2cs2: same for auto-promoted artifacts (diagrams + chart PNGs).
    let auto_rels: Vec<PathBuf> = auto_promoted
        .diagrams
        .iter()
        .chain(auto_promoted.charts.iter())
        .filter_map(|abs| {
            abs.strip_prefix(root)
                .ok()
                .map(std::path::Path::to_path_buf)
        })
        .collect();
    for rel in &auto_rels {
        question_output_paths.push(rel.clone());
    }

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
            output_paths: question_output_paths,
            status: artifact_status,
        },
        raw_query: query.to_string(),
        requested_format: requested_format.to_string(),
        requesting_context: QuestionContext::ProjectKb,
        retrieval_plan: base_dir
            .join("retrieval_plan.json")
            .to_string_lossy()
            .into_owned(),
        token_budget: Some(cfg.ask.token_budget),
    };

    let mut artifact_output_paths = vec![answer_rel];
    if let Some(rel) = chart_rel.as_ref() {
        artifact_output_paths.push(rel.clone());
    }
    for rel in &excalidraw_rels {
        artifact_output_paths.push(rel.clone());
    }
    for rel in &auto_rels {
        artifact_output_paths.push(rel.clone());
    }

    let artifact = Artifact {
        metadata: EntityMetadata {
            id: format!(
                "art-{}",
                question_id.strip_prefix("q-").unwrap_or(&question_id)
            ),
            created_at_millis: timestamp,
            updated_at_millis: now_millis()?,
            source_hashes: Vec::new(),
            model_version,
            tool_version: Some(format!("kb/{}", env!("CARGO_PKG_VERSION"))),
            prompt_template_hash: None,
            dependencies: vec![question_id.clone()],
            output_paths: artifact_output_paths,
            status: artifact_status,
        },
        question_id: question_id.clone(),
        artifact_kind: match requested_format {
            "png" | "chart" | "excalidraw" => ArtifactKind::Figure,
            "marp" => ArtifactKind::SlideDeck,
            "json" => ArtifactKind::JsonSpec,
            _ => ArtifactKind::AnswerNote,
        },
        format: requested_format.to_string(),
        output_path: base_dir.join(answer_file_name),
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

    // bn-166d: verify quoted spans actually appear in their cited sources
    // and append a footer summarizing the result. Only runs when the LLM
    // produced a real body; placeholder artifacts are skipped because
    // there's nothing to verify. The footer is appended to `artifact_body`
    // *before* write_artifact so it lands inside answer.md and the
    // structured answer.json `body` field alike.
    let quote_verifications: Vec<kb_core::QuoteVerification> = if llm_info.is_some()
        && cfg.lint.citation_verification.enabled
    {
        kb_core::verify_body_quotes(
            &artifact_body,
            cfg.lint.citation_verification.fuzz_per_100_chars,
            |src_id| {
                kb_compile::source_page::resolve_source_page_path(root, src_id)
                    .and_then(|rel| std::fs::read_to_string(root.join(rel)).ok())
            },
        )
    } else {
        Vec::new()
    };
    if !quote_verifications.is_empty() {
        artifact_body.push_str(&render_quote_verification_footer(&quote_verifications));
    }

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
        question_dir_name: Some(&question_dir_name),
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
    } else {
        // Render the answer body to stdout before the footer. Skipped when the
        // caller opted out (`--quiet`, `--no-render`) or when the requested
        // format is `json` (body is structured, not markdown) or when the
        // LLM produced no real body (placeholder artifact).
        // `artifact_body` is the raw LLM output (no frontmatter), already in
        // memory; no need to re-read answer.md.
        let render_body =
            !quiet && !no_render && requested_format != "json" && llm_info.is_some();
        if render_body {
            if std::io::stdout().is_terminal() {
                // TTY: colorized markdown via termimad.
                termimad::print_text(&artifact_body);
            } else {
                // Piped: raw markdown bytes exactly as persisted to the body.
                print!("{artifact_body}");
                if !artifact_body.ends_with('\n') {
                    println!();
                }
            }
            println!();
        }
        println!("Artifact written: {artifact_path}");
        if let Some((result, provenance)) = &llm_info {
            println!(
                "Citations: {} valid, {} unresolved",
                result.valid_citations.len(),
                result.invalid_citations.len()
            );
            if result.has_uncertainty_banner {
                println!("Note: low source coverage — uncertainty banner added.");
            }
            println!("Model: {} ({}ms)", provenance.model, provenance.latency_ms);
        } else {
            println!("LLM backend unavailable — placeholder artifact created.");
        }
        if promote {
            println!("ReviewItem created for promotion.");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// bn-o6wv: conversational `kb ask --session <id>` and `kb session ...`
// ---------------------------------------------------------------------------

/// Maximum number of prior turns to include in the rewrite prompt. The
/// rewrite only needs the recent context — a long-running session pays
/// for nothing by sending its first 50 turns to the rewriter every time.
const SESSION_REWRITE_HISTORY_TURNS: usize = 8;

/// Maximum number of prior turns to include in the answer prompt's
/// `{{conversation}}` block. Older turns get trimmed first so the
/// retrieval context still fits the token budget. Mirrors the rewrite
/// budget so the model sees the same history shape across both calls.
const SESSION_ANSWER_HISTORY_TURNS: usize = 8;

/// One round-trip of a session-mode ask: rewrite (if needed), retrieve,
/// answer, and append two turns to the on-disk session file.
///
/// `query.is_some()` does one round and exits. `query.is_none()` opens
/// an interactive REPL on the session, looping until the user enters
/// `:q` or sends EOF (Ctrl-D).
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn run_ask_session(
    root: &Path,
    session_id: &str,
    query: Option<&str>,
    editor: bool,
    cli_model: Option<&str>,
    json: bool,
    dry_run: bool,
    quiet: bool,
    no_render: bool,
    promote: bool,
    requested_format: Option<&str>,
) -> Result<()> {
    if dry_run {
        return Err(ValidationError::new(
            "kb ask --session does not support --dry-run yet",
        )
        .into());
    }
    if promote {
        return Err(ValidationError::new(
            "kb ask --session does not support --promote (sessions are scratch space; \
             ask without --session to promote a single-turn answer)",
        )
        .into());
    }

    if let Some(q) = query {
        // Single-shot turn against the named session.
        let q = if editor {
            // Editor mode is only sensible when there's no positional
            // query — but if both are passed, prefer the editor (matches
            // the historical resolve_query semantics).
            read_from_editor()?
        } else {
            q.to_string()
        };
        run_session_turn(
            root,
            session_id,
            &q,
            cli_model,
            json,
            quiet,
            no_render,
            requested_format,
        )?;
        return Ok(());
    }

    // No query provided — interactive REPL on the session.
    if editor {
        // `--editor` + no positional + interactive session would fight over
        // stdin between the REPL and the editor. Reject early.
        return Err(ValidationError::new(
            "kb ask --session --editor requires a positional question; drop --editor for the REPL",
        )
        .into());
    }
    run_session_repl(
        root,
        session_id,
        cli_model,
        json,
        quiet,
        no_render,
        requested_format,
    )
}

/// Drive the REPL loop for `kb ask --session <id>` with no query arg.
fn run_session_repl(
    root: &Path,
    session_id: &str,
    cli_model: Option<&str>,
    json: bool,
    quiet: bool,
    no_render: bool,
    requested_format: Option<&str>,
) -> Result<()> {
    use std::io::IsTerminal;

    if !std::io::stdin().is_terminal() {
        return Err(ValidationError::new(
            "kb ask --session with no positional question expects an interactive TTY; \
             pass a question on the command line or pipe one with `kb ask --session <id> -`",
        )
        .into());
    }

    eprintln!(
        "kb session {session_id}: type a question and press Enter, `:q` or Ctrl-D to exit.",
    );
    loop {
        eprint!(">>> ");
        std::io::Write::flush(&mut std::io::stderr()).ok();
        let mut buf = String::new();
        let read = std::io::stdin()
            .read_line(&mut buf)
            .context("read repl line")?;
        if read == 0 {
            // Ctrl-D / EOF.
            eprintln!();
            break;
        }
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == ":q" || trimmed == ":quit" || trimmed == ":exit" {
            break;
        }
        if let Err(err) = run_session_turn(
            root,
            session_id,
            trimmed,
            cli_model,
            json,
            quiet,
            no_render,
            requested_format,
        ) {
            eprintln!("error: {err:#}");
            // Don't bail — let the user retry with a new question.
        }
    }
    Ok(())
}

/// Execute a single conversational turn:
///
/// 1. Load (or create) the on-disk session.
/// 2. If prior turns exist, ask the LLM to rewrite the new question into
///    a standalone retrieval query. On rewrite failure, fall back to the
///    raw question — a flaky rewrite must not break the answer path.
/// 3. Run hybrid retrieval against the (possibly rewritten) query.
/// 4. Build an answer prompt that includes the prior turns plus the
///    retrieved chunks; send to the LLM.
/// 5. Persist a User turn (raw question + retrieval ids) and an
///    Assistant turn (answer text + citation labels) to the session.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_session_turn(
    root: &Path,
    session_id: &str,
    raw_question: &str,
    cli_model: Option<&str>,
    json: bool,
    quiet: bool,
    no_render: bool,
    requested_format: Option<&str>,
) -> Result<()> {
    let raw_question = raw_question.trim();
    if raw_question.is_empty() {
        return Err(ValidationError::new("ask: question cannot be empty").into());
    }

    let cfg = Config::load_from_root(root, cli_model)?;
    // bn-o6wv: sessions are conversational scratch — keep them on the
    // markdown-only path so each turn produces a clean text answer
    // without tool-driving (chart/excalidraw) side effects. Callers can
    // still override with `--format` (validated below) but the default
    // is `md` rather than the fancier `auto`.
    let format = requested_format.unwrap_or("md");
    let format = normalize_ask_format(format)?;
    if !matches!(format, "md" | "marp" | "json") {
        return Err(ValidationError::new(format!(
            "kb ask --session only supports text formats (md, marp, json); got {format}"
        ))
        .into());
    }

    let mut sess = match session::load(root, session_id)? {
        Some(s) => s,
        None => session::Session::new(session_id)?,
    };

    // Step 1: rewrite when there's prior context.
    let retrieval_query = if sess.turns.is_empty() {
        raw_question.to_string()
    } else {
        match rewrite_query_with_history(&cfg, root, raw_question, &sess) {
            Ok(rewritten) => {
                if rewritten.trim().is_empty() {
                    // Rewriter returned nothing usable — fall back.
                    raw_question.to_string()
                } else {
                    rewritten
                }
            }
            Err(err) => {
                tracing::warn!("session rewrite failed: {err}; falling back to raw question");
                raw_question.to_string()
            }
        }
    };

    // Step 2: hybrid retrieval against the rewritten query.
    let backend_config = cfg.semantic.to_backend_config();
    let backend = kb_query::SemanticBackend::from_config(&backend_config)?;
    let retrieval_plan = kb_query::plan_retrieval_hybrid_with_backend(
        root,
        &retrieval_query,
        cfg.ask.token_budget,
        cfg.retrieval.to_hybrid_options(backend_config.kind),
        &backend,
    )?;
    let retrieved_ids: Vec<String> = retrieval_plan
        .candidates
        .iter()
        .map(|c| c.id.clone())
        .collect();

    // Step 3: assemble context + manifest, then call the LLM with the
    // session-aware template that includes a `{{conversation}}` block.
    let assembled = kb_query::assemble_context(root, &retrieval_plan)?;
    let citation_manifest = kb_query::build_citation_manifest(&assembled);
    let manifest_text = kb_query::render_manifest_for_prompt(&citation_manifest);
    let conversation_text = render_conversation_for_prompt(
        &sess.turns,
        SESSION_ANSWER_HISTORY_TURNS,
    );

    let llm_outcome = try_generate_session_answer(
        &cfg,
        root,
        raw_question,
        &assembled,
        &manifest_text,
        &conversation_text,
    );

    let (answer_body, citations, model_label, latency_ms, fallback_note) = match llm_outcome {
        Ok((result, provenance)) => {
            let citations: Vec<String> = result
                .valid_citations
                .iter()
                .filter_map(|key| citation_manifest.entries.get(key))
                .map(|entry| entry.label.clone())
                .collect();
            (
                result.body,
                citations,
                Some(provenance.model),
                Some(provenance.latency_ms),
                None,
            )
        }
        Err(err) => (
            format!(
                "> **LLM unavailable:** {err}\n\n\
                 Question recorded in session `{session_id}`. Re-run when a backend is available.\n",
            ),
            Vec::new(),
            None,
            None,
            Some(format!("LLM unavailable: {err}")),
        ),
    };

    // Step 4: append both turns and persist.
    sess.push_turn(session::Turn {
        role: session::TurnRole::User,
        text: raw_question.to_string(),
        retrieved_ids: retrieved_ids.clone(),
        citations: Vec::new(),
    })?;
    sess.push_turn(session::Turn {
        role: session::TurnRole::Assistant,
        text: answer_body.clone(),
        retrieved_ids: Vec::new(),
        citations: citations.clone(),
    })?;
    session::save(root, &sess)?;

    // Step 5: present.
    let path = session::session_path(root, session_id);
    if json {
        emit_json(
            "ask",
            serde_json::json!({
                "session_id": session_id,
                "session_path": path,
                "rewritten_query": retrieval_query,
                "retrieved_ids": retrieved_ids,
                "citations": citations,
                "answer": answer_body,
                "turn_count": sess.turns.len(),
                "fallback_reason": fallback_note,
            }),
        )?;
    } else {
        let render_body = !quiet && !no_render;
        if render_body {
            if std::io::stdout().is_terminal() {
                termimad::print_text(&answer_body);
            } else {
                print!("{answer_body}");
                if !answer_body.ends_with('\n') {
                    println!();
                }
            }
            println!();
        }
        println!("Session: {} ({} turns)", session_id, sess.turns.len());
        if !citations.is_empty() {
            println!("Citations: {}", citations.join(", "));
        }
        if let (Some(model), Some(latency)) = (model_label.as_deref(), latency_ms) {
            println!("Model: {model} ({latency}ms)");
        } else {
            println!("LLM backend unavailable — placeholder turn recorded.");
        }
    }
    Ok(())
}

/// Render the conversation transcript that goes into the rewrite and
/// answer prompts. Keeps the most recent `max_turns` turns and drops
/// older ones — the oldest turns rarely matter for "what about X?"
/// follow-ups, and trimming keeps the prompt budget under control.
fn render_conversation_for_prompt(turns: &[session::Turn], max_turns: usize) -> String {
    if turns.is_empty() {
        return String::new();
    }
    let start = turns.len().saturating_sub(max_turns);
    let mut out = String::new();
    for turn in &turns[start..] {
        let role = match turn.role {
            session::TurnRole::User => "user",
            session::TurnRole::Assistant => "assistant",
        };
        // Indent each turn's text so multi-paragraph answers stay
        // readable inside the rendered prompt.
        out.push('[');
        out.push_str(role);
        out.push_str("] ");
        out.push_str(turn.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// Ask the LLM to rewrite `new_question` into a standalone retrieval
/// query given the recent session history. Uses the bundled
/// `rewrite_query.md` template so the prompt lives next to the other
/// adapter prompts.
fn rewrite_query_with_history(
    cfg: &Config,
    root: &Path,
    new_question: &str,
    sess: &session::Session,
) -> Result<String> {
    let adapter = create_ask_adapter(cfg, root)?;
    let history = render_conversation_for_prompt(&sess.turns, SESSION_REWRITE_HISTORY_TURNS);
    let request = kb_llm::AnswerQuestionRequest {
        question: new_question.to_string(),
        // The rewrite template uses `{{sources}}` for the conversation
        // block to keep the placeholder set the adapter already
        // populates — no need to grow the context map for one prompt.
        context: vec![history],
        format: Some(String::new()),
        template_name: Some("rewrite_query.md".to_string()),
        output_path: None,
        image_paths: Vec::new(),
        structured_output: false,
        conversation: String::new(),
    };
    let (response, _provenance) = adapter
        .answer_question(request)
        .map_err(|err| anyhow!("rewrite query: {err}"))?;
    // Models sometimes wrap the rewrite in quotes or fenced blocks even
    // though the prompt asks for a bare line — strip those defensively.
    let cleaned = response
        .answer
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
        .to_string();
    Ok(cleaned)
}

/// Send the session-aware answer prompt to the LLM. Mirrors the shape
/// of `try_generate_answer` but uses `ask_session.md` and threads the
/// rendered conversation through the new `conversation` field.
fn try_generate_session_answer(
    cfg: &Config,
    root: &Path,
    question: &str,
    assembled: &kb_query::AssembledContext,
    manifest_text: &str,
    conversation_text: &str,
) -> Result<(kb_query::ArtifactResult, kb_llm::ProvenanceRecord)> {
    let adapter = create_ask_adapter(cfg, root)?;
    let request = kb_llm::AnswerQuestionRequest {
        question: question.to_string(),
        context: vec![assembled.text.clone()],
        format: Some(manifest_text.to_string()),
        template_name: Some("ask_session.md".to_string()),
        output_path: None,
        image_paths: Vec::new(),
        structured_output: false,
        conversation: conversation_text.to_string(),
    };
    let (response, provenance) = adapter
        .answer_question(request)
        .map_err(|err| anyhow!("{err}"))?;
    let citation_manifest = kb_query::build_citation_manifest(assembled);
    let result = kb_query::postprocess_answer(&response.answer, &citation_manifest, assembled);
    Ok((result, provenance))
}

/// Dispatch `kb session list | show | new`.
#[allow(clippy::too_many_lines)]
fn run_session_subcommand(
    root: &Path,
    action: SessionAction,
    json: bool,
) -> Result<()> {
    match action {
        SessionAction::List => {
            let entries = session::list(root)?;
            if json {
                let payload: Vec<_> = entries
                    .iter()
                    .map(|e| {
                        serde_json::json!({
                            "id": e.id,
                            "updated_at_millis": e.updated_at_millis,
                            "turn_count": e.turn_count,
                        })
                    })
                    .collect();
                emit_json(
                    "session list",
                    serde_json::json!({ "sessions": payload }),
                )?;
            } else if entries.is_empty() {
                println!("(no sessions yet — start one with `kb ask --session <id> \"...\"`)");
            } else {
                println!("{:<24}  {:>6}  UPDATED", "ID", "TURNS");
                for e in &entries {
                    println!(
                        "{:<24}  {:>6}  {}",
                        e.id,
                        e.turn_count,
                        format_millis_iso8601(e.updated_at_millis)
                    );
                }
            }
            Ok(())
        }
        SessionAction::Show { id } => {
            session::validate_session_id(&id)
                .map_err(|err| ValidationError::new(format!("{err}")))?;
            let Some(sess) = session::load(root, &id)? else {
                return Err(ValidationError::new(format!(
                    "no such session: {id} (sessions live under .kb/sessions/<id>.json)"
                ))
                .into());
            };
            if json {
                emit_json("session show", &sess)?;
            } else {
                println!("# Session {} ({} turns)", sess.id, sess.turns.len());
                println!(
                    "Created: {}  Updated: {}",
                    format_millis_iso8601(sess.created_at_millis),
                    format_millis_iso8601(sess.updated_at_millis)
                );
                println!();
                for (i, turn) in sess.turns.iter().enumerate() {
                    let role = match turn.role {
                        session::TurnRole::User => "user",
                        session::TurnRole::Assistant => "assistant",
                    };
                    println!("## Turn {} [{}]", i + 1, role);
                    println!();
                    println!("{}", turn.text.trim());
                    println!();
                    if !turn.retrieved_ids.is_empty() {
                        println!("retrieved: {}", turn.retrieved_ids.join(", "));
                    }
                    if !turn.citations.is_empty() {
                        println!("citations: {}", turn.citations.join(", "));
                    }
                    println!();
                }
            }
            Ok(())
        }
        SessionAction::New { id } => {
            let id = match id {
                Some(custom) => {
                    session::validate_session_id(&custom)
                        .map_err(|err| ValidationError::new(format!("{err}")))?;
                    custom
                }
                None => session::generate_session_id(root),
            };
            if session::load(root, &id)?.is_some() {
                return Err(ValidationError::new(format!(
                    "session `{id}` already exists; pick a different id or run `kb session show {id}`"
                ))
                .into());
            }
            let sess = session::Session::new(&id)?;
            session::save(root, &sess)?;
            if json {
                emit_json(
                    "session new",
                    serde_json::json!({
                        "id": id,
                        "session_path": session::session_path(root, &id),
                    }),
                )?;
            } else {
                println!("Created session: {id}");
                println!(
                    "Add the first turn with: kb ask --session {id} \"your question\""
                );
            }
            Ok(())
        }
    }
}

/// Format a wall-clock millisecond timestamp as a relative "N units ago"
/// string anchored at the current time. Picks the largest unit that
/// renders to at least 1 (seconds → minutes → hours → days). Falls back
/// to the raw epoch-millis for dates in the future or before the epoch.
fn format_millis_iso8601(millis: i64) -> String {
    let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return format!("(t={millis})");
    };
    let Ok(now_ms) = i64::try_from(now.as_millis()) else {
        return format!("(t={millis})");
    };
    if millis <= 0 || millis > now_ms {
        return format!("(t={millis})");
    }
    let delta_secs = (now_ms - millis) / 1000;
    if delta_secs < 60 {
        format!("{delta_secs}s ago")
    } else if delta_secs < 3600 {
        format!("{}m ago", delta_secs / 60)
    } else if delta_secs < 86_400 {
        format!("{}h ago", delta_secs / 3600)
    } else {
        format!("{}d ago", delta_secs / 86_400)
    }
}

// bn-3dkw: `image_paths` brings the arity to 8. The helper only has one caller
// (`run_ask`) and each param is a distinct concern, so bundling them into a
// struct is more ceremony than the site is worth.
// bn-1ikn: `structured_output` is the ninth — set only for `--format=chart` so
// the opencode adapter pulls the final assistant message out of the JSON event
// stream instead of relying on post-hoc narration stripping.
#[allow(clippy::too_many_arguments)]
fn try_generate_answer(
    cfg: &Config,
    root: &Path,
    query: &str,
    assembled: &kb_query::AssembledContext,
    manifest_text: &str,
    template_name: Option<&str>,
    output_path: Option<&str>,
    image_paths: &[PathBuf],
    structured_output: bool,
) -> Result<(kb_query::ArtifactResult, kb_llm::ProvenanceRecord)> {
    let adapter = create_ask_adapter(cfg, root)?;

    let llm_request = kb_llm::AnswerQuestionRequest {
        question: query.to_string(),
        context: vec![assembled.text.clone()],
        format: Some(manifest_text.to_string()),
        template_name: template_name.map(str::to_string),
        output_path: output_path.map(str::to_string),
        image_paths: image_paths.to_vec(),
        structured_output,
        conversation: String::new(),
    };

    let (llm_response, provenance) = adapter
        .answer_question(llm_request)
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    let citation_manifest = kb_query::build_citation_manifest(assembled);
    let result =
        kb_query::postprocess_answer(&llm_response.answer, &citation_manifest, assembled);

    Ok((result, provenance))
}

/// Render the `verified: N/M quotes found in sources` footer (bn-166d).
///
/// Always emits the count so readers can see the verifier ran, even
/// when every quote checked out. When some quotes failed verification
/// it lists each failure with its src-id and a truncated copy of the
/// quote for at-a-glance audit.
fn render_quote_verification_footer(verifications: &[kb_core::QuoteVerification]) -> String {
    let total = verifications.len();
    let verified = verifications.iter().filter(|v| v.is_verified()).count();
    let failures: Vec<&kb_core::QuoteVerification> = verifications
        .iter()
        .filter(|v| !v.is_verified())
        .collect();

    let mut out = String::new();
    out.push_str("\n\n---\n\n");
    let _ = std::fmt::Write::write_fmt(
        &mut out,
        format_args!("**verified: {verified}/{total} quotes found in sources**\n"),
    );
    if !failures.is_empty() {
        out.push_str("\nUnverified quotes:\n");
        for v in failures {
            let reason = match v.outcome {
                kb_core::VerificationKind::SourceNotFound => "source not found",
                kb_core::VerificationKind::QuoteNotInSource => "not in source",
                _ => "unverified",
            };
            let preview = if v.quote.chars().count() > 80 {
                let mut s: String = v.quote.chars().take(80).collect();
                s.push('…');
                s
            } else {
                v.quote.clone()
            };
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!("- `{}` — \"{preview}\" ({reason})\n", v.src_id),
            );
        }
    }
    out
}

/// Execute one `kb compile` invocation (dry-run or real) against `root`.
///
/// Split out of the `Command::Compile` dispatch arm so the dry-run path can
/// skip `execute_mutating_command_with_handle` entirely — dry-run never
/// writes, so it must not acquire the root lock or emit a job manifest.
///
/// `log_sink` is `Some` only for real compiles (streamed into the job's
/// on-disk log); dry-run passes `None`.
#[allow(clippy::fn_params_excessive_bools)]
fn run_compile_action(
    compile_root: &Path,
    force: bool,
    dry_run: bool,
    json: bool,
    quiet: bool,
    cli_model: Option<&str>,
    log_sink: Option<std::sync::Arc<dyn kb_compile::pipeline::LogSink>>,
) -> Result<()> {
    // Choose the renderer by the effective stdout context:
    //   TTY + no --json + no --quiet  → indicatif multi-progress bars
    //   everything else               → line-by-line [run]/[ok] on stderr
    // --quiet uses a reduced LineLogReporter that suppresses per-item lines
    // (banner + final render still print).
    let reporter: std::sync::Arc<dyn kb_compile::progress::ProgressReporter> =
        if !json && !quiet && std::io::stdout().is_terminal() {
            std::sync::Arc::new(kb_compile::progress::IndicatifReporter::new())
        } else if quiet {
            std::sync::Arc::new(kb_compile::progress::LineLogReporter::quiet())
        } else {
            std::sync::Arc::new(kb_compile::progress::LineLogReporter::new())
        };

    // Pull `[semantic]` so the embedding-sync pass writes the user-selected
    // backend's vectors. A broken kb.toml would already have been rejected by
    // the validity check at the top of `main`, so falling back to defaults on
    // a load error here is intentional — the typical cause is that the file
    // is missing entirely (legacy or freshly-init'd vault).
    let loaded_config = Config::load_from_root(compile_root, None).ok();
    let semantic_backend = loaded_config
        .as_ref()
        .map(|cfg| cfg.semantic.to_backend_config())
        .unwrap_or_default();
    // bn-2qda: pull `[compile.captions]` so the captions pass picks up the
    // user-configured allow_paths / enabled flag. Defaults match the
    // `Default` impl on `CaptionsConfig`.
    let captions = loaded_config
        .as_ref()
        .map(|cfg| cfg.compile.captions.to_pipeline_config())
        .unwrap_or_default();

    let options = kb_compile::pipeline::CompileOptions {
        force,
        dry_run,
        // Progress rendering is now owned by `reporter`; the legacy
        // `progress: bool` is suppressed under --json and --quiet to match
        // the old stderr-suppression semantics for any code path that still
        // looks at the flag.
        progress: !json && !quiet,
        log_sink,
        reporter: Some(reporter),
        semantic_backend,
        captions,
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

fn resolve_query(query: Option<String>, editor: bool) -> Result<String> {
    // bn-1dar: `--editor`/`-e` always opens $VISUAL/$EDITOR/vi on a tempfile,
    // regardless of whether a query arg, stdin, or TTY is present. This is
    // the dedicated multi-line authoring path; the reedline quick-path
    // (bn-3iq4, bn-ozj5) stays the default when `--editor` is not set.
    if editor {
        return read_from_editor();
    }
    // bn-3iq4: when invoked with no query arg on a TTY, open an interactive
    // multi-line readline editor. Piped stdin and explicit `-` keep the
    // original read-to-EOF behavior.
    match query {
        // Non-empty query arg — fast path. (run_ask still enforces the
        // whitespace-only bail via ValidationError, so we don't duplicate
        // it here.)
        Some(q) if q != "-" => Ok(q),
        // Explicit `-` — always read stdin, regardless of TTY.
        Some(_) => read_stdin_to_end(),
        None => {
            if std::io::stdin().is_terminal() {
                read_interactive_multiline()
            } else {
                read_stdin_to_end()
            }
        }
    }
}

/// bn-1dar: compose the question in `$VISUAL`/`$EDITOR` (falling back to
/// `vi`) on a tempfile. Lines starting with `#` are stripped (git-commit
/// style). Empty content — both at the "user saved blank" and "user wrote
/// only comments" levels — is a `ValidationError` so it does not leave a
/// failed-job manifest behind (bn-1jx).
fn read_from_editor() -> Result<String> {
    // $VISUAL wins over $EDITOR, per longstanding Unix convention.
    let editor = std::env::var("VISUAL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("EDITOR").ok().filter(|s| !s.trim().is_empty()))
        .unwrap_or_else(|| "vi".to_string());

    let tmp = tempfile::Builder::new()
        .prefix("kb-ask-")
        .suffix(".md")
        .tempfile()
        .context("create editor tempfile")?;

    let header = "\
# Write your question below. Lines starting with # are ignored.
# Save and close to submit; close with empty content to cancel.

";
    fs::write(tmp.path(), header).context("seed editor tempfile")?;

    // Split on whitespace so `EDITOR=\"vim -u NONE\"` and similar work.
    let mut parts = editor.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty $EDITOR/$VISUAL value"))?;
    let status = std::process::Command::new(program)
        .args(parts)
        .arg(tmp.path())
        .status()
        .with_context(|| format!("launch editor: {editor}"))?;
    if !status.success() {
        bail!("editor exited with non-zero status: {status}");
    }

    let raw = fs::read_to_string(tmp.path()).context("read editor tempfile")?;
    // Strip git-style comment lines, then trim. `#` only counts as a comment
    // at the start of a line (after optional whitespace).
    let body = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let body = body.trim().to_string();

    if body.is_empty() {
        return Err(ValidationError::new("ask: question cannot be empty").into());
    }

    // RAII drops `tmp` here, unlinking the tempfile on success.
    Ok(body)
}

fn read_stdin_to_end() -> Result<String> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let trimmed = buf.trim().to_string();
    if trimmed.is_empty() {
        bail!("no question provided (pass as argument or pipe via stdin)");
    }
    Ok(trimmed)
}

/// Interactive multi-line readline editor for `kb ask` with no query arg
/// on a TTY (bn-3iq4, bn-ozj5). Enter inserts a newline, Ctrl-D submits,
/// Ctrl-C aborts. Arrow-up/down move the cursor within the current buffer
/// — there is no cross-invocation history, so up/down are bound to plain
/// line movement (not history navigation).
///
/// Any error here bubbles up out of `resolve_query` BEFORE
/// `execute_mutating_command` is invoked, so the empty/aborted cases
/// never leave a failed-job manifest behind (bn-1jx semantics).
fn read_interactive_multiline() -> Result<String> {
    use reedline::{
        default_emacs_keybindings, EditCommand, Emacs, KeyCode, KeyModifiers, Prompt,
        PromptEditMode, PromptHistorySearch, Reedline, ReedlineEvent, Signal,
    };
    use std::borrow::Cow;

    // Reedline's DefaultPrompt renders "<segment>〉". We want a plain `>>> `
    // prompt for the first line and `... ` as the continuation indicator —
    // standard REPL conventions that are also familiar to Python users.
    struct AskPrompt;
    impl Prompt for AskPrompt {
        fn render_prompt_left(&self) -> Cow<'_, str> {
            Cow::Borrowed("")
        }
        fn render_prompt_right(&self) -> Cow<'_, str> {
            Cow::Borrowed("")
        }
        fn render_prompt_indicator(&self, _: PromptEditMode) -> Cow<'_, str> {
            Cow::Borrowed(">>> ")
        }
        fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
            Cow::Borrowed("... ")
        }
        fn render_prompt_history_search_indicator(
            &self,
            _: PromptHistorySearch,
        ) -> Cow<'_, str> {
            Cow::Borrowed("(search) ")
        }
    }

    eprintln!("Enter your question (multi-line; Ctrl-D to submit, Ctrl-C to abort):");

    let mut keybindings = default_emacs_keybindings();
    // Enter inserts a newline instead of submitting. Reedline's default is
    // `ReedlineEvent::Enter` which submits (or runs the validator, if one is
    // configured); replace it with an explicit InsertNewline edit.
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );
    // Ctrl-D unconditionally submits (reedline's default `CtrlD` only submits
    // on an *empty* line — on a non-empty line it deletes a character). We
    // want "submit whatever you've got" regardless of cursor position.
    keybindings.add_binding(
        KeyModifiers::CONTROL,
        KeyCode::Char('d'),
        ReedlineEvent::Submit,
    );
    // Arrow-up/down stay as cursor movement only. Reedline's default `Up`/
    // `Down` events fall back to history navigation when the cursor is on
    // the first/last line of the buffer; we never want that here (no history
    // is attached anyway, but being explicit is safer).
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Up,
        ReedlineEvent::Edit(vec![EditCommand::MoveLineUp { select: false }]),
    );
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Down,
        ReedlineEvent::Edit(vec![EditCommand::MoveLineDown { select: false }]),
    );

    let edit_mode = Box::new(Emacs::new(keybindings));
    let mut line_editor = Reedline::create().with_edit_mode(edit_mode);
    let prompt = AskPrompt;

    match line_editor.read_line(&prompt) {
        Ok(Signal::Success(text)) => {
            if text.trim().is_empty() {
                bail!("ask: question cannot be empty");
            }
            Ok(text)
        }
        Ok(Signal::CtrlD) => bail!("ask: question cannot be empty"),
        // `Signal` is non_exhaustive (reedline reserves room for future
        // variants). Treat Ctrl-C and anything new alike as a clean abort
        // rather than falling through.
        Ok(Signal::CtrlC | _) => bail!("ask: aborted"),
        Err(e) => Err(anyhow!("readline: {e}")),
    }
}

fn normalize_ask_format(format: &str) -> Result<&str> {
    match format {
        "markdown" => Ok("md"),
        // `figure` is an alias for `chart`; both map to the matplotlib pipeline.
        "figure" => Ok("chart"),
        // `diagram` is an alias for `excalidraw`; both produce one or more
        // `.excalidraw` JSON files alongside `answer.md`.
        "diagram" => Ok("excalidraw"),
        "auto" | "md" | "marp" | "json" | "chart" | "excalidraw" | "png" => Ok(format),
        other => bail!("unsupported ask format: {other}"),
    }
}

fn now_millis() -> Result<u64> {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(u64::try_from(millis)?)
}

/// Generate a terse, collision-free id for a new question under `outputs/questions/`.
///
/// Uses `terseid` with prefix `"q"`. The seed is derived from
/// `{timestamp_ns}|{query}` so repeated asks of the same query at different
/// times produce distinct seeds (and therefore different ids). The exists
/// closure walks `outputs/questions/` so we never collide with an existing
/// question directory.
fn generate_question_id(root: &Path, timestamp_ms: u64, query: &str) -> String {
    let questions_dir = root.join("outputs/questions");
    let item_count = if questions_dir.exists() {
        fs::read_dir(&questions_dir).map_or(0, |rd| rd.filter_map(Result::ok).count())
    } else {
        0
    };
    // Combine the timestamp (in ns for extra entropy across sub-ms asks) with
    // the raw query text to build the seed.
    let ts_nanos = u128::from(timestamp_ms).saturating_mul(1_000_000);
    let seed_base = format!("{ts_nanos}|{query}");
    let generator = terseid::IdGenerator::new(terseid::IdConfig::new("q"));
    generator.generate(
        |nonce| {
            let mut bytes = seed_base.as_bytes().to_vec();
            bytes.extend_from_slice(&nonce.to_le_bytes());
            bytes
        },
        item_count,
        |candidate| questions_dir.join(candidate).exists(),
    )
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

#[derive(Debug, Serialize)]
struct ResolveReport {
    uri: String,
    target: String,
    stable_id: Option<String>,
    current_path: Option<String>,
    title: Option<String>,
    content_hash: Option<String>,
    freshness: String,
    broken: bool,
    broken_reason: Option<String>,
    kind: Option<String>,
}

fn run_resolve(root: &Path, uri: &str, json: bool) -> Result<()> {
    let target = kb_uri_target(uri)?;
    let report = if let Some(reason) = path_outside_root(root, &target) {
        ResolveReport {
            uri: uri.to_string(),
            target,
            stable_id: None,
            current_path: None,
            title: None,
            content_hash: None,
            freshness: "missing".to_string(),
            broken: true,
            broken_reason: Some(format!(
                "target is outside the KB root ({reason}); resolve only accepts kb:// references under {}",
                root.display()
            )),
            kind: None,
        }
    } else {
        match build_inspect_report_for_target(root, &target) {
            Ok(report) => resolve_report_from_inspect(root, uri, &target, &report),
            Err(err) => ResolveReport {
                uri: uri.to_string(),
                target,
                stable_id: None,
                current_path: None,
                title: None,
                content_hash: None,
                freshness: "missing".to_string(),
                broken: true,
                broken_reason: Some(err.to_string()),
                kind: None,
            },
        }
    };

    if json {
        emit_json("resolve", &report)?;
    } else {
        println!("{}", render_resolve_report(&report));
    }
    Ok(())
}

fn kb_uri_target(uri: &str) -> Result<String> {
    let Some(rest) = uri.strip_prefix("kb://") else {
        bail!("resolve: URI must start with kb://");
    };
    let no_fragment = rest.split_once('#').map_or(rest, |(head, _)| head);
    let no_query = no_fragment
        .split_once('?')
        .map_or(no_fragment, |(head, _)| head);
    let target = no_query.trim_start_matches('/').trim();
    if target.is_empty() {
        bail!("resolve: kb:// URI target cannot be empty");
    }
    Ok(target.to_string())
}

fn resolve_report_from_inspect(
    root: &Path,
    uri: &str,
    target: &str,
    report: &InspectReport,
) -> ResolveReport {
    let current_path = report.metadata.file_path.clone();
    let file_path = current_path.as_deref().map(|path| root.join(path));
    let (frontmatter, body) = file_path
        .as_ref()
        .and_then(|path| path.is_file().then_some(path))
        .and_then(|path| kb_core::frontmatter::read_frontmatter(path).ok())
        .map_or_else(
            || (serde_yaml::Mapping::new(), String::new()),
            |(frontmatter, body)| (frontmatter, body),
        );
    let stable_id = frontmatter_string(&frontmatter, "id")
        .or_else(|| frontmatter_string(&frontmatter, "source_document_id"))
        .or_else(|| current_path.as_deref().and_then(stable_id_from_path))
        .or_else(|| stable_id_from_resolved_id(&report.resolved_id));
    let title = frontmatter_string(&frontmatter, "title")
        .or_else(|| frontmatter_string(&frontmatter, "name"))
        .or_else(|| first_markdown_heading(&body))
        .or_else(|| current_path.as_deref().and_then(title_from_path));
    let content_hash = file_path
        .as_ref()
        .and_then(|path| path.is_file().then_some(path))
        .and_then(|path| hash_file(path).ok())
        .map(|hash| hash.to_hex());

    ResolveReport {
        uri: uri.to_string(),
        target: target.to_string(),
        stable_id,
        current_path,
        title,
        content_hash,
        freshness: report.freshness.clone(),
        broken: !report.metadata.exists_on_disk || report.freshness == "missing",
        broken_reason: if report.metadata.exists_on_disk {
            None
        } else {
            Some("resolved target does not exist on disk".to_string())
        },
        kind: Some(report.kind.clone()),
    }
}

fn frontmatter_string(frontmatter: &serde_yaml::Mapping, key: &str) -> Option<String> {
    frontmatter
        .get(serde_yaml::Value::String(key.to_string()))
        .and_then(serde_yaml::Value::as_str)
        .map(str::to_string)
}

fn stable_id_from_path(path: &str) -> Option<String> {
    let stem = Path::new(path).file_stem()?.to_str()?;
    if let Some(rest) = stem.strip_prefix("src-") {
        let suffix = rest.split_once('-').map_or(rest, |(id, _)| id);
        return Some(format!("src-{suffix}"));
    }
    if let Some(rest) = stem.strip_prefix("q-") {
        let suffix = rest.split_once('-').map_or(rest, |(id, _)| id);
        return Some(format!("q-{suffix}"));
    }
    if path.starts_with("wiki/concepts/") && !stem.is_empty() {
        return Some(stem.to_string());
    }
    None
}

fn stable_id_from_resolved_id(id: &str) -> Option<String> {
    if is_source_id(id) || id.starts_with("q-") || id.starts_with("build:") {
        Some(id.to_string())
    } else {
        None
    }
}

fn first_markdown_heading(body: &str) -> Option<String> {
    body.lines()
        .find_map(|line| line.trim_start().strip_prefix("# "))
        .map(str::trim)
        .filter(|heading| !heading.is_empty())
        .map(str::to_string)
}

fn title_from_path(path: &str) -> Option<String> {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(str::to_string)
}

fn render_resolve_report(report: &ResolveReport) -> String {
    [
        format!("uri: {}", report.uri),
        format!("target: {}", report.target),
        format!(
            "stable_id: {}",
            report.stable_id.as_deref().unwrap_or("(none)")
        ),
        format!(
            "current_path: {}",
            report.current_path.as_deref().unwrap_or("(none)")
        ),
        format!("title: {}", report.title.as_deref().unwrap_or("(none)")),
        format!(
            "content_hash: {}",
            report.content_hash.as_deref().unwrap_or("(none)")
        ),
        format!("freshness: {}", report.freshness),
        format!("broken: {}", report.broken),
        format!(
            "broken_reason: {}",
            report.broken_reason.as_deref().unwrap_or("(none)")
        ),
        format!("kind: {}", report.kind.as_deref().unwrap_or("(none)")),
    ]
    .join("\n")
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

    let mut report = build_inspect_report_for_target(root, target)?;

    if trace {
        if let Some(graph_data) = &report.graph {
            let _ = graph_data;
            let graph = Graph::load_from(root)?;
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

#[allow(clippy::too_many_lines)]
fn build_inspect_report_for_target(root: &Path, target: &str) -> Result<InspectReport> {
    let graph = Graph::load_from(root)?;
    let hash_state = kb_compile::HashState::load_from_root(root)?;
    let changed_inputs = find_changed_inputs(root, &hash_state)?;
    let jobs = jobs::recent_jobs(root, 1_000)?;

    if let Some(path) = resolve_source_id(root, target) {
        // I1: bare `src-<hex>` identifiers resolve to their wiki/sources page
        // (preferred) or their normalized/<id>/source.md.
        return build_file_report(
            root,
            target,
            &path,
            &jobs,
            &graph,
            &changed_inputs,
            &hash_state,
        );
    }
    if let Some(id) = graph.resolve_node_id(target) {
        return build_graph_inspect_report(root, &graph, target, &id, &changed_inputs, &jobs);
    }
    if let Some(record) = kb_core::load_build_record(root, target)? {
        return build_build_record_report(root, target, &record, &jobs);
    }
    if let Some(job) = jobs.iter().find(|job| job.metadata.id == target) {
        return Ok(build_job_report(target, job));
    }

    let candidate = root.join(target);
    let resolved = if candidate.exists() {
        Some(candidate)
    } else {
        resolve_wiki_missing_md(root, target)
    };
    if let Some(path) = resolved {
        return build_file_report(
            root,
            target,
            &path,
            &jobs,
            &graph,
            &changed_inputs,
            &hash_state,
        );
    }

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
            )
        }
        0 => {
            // bn-1525: last-resort short-prefix resolution. If `target`
            // is a partial id (e.g. `src-a7` for `src-a7x3q9`), delegate
            // to `id_resolve`, which tries src → concept → question and
            // propagates an ambiguity error verbatim when two full ids
            // share the prefix. Ambiguity here ALWAYS wins over "not
            // found" even if a later kind would uniquely match — users
            // who typed a prefix expect "longer prefix, please", not a
            // silent fall-through.
            match id_resolve::resolve(root, target) {
                Ok(resolved) => build_report_for_resolved(
                    root,
                    target,
                    &resolved,
                    &jobs,
                    &graph,
                    &changed_inputs,
                    &hash_state,
                ),
                Err(err) => {
                    let msg = err.to_string();
                    if msg.contains("ambiguous") {
                        // Short-circuit: re-raise the candidate list
                        // instead of the generic "was not found"
                        // message.
                        return Err(err);
                    }
                    bail!(
                        "'{target}' was not found. Try an exact ID, a unique prefix (e.g. 'src-a7' for 'src-a7x3q9'), a unique graph suffix, a build record ID, a job ID, a frontmatter id, or a path under the KB root. Run 'kb compile' first if the dependency graph has not been created yet."
                    );
                }
            }
        }
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
/// by their frontmatter id (e.g. `wiki-source-manual`, `art-a7x`)
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

/// bn-1525: dispatch a resolved [`id_resolve::ResolvedId`] to the right
/// report-building path. Each kind maps to a concrete on-disk file the
/// existing [`build_file_report`] understands.
fn build_report_for_resolved(
    root: &Path,
    target: &str,
    resolved: &id_resolve::ResolvedId,
    jobs: &[JobRun],
    graph: &Graph,
    changed_inputs: &[ChangedInput],
    hash_state: &kb_compile::HashState,
) -> Result<InspectReport> {
    let path = match resolved.kind {
        id_resolve::IdKind::Source => resolve_source_id(root, &resolved.id).unwrap_or_else(|| {
            // If neither the wiki page nor the normalized source.md exists yet
            // (rare — implies an in-progress compile), fall back to the
            // normalized dir itself so the report still surfaces something.
            normalized_dir(root).join(&resolved.id)
        }),
        id_resolve::IdKind::Concept => root.join("wiki/concepts").join(format!("{}.md", resolved.id)),
        id_resolve::IdKind::Question => {
            // Prefer the answer artifact when it exists, otherwise point at
            // the output dir. bn-nlw9: the dir may carry a `-<slug>` suffix
            // so resolve via `resolve_question_dir` rather than join by id.
            let q_dir = id_resolve::resolve_question_dir(root, &resolved.id)
                .unwrap_or_else(|| root.join("outputs/questions").join(&resolved.id));
            let answer = q_dir.join("answer.md");
            if answer.is_file() { answer } else { q_dir }
        }
    };
    build_file_report(root, target, &path, jobs, graph, changed_inputs, hash_state)
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
    // bn-nlw9: page may be at either `wiki/sources/<src>.md` (legacy) or
    // `wiki/sources/<src>-<slug>.md` (current). Use the resolver.
    if let Some(wiki_page) = kb_compile::source_page::resolve_source_page_path(root, target) {
        return Some(wiki_page);
    }
    let normalized = normalized_dir(root).join(target).join("source.md");
    if normalized.exists() {
        return Some(normalized);
    }
    None
}

fn is_source_id(s: &str) -> bool {
    // terseid hashes are lowercase base36. Legacy hex ids are a strict
    // subset of base36, so this accepts both.
    let Some(hash) = s.strip_prefix("src-") else {
        return false;
    };
    !hash.is_empty()
        && hash
            .chars()
            .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
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

    let metadata_path = normalized_dir(root)
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
    } else if id.starts_with("q-") {
        "question".to_string()
    } else if id.starts_with("art-") {
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

/// Peek at the root lock before a lint pass. If another process is mid-compile
/// the on-disk state is a moving target (half-written source pages, not-yet-
/// rendered concept pages), and running lint against that snapshot produces
/// scary false positives that resolve themselves silently once compile
/// finishes.
///
/// * Default mode: warn on stderr and keep going — lint is still useful advice.
/// * `--strict` mode: refuse to run. A strict run that fires off stale warnings
///   defeats the point of `--strict` (it's used in CI to gate merges on a clean
///   tree).
///
/// The peek deliberately does not acquire the lock; lint is read-only and must
/// remain runnable concurrently when no one is writing.
fn guard_lint_against_concurrent_compile(root: &Path, strict: bool) -> Result<()> {
    let Some(holder) = jobs::peek_root_lock(root) else {
        return Ok(());
    };
    if !holder.command.contains("compile") {
        return Ok(());
    }
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
    Ok(())
}

fn run_lint(
    root: &Path,
    json: bool,
    check: Option<&str>,
    strict: bool,
    impute: bool,
) -> Result<()> {
    guard_lint_against_concurrent_compile(root, strict)?;

    let cfg = Config::load_from_root(root, None)?;
    let check = kb_lint::LintRule::parse(check)?;
    let LintRunOptions {
        options,
        contradictions_cfg,
        missing_concepts_cfg,
        rules_args,
    } = build_lint_run_options(&cfg)?;

    let rules = if matches!(check, kb_lint::LintRule::All) {
        lint_rules_for_root(rules_args)
    } else {
        vec![check]
    };

    persist_concept_candidate_review_items(root, &rules, &missing_concepts_cfg);

    let mut total_warnings = 0;
    let mut total_errors = 0;
    let mut reports = Vec::new();

    for rule in &rules {
        let rule = *rule;
        let report = if matches!(rule, kb_lint::LintRule::Contradictions) {
            // Contradictions is the only rule that calls the LLM — it has a
            // dedicated dispatch path that builds the adapter and persists
            // review items in one place. Everything else routes through the
            // pure `run_lint_with_options`.
            run_contradictions_lint(root, &cfg, &contradictions_cfg)?
        } else {
            kb_lint::run_lint_with_options(root, rule, &options)?
        };
        accumulate_lint_report(
            &mut reports,
            &mut total_warnings,
            &mut total_errors,
            report,
            rule,
            json,
        );
    }

    // bn-xt4o: `--impute` runs after the normal lint sweep. It calls a
    // web-search-capable LLM to draft fill-in content for missing
    // concepts and thin concept bodies, then queues each draft as a
    // `ReviewKind::ImputedFix` review item. Never applies changes
    // directly — the user approves via `kb review approve
    // lint:imputed-fix:<...>`.
    if impute {
        let report = run_impute_mode(root, &cfg, &missing_concepts_cfg)?;
        accumulate_lint_report(
            &mut reports,
            &mut total_warnings,
            &mut total_errors,
            report,
            kb_lint::LintRule::MissingConcepts,
            json,
        );
    }

    if json {
        emit_json("lint", LintReportPayload {
            checks: reports,
            checks_ran: rules.len() + usize::from(impute),
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
/// Tally a single `LintReport`, print it to stdout (when not in JSON
/// mode), and push its `LintCheckReport` shape onto the running reports
/// list. Factored out of `run_lint` so the function stays under clippy's
/// line cap.
fn accumulate_lint_report(
    reports: &mut Vec<LintCheckReport>,
    total_warnings: &mut usize,
    total_errors: &mut usize,
    report: kb_lint::LintReport,
    rule: kb_lint::LintRule,
    json: bool,
) {
    let mut warning_count = 0;
    let mut error_count = 0;
    for issue in &report.issues {
        if matches!(issue.severity, kb_lint::IssueSeverity::Warning) {
            warning_count += 1;
        } else {
            error_count += 1;
        }
    }
    *total_warnings += warning_count;
    *total_errors += error_count;
    print_lint_check_report(&report, rule, warning_count, error_count, json);
    reports.push(LintCheckReport {
        check: report.rule,
        issue_count: report.issue_count,
        warning_count,
        error_count,
        issues: report.issues,
    });
}

/// Translate the parsed `[lint.citation_verification]` TOML section
/// into the runtime [`kb_lint::CitationVerificationConfig`]. Lifted out
/// of `run_lint` so the dispatch function stays under clippy's
/// 100-line cap. Returns an error when `level` is misspelled.
fn build_citation_verification_cfg(cfg: &Config) -> Result<kb_lint::CitationVerificationConfig> {
    let level = match cfg.lint.citation_verification.level.as_str() {
        "warn" | "warning" => kb_lint::MissingCitationsLevel::Warn,
        "error" => kb_lint::MissingCitationsLevel::Error,
        other => bail!(
            "invalid lint.citation_verification.level in kb.toml: {other} (expected warn or error)"
        ),
    };
    Ok(kb_lint::CitationVerificationConfig {
        enabled: cfg.lint.citation_verification.enabled,
        level,
        fuzz_per_100_chars: cfg.lint.citation_verification.fuzz_per_100_chars,
    })
}

/// Bundle of toggles `lint_rules_for_root` consults to assemble the
/// default rule list. Bagged into a struct (rather than long argument
/// list) because clippy's `too_many_arguments` lint fires once we add
/// orphan-sources / stale-citations / drift switches. The "many bools"
/// pattern is unavoidable here — each is a TOML knob that genuinely
/// gates a distinct lint pass.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy)]
struct LintRulesForRootArgs {
    require_citations: bool,
    missing_concepts_enabled: bool,
    citation_verification_enabled: bool,
    orphan_sources_enabled: bool,
    stale_citations_enabled: bool,
    drift_enabled: bool,
}

fn lint_rules_for_root(args: LintRulesForRootArgs) -> Vec<kb_lint::LintRule> {
    let mut rules = vec![
        kb_lint::LintRule::BrokenLinks,
        kb_lint::LintRule::Orphans,
        kb_lint::LintRule::StaleRevision,
        kb_lint::LintRule::StaleArtifacts,
    ];
    if args.require_citations {
        rules.push(kb_lint::LintRule::MissingCitations);
    }
    if args.missing_concepts_enabled {
        rules.push(kb_lint::LintRule::MissingConcepts);
    }
    if args.citation_verification_enabled {
        rules.push(kb_lint::LintRule::UnverifiedQuote);
    }
    if args.orphan_sources_enabled {
        rules.push(kb_lint::LintRule::OrphanSources);
    }
    if args.stale_citations_enabled {
        rules.push(kb_lint::LintRule::StaleCitations);
    }
    if args.drift_enabled {
        rules.push(kb_lint::LintRule::Drift);
    }
    // Deliberately exclude LintRule::Contradictions: it's LLM-powered and
    // expensive, so it's opt-in per-command via `kb lint --check contradictions`,
    // never part of the default lint sweep.
    rules
}

/// Translate the parsed `[lint.orphans]` TOML section into the runtime
/// [`kb_lint::OrphanSourcesConfig`] (bn-asr2). Returns an error when
/// `level` is misspelled.
fn build_orphan_sources_cfg(cfg: &Config) -> Result<kb_lint::OrphanSourcesConfig> {
    let level = parse_lint_level("lint.orphans.level", &cfg.lint.orphans.level)?;
    Ok(kb_lint::OrphanSourcesConfig {
        enabled: cfg.lint.orphans.enabled,
        level,
        exempt_globs: cfg.lint.orphans.exempt_globs.clone(),
    })
}

/// Translate the parsed `[lint.stale_citations]` TOML section into the
/// runtime [`kb_lint::StaleCitationsConfig`] (bn-asr2). Returns an
/// error when `level` is misspelled.
fn build_stale_citations_cfg(cfg: &Config) -> Result<kb_lint::StaleCitationsConfig> {
    let level = parse_lint_level("lint.stale_citations.level", &cfg.lint.stale_citations.level)?;
    Ok(kb_lint::StaleCitationsConfig {
        enabled: cfg.lint.stale_citations.enabled,
        level,
    })
}

/// Translate the parsed `[lint.drift]` TOML section into the runtime
/// [`kb_lint::DriftConfig`] (bn-asr2). Returns an error when `level`
/// is misspelled.
fn build_drift_cfg(cfg: &Config) -> Result<kb_lint::DriftConfig> {
    let level = parse_lint_level("lint.drift.level", &cfg.lint.drift.level)?;
    Ok(kb_lint::DriftConfig {
        enabled: cfg.lint.drift.enabled,
        level,
        fuzz_per_100_chars: cfg.lint.drift.fuzz_per_100_chars,
    })
}

fn parse_lint_level(label: &str, raw: &str) -> Result<kb_lint::MissingCitationsLevel> {
    match raw {
        "warn" | "warning" => Ok(kb_lint::MissingCitationsLevel::Warn),
        "error" => Ok(kb_lint::MissingCitationsLevel::Error),
        other => bail!("invalid {label} in kb.toml: {other} (expected warn or error)"),
    }
}

/// Bundle the runtime configs `run_lint` produces from the parsed
/// `kb.toml`. Extracted so `run_lint` itself stays under clippy's
/// 100-line cap.
struct LintRunOptions {
    options: kb_lint::LintOptions,
    contradictions_cfg: kb_lint::ContradictionsConfig,
    missing_concepts_cfg: kb_lint::MissingConceptsConfig,
    rules_args: LintRulesForRootArgs,
}

fn build_lint_run_options(cfg: &Config) -> Result<LintRunOptions> {
    let missing_citations_level = parse_lint_level(
        "lint.missing_citations_level",
        &cfg.lint.missing_citations_level,
    )?;
    let missing_concepts_cfg = kb_lint::MissingConceptsConfig {
        enabled: cfg.lint.missing_concepts.enabled,
        min_sources: cfg.lint.missing_concepts.min_sources,
        min_mentions: cfg.lint.missing_concepts.min_mentions,
    };
    let contradictions_cfg = kb_lint::ContradictionsConfig {
        // bn-3axp: when the user asks for `--check contradictions` we
        // honor the request even if the TOML says `enabled = false`.
        // The config flag only controls whether it runs as part of
        // `LintRule::All` (which currently never includes it). Run-mode
        // override happens at dispatch time.
        enabled: cfg.lint.contradictions.enabled,
        min_sources: cfg.lint.contradictions.min_sources,
    };
    let citation_verification_cfg = build_citation_verification_cfg(cfg)?;
    let orphan_sources_cfg = build_orphan_sources_cfg(cfg)?;
    let stale_citations_cfg = build_stale_citations_cfg(cfg)?;
    let drift_cfg = build_drift_cfg(cfg)?;
    let rules_args = LintRulesForRootArgs {
        require_citations: cfg.lint.require_citations,
        missing_concepts_enabled: missing_concepts_cfg.enabled,
        citation_verification_enabled: citation_verification_cfg.enabled,
        orphan_sources_enabled: orphan_sources_cfg.enabled,
        stale_citations_enabled: stale_citations_cfg.enabled,
        drift_enabled: drift_cfg.enabled,
    };
    let options = kb_lint::LintOptions {
        require_citations: cfg.lint.require_citations,
        missing_citations_level,
        missing_concepts: missing_concepts_cfg.clone(),
        citation_verification: citation_verification_cfg,
        orphan_sources: orphan_sources_cfg,
        stale_citations: stale_citations_cfg,
        drift: drift_cfg,
    };
    Ok(LintRunOptions {
        options,
        contradictions_cfg,
        missing_concepts_cfg,
        rules_args,
    })
}

/// Dispatch the LLM-powered contradictions check.
///
/// Builds the configured adapter, runs the pass, persists any flagged
/// concepts as `ReviewKind::Contradiction` items so `kb review` can see
/// them, and returns a `LintReport` for the normal lint printer.
fn run_contradictions_lint(
    root: &Path,
    cfg: &Config,
    contradictions_cfg: &kb_lint::ContradictionsConfig,
) -> Result<kb_lint::LintReport> {
    // The user explicitly selected this rule. Honor the request even if
    // `[lint.contradictions] enabled = false` — the TOML flag is about
    // inclusion in `LintRule::All` (which we never populate with this
    // rule), not about whether `--check contradictions` is allowed.
    let forced_cfg = kb_lint::ContradictionsConfig {
        enabled: true,
        min_sources: contradictions_cfg.min_sources,
    };

    let adapter = create_ask_adapter(cfg, root)?;

    let items = kb_lint::check_contradictions(root, adapter.as_ref(), &forced_cfg)
        .context("run contradictions check")?;

    for item in &items {
        if let Err(err) = kb_core::save_review_item(root, item) {
            tracing::warn!(
                "failed to persist contradiction review item '{}': {err}",
                item.metadata.id
            );
        }
    }

    // Produce the same shape of issues as `detect_contradictions_issues`
    // but reuse the already-computed review items so we don't double-call
    // the LLM.
    let issues: Vec<kb_lint::LintIssue> = items
        .into_iter()
        .map(|item| kb_lint::LintIssue {
            severity: kb_lint::IssueSeverity::Warning,
            kind: kb_lint::IssueKind::Contradiction,
            referring_page: item
                .affected_pages
                .first()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            line: 0,
            target: item.target_entity_id.clone(),
            message: item.comment.clone(),
            suggested_fix: Some(format!(
                "run 'kb review show {}' to inspect, then approve (acknowledge) or reject (mark intended nuance)",
                item.metadata.id
            )),
        })
        .collect();

    Ok(kb_lint::LintReport {
        rule: kb_lint::LintRule::Contradictions.as_str().to_string(),
        issue_count: issues.len(),
        issues,
    })
}

/// bn-xt4o: dispatch the `--impute` pass. Builds the configured LLM
/// adapter, runs [`kb_lint::run_impute_pass`], persists every successful
/// outcome as a `ReviewKind::ImputedFix` review item (with its payload
/// sidecar), and returns a `LintReport` shaped like the rest of the lint
/// output. Skipped outcomes surface as warning-level issues too so the
/// user knows the pass ran even when every call failed.
fn run_impute_mode(
    root: &Path,
    cfg: &Config,
    missing_concepts_cfg: &kb_lint::MissingConceptsConfig,
) -> Result<kb_lint::LintReport> {
    let adapter = create_ask_adapter(cfg, root)?;
    let impute_cfg = kb_lint::ImputeConfig {
        missing_concepts: missing_concepts_cfg.clone(),
        ..Default::default()
    };
    let outcomes = kb_lint::run_impute_pass(root, adapter.as_ref(), &impute_cfg)
        .context("run impute pass")?;

    for outcome in &outcomes {
        if let kb_lint::ImputeOutcome::Item(imputed) = outcome {
            if let Err(err) = kb_core::save_review_item(root, &imputed.item) {
                tracing::warn!(
                    "failed to persist imputed-fix review item '{}': {err}",
                    imputed.item.metadata.id
                );
                continue;
            }
            if let Err(err) = kb_lint::save_imputed_fix_payload(root, &imputed.item, &imputed.payload) {
                tracing::warn!(
                    "failed to persist imputed-fix payload sidecar for '{}': {err}",
                    imputed.item.metadata.id
                );
            }
        }
    }

    let issues = kb_lint::outcomes_to_lint_issues(&outcomes);
    Ok(kb_lint::LintReport {
        rule: "impute".to_string(),
        issue_count: issues.len(),
        issues,
    })
}

/// Persist concept-candidate review items to `reviews/concept_candidates/`
/// whenever the `missing_concepts` rule is part of this lint pass.
///
/// Mirrors how `kb compile` saves duplicate-concepts review items: lint is
/// the discovery surface, and the queue is what makes findings approvable
/// via `kb review`. Errors are logged as warnings rather than propagated —
/// the lint itself already succeeded, and queue-write failures shouldn't
/// mask the lint output.
fn persist_concept_candidate_review_items(
    root: &Path,
    rules: &[kb_lint::LintRule],
    missing_concepts_cfg: &kb_lint::MissingConceptsConfig,
) {
    if !missing_concepts_cfg.enabled {
        return;
    }
    let should_run = rules
        .iter()
        .any(|r| matches!(r, kb_lint::LintRule::MissingConcepts | kb_lint::LintRule::All));
    if !should_run {
        return;
    }
    match kb_lint::check_missing_concepts(root, missing_concepts_cfg) {
        Ok(items) => {
            for item in &items {
                if let Err(err) = kb_core::save_review_item(root, item) {
                    tracing::warn!(
                        "failed to persist concept-candidate review item '{}': {err}",
                        item.metadata.id
                    );
                }
            }
        }
        Err(err) => {
            tracing::warn!("missing-concepts review-item write skipped: {err}");
        }
    }
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
        // Some lint checks (e.g. concept-candidate) don't have a source
        // location — they fire at the corpus level, not at a particular
        // `file:line`. In that case skip the `:line` suffix entirely so
        // the output doesn't render a stray `:0` that reads like a
        // column number.
        let location = if issue.referring_page.is_empty() && issue.line == 0 {
            String::new()
        } else if issue.line == 0 {
            issue.referring_page.clone()
        } else {
            format!("{}:{}", issue.referring_page, issue.line)
        };
        if location.is_empty() {
            println!("- [{:?}] {}", issue.severity, issue.message);
        } else {
            println!("- [{:?}] {} {}", issue.severity, location, issue.message);
        }
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
    /// Chief-facing alias for the total number of ingested source roots.
    total_sources: usize,
    /// Chief-facing alias for sources whose compiled wiki artifacts are stale.
    stale_sources: usize,
    /// Chief-facing count of generated wiki pages known from disk.
    wiki_page_count: usize,
    /// Most recent successful compile completion/start time, if recorded.
    last_compile_at_millis: Option<u64>,
    /// Count of ingested sources discovered under `normalized/<id>/`.
    normalized_source_count: usize,
    /// Wiki source pages (`wiki-page-*` nodes referencing a source document),
    /// and a by-kind breakdown carried over from the pre-compile graph.
    sources: SourceCounts,
    wiki_pages: usize,
    concepts: usize,
    stale_count: usize,
    /// Snapshot of the semantic embedding store, populated when
    /// `.kb/state/embeddings.db` exists. Both fields are 0 on a fresh kb
    /// that has not been compiled.
    semantic_index: SemanticIndexStatus,
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
    let wiki_page_count = wiki_pages + concepts;

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
    let last_compile_at_millis = all_jobs
        .iter()
        .filter(|job| job.command == "compile" && job.status == JobRunStatus::Succeeded)
        .find_map(|job| job.ended_at_millis.or(Some(job.started_at_millis)));

    // Best-effort: a missing/locked embedding DB doesn't fail status.
    let semantic_index = match kb_query::semantic_index_stats(root) {
        Ok(stats) => SemanticIndexStatus {
            embeddings: stats.embeddings,
            stale: stats.stale,
        },
        Err(err) => {
            tracing::debug!("semantic index status unavailable: {err}");
            SemanticIndexStatus::default()
        }
    };

    Ok(StatusPayload {
        total_sources: normalized_source_count,
        stale_sources: stale_count,
        wiki_page_count,
        last_compile_at_millis,
        normalized_source_count,
        sources: source_counts,
        wiki_pages,
        concepts,
        stale_count,
        semantic_index,
        recent_jobs,
        failed_jobs,
        failed_jobs_total,
        interrupted_jobs,
        interrupted_jobs_total,
        changed_inputs_not_compiled,
        sources_with_missing_origin,
    })
}

/// Semantic index summary surfaced by `kb status`.
#[derive(Debug, Default, Serialize)]
struct SemanticIndexStatus {
    embeddings: usize,
    stale: usize,
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
    let normalized_root = normalized_dir(root);
    if !normalized_root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&normalized_root)
        .with_context(|| format!("read {}", normalized_root.display()))?
    {
        let entry = entry
            .with_context(|| format!("read entry in {}", normalized_root.display()))?;
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
    let dir = normalized_dir(root);
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
    let normalized_root = normalized_dir(root);
    if !normalized_root.exists() {
        return Ok(ChangedInputsScan {
            entries: Vec::new(),
            revision_mismatched: 0,
        });
    }

    let mut entries = Vec::new();
    let mut revision_mismatched = 0;
    for entry in std::fs::read_dir(&normalized_root)
        .with_context(|| format!("read normalized dir {}", normalized_root.display()))?
    {
        let entry = entry
            .with_context(|| format!("read normalized entry in {}", normalized_root.display()))?;
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
    let metadata_path = normalized_dir(root)
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

    let Some(wiki_page) =
        kb_compile::source_page::resolve_source_page_path(root, normalized_id)
    else {
        // No wiki page yet — not a "mismatch"; already caught by hash state
        // in the common never-compiled case. If the wiki page was deleted
        // out from under us, the next compile will regenerate it.
        return Ok(false);
    };
    let Ok(markdown) = std::fs::read_to_string(&wiki_page) else {
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
    println!(
        "semantic index: {} embeddings, {} stale",
        status.semantic_index.embeddings, status.semantic_index.stale
    );
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
    fn days_to_ymd_known_dates() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
        // 2026-04-27 is 20_570 days past the epoch (date used as default
        // recording_date in `kb ingest --audio`).
        assert_eq!(days_to_ymd(20_570), (2026, 4, 27));
        // Pre-epoch sanity check: 1969-12-31.
        assert_eq!(days_to_ymd(-1), (1969, 12, 31));
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

        let normalized = normalized_dir(root);
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
        fs::create_dir_all(normalized_dir(&root).join("src-42"))
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
        let dir = normalized_dir(root).join(id);
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
            vec![normalized_dir(root).join("src-1")],
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
        assert_eq!(changed_paths, vec![normalized_dir(root).join("src-fresh")]);
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
        let jobs_dir = state_dir(&root).join("jobs");
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

    /// bn-28ob: `excalidraw` and `diagram` are accepted spellings; `diagram`
    /// is normalized into `excalidraw` so the rest of the dispatch only
    /// has to handle one canonical name.
    #[test]
    fn normalize_ask_format_accepts_excalidraw_and_diagram_alias() {
        assert_eq!(
            normalize_ask_format("excalidraw").expect("excalidraw"),
            "excalidraw",
        );
        assert_eq!(
            normalize_ask_format("diagram").expect("diagram alias"),
            "excalidraw",
        );
        // Existing surface still works.
        assert_eq!(normalize_ask_format("md").expect("md"), "md");
        assert_eq!(
            normalize_ask_format("figure").expect("figure alias"),
            "chart",
        );
        assert!(normalize_ask_format("not-a-format").is_err());
    }

    /// bn-2cs2: `auto` is now both the default config value and an accepted
    /// `--format` argument so users can pass `--format=auto` explicitly to
    /// document the choice in scripts.
    #[test]
    fn normalize_ask_format_accepts_auto() {
        assert_eq!(normalize_ask_format("auto").expect("auto"), "auto");
        assert_eq!(
            Config::default().ask.artifact_default_format,
            "auto",
            "default format should be auto so the model can multiplex outputs",
        );
    }

    /// bn-2cs2: the auto promoter classifies sandbox files into diagrams
    /// (`.excalidraw[.md]`) and charts (`.png`) and leaves anything else
    /// behind so the post-cleanup `remove_dir_all` discards it.
    #[test]
    fn promote_auto_files_classifies_diagrams_and_charts() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let sandbox = root.join("sandbox");
        let dest = root.join("dest");
        fs::create_dir_all(&sandbox).expect("create sandbox");
        fs::create_dir_all(&dest).expect("create dest");

        fs::write(sandbox.join("system.excalidraw"), b"{\"type\":\"excalidraw\"}")
            .expect("write system.excalidraw");
        fs::write(
            sandbox.join("flow.excalidraw.md"),
            b"---\nexcalidraw-plugin: true\n---\n",
        )
        .expect("write flow.excalidraw.md");
        // PNG bytes (the magic number is enough — we don't validate content).
        fs::write(sandbox.join("trend.png"), b"\x89PNG\r\n\x1a\n").expect("write trend.png");
        // Stray junk that must NOT promote.
        fs::write(sandbox.join("notes.txt"), b"ignored").expect("write notes.txt");
        fs::write(sandbox.join("script.py"), b"# helper").expect("write script.py");
        fs::create_dir_all(sandbox.join("scratch")).expect("scratch sub-dir");

        let promoted = promote_auto_files(&sandbox, &dest).expect("promote");

        let mut diagram_names: Vec<String> = promoted
            .diagrams
            .iter()
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(str::to_string))
            .collect();
        diagram_names.sort();
        assert_eq!(
            diagram_names,
            vec!["flow.excalidraw.md".to_string(), "system.excalidraw".to_string()],
        );
        let chart_names: Vec<String> = promoted
            .charts
            .iter()
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(str::to_string))
            .collect();
        assert_eq!(chart_names, vec!["trend.png".to_string()]);

        assert!(dest.join("system.excalidraw").exists());
        assert!(dest.join("flow.excalidraw.md").exists());
        assert!(dest.join("trend.png").exists());
        // Junk stays in the sandbox so the caller can sweep it.
        assert!(sandbox.join("notes.txt").exists());
        assert!(sandbox.join("script.py").exists());
        assert!(!dest.join("notes.txt").exists());
        assert!(!dest.join("script.py").exists());
    }

    /// bn-2cs2: empty sandbox is success, not failure — auto can produce
    /// zero artifacts when plain text answers the question.
    #[test]
    fn promote_auto_files_returns_empty_when_sandbox_is_empty() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let sandbox = root.join("sandbox");
        let dest = root.join("dest");
        fs::create_dir_all(&sandbox).expect("create sandbox");
        fs::create_dir_all(&dest).expect("create dest");

        let promoted = promote_auto_files(&sandbox, &dest).expect("promote");
        assert!(promoted.diagrams.is_empty());
        assert!(promoted.charts.is_empty());
    }

    /// bn-28ob: `promote_excalidraw_files` should move accepted Excalidraw
    /// files out of the sandbox into the destination dir, drop everything
    /// else, and report what landed.
    #[test]
    fn promote_excalidraw_files_moves_only_excalidraw_extensions() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let sandbox = root.join("sandbox");
        let dest = root.join("dest");
        fs::create_dir_all(&sandbox).expect("create sandbox");
        fs::create_dir_all(&dest).expect("create dest");

        // Two valid diagrams, plus one stray file the model shouldn't have
        // written. The stray file must NOT be promoted.
        fs::write(sandbox.join("system.excalidraw"), b"{\"type\":\"excalidraw\"}")
            .expect("write system.excalidraw");
        fs::write(
            sandbox.join("flow.excalidraw.md"),
            b"---\nexcalidraw-plugin: true\n---\n# excalidraw\n",
        )
        .expect("write flow.excalidraw.md");
        fs::write(sandbox.join("notes.txt"), b"ignored").expect("write notes.txt");
        // Non-file entries (a sub-directory) must also be ignored.
        fs::create_dir_all(sandbox.join("scratch")).expect("create scratch dir");

        let moved = promote_excalidraw_files(&sandbox, &dest).expect("promote");
        let mut moved_names: Vec<String> = moved
            .iter()
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(str::to_string))
            .collect();
        moved_names.sort();
        assert_eq!(
            moved_names,
            vec!["flow.excalidraw.md".to_string(), "system.excalidraw".to_string()],
        );

        assert!(dest.join("system.excalidraw").exists());
        assert!(dest.join("flow.excalidraw.md").exists());
        // Stray file stayed behind in the sandbox so it can be cleaned up.
        assert!(sandbox.join("notes.txt").exists());
        assert!(!dest.join("notes.txt").exists());
    }

    /// bn-28ob: empty sandbox returns an empty Vec rather than erroring,
    /// so the caller can decide what "no files produced" means (the
    /// `run_ask` caller turns it into a clean failure with persisted
    /// artifacts).
    #[test]
    fn promote_excalidraw_files_returns_empty_when_sandbox_is_empty() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let sandbox = root.join("sandbox");
        let dest = root.join("dest");
        fs::create_dir_all(&sandbox).expect("create sandbox");
        fs::create_dir_all(&dest).expect("create dest");

        let moved = promote_excalidraw_files(&sandbox, &dest).expect("promote");
        assert!(moved.is_empty());
    }
}
