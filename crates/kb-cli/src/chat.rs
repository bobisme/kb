//! `kb chat` — launch an interactive opencode TUI session with a
//! custom read-only KB agent injected via `OPENCODE_CONFIG`.
//!
//! Pattern mirrors chief's COO runner (see `chief::runner::opencode`):
//! we write a system prompt to disk, generate a one-off opencode config
//! that references that prompt file and names a `kb-chat` primary agent,
//! then spawn `opencode --agent kb-chat` with stdio inherited so the TUI
//! takes over the terminal. On exit we clean up the temp system-prompt
//! file (opencode.json stays — it's harmless and helpful for debugging).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use kb_core::state_dir;
use serde::Serialize;
use serde_json::json;

use crate::ValidationError;
use crate::config::{Config, LlmRunnerConfig};

/// Per-runner inputs the chat session would launch with — emitted by
/// `kb chat --dry-run` so the user (or an agent driving the CLI) can
/// see exactly what `opencode --agent kb-chat` would be invoked with
/// without actually spawning the TUI (bn-1588).
#[derive(Debug, Serialize)]
pub struct ChatDryRunReport {
    pub runner: &'static str,
    pub model: String,
    pub program: String,
    pub args: Vec<String>,
    pub opencode_config_path: PathBuf,
    pub system_prompt_path: PathBuf,
    pub tools: ChatDryRunTools,
}

#[derive(Debug, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ChatDryRunTools {
    pub read: bool,
    pub write: bool,
    pub edit: bool,
    pub bash: bool,
}

/// Entry point wired from `main::run()`. Validates the runner, builds the
/// prompt + opencode config, spawns the TUI (or, with `dry_run`, prints
/// the planned invocation), and cleans up afterwards.
///
/// bn-1588: when `dry_run` is true, the function builds the system
/// prompt and opencode config (so users can inspect them) but skips the
/// `opencode --agent kb-chat` spawn. With `json`, the report rides
/// under the standard CLI JSON envelope.
pub fn run_chat(
    root: &Path,
    model_override: Option<&str>,
    dry_run: bool,
    json: bool,
) -> Result<()> {
    let cfg = Config::load_from_root(root, None)?;

    let Some(opencode) = cfg.llm.runners.get("opencode") else {
        return Err(ValidationError::new(
            "kb chat requires an opencode runner. Add to kb.toml:\n  \
             [llm.runners.opencode]\n  command = \"opencode run\"",
        )
        .into());
    };

    let model = model_override
        .map(str::to_owned)
        .or_else(|| opencode.model.clone())
        .unwrap_or_else(|| cfg.llm.default_model.clone());

    let prompt_path = write_system_prompt(root)?;
    let oc_config_path = generate_opencode_config(root, &prompt_path, &model, opencode)?;

    if dry_run {
        let report = build_dry_run_report(opencode, &model, &oc_config_path, &prompt_path);
        if json {
            crate::emit_json("chat", &report)?;
        } else {
            print_dry_run_report(&report);
        }
        // Same cleanup discipline as the live path.
        let _ = std::fs::remove_file(&prompt_path);
        return Ok(());
    }

    let spawn_result = spawn_interactive(opencode, &oc_config_path, root);

    // Always try to clean up the temp prompt file, even on error.
    let _ = std::fs::remove_file(&prompt_path);

    spawn_result
}

/// Build the planned-invocation report for `--dry-run`. Mirrors the
/// argv splitting that `spawn_interactive` performs (drop trailing
/// `run`, prepend `--agent kb-chat`) so the JSON exactly describes
/// what the live path would have spawned.
fn build_dry_run_report(
    runner: &LlmRunnerConfig,
    model: &str,
    oc_config_path: &Path,
    prompt_path: &Path,
) -> ChatDryRunReport {
    let mut parts = runner.command.split_whitespace();
    let program = parts.next().unwrap_or("opencode").to_string();
    let mut args: Vec<String> = parts
        .filter(|arg| *arg != "run")
        .map(str::to_owned)
        .collect();
    args.push("--agent".to_string());
    args.push("kb-chat".to_string());
    ChatDryRunReport {
        runner: "opencode",
        model: model.to_string(),
        program,
        args,
        opencode_config_path: oc_config_path.to_path_buf(),
        system_prompt_path: prompt_path.to_path_buf(),
        tools: ChatDryRunTools {
            read: runner.tools_read,
            write: false,
            edit: false,
            bash: runner.tools_bash,
        },
    }
}

fn print_dry_run_report(report: &ChatDryRunReport) {
    println!("kb chat (dry run): would launch the opencode TUI without spawning it.");
    println!("  runner:   {}", report.runner);
    println!("  model:    {}", report.model);
    println!(
        "  command:  {} {}",
        report.program,
        report.args.join(" ")
    );
    println!("  config:   {}", report.opencode_config_path.display());
    println!("  prompt:   {}", report.system_prompt_path.display());
    println!(
        "  tools:    read={} write={} edit={} bash={}",
        report.tools.read, report.tools.write, report.tools.edit, report.tools.bash,
    );
}

/// Returns the content of the system prompt injected into the chat agent.
/// Separated so tests can assert its shape.
pub fn system_prompt_content(root: &Path) -> String {
    format!(
        "You are a read-only assistant for a personal knowledge base rooted at {root}.\n\
         \n\
         The KB is organized as:\n\
         - wiki/concepts/<name>.md — distilled concepts with aliases and cross-links\n\
         - wiki/sources/src-<id>.md — summaries of each ingested source document\n\
         - wiki/questions/<slug>.md — previously promoted Q&A pages\n\
         - wiki/index.md — top-level navigation (lists sources, concepts, questions)\n\
         - .kb/normalized/src-<id>/source.md — full original text of each source\n\
         - outputs/questions/q-<id>/ — historical ask artifacts\n\
         \n\
         Use your read and grep/bash tools to answer the user's questions. Start with\n\
         wiki/index.md to see what's available. For a topic, open wiki/concepts/<topic>.md\n\
         first, then drill into linked source pages.\n\
         \n\
         When you cite information, include the path inline, e.g. \"per wiki/sources/src-abc.md\".\n\
         \n\
         Do not modify any files. This is a read-only session.\n",
        root = root.display(),
    )
}

/// Write the system prompt to `.kb/state/chat/system-prompt-<terseid>.md` and
/// return the path. Caller is responsible for removing it on exit.
pub fn write_system_prompt(root: &Path) -> Result<PathBuf> {
    let dir = state_dir(root).join("chat");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;

    let id = generate_prompt_id();
    let path = dir.join(format!("system-prompt-{id}.md"));
    let content = system_prompt_content(root);
    std::fs::write(&path, content)
        .with_context(|| format!("failed to write system prompt to {}", path.display()))?;
    Ok(path)
}

/// Generate a short random-ish id for the prompt filename. We don't need
/// cryptographic strength — the file lives for the duration of one session,
/// in a per-root directory, and we just want to avoid collisions between
/// two `kb chat` invocations started at the same nanosecond.
fn generate_prompt_id() -> String {
    let prefix = "chat".to_string();
    let generator = terseid::IdGenerator::new(terseid::IdConfig::new(prefix));
    let pid = std::process::id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    generator.generate(
        |nonce| format!("{now}|{pid}|chat|{nonce}").into_bytes(),
        0,
        |_| false,
    )
}

/// Generate `.kb/opencode.json` with a single `kb-chat` primary agent that
/// references the system prompt by file path. Returns the config path.
pub fn generate_opencode_config(
    root: &Path,
    system_prompt_path: &Path,
    model: &str,
    runner: &LlmRunnerConfig,
) -> Result<PathBuf> {
    let agent_config = json!({
        "description": "Personal knowledge base assistant",
        "mode": "primary",
        "model": model,
        "prompt": format!("{{file:{}}}", system_prompt_path.display()),
        "tools": {
            "read": runner.tools_read,
            "write": false,
            "edit": false,
            "bash": runner.tools_bash,
        }
    });

    let mut agent_map = serde_json::Map::new();
    agent_map.insert("kb-chat".to_string(), agent_config);

    let config = json!({
        "$schema": "https://opencode.ai/config.json",
        "agent": agent_map,
    });

    let config_path = root.join(".kb").join("opencode.json");
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&config_path, serde_json::to_string_pretty(&config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    Ok(config_path)
}

/// Spawn `opencode --agent kb-chat` with stdio inherited so the TUI takes
/// over the terminal. Splits `runner.command` on whitespace so configs
/// like `command = "opencode run"` work — we drop the trailing `run`
/// positional (opencode defaults to TUI without it) and treat the first
/// token as the binary.
fn spawn_interactive(
    runner: &LlmRunnerConfig,
    config_path: &Path,
    root: &Path,
) -> Result<()> {
    let mut parts = runner.command.split_whitespace();
    let program = parts
        .next()
        .context("opencode runner command is empty in kb.toml")?;

    // Pass through any leading args the user configured *except* a trailing
    // `run` — that forces non-interactive mode in opencode, but `kb chat`
    // specifically wants the TUI. kb.toml's default runner is
    // `"opencode run"` because the non-interactive runner uses `run`, so
    // stripping it here lets the same `[llm.runners.opencode]` block
    // serve both flows without the user needing a second entry.
    let extra_args: Vec<&str> = parts.filter(|arg| *arg != "run").collect();

    let mut cmd = Command::new(program);
    for arg in &extra_args {
        cmd.arg(arg);
    }
    cmd.arg("--agent").arg("kb-chat");
    cmd.env("OPENCODE_CONFIG", config_path);
    cmd.current_dir(root);

    let status = cmd
        .status()
        .with_context(|| format!("failed to launch opencode (`{program}`). Is it installed?"))?;
    if !status.success() {
        bail!("opencode exited with status {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn default_runner() -> LlmRunnerConfig {
        LlmRunnerConfig::default()
    }

    #[test]
    fn generate_opencode_config_writes_expected_shape() {
        let tmp = TempDir::new().expect("create tempdir");
        let runner = default_runner();
        let prompt_path = tmp.path().join("state/chat/system-prompt-xxx.md");

        let config_path = generate_opencode_config(
            tmp.path(),
            &prompt_path,
            "openai/gpt-5.4",
            &runner,
        )
        .expect("generate config");

        assert_eq!(config_path, tmp.path().join(".kb/opencode.json"));
        let raw = std::fs::read_to_string(&config_path).expect("read config");
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse config");

        assert_eq!(
            parsed["$schema"].as_str(),
            Some("https://opencode.ai/config.json")
        );
        let agent = &parsed["agent"]["kb-chat"];
        assert!(agent.is_object(), "expected kb-chat agent to be present");
        assert_eq!(agent["mode"].as_str(), Some("primary"));
        assert_eq!(agent["model"].as_str(), Some("openai/gpt-5.4"));
        assert!(
            agent["prompt"]
                .as_str()
                .expect("prompt field is string")
                .starts_with("{file:"),
            "prompt should use opencode's file: reference form"
        );
        assert_eq!(agent["tools"]["read"].as_bool(), Some(true));
        assert_eq!(agent["tools"]["write"].as_bool(), Some(false));
        assert_eq!(agent["tools"]["edit"].as_bool(), Some(false));
        assert_eq!(agent["tools"]["bash"].as_bool(), Some(true));
    }

    #[test]
    fn generate_opencode_config_threads_model_override() {
        let tmp = TempDir::new().expect("create tempdir");
        let runner = default_runner();
        let prompt_path = tmp.path().join("p.md");

        let config_path = generate_opencode_config(
            tmp.path(),
            &prompt_path,
            "claude-sonnet-4-6",
            &runner,
        )
        .expect("generate config");

        let raw = std::fs::read_to_string(&config_path).expect("read config");
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse config");
        assert_eq!(
            parsed["agent"]["kb-chat"]["model"].as_str(),
            Some("claude-sonnet-4-6")
        );
    }

    #[test]
    fn write_system_prompt_creates_file_and_directory() {
        let tmp = TempDir::new().expect("create tempdir");
        let path = write_system_prompt(tmp.path()).expect("write prompt");

        assert!(path.exists(), "prompt file should exist after write");
        assert!(
            path.starts_with(state_dir(tmp.path()).join("chat")),
            "prompt should live under .kb/state/chat/"
        );
        let content = std::fs::read_to_string(&path).expect("read prompt");
        assert!(
            content.contains("read-only assistant"),
            "prompt should describe its read-only nature"
        );
        assert!(
            content.contains(&tmp.path().display().to_string()),
            "prompt should mention the KB root"
        );
        assert!(
            content.contains("wiki/concepts/"),
            "prompt should describe the KB layout"
        );
    }

    /// bn-1588: --dry-run must NOT spawn opencode. The command should
    /// succeed and the temp system-prompt file must be cleaned up,
    /// matching the live path's cleanup discipline.
    #[test]
    fn run_chat_dry_run_does_not_spawn_and_cleans_up_prompt() {
        let tmp = TempDir::new().expect("create tempdir");

        // A kb.toml with an opencode runner that points at a nonexistent
        // binary — if dry-run accidentally spawned, the test would fail
        // with a "failed to launch opencode" error.
        let toml = r#"
[llm]
default_runner = "opencode"
default_model = "openai/gpt-5.4"

[llm.runners.opencode]
command = "/nonexistent/opencode-binary"
tools_read = true
tools_write = false
tools_edit = false
tools_bash = false
"#;
        std::fs::write(tmp.path().join("kb.toml"), toml).expect("write kb.toml");

        run_chat(tmp.path(), None, true, false).expect("dry-run should succeed");

        // Prompt file must be cleaned up — same invariant the live path
        // promises. The chat dir may still exist; the prompt itself
        // (system-prompt-*.md) must not.
        let chat_dir = state_dir(tmp.path()).join("chat");
        if chat_dir.exists() {
            for entry in std::fs::read_dir(&chat_dir).expect("read chat dir") {
                let entry = entry.expect("entry");
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                assert!(
                    !name_str.starts_with("system-prompt-"),
                    "dry-run must clean up the temp prompt file, found: {name_str}"
                );
            }
        }
    }

    #[test]
    fn build_dry_run_report_strips_trailing_run_and_appends_agent_flag() {
        let runner = LlmRunnerConfig {
            command: "opencode run".to_string(),
            tools_read: true,
            tools_bash: true,
            ..LlmRunnerConfig::default()
        };
        let report = build_dry_run_report(
            &runner,
            "openai/gpt-5.4",
            Path::new("/kb/.kb/opencode.json"),
            Path::new("/kb/.kb/state/chat/system-prompt-x.md"),
        );
        assert_eq!(report.program, "opencode");
        // 'run' is stripped; --agent kb-chat is appended.
        assert_eq!(report.args, vec!["--agent".to_string(), "kb-chat".to_string()]);
        assert_eq!(report.model, "openai/gpt-5.4");
        assert_eq!(report.runner, "opencode");
        // Defense-in-depth: write/edit always false in the chat agent's
        // tool surface, regardless of what the runner config says.
        assert!(!report.tools.write);
        assert!(!report.tools.edit);
        assert!(report.tools.read);
        assert!(report.tools.bash);
    }

    #[test]
    fn run_chat_errors_cleanly_when_opencode_runner_missing() {
        let tmp = TempDir::new().expect("create tempdir");

        // Write a kb.toml that explicitly drops the opencode runner. The
        // `default` impl populates both opencode and claude, so we need a
        // real config file to test the missing-runner branch.
        let toml = r#"
[llm]
default_runner = "claude"
default_model = "claude-sonnet-4-6"

[llm.runners.claude]
command = "claude"
tools_read = true
tools_write = false
tools_edit = false
tools_bash = false
"#;
        std::fs::write(tmp.path().join("kb.toml"), toml).expect("write kb.toml");

        let err =
            run_chat(tmp.path(), None, false, false).expect_err("expected missing-runner error");
        // It must be a ValidationError so bn-1jx's failed-jobs filter
        // leaves it alone.
        let validation = err
            .downcast_ref::<ValidationError>()
            .expect("missing opencode runner should be a ValidationError");
        assert!(
            validation.to_string().contains("opencode runner"),
            "error should mention the missing opencode runner, got: {validation}"
        );
    }
}
