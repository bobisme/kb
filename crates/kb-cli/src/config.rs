#![allow(dead_code)]

use std::{collections::HashMap, env, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

/// Top-level CLI configuration parsed from `kb.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub data: DataConfig,
    pub llm: LlmConfig,
    pub compile: CompileConfig,
    pub ask: AskConfig,
    pub lint: LintConfig,
    pub publish: PublishConfig,
    pub lock: LockConfig,
}

impl Config {
    pub const FILE_NAME: &'static str = "kb.toml";

    pub fn load_from_root(root: &Path, cli_model: Option<&str>) -> Result<Self> {
        let path = root.join(Self::FILE_NAME);
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;

        Self::from_toml(&contents)
            .and_then(Self::with_env_overrides)
            .map(|cfg| cfg.with_cli_overrides(cli_model))
            .with_context(|| format!("invalid config file {}", path.display()))
    }

    pub fn from_toml(contents: &str) -> Result<Self> {
        toml::from_str(contents).with_context(|| "invalid kb.toml".to_string())
    }

    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string_pretty(self).context("failed to serialize config to toml")
    }

    #[allow(clippy::too_many_lines)]
    pub fn with_env_overrides(mut self) -> Result<Self> {
        if let Ok(default_runner) = env::var("KB_LLM_DEFAULT_RUNNER") {
            self.llm.default_runner = default_runner;
        }

        if let Ok(default_model) = env::var("KB_LLM_DEFAULT_MODEL") {
            self.llm.default_model = default_model;
        }

        if let Ok(command) = env::var("KB_LLM_RUNNER_OPENCODE_COMMAND") {
            self.llm
                .runners
                .entry("opencode".to_string())
                .or_default()
                .command = command;
        }

        if let Ok(model) = env::var("KB_LLM_RUNNER_OPENCODE_MODEL") {
            self.llm
                .runners
                .entry("opencode".to_string())
                .or_default()
                .model = Some(model);
        }

        if let Ok(tools_read) = env::var("KB_LLM_RUNNER_OPENCODE_TOOLS_READ") {
            self.llm
                .runners
                .entry("opencode".to_string())
                .or_default()
                .tools_read = parse_bool_var("KB_LLM_RUNNER_OPENCODE_TOOLS_READ", &tools_read)?;
        }

        if let Ok(tools_write) = env::var("KB_LLM_RUNNER_OPENCODE_TOOLS_WRITE") {
            self.llm
                .runners
                .entry("opencode".to_string())
                .or_default()
                .tools_write = parse_bool_var("KB_LLM_RUNNER_OPENCODE_TOOLS_WRITE", &tools_write)?;
        }

        if let Ok(tools_edit) = env::var("KB_LLM_RUNNER_OPENCODE_TOOLS_EDIT") {
            self.llm
                .runners
                .entry("opencode".to_string())
                .or_default()
                .tools_edit = parse_bool_var("KB_LLM_RUNNER_OPENCODE_TOOLS_EDIT", &tools_edit)?;
        }

        if let Ok(tools_bash) = env::var("KB_LLM_RUNNER_OPENCODE_TOOLS_BASH") {
            self.llm
                .runners
                .entry("opencode".to_string())
                .or_default()
                .tools_bash = parse_bool_var("KB_LLM_RUNNER_OPENCODE_TOOLS_BASH", &tools_bash)?;
        }

        if let Ok(command) = env::var("KB_LLM_RUNNER_CLAUDE_COMMAND") {
            self.llm
                .runners
                .entry("claude".to_string())
                .or_default()
                .command = command;
        }

        if let Ok(model) = env::var("KB_LLM_RUNNER_CLAUDE_MODEL") {
            self.llm
                .runners
                .entry("claude".to_string())
                .or_default()
                .model = Some(model);
        }

        if let Ok(permission_mode) = env::var("KB_LLM_RUNNER_CLAUDE_PERMISSION_MODE") {
            self.llm
                .runners
                .entry("claude".to_string())
                .or_default()
                .permission_mode = Some(permission_mode);
        }

        if let Ok(token_budget) = env::var("KB_COMPILE_TOKEN_BUDGET") {
            self.compile.token_budget = parse_u32_var("KB_COMPILE_TOKEN_BUDGET", &token_budget)?;
        }

        if let Ok(token_budget) = env::var("KB_ASK_TOKEN_BUDGET") {
            self.ask.token_budget = parse_u32_var("KB_ASK_TOKEN_BUDGET", &token_budget)?;
        }

        if let Ok(timeout_ms) = env::var("KB_LOCK_TIMEOUT_MS") {
            self.lock.timeout_ms = parse_u64_var("KB_LOCK_TIMEOUT_MS", &timeout_ms)?;
        }

        Ok(self)
    }

    pub fn with_cli_overrides(mut self, cli_model: Option<&str>) -> Self {
        if let Some(model) = cli_model {
            self.llm.default_model = model.to_string();
            self.llm
                .runners
                .entry(self.llm.default_runner.clone())
                .or_default()
                .model = Some(model.to_string());
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Directory (relative to KB root) where prompt templates live. The
    /// LLM adapter loads `<root>/<prompt_templates>/<name>.md`, falling
    /// back to bundled defaults when the file is absent.
    #[serde(rename = "prompt_templates_dir")]
    pub prompt_templates: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            prompt_templates: "prompts".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    pub default_runner: String,
    pub default_model: String,
    pub runners: HashMap<String, LlmRunnerConfig>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let mut runners = HashMap::new();
        runners.insert(
            "opencode".to_string(),
            LlmRunnerConfig {
                command: "opencode run".to_string(),
                model: None,
                permission_mode: None,
                tools_read: true,
                tools_write: true,
                tools_edit: true,
                tools_bash: true,
                timeout_seconds: Some(900),
            },
        );
        runners.insert(
            "claude".to_string(),
            LlmRunnerConfig {
                command: "claude".to_string(),
                model: None,
                permission_mode: Some("ask".to_string()),
                tools_read: true,
                tools_write: true,
                tools_edit: true,
                tools_bash: true,
                timeout_seconds: Some(900),
            },
        );

        Self {
            default_runner: "opencode".to_string(),
            default_model: "openai/gpt-5.4".to_string(),
            runners,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct LlmRunnerConfig {
    pub command: String,
    pub model: Option<String>,
    pub permission_mode: Option<String>,
    pub tools_read: bool,
    pub tools_write: bool,
    pub tools_edit: bool,
    pub tools_bash: bool,
    pub timeout_seconds: Option<u64>,
}

impl Default for LlmRunnerConfig {
    fn default() -> Self {
        Self {
            command: "opencode run".to_string(),
            model: None,
            permission_mode: None,
            tools_read: true,
            tools_write: true,
            tools_edit: true,
            tools_bash: true,
            timeout_seconds: Some(900),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct CompileConfig {
    pub token_budget: u32,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            token_budget: 25_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct AskConfig {
    pub token_budget: u32,
    pub artifact_default_format: String,
}

impl Default for AskConfig {
    fn default() -> Self {
        Self {
            token_budget: 20_000,
            artifact_default_format: "markdown".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LintConfig {
    pub require_citations: bool,
    pub missing_citations_level: String,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            require_citations: true,
            missing_citations_level: "warn".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct PublishTargetConfig {
    /// Destination directory (absolute or relative to KB root)
    pub path: String,
    /// Glob-style filter: e.g. "wiki/**" or "outputs/questions/**"
    pub filter: Option<String>,
    /// Output format (currently only "markdown" is supported)
    pub format: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct PublishConfig {
    pub targets: HashMap<String, PublishTargetConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LockConfig {
    pub timeout_ms: u64,
}

impl Default for LockConfig {
    fn default() -> Self {
        Self { timeout_ms: 5_000 }
    }
}

fn parse_bool_var(name: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(anyhow!("invalid boolean for {name}: {value}")),
    }
}

fn parse_u32_var(name: &str, value: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|_| anyhow!("invalid integer for {name}: {value}"))
}

fn parse_u64_var(name: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| anyhow!("invalid integer for {name}: {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn parse_minimal_config_with_defaults() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join(Config::FILE_NAME);
        let toml = r#"
[llm]
default_model = "test-model"
[compile]
token_budget = 12000
"#;
        fs::write(&path, toml)?;

        let cfg = Config::load_from_root(dir.path(), None)?;

        assert_eq!(cfg.llm.default_model, "test-model");
        assert_eq!(cfg.llm.default_runner, Config::default().llm.default_runner);
        assert_eq!(cfg.compile.token_budget, 12_000);
        assert_eq!(
            cfg.ask.artifact_default_format,
            Config::default().ask.artifact_default_format
        );
        assert_eq!(cfg.lock.timeout_ms, Config::default().lock.timeout_ms);
        Ok(())
    }

    #[test]
    fn invalid_config_reports_unknown_key() {
        let bad = "[llm]\nunknown_key = \"value\"\n";
        let message = match Config::from_toml(bad) {
            Ok(_) => "expected error".to_string(),
            Err(err) => err.to_string(),
        };
        assert!(
            message.contains("unknown field `unknown_key`") || message.contains("invalid kb.toml")
        );
    }

    #[test]
    fn serialize_and_round_trip() -> Result<()> {
        let cfg = Config::default();
        let rendered = cfg.to_toml_string()?;
        let parsed = Config::from_toml(&rendered)?;
        assert_eq!(cfg, parsed);
        Ok(())
    }

    #[test]
    fn cli_model_override_is_applied() {
        let cfg = Config::default().with_cli_overrides(Some("cli-model"));
        assert_eq!(cfg.llm.default_model, "cli-model");
        assert_eq!(
            cfg.llm
                .runners
                .get("opencode")
                .and_then(|runner| runner.model.as_deref()),
            Some("cli-model")
        );
    }
}
