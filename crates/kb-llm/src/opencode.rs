use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::json;

use crate::adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter, LlmAdapterError,
    MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, RunHealthCheckRequest,
    RunHealthCheckResponse, SummarizeDocumentRequest, SummarizeDocumentResponse,
};
use crate::provenance::ProvenanceRecord;
use crate::subprocess::{SubprocessError, run_shell_command};
use crate::templates::Template;

/// Configuration for the opencode subprocess backend.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct OpencodeConfig {
    /// Path to the `opencode` binary.
    pub command: String,
    /// Model identifier passed in the generated config (e.g. `"openai/gpt-5.4"`).
    pub model: String,
    /// Agent name used as the JSON config key and `--agent` argument.
    pub agent_name: String,
    /// Optional session ID for stateful sessions.
    pub session_id: Option<String>,
    /// Optional variant flag.
    pub variant: Option<String>,
    /// Allow the agent to read files.
    pub tools_read: bool,
    /// Allow the agent to write files.
    pub tools_write: bool,
    /// Allow the agent to edit files.
    pub tools_edit: bool,
    /// Allow the agent to run bash commands.
    pub tools_bash: bool,
    /// Subprocess timeout.
    pub timeout: Duration,
    /// Project root for template loading.
    pub project_root: Option<PathBuf>,
}

impl Default for OpencodeConfig {
    fn default() -> Self {
        Self {
            command: "opencode".to_string(),
            model: "openai/gpt-5.4".to_string(),
            agent_name: "kb".to_string(),
            session_id: None,
            variant: None,
            tools_read: true,
            tools_write: false,
            tools_edit: false,
            tools_bash: false,
            timeout: Duration::from_secs(900),
            project_root: None,
        }
    }
}

/// LLM adapter that delegates to the `opencode` CLI.
#[derive(Debug)]
pub struct OpencodeAdapter {
    config: OpencodeConfig,
}

impl OpencodeAdapter {
    #[must_use]
    pub const fn new(config: OpencodeConfig) -> Self {
        Self { config }
    }

    /// Write a per-call `opencode.json` config to a unique temp directory.
    /// Returns `(temp_dir, config_file_path)`.
    fn write_config(&self) -> Result<(PathBuf, PathBuf), LlmAdapterError> {
        let dir = std::env::temp_dir().join(format!(
            "kb-opencode-{}-{}",
            std::process::id(),
            unix_time_ms()?
        ));
        std::fs::create_dir_all(&dir)
            .map_err(|e| LlmAdapterError::Other(format!("create config dir: {e}")))?;

        let agent_config = json!({
            "model": self.config.model,
            "tools": {
                "read": self.config.tools_read,
                "write": self.config.tools_write,
                "edit": self.config.tools_edit,
                "bash": self.config.tools_bash,
            }
        });
        let mut agent_map = serde_json::Map::new();
        agent_map.insert(self.config.agent_name.clone(), agent_config);
        let config = json!({
            "$schema": "https://opencode.ai/config.json",
            "agent": agent_map,
        });

        let config_path = dir.join("opencode.json");
        let json_str = serde_json::to_string_pretty(&config)
            .map_err(|e| LlmAdapterError::Other(format!("serialize opencode config: {e}")))?;
        std::fs::write(&config_path, json_str)
            .map_err(|e| LlmAdapterError::Other(format!("write opencode config: {e}")))?;

        Ok((dir, config_path))
    }

    /// Build the shell command string for `opencode run`.
    fn build_command(&self, config_path: &Path, prompt: &str) -> String {
        // OPENCODE_CONFIG env var is set inline before the command (sh -c handles this)
        let mut parts = vec![
            format!("OPENCODE_CONFIG={}", shell_quote(&config_path.display().to_string())),
            self.config.command.clone(),
            "run".to_string(),
            "--agent".to_string(),
            shell_quote(&self.config.agent_name),
        ];

        if let Some(ref variant) = self.config.variant {
            parts.push("--variant".to_string());
            parts.push(shell_quote(variant));
        }

        if let Some(ref session_id) = self.config.session_id {
            parts.push("--session".to_string());
            parts.push(shell_quote(session_id));
        }

        parts.push(shell_quote(prompt));
        parts.join(" ")
    }

    /// Generate config, invoke opencode, and return the stripped response text.
    fn run_prompt(&self, prompt: &str) -> Result<String, LlmAdapterError> {
        let (temp_dir, config_path) = self.write_config()?;
        let command = self.build_command(&config_path, prompt);

        let result = run_shell_command(&command, self.config.timeout);
        // Clean up temp dir regardless of outcome
        let _ = std::fs::remove_dir_all(&temp_dir);

        let output = result.map_err(|e| match e {
            SubprocessError::TimedOut { timeout, .. } => {
                LlmAdapterError::Timeout(format!("opencode exceeded timeout of {timeout:?}"))
            }
            SubprocessError::Other(err) => {
                LlmAdapterError::Transport(format!("failed to invoke opencode: {err}"))
            }
        })?;

        if output.exit_code != Some(0) {
            let stderr = output.stderr.trim().to_string();
            let details = if stderr.is_empty() {
                output.stdout.trim().to_string()
            } else {
                stderr
            };
            return Err(LlmAdapterError::Other(format!(
                "opencode exited with error: {details}"
            )));
        }

        Ok(strip_ansi_header(&output.stdout))
    }
}

impl LlmAdapter for OpencodeAdapter {
    fn summarize_document(
        &self,
        request: SummarizeDocumentRequest,
    ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
        let template = Template::load("summarize_document.md", self.config.project_root.as_deref())
            .map_err(|err| LlmAdapterError::Other(format!("load summarize template: {err}")))?;

        let mut context = HashMap::new();
        context.insert("document_text".to_string(), request.document_text.clone());
        context.insert(
            "style".to_string(),
            request
                .style
                .unwrap_or_else(|| "Provide a concise prose summary.".to_string()),
        );

        let rendered = template
            .render(&context)
            .map_err(|err| LlmAdapterError::Other(format!("render summarize template: {err}")))?;

        let started_at = unix_time_ms()?;
        let summary = self.run_prompt(&rendered.content)?;
        let ended_at = unix_time_ms()?;

        let provenance = ProvenanceRecord {
            harness: "opencode".to_string(),
            harness_version: None,
            model: self.config.model.clone(),
            prompt_template_name: template.name,
            prompt_template_hash: template.template_hash,
            prompt_render_hash: rendered.render_hash,
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: None,
            cost_estimate: None,
        };

        Ok((SummarizeDocumentResponse { summary }, provenance))
    }

    fn extract_concepts(
        &self,
        _request: ExtractConceptsRequest,
    ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "opencode adapter extract_concepts is not implemented yet".to_string(),
        ))
    }

    fn merge_concept_candidates(
        &self,
        _request: MergeConceptCandidatesRequest,
    ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "opencode adapter merge_concept_candidates is not implemented yet".to_string(),
        ))
    }

    fn answer_question(
        &self,
        _request: AnswerQuestionRequest,
    ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "opencode adapter answer_question is not implemented yet".to_string(),
        ))
    }

    fn generate_slides(
        &self,
        _request: GenerateSlidesRequest,
    ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "opencode adapter generate_slides is not implemented yet".to_string(),
        ))
    }

    fn run_health_check(
        &self,
        _request: RunHealthCheckRequest,
    ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "opencode adapter run_health_check is not implemented yet".to_string(),
        ))
    }
}

/// Strip ANSI escape codes and the opencode header from output.
/// Opencode prepends: `\x1b[0m\n> agent · model\n\x1b[0m\n` before the response text.
fn strip_ansi_header(text: &str) -> String {
    let ansi_re = Regex::new(r"\x1b\[[0-9;]*m").expect("valid ANSI regex");
    let clean = ansi_re.replace_all(text, "");
    let lines: Vec<&str> = clean
        .lines()
        .skip_while(|line| {
            let trimmed = line.trim();
            trimmed.is_empty() || trimmed.starts_with('>')
        })
        .collect();
    lines.join("\n").trim().to_string()
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn unix_time_ms() -> Result<u64, LlmAdapterError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| LlmAdapterError::Other(format!("system clock before unix epoch: {err}")))?;
    u64::try_from(duration.as_millis())
        .map_err(|err| LlmAdapterError::Other(format!("timestamp overflow: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn test_adapter_with_script(script: &str) -> (OpencodeAdapter, TempDir) {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        fs::write(&script_path, script).expect("write script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).expect("metadata").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).expect("chmod");
        }

        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: script_path.display().to_string(),
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        (adapter, tmp)
    }

    #[test]
    fn strip_ansi_header_removes_codes_and_header_lines() {
        let input = "\x1b[0m\n> kb · openai/gpt-5.4\n\x1b[0m\nHere is the summary.";
        assert_eq!(strip_ansi_header(input), "Here is the summary.");
    }

    #[test]
    fn strip_ansi_header_passes_through_plain_text() {
        let input = "Plain text response without header.";
        assert_eq!(strip_ansi_header(input), "Plain text response without header.");
    }

    #[test]
    fn strip_ansi_header_trims_trailing_whitespace() {
        let input = "Response text.  \n\n";
        assert_eq!(strip_ansi_header(input), "Response text.");
    }

    #[test]
    fn write_config_generates_correct_json() {
        let adapter = OpencodeAdapter::new(OpencodeConfig::default());
        let (_dir, config_path) = adapter.write_config().expect("write config");
        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&config_path).expect("read config"))
                .expect("parse config");

        assert_eq!(config["agent"]["kb"]["model"], "openai/gpt-5.4");
        assert_eq!(config["agent"]["kb"]["tools"]["read"], true);
        assert_eq!(config["agent"]["kb"]["tools"]["write"], false);
        assert_eq!(config["agent"]["kb"]["tools"]["edit"], false);
        assert_eq!(config["agent"]["kb"]["tools"]["bash"], false);
        assert!(config["$schema"]
            .as_str()
            .expect("$schema should be a string")
            .contains("opencode.ai"));
    }

    #[test]
    fn summarize_document_invokes_opencode_run_and_parses_output() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf 'Summary from opencode.'\n",
                args_path.display(),
            ),
        )
        .expect("write fake opencode script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).expect("metadata").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).expect("chmod");
        }

        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: script_path.display().to_string(),
            model: "openai/gpt-5.4".to_string(),
            agent_name: "kb".to_string(),
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        let (response, provenance) = adapter
            .summarize_document(SummarizeDocumentRequest {
                document_text: "A long source document.".to_string(),
                style: Some("concise bullet points".to_string()),
            })
            .expect("summarize document");

        assert_eq!(response.summary, "Summary from opencode.");
        assert_eq!(provenance.harness, "opencode");
        assert_eq!(provenance.model, "openai/gpt-5.4");
        assert_eq!(provenance.prompt_template_name, "summarize_document.md");
        assert!(provenance.tokens.is_none());

        let args = fs::read_to_string(&args_path).expect("read captured args");
        assert!(args.contains("run"), "args should contain 'run': {args}");
        assert!(args.contains("--agent"), "args should contain '--agent': {args}");
        assert!(args.contains("kb"), "args should contain agent name: {args}");
        assert!(
            args.contains("A long source document."),
            "args should contain document text: {args}"
        );
        assert!(
            args.contains("concise bullet points"),
            "args should contain style: {args}"
        );
    }

    #[test]
    fn nonzero_exit_surfaces_stderr_in_error() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fail-opencode.sh");
        fs::write(
            &script_path,
            "#!/bin/sh\nprintf 'opencode-error-output' >&2\nexit 1\n",
        )
        .expect("write fail script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).expect("metadata").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).expect("chmod");
        }

        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: script_path.display().to_string(),
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        let err = adapter.run_prompt("hello").expect_err("should fail");
        match err {
            LlmAdapterError::Other(msg) => {
                assert!(msg.contains("opencode-error-output"), "error: {msg}");
            }
            other => panic!("expected Other error, got {other:?}"),
        }
    }

    #[test]
    fn variant_and_session_flags_appear_in_command() {
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: "opencode".to_string(),
            agent_name: "kb".to_string(),
            variant: Some("fast".to_string()),
            session_id: Some("sess-123".to_string()),
            ..Default::default()
        });

        let config_path = PathBuf::from("/tmp/opencode.json");
        let cmd = adapter.build_command(&config_path, "test prompt");
        assert!(cmd.contains("--variant"), "missing --variant: {cmd}");
        assert!(cmd.contains("fast"), "missing variant value: {cmd}");
        assert!(cmd.contains("--session"), "missing --session: {cmd}");
        assert!(cmd.contains("sess-123"), "missing session value: {cmd}");
    }

    #[test]
    fn summarize_document_maps_timeouts() {
        let (adapter, _tmp) = test_adapter_with_script("#!/bin/sh\nsleep 100\n");
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: adapter.config.command,
            timeout: Duration::from_millis(100),
            ..Default::default()
        });

        let err = adapter.run_prompt("hello").expect_err("should time out");
        assert!(matches!(err, LlmAdapterError::Timeout(_)), "expected Timeout, got {err:?}");
    }
}
