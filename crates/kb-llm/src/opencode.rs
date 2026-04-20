use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::json;
use tempfile::TempDir;

use crate::adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, DetectContradictionsRequest,
    DetectContradictionsResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    GenerateConceptBodyRequest, GenerateConceptBodyResponse, GenerateSlidesRequest,
    GenerateSlidesResponse, LlmAdapter, LlmAdapterError, MergeConceptCandidatesRequest,
    MergeConceptCandidatesResponse, RunHealthCheckRequest, RunHealthCheckResponse,
    SummarizeDocumentRequest, SummarizeDocumentResponse, parse_detect_contradictions_json,
    parse_extract_concepts_json, parse_merge_concept_candidates_json,
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

    /// Write a per-call `opencode.json` config into a new RAII temp directory.
    ///
    /// The returned [`TempDir`] must be kept alive until the subprocess finishes;
    /// dropping it (including on panic) removes the directory. This replaces an
    /// earlier manual `remove_dir_all` call that could leak the directory if the
    /// process was killed between config write and cleanup.
    fn write_config(&self) -> Result<(TempDir, PathBuf), LlmAdapterError> {
        let dir = tempfile::Builder::new()
            .prefix("kb-opencode-")
            .tempdir()
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

        let config_path = dir.path().join("opencode.json");
        let json_str = serde_json::to_string_pretty(&config)
            .map_err(|e| LlmAdapterError::Other(format!("serialize opencode config: {e}")))?;
        std::fs::write(&config_path, json_str)
            .map_err(|e| LlmAdapterError::Other(format!("write opencode config: {e}")))?;

        Ok((dir, config_path))
    }

    /// Build the shell command string for `opencode run`.
    fn build_command(&self, config_path: &Path, prompt: &str) -> String {
        self.build_command_with_attachments(config_path, prompt, &[])
    }

    /// Build the shell command string for `opencode run`, appending one `-f`
    /// flag per image attachment. `opencode run -f <path>` is repeatable
    /// (declared as `[array]` in `opencode run --help`), and opencode is
    /// routed to openai/gpt-5.4 which accepts multimodal inputs natively —
    /// so each image path flows through as an attachment the model can see.
    fn build_command_with_attachments(
        &self,
        config_path: &Path,
        prompt: &str,
        image_paths: &[PathBuf],
    ) -> String {
        // OPENCODE_CONFIG env var is set inline before the command (sh -c handles this)
        let mut parts = vec![
            format!(
                "OPENCODE_CONFIG={}",
                shell_quote(&config_path.display().to_string())
            ),
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

        for image in image_paths {
            parts.push("-f".to_string());
            parts.push(shell_quote(&image.display().to_string()));
        }

        parts.push(shell_quote(prompt));
        parts.join(" ")
    }

    fn render_extract_concepts_prompt(
        &self,
        request: &ExtractConceptsRequest,
    ) -> Result<
        (
            crate::templates::RenderedTemplate,
            crate::templates::Template,
        ),
        LlmAdapterError,
    > {
        let template = Template::load("extract_concepts.md", self.config.project_root.as_deref())
            .map_err(|err| {
            LlmAdapterError::Other(format!("load extract concepts template: {err}"))
        })?;

        let mut context = HashMap::new();
        context.insert("title".to_string(), request.title.clone());
        context.insert("body".to_string(), request.body.clone());
        context.insert(
            "summary".to_string(),
            request.summary.clone().unwrap_or_default(),
        );
        context.insert(
            "max_concepts".to_string(),
            request
                .max_concepts
                .map_or_else(|| "no limit".to_string(), |value| value.to_string()),
        );

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render extract concepts template: {err}"))
        })?;

        Ok((rendered, template))
    }

    /// Generate config, invoke opencode, and return the stripped response text.
    fn run_prompt(&self, prompt: &str) -> Result<String, LlmAdapterError> {
        self.run_prompt_with_attachments(prompt, &[])
    }

    /// Like [`run_prompt`](Self::run_prompt) but forwards `image_paths` as
    /// `-f <path>` flags on the `opencode run` invocation.
    fn run_prompt_with_attachments(
        &self,
        prompt: &str,
        image_paths: &[PathBuf],
    ) -> Result<String, LlmAdapterError> {
        // `_temp_dir` is held for the duration of the subprocess; its Drop removes
        // the config directory even if we panic or unwind here.
        let (_temp_dir, config_path) = self.write_config()?;
        let command = self.build_command_with_attachments(&config_path, prompt, image_paths);
        let result = run_shell_command(&command, self.config.timeout);

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
        context.insert("title".to_string(), request.title.clone());
        context.insert("body".to_string(), request.body.clone());
        context.insert("max_words".to_string(), request.max_words.to_string());

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
        request: ExtractConceptsRequest,
    ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
        let (rendered, template) = self.render_extract_concepts_prompt(&request)?;

        let started_at = unix_time_ms()?;
        let raw = self.run_prompt(&rendered.content)?;
        let response = parse_extract_concepts_json(&raw)?;
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

        Ok((response, provenance))
    }

    fn merge_concept_candidates(
        &self,
        request: MergeConceptCandidatesRequest,
    ) -> Result<(MergeConceptCandidatesResponse, ProvenanceRecord), LlmAdapterError> {
        let template = Template::load(
            "merge_concept_candidates.md",
            self.config.project_root.as_deref(),
        )
        .map_err(|err| {
            LlmAdapterError::Other(format!("load merge_concept_candidates template: {err}"))
        })?;

        let candidates_json = serde_json::to_string_pretty(&request.candidates)
            .map_err(|err| LlmAdapterError::Other(format!("serialize candidates: {err}")))?;

        let mut context = HashMap::new();
        context.insert("candidates_json".to_string(), candidates_json);

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render merge_concept_candidates template: {err}"))
        })?;

        let started_at = unix_time_ms()?;
        let raw = self.run_prompt(&rendered.content)?;
        let response = parse_merge_concept_candidates_json(&raw)?;
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

        Ok((response, provenance))
    }

    fn generate_concept_body(
        &self,
        request: GenerateConceptBodyRequest,
    ) -> Result<(GenerateConceptBodyResponse, ProvenanceRecord), LlmAdapterError> {
        let template = Template::load("concept_body.md", self.config.project_root.as_deref())
            .map_err(|err| {
                LlmAdapterError::Other(format!("load concept_body template: {err}"))
            })?;

        let mut context = HashMap::new();
        context.insert("canonical".to_string(), request.canonical_name.clone());
        context.insert(
            "aliases".to_string(),
            format_aliases_for_prompt(&request.aliases),
        );
        context.insert(
            "quotes".to_string(),
            format_quotes_for_prompt(&request.candidate_quotes),
        );

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render concept_body template: {err}"))
        })?;

        let started_at = unix_time_ms()?;
        let raw = self.run_prompt(&rendered.content)?;
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

        Ok((
            GenerateConceptBodyResponse {
                body: strip_plain_text_wrappers(&raw),
            },
            provenance,
        ))
    }

    fn answer_question(
        &self,
        request: AnswerQuestionRequest,
    ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
        let template_name = request.template_name.as_deref().unwrap_or("ask.md");
        let template =
            Template::load(template_name, self.config.project_root.as_deref()).map_err(|err| {
                LlmAdapterError::Other(format!("load ask template: {err}"))
            })?;

        let mut context = HashMap::new();
        context.insert("query".to_string(), request.question.clone());
        context.insert("sources".to_string(), request.context.join("\n\n"));
        context.insert(
            "citation_manifest".to_string(),
            request.format.unwrap_or_default(),
        );
        if let Some(path) = request.output_path.as_ref() {
            context.insert("output_path".to_string(), path.clone());
        }

        let rendered = template
            .render(&context)
            .map_err(|err| LlmAdapterError::Other(format!("render ask template: {err}")))?;

        let started_at = unix_time_ms()?;
        // bn-3dkw: forward image attachments as `-f <path>` flags so the
        // multimodal model (gpt-5.4) can actually see them instead of just the
        // markdown reference in the prompt text.
        let answer = self.run_prompt_with_attachments(&rendered.content, &request.image_paths)?;
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

        Ok((
            AnswerQuestionResponse {
                answer,
                references: None,
            },
            provenance,
        ))
    }

    fn detect_contradictions(
        &self,
        request: DetectContradictionsRequest,
    ) -> Result<(DetectContradictionsResponse, ProvenanceRecord), LlmAdapterError> {
        let template = Template::load(
            "detect_contradictions.md",
            self.config.project_root.as_deref(),
        )
        .map_err(|err| {
            LlmAdapterError::Other(format!("load detect_contradictions template: {err}"))
        })?;

        let mut context = HashMap::new();
        context.insert("concept_name".to_string(), request.concept_name.clone());
        context.insert(
            "quotes".to_string(),
            format_contradiction_quotes_for_prompt(&request.quotes),
        );

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render detect_contradictions template: {err}"))
        })?;

        let started_at = unix_time_ms()?;
        let raw = self.run_prompt(&rendered.content)?;
        let response = parse_detect_contradictions_json(&raw)?;
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

        Ok((response, provenance))
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
        let prompt = "respond with OK";
        let timeout = health_check_timeout(self.config.timeout);

        let (temp_dir, config_path) = self.write_config()?;
        let command = self.build_command(&config_path, prompt);

        let started_at = unix_time_ms()?;
        let result = run_shell_command(&command, timeout);
        let ended_at = unix_time_ms()?;

        // Clean up temp dir regardless of outcome
        let _ = std::fs::remove_dir_all(&temp_dir);

        let output = result.map_err(|e| match e {
            SubprocessError::TimedOut { timeout, .. } => LlmAdapterError::Timeout(format!(
                "opencode health check exceeded timeout of {}s \u{2014} this can mean the harness is slow to start (try again), the runner isn't authenticated, or the model isn't available.",
                timeout.as_secs()
            )),
            SubprocessError::Other(err) => {
                let msg = err.to_string();
                if msg.contains("not found") || msg.contains("No such file or directory") {
                    LlmAdapterError::Transport(format!("opencode not on PATH: {msg}"))
                } else {
                    LlmAdapterError::Transport(format!("failed to invoke opencode: {msg}"))
                }
            }
        })?;

        if output.exit_code != Some(0) {
            let stderr = output.stderr.trim().to_string();
            let details = if stderr.is_empty() {
                output.stdout.trim().to_string()
            } else {
                stderr
            };
            if details.contains("not found") || details.contains("No such file or directory") {
                return Err(LlmAdapterError::Transport(format!(
                    "opencode not on PATH: {details}"
                )));
            }

            return Err(LlmAdapterError::Other(format!(
                "opencode health check exited with error: {details}"
            )));
        }

        let response_text = strip_ansi_header(&output.stdout);

        let status = if response_text.contains("OK") {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        };

        let provenance = ProvenanceRecord {
            harness: "opencode".to_string(),
            harness_version: None,
            model: self.config.model.clone(),
            prompt_template_name: "health_check".to_string(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::hash_bytes(prompt.as_bytes()),
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: None,
            cost_estimate: None,
        };

        Ok((
            RunHealthCheckResponse {
                status,
                details: Some(response_text),
            },
            provenance,
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

/// Format an alias list for the `concept_body.md` prompt.
/// Empty list renders as "(none)" so the template line "Aliases: " stays grammatical.
fn format_aliases_for_prompt(aliases: &[String]) -> String {
    if aliases.is_empty() {
        "(none)".to_string()
    } else {
        aliases.join(", ")
    }
}

/// Format candidate quotes as bullet lines for the `concept_body.md` prompt.
/// Empty list renders as a single "(no quotes available)" line so the prompt
/// still parses cleanly without dangling whitespace.
fn format_quotes_for_prompt(quotes: &[String]) -> String {
    if quotes.is_empty() {
        "- (no quotes available)".to_string()
    } else {
        quotes
            .iter()
            .map(|q| format!("- {}", q.trim()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Format quotes for the `detect_contradictions.md` prompt. Mirrors the Claude
/// adapter's helper — numbered zero-based entries with source labels so the
/// model's `conflicting_quotes: [index, …]` output can be mapped back to the
/// input list.
fn format_contradiction_quotes_for_prompt(
    quotes: &[crate::adapter::ContradictionQuote],
) -> String {
    if quotes.is_empty() {
        return "(no quotes)".to_string();
    }
    quotes
        .iter()
        .enumerate()
        .map(|(idx, q)| {
            format!(
                "[{idx}] source: {source}\n    quote: {text}",
                source = q.source_label.trim(),
                text = q.text.trim(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strip wrapping code fences from a plain-text LLM response. Defensive —
/// the prompt tells the model to return plain text, but models occasionally
/// wrap short bodies in triple backticks anyway.
fn strip_plain_text_wrappers(text: &str) -> String {
    let trimmed = text.trim();
    let without_fence = trimmed.strip_prefix("```").map_or(trimmed, |rest| {
        let after_header = rest.split_once('\n').map_or(rest, |(_, body)| body);
        after_header
            .rsplit_once("```")
            .map_or(after_header, |(body, _)| body)
            .trim()
    });
    without_fence.to_string()
}

/// Clamp the configured runner timeout into `[30s, 60s]` for health checks.
///
/// `kb doctor` shouldn't hang for 900s on a misconfigured harness, but a cold
/// `opencode` invocation can take 10-15s on first call due to mise/TS bootup.
/// The `[30s, 60s]` range keeps doctor snappy while tolerating cold starts.
fn health_check_timeout(configured: Duration) -> Duration {
    configured
        .min(Duration::from_secs(60))
        .max(Duration::from_secs(30))
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
        assert_eq!(
            strip_ansi_header(input),
            "Plain text response without header."
        );
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
        assert!(
            config["$schema"]
                .as_str()
                .expect("$schema should be a string")
                .contains("opencode.ai")
        );
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
                title: "Example Source".to_string(),
                body: "A long source document.".to_string(),
                max_words: 80,
            })
            .expect("summarize document");

        assert_eq!(response.summary, "Summary from opencode.");
        assert_eq!(provenance.harness, "opencode");
        assert_eq!(provenance.model, "openai/gpt-5.4");
        assert_eq!(provenance.prompt_template_name, "summarize_document.md");
        assert!(provenance.tokens.is_none());

        let args = fs::read_to_string(&args_path).expect("read captured args");
        assert!(args.contains("run"), "args should contain 'run': {args}");
        assert!(
            args.contains("--agent"),
            "args should contain '--agent': {args}"
        );
        assert!(
            args.contains("kb"),
            "args should contain agent name: {args}"
        );
        assert!(
            args.contains("Example Source"),
            "args should contain source title: {args}"
        );
        assert!(
            args.contains("A long source document."),
            "args should contain document text: {args}"
        );
        assert!(
            args.contains("80 words"),
            "args should contain max word budget: {args}"
        );
    }

    #[test]
    fn extract_concepts_invokes_opencode_run_and_parses_json_payload() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf '{{\"concepts\":[{{\"name\":\"Borrow checker\",\"aliases\":[\"borrowck\"],\"definition_hint\":\"Rust''s reference safety analysis.\",\"source_anchors\":[{{\"heading_anchor\":\"ownership\",\"quote\":\"The borrow checker validates references.\"}}]}}]}}'\n",
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
            .extract_concepts(ExtractConceptsRequest {
                title: "Ownership Notes".to_string(),
                body: "The borrow checker validates references.".to_string(),
                summary: Some("Rust ownership overview".to_string()),
                max_concepts: Some(5),
            })
            .expect("extract concepts");

        assert_eq!(response.concepts.len(), 1);
        assert_eq!(response.concepts[0].name, "Borrow checker");
        assert_eq!(response.concepts[0].aliases, vec!["borrowck"]);
        assert_eq!(provenance.prompt_template_name, "extract_concepts.md");

        let args = fs::read_to_string(&args_path).expect("read captured args");
        assert!(args.contains("Ownership Notes"));
        assert!(args.contains("Rust ownership overview"));
        assert!(args.contains('5'));
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
    fn build_command_with_attachments_emits_f_flag_per_image() {
        // bn-3dkw: each image must appear as its own `-f <path>` pair so
        // opencode's yargs parser picks them up as an array.
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: "opencode".to_string(),
            agent_name: "kb".to_string(),
            ..Default::default()
        });

        let config_path = PathBuf::from("/tmp/opencode.json");
        let images = [
            PathBuf::from("/abs/a.png"),
            PathBuf::from("/abs/b.jpg"),
        ];
        let cmd = adapter.build_command_with_attachments(&config_path, "prompt", &images);

        let f_flag_count = cmd.matches("-f ").count();
        assert_eq!(f_flag_count, 2, "expected two -f flags in: {cmd}");
        assert!(cmd.contains("/abs/a.png"), "missing a.png in: {cmd}");
        assert!(cmd.contains("/abs/b.jpg"), "missing b.jpg in: {cmd}");
    }

    #[test]
    fn build_command_omits_f_flag_when_no_attachments() {
        let adapter = OpencodeAdapter::new(OpencodeConfig::default());
        let config_path = PathBuf::from("/tmp/opencode.json");
        let cmd = adapter.build_command_with_attachments(&config_path, "prompt", &[]);
        assert!(
            !cmd.contains(" -f "),
            "no -f flag expected when image_paths empty: {cmd}"
        );
    }

    #[test]
    fn answer_question_forwards_image_paths_to_opencode_cli() {
        // bn-3dkw integration: the image paths on AnswerQuestionRequest reach
        // the opencode subprocess invocation as `-f <path>` args. Uses a
        // fake-opencode script that captures argv to disk.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf 'answer from opencode'\n",
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

        let image_one = tmp.path().join("pic1.png");
        let image_two = tmp.path().join("pic2.png");
        fs::write(&image_one, b"\x89PNG1").expect("pic1");
        fs::write(&image_two, b"\x89PNG2").expect("pic2");

        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: script_path.display().to_string(),
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        let (response, _provenance) = adapter
            .answer_question(AnswerQuestionRequest {
                question: "what does the diagram show?".to_string(),
                context: vec!["source body with ![d](pic.png)".to_string()],
                format: Some(String::new()),
                template_name: None,
                output_path: None,
                image_paths: vec![image_one.clone(), image_two.clone()],
            })
            .expect("answer");

        assert_eq!(response.answer, "answer from opencode");

        let args = fs::read_to_string(&args_path).expect("captured args");
        // Each line is one positional argument. Count the `-f` flags and
        // verify both image paths appear in argv.
        let f_lines = args.lines().filter(|line| line == &"-f").count();
        assert_eq!(f_lines, 2, "expected two -f flags in captured args: {args}");
        assert!(
            args.contains(image_one.to_str().expect("utf8")),
            "missing first image in args: {args}"
        );
        assert!(
            args.contains(image_two.to_str().expect("utf8")),
            "missing second image in args: {args}"
        );
    }

    #[test]
    fn answer_question_with_no_images_omits_f_flag() {
        // bn-3dkw zero-cost fast path: image_paths empty => no `-f` in argv.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nprintf 'answer without images'\n",
                args_path.display(),
            ),
        )
        .expect("write fake opencode");

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

        let _ = adapter
            .answer_question(AnswerQuestionRequest {
                question: "text-only?".to_string(),
                context: vec!["plain source text".to_string()],
                format: Some(String::new()),
                template_name: None,
                output_path: None,
                image_paths: Vec::new(),
            })
            .expect("answer");

        let args = fs::read_to_string(&args_path).expect("captured args");
        assert!(
            !args.lines().any(|l| l == "-f"),
            "no -f flag expected for empty image_paths: {args}"
        );
    }

    #[test]
    fn run_health_check_returns_healthy_when_output_contains_ok() {
        let (adapter, _tmp) = test_adapter_with_script("#!/bin/sh\nprintf 'OK'");

        let (response, _provenance) = adapter
            .run_health_check(RunHealthCheckRequest {
                check_details: None,
            })
            .expect("health check");

        assert_eq!(response.status, "healthy");
        assert_eq!(response.details, Some("OK".to_string()));
    }

    #[test]
    fn run_health_check_returns_degraded_when_output_does_not_contain_ok() {
        let (adapter, _tmp) =
            test_adapter_with_script("#!/bin/sh\nprintf 'Error: something went wrong'");

        let (response, _provenance) = adapter
            .run_health_check(RunHealthCheckRequest {
                check_details: None,
            })
            .expect("health check");

        assert_eq!(response.status, "degraded");
        assert_eq!(
            response.details,
            Some("Error: something went wrong".to_string())
        );
    }

    #[test]
    fn run_prompt_returns_timeout_error() {
        let (adapter, _tmp) = test_adapter_with_script("#!/bin/sh\nsleep 100");
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: adapter.config.command,
            timeout: Duration::from_millis(50),
            ..Default::default()
        });

        let err = adapter.run_prompt("hello").expect_err("should time out");
        assert!(
            matches!(err, LlmAdapterError::Timeout(_)),
            "expected Timeout, got {err:?}"
        );
    }

    #[test]
    fn health_check_timeout_clamps_to_thirty_sixty_range() {
        // Below-floor runner timeouts get lifted to the 30s floor so cold-start
        // opencode invocations don't false-negative.
        assert_eq!(
            health_check_timeout(Duration::from_secs(5)),
            Duration::from_secs(30)
        );
        assert_eq!(
            health_check_timeout(Duration::from_millis(50)),
            Duration::from_secs(30)
        );

        // Above-ceiling runner timeouts (typically 900s) get clamped to the 60s
        // ceiling so `kb doctor` stays snappy on a genuinely-broken harness.
        assert_eq!(
            health_check_timeout(Duration::from_secs(900)),
            Duration::from_secs(60)
        );
        assert_eq!(
            health_check_timeout(Duration::from_secs(60)),
            Duration::from_secs(60)
        );

        // Values inside the band pass through unchanged.
        assert_eq!(
            health_check_timeout(Duration::from_secs(45)),
            Duration::from_secs(45)
        );
        assert_eq!(
            health_check_timeout(Duration::from_secs(30)),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn run_health_check_returns_path_error_when_binary_missing() {
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: "/nonexistent/opencode".to_string(),
            ..Default::default()
        });

        let err = adapter
            .run_health_check(RunHealthCheckRequest {
                check_details: None,
            })
            .expect_err("should fail");

        match err {
            LlmAdapterError::Transport(msg) => {
                assert!(msg.contains("opencode not on PATH"));
            }
            other => panic!("expected Transport error with PATH message, got {other:?}"),
        }
    }
}
