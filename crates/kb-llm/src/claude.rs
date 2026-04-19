use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, LlmAdapter, LlmAdapterError,
    MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, RunHealthCheckRequest,
    RunHealthCheckResponse, SummarizeDocumentRequest, SummarizeDocumentResponse,
    parse_extract_concepts_json, parse_merge_concept_candidates_json,
};
use crate::provenance::{ProvenanceRecord, TokenUsage};
use crate::subprocess::{SubprocessError, run_shell_command};
use crate::templates::Template;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeCliConfig {
    pub command: String,
    pub model: Option<String>,
    pub permission_mode: Option<String>,
    pub timeout: Duration,
    pub project_root: Option<PathBuf>,
}

impl Default for ClaudeCliConfig {
    fn default() -> Self {
        Self {
            command: "claude".to_string(),
            model: None,
            permission_mode: Some("default".to_string()),
            timeout: Duration::from_secs(900),
            project_root: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClaudeCliAdapter {
    config: ClaudeCliConfig,
}

impl ClaudeCliAdapter {
    #[must_use]
    pub const fn new(config: ClaudeCliConfig) -> Self {
        Self { config }
    }

    fn render_summary_prompt(
        &self,
        request: &SummarizeDocumentRequest,
    ) -> Result<RenderedPrompt, LlmAdapterError> {
        let template = Template::load("summarize_document.md", self.config.project_root.as_deref())
            .map_err(|err| LlmAdapterError::Other(format!("load summarize template: {err}")))?;

        let mut context = HashMap::new();
        context.insert("title".to_string(), request.title.clone());
        context.insert("body".to_string(), request.body.clone());
        context.insert("max_words".to_string(), request.max_words.to_string());

        let rendered = template
            .render(&context)
            .map_err(|err| LlmAdapterError::Other(format!("render summarize template: {err}")))?;

        Ok(RenderedPrompt {
            template_name: template.name,
            template_hash: template.template_hash,
            content: rendered.content,
            render_hash: rendered.render_hash,
        })
    }

    fn render_extract_concepts_prompt(
        &self,
        request: &ExtractConceptsRequest,
    ) -> Result<RenderedPrompt, LlmAdapterError> {
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

        Ok(RenderedPrompt {
            template_name: template.name,
            template_hash: template.template_hash,
            content: rendered.content,
            render_hash: rendered.render_hash,
        })
    }

    fn summarize_with_rendered_prompt(
        &self,
        rendered: RenderedPrompt,
    ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
        let started_at = unix_time_ms()?;
        let command = self.build_command(&rendered.content);

        let output =
            run_shell_command(&command, self.config.timeout).map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;
        let ended_at = unix_time_ms()?;

        let model = parsed
            .model
            .or_else(|| self.config.model.clone())
            .unwrap_or_else(|| "claude".to_string());

        let provenance = ProvenanceRecord {
            harness: "claude".to_string(),
            harness_version: None,
            model,
            prompt_template_name: rendered.template_name,
            prompt_template_hash: rendered.template_hash,
            prompt_render_hash: rendered.render_hash,
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: parsed.tokens,
            cost_estimate: parsed.cost_estimate,
        };

        Ok((
            SummarizeDocumentResponse {
                summary: parsed.text,
            },
            provenance,
        ))
    }

    fn build_command(&self, prompt: &str) -> String {
        let mut parts = vec![self.config.command.clone(), "-p".to_string()];

        if let Some(model) = &self.config.model {
            parts.push("--model".to_string());
            parts.push(shell_quote(model));
        }

        if let Some(permission_mode) = &self.config.permission_mode {
            parts.push("--permission-mode".to_string());
            parts.push(shell_quote(permission_mode));
        }

        parts.push("--output-format".to_string());
        parts.push("json".to_string());
        parts.push(shell_quote(prompt));

        parts.join(" ")
    }
}

impl LlmAdapter for ClaudeCliAdapter {
    fn summarize_document(
        &self,
        request: SummarizeDocumentRequest,
    ) -> Result<(SummarizeDocumentResponse, ProvenanceRecord), LlmAdapterError> {
        let rendered = self.render_summary_prompt(&request)?;
        self.summarize_with_rendered_prompt(rendered)
    }

    fn extract_concepts(
        &self,
        request: ExtractConceptsRequest,
    ) -> Result<(ExtractConceptsResponse, ProvenanceRecord), LlmAdapterError> {
        let rendered = self.render_extract_concepts_prompt(&request)?;
        let started_at = unix_time_ms()?;
        let command = self.build_command(&rendered.content);

        let output =
            run_shell_command(&command, self.config.timeout).map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;
        let response = parse_extract_concepts_json(&parsed.text)?;
        let ended_at = unix_time_ms()?;

        let model = parsed
            .model
            .or_else(|| self.config.model.clone())
            .unwrap_or_else(|| "claude".to_string());

        let provenance = ProvenanceRecord {
            harness: "claude".to_string(),
            harness_version: None,
            model,
            prompt_template_name: rendered.template_name,
            prompt_template_hash: rendered.template_hash,
            prompt_render_hash: rendered.render_hash,
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: parsed.tokens,
            cost_estimate: parsed.cost_estimate,
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
        let command = self.build_command(&rendered.content);
        let output =
            run_shell_command(&command, self.config.timeout).map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        // Claude wraps responses in JSON ({"result": "...", "usage": {...}}). Parse the
        // envelope first, then parse the assistant text as the merge response payload.
        let parsed = parse_claude_json(&output.stdout)?;
        let response = parse_merge_concept_candidates_json(&parsed.text)?;
        let ended_at = unix_time_ms()?;

        let model = parsed
            .model
            .or_else(|| self.config.model.clone())
            .unwrap_or_else(|| "claude".to_string());

        let provenance = ProvenanceRecord {
            harness: "claude".to_string(),
            harness_version: None,
            model,
            prompt_template_name: template.name,
            prompt_template_hash: template.template_hash,
            prompt_render_hash: rendered.render_hash,
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: parsed.tokens,
            cost_estimate: parsed.cost_estimate,
        };

        Ok((response, provenance))
    }

    fn answer_question(
        &self,
        request: AnswerQuestionRequest,
    ) -> Result<(AnswerQuestionResponse, ProvenanceRecord), LlmAdapterError> {
        let template =
            Template::load("ask.md", self.config.project_root.as_deref()).map_err(|err| {
                LlmAdapterError::Other(format!("load ask template: {err}"))
            })?;

        let mut context = HashMap::new();
        context.insert("query".to_string(), request.question.clone());
        context.insert("sources".to_string(), request.context.join("\n\n"));
        context.insert(
            "citation_manifest".to_string(),
            request.format.unwrap_or_default(),
        );

        let rendered = template
            .render(&context)
            .map_err(|err| LlmAdapterError::Other(format!("render ask template: {err}")))?;

        let prompt = RenderedPrompt {
            template_name: template.name,
            template_hash: template.template_hash,
            content: rendered.content,
            render_hash: rendered.render_hash,
        };

        let started_at = unix_time_ms()?;
        let command = self.build_command(&prompt.content);

        let output =
            run_shell_command(&command, self.config.timeout).map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;
        let ended_at = unix_time_ms()?;

        let model = parsed
            .model
            .or_else(|| self.config.model.clone())
            .unwrap_or_else(|| "claude".to_string());

        let provenance = ProvenanceRecord {
            harness: "claude".to_string(),
            harness_version: None,
            model,
            prompt_template_name: prompt.template_name,
            prompt_template_hash: prompt.template_hash,
            prompt_render_hash: prompt.render_hash,
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: parsed.tokens,
            cost_estimate: parsed.cost_estimate,
        };

        Ok((
            AnswerQuestionResponse {
                answer: parsed.text,
                references: None,
            },
            provenance,
        ))
    }

    fn generate_slides(
        &self,
        _request: GenerateSlidesRequest,
    ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "Claude CLI adapter generate_slides is not implemented yet".to_string(),
        ))
    }

    fn run_health_check(
        &self,
        _request: RunHealthCheckRequest,
    ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
        let prompt = "respond with OK";
        let timeout = Duration::from_secs(10);
        let command = self.build_command(prompt);

        let started_at = unix_time_ms()?;
        let result = run_shell_command(&command, timeout);
        let ended_at = unix_time_ms()?;

        let output = result.map_err(|e| match e {
            SubprocessError::TimedOut { timeout, .. } => LlmAdapterError::Timeout(format!(
                "Claude CLI health check exceeded timeout of {timeout:?}"
            )),
            SubprocessError::Other(err) => {
                let msg = err.to_string();
                if msg.contains("not found") || msg.contains("No such file or directory") {
                    LlmAdapterError::Transport(format!("claude not on PATH: {msg}"))
                } else {
                    LlmAdapterError::Transport(format!("Failed to invoke Claude CLI: {msg}"))
                }
            }
        })?;

        if output.exit_code != Some(0) {
            let stderr = output.stderr.trim();
            let details = if stderr.is_empty() {
                output.stdout.trim()
            } else {
                stderr
            };
            if details.contains("not found") || details.contains("No such file or directory") {
                return Err(LlmAdapterError::Transport(format!(
                    "claude not on PATH: {details}"
                )));
            }

            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;

        let status = if parsed.text.contains("OK") {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        };

        let model = parsed
            .model
            .or_else(|| self.config.model.clone())
            .unwrap_or_else(|| "claude".to_string());

        let provenance = ProvenanceRecord {
            harness: "claude".to_string(),
            harness_version: None,
            model,
            prompt_template_name: "health_check".to_string(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::hash_bytes(prompt.as_bytes()),
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: parsed.tokens,
            cost_estimate: parsed.cost_estimate,
        };

        Ok((
            RunHealthCheckResponse {
                status,
                details: Some(parsed.text),
            },
            provenance,
        ))
    }
}

#[derive(Debug, Clone)]
struct RenderedPrompt {
    template_name: String,
    template_hash: kb_core::Hash,
    content: String,
    render_hash: kb_core::Hash,
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedClaudeResponse {
    text: String,
    model: Option<String>,
    tokens: Option<TokenUsage>,
    cost_estimate: Option<f64>,
}

fn parse_claude_json(stdout: &str) -> Result<ParsedClaudeResponse, LlmAdapterError> {
    let payloads = parse_json_payloads(stdout)?;

    let mut text = None;
    let mut model = None;
    let mut tokens = None;
    let mut cost_estimate = None;

    for payload in payloads {
        if text.is_none() {
            text = extract_text(&payload);
        }
        if model.is_none() {
            model = extract_model(&payload);
        }
        if tokens.is_none() {
            tokens = extract_tokens(&payload);
        }
        if cost_estimate.is_none() {
            cost_estimate = extract_cost(&payload);
        }
    }

    let text = text
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            LlmAdapterError::Parse(
                "Claude JSON output did not contain assistant text content".to_string(),
            )
        })?;

    Ok(ParsedClaudeResponse {
        text,
        model,
        tokens,
        cost_estimate,
    })
}

fn parse_json_payloads(stdout: &str) -> Result<Vec<Value>, LlmAdapterError> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err(LlmAdapterError::Parse(
            "Claude CLI returned empty stdout".to_string(),
        ));
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Ok(match value {
            Value::Array(items) => items,
            other => vec![other],
        });
    }

    let mut payloads = Vec::new();
    for line in trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            payloads.push(value);
        }
    }

    if payloads.is_empty() {
        return Err(LlmAdapterError::Parse(
            "Claude CLI stdout was not valid JSON or JSON lines".to_string(),
        ));
    }

    Ok(payloads)
}

fn extract_text(value: &Value) -> Option<String> {
    string_at_path(value, &["result"])
        .or_else(|| string_at_path(value, &["content"]))
        .or_else(|| string_at_path(value, &["message", "content"]))
        .or_else(|| text_from_content_array(value.get("content")))
        .or_else(|| text_from_content_array(value.pointer("/message/content")))
        .or_else(|| string_at_path(value, &["message", "text"]))
        .or_else(|| string_at_path(value, &["text"]))
}

fn text_from_content_array(value: Option<&Value>) -> Option<String> {
    let Value::Array(items) = value? else {
        return None;
    };

    let text_parts: Vec<&str> = items
        .iter()
        .filter_map(|item| {
            if item.get("type").and_then(Value::as_str) == Some("text") {
                item.get("text").and_then(Value::as_str)
            } else {
                None
            }
        })
        .collect();

    if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    }
}

fn extract_model(value: &Value) -> Option<String> {
    string_at_path(value, &["model"])
        .or_else(|| string_at_path(value, &["message", "model"]))
        .or_else(|| string_at_path(value, &["metadata", "model"]))
}

fn extract_tokens(value: &Value) -> Option<TokenUsage> {
    let usage = value
        .get("usage")
        .or_else(|| value.pointer("/message/usage"))
        .or_else(|| value.pointer("/metadata/usage"))?;

    let prompt_tokens = usage
        .get("input_tokens")
        .or_else(|| usage.get("prompt_tokens"))
        .or_else(|| usage.get("inputTokens"))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())?;

    let completion_tokens = usage
        .get("output_tokens")
        .or_else(|| usage.get("completion_tokens"))
        .or_else(|| usage.get("outputTokens"))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())?;

    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
    })
}

fn extract_cost(value: &Value) -> Option<f64> {
    value
        .get("cost_usd")
        .or_else(|| value.get("cost"))
        .or_else(|| value.pointer("/metadata/cost_usd"))
        .or_else(|| value.pointer("/usage/cost_usd"))
        .and_then(Value::as_f64)
}

fn string_at_path(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    current.as_str().map(ToOwned::to_owned)
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn map_subprocess_error(error: SubprocessError) -> LlmAdapterError {
    match error {
        SubprocessError::TimedOut { timeout, .. } => {
            LlmAdapterError::Timeout(format!("Claude CLI exceeded timeout of {timeout:?}"))
        }
        SubprocessError::Other(err) => {
            let msg = err.to_string();
            if msg.contains("not found") || msg.contains("No such file or directory") {
                LlmAdapterError::Transport(format!("claude not on PATH: {msg}"))
            } else {
                LlmAdapterError::Transport(format!("Failed to invoke Claude CLI: {msg}"))
            }
        }
    }
}

fn classify_nonzero_exit(exit_code: Option<i32>, stderr: &str, stdout: &str) -> LlmAdapterError {
    let details = if stderr.trim().is_empty() {
        stdout.trim().to_string()
    } else {
        stderr.trim().to_string()
    };
    let exit = exit_code.map_or_else(|| "signal".to_string(), |code| code.to_string());
    let lower = details.to_lowercase();

    if lower.contains("auth") || lower.contains("unauthorized") || lower.contains("forbidden") {
        return LlmAdapterError::Auth(format!("Claude CLI exited with {exit}: {details}"));
    }

    if lower.contains("rate limit") || lower.contains("quota") {
        return LlmAdapterError::RateLimit(format!("Claude CLI exited with {exit}: {details}"));
    }

    LlmAdapterError::Other(format!("Claude CLI exited with {exit}: {details}"))
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

    #[test]
    fn parses_single_json_object_with_usage() {
        let parsed = parse_claude_json(
            r#"{
                "result": "short summary",
                "model": "claude-sonnet-4-5",
                "usage": {"input_tokens": 11, "output_tokens": 7},
                "cost_usd": 0.42
            }"#,
        )
        .expect("parse claude json");

        assert_eq!(parsed.text, "short summary");
        assert_eq!(parsed.model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(
            parsed.tokens,
            Some(TokenUsage {
                prompt_tokens: 11,
                completion_tokens: 7,
            })
        );
        assert_eq!(parsed.cost_estimate, Some(0.42));
    }

    #[test]
    fn parses_ndjson_assistant_message_shape() {
        let parsed = parse_claude_json(
            "{\"type\":\"system\",\"message\":\"ignored\"}\n{\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"line one\"},{\"type\":\"text\",\"text\":\"line two\"}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":4}}}",
        )
        .expect("parse ndjson");

        assert_eq!(parsed.text, "line one\nline two");
        assert_eq!(
            parsed.tokens,
            Some(TokenUsage {
                prompt_tokens: 9,
                completion_tokens: 4,
            })
        );
    }

    #[test]
    fn summarize_document_invokes_claude_cli_and_builds_provenance() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        let args_path = tmp.path().join("args.txt");
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\n' \"$@\" > '{}'\nprintf '%s' '{}'\n",
                args_path.display(),
                r#"{"result":"summary from claude","model":"claude-haiku","usage":{"input_tokens":21,"output_tokens":6}}"#
            ),
        )
        .expect("write fake claude script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&script_path).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&script_path, permissions).expect("chmod");
        }

        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: script_path.display().to_string(),
            model: Some("claude-haiku".to_string()),
            permission_mode: Some("default".to_string()),
            timeout: Duration::from_secs(5),
            project_root: None,
        });

        let (response, provenance) = adapter
            .summarize_document(SummarizeDocumentRequest {
                title: "Example Source".to_string(),
                body: "A long source document.".to_string(),
                max_words: 80,
            })
            .expect("summarize document");

        assert_eq!(response.summary, "summary from claude");
        assert_eq!(provenance.harness, "claude");
        assert_eq!(provenance.model, "claude-haiku");
        assert_eq!(
            provenance.tokens,
            Some(TokenUsage {
                prompt_tokens: 21,
                completion_tokens: 6,
            })
        );
        assert_eq!(provenance.prompt_template_name, "summarize_document.md");

        let args = fs::read_to_string(&args_path).expect("read fake claude args");
        assert!(args.contains("-p"));
        assert!(args.contains("--output-format"));
        assert!(args.contains("json"));
        assert!(args.contains("--permission-mode"));
        assert!(args.contains("default"));
        assert!(args.contains("--model"));
        assert!(args.contains("claude-haiku"));
        assert!(args.contains("Example Source"));
        assert!(args.contains("A long source document."));
        assert!(args.contains("80 words"));
    }

    #[test]
    fn extract_concepts_invokes_claude_cli_and_parses_json_payload() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        let args_path = tmp.path().join("args.txt");
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\n' \"$@\" > '{}'\ncat <<'EOF'\n{}\nEOF\n",
                args_path.display(),
                r#"{"result":"{\"concepts\":[{\"name\":\"Borrow checker\",\"aliases\":[\"borrowck\"],\"definition_hint\":\"Rust's reference safety analysis.\",\"source_anchors\":[{\"heading_anchor\":\"ownership\",\"quote\":\"The borrow checker validates references.\"}]}]}","model":"claude-haiku","usage":{"input_tokens":30,"output_tokens":9}}"#
            ),
        )
        .expect("write fake claude script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&script_path).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&script_path, permissions).expect("chmod");
        }

        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: script_path.display().to_string(),
            model: Some("claude-haiku".to_string()),
            permission_mode: Some("default".to_string()),
            timeout: Duration::from_secs(5),
            project_root: None,
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

        let args = fs::read_to_string(&args_path).expect("read fake claude args");
        assert!(args.contains("Ownership Notes"));
        assert!(args.contains("Rust ownership overview"));
        assert!(args.contains('5'));
    }

    #[test]
    fn run_health_check_returns_healthy_when_output_contains_ok() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        fs::write(&script_path, "#!/bin/sh\nprintf '{\"result\":\"OK\"}'")
            .expect("write fake claude script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&script_path).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&script_path, permissions).expect("chmod");
        }

        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: script_path.display().to_string(),
            ..Default::default()
        });

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
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        fs::write(&script_path, "#!/bin/sh\nprintf '{\"result\":\"Error\"}'")
            .expect("write fake claude script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&script_path).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&script_path, permissions).expect("chmod");
        }

        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: script_path.display().to_string(),
            ..Default::default()
        });

        let (response, _provenance) = adapter
            .run_health_check(RunHealthCheckRequest {
                check_details: None,
            })
            .expect("health check");

        assert_eq!(response.status, "degraded");
        assert_eq!(response.details, Some("Error".to_string()));
    }

    #[test]
    fn run_health_check_returns_path_error_when_binary_missing() {
        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: "/nonexistent/claude".to_string(),
            ..Default::default()
        });

        let err = adapter
            .run_health_check(RunHealthCheckRequest {
                check_details: None,
            })
            .expect_err("should fail");

        match err {
            LlmAdapterError::Transport(msg) => {
                assert!(msg.contains("claude not on PATH"));
            }
            other => panic!("expected Transport error with PATH message, got {other:?}"),
        }
    }
}
