use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::json;
use tempfile::TempDir;

use crate::adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, DetectContradictionsRequest,
    DetectContradictionsResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    GenerateConceptBodyRequest, GenerateConceptBodyResponse, GenerateConceptFromCandidateRequest,
    GenerateConceptFromCandidateResponse, GenerateSlidesRequest, GenerateSlidesResponse,
    ImputeGapRequest, ImputeGapResponse, LlmAdapter, LlmAdapterError,
    MergeConceptCandidatesRequest, MergeConceptCandidatesResponse, RunHealthCheckRequest,
    RunHealthCheckResponse, SummarizeDocumentRequest, SummarizeDocumentResponse,
    parse_detect_contradictions_json, parse_extract_concepts_json,
    parse_generate_concept_from_candidate_json, parse_impute_gap_json,
    parse_merge_concept_candidates_json,
};
use crate::provenance::ProvenanceRecord;
use crate::subprocess::{SubprocessError, run_command_with_stdin};
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
        self.write_config_with_extra_tools(&[])
    }

    /// Like [`write_config`](Self::write_config) but merges extra boolean tool
    /// entries into the agent's `tools` map. bn-xt4o: the `impute_gap` call
    /// turns on opencode's `webfetch` tool here so the model can browse
    /// external sources, without affecting other calls.
    fn write_config_with_extra_tools(
        &self,
        extra_tools: &[(&str, bool)],
    ) -> Result<(TempDir, PathBuf), LlmAdapterError> {
        let dir = tempfile::Builder::new()
            .prefix("kb-opencode-")
            .tempdir()
            .map_err(|e| LlmAdapterError::Other(format!("create config dir: {e}")))?;

        let mut tools = serde_json::Map::new();
        tools.insert("read".to_string(), json!(self.config.tools_read));
        tools.insert("write".to_string(), json!(self.config.tools_write));
        tools.insert("edit".to_string(), json!(self.config.tools_edit));
        tools.insert("bash".to_string(), json!(self.config.tools_bash));
        for (name, enabled) in extra_tools {
            tools.insert((*name).to_string(), json!(*enabled));
        }

        let agent_config = json!({
            "model": self.config.model,
            "tools": tools,
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

    /// Build the argv for `opencode run` (no shell, no quoting).
    ///
    /// The prompt is **not** included here — it travels via the child's
    /// stdin so we sidestep `ARG_MAX` entirely and avoid handing the model's
    /// rendered prompt to `sh` for re-parsing. Caller writes
    /// `prompt.as_bytes()` to stdin via [`run_command_with_stdin`] and
    /// passes `OPENCODE_CONFIG` via [`Self::build_env`].
    fn build_argv(&self) -> Vec<String> {
        self.build_argv_with_attachments(&[], false)
    }

    /// Build the argv for `opencode run`, appending one `-f` flag per image
    /// attachment. `opencode run -f <path>` is repeatable (declared as
    /// `[array]` in `opencode run --help`), and opencode is routed to
    /// openai/gpt-5.4 which accepts multimodal inputs natively — so each
    /// image path flows through as an attachment the model can see.
    ///
    /// bn-1ikn: when `json_events` is `true`, passes `--format json` so
    /// opencode emits its NDJSON event stream instead of the rendered
    /// plain-text transcript. Caller is responsible for parsing the stream
    /// via [`extract_final_answer_from_json_events`].
    ///
    /// bn-18xw: the prompt is no longer included in argv — it's piped to
    /// the child's stdin. `opencode run` reads stdin when no positional
    /// `message` is provided, which means we never pay the shell-quoting
    /// tax or the `ARG_MAX` penalty for large prompts (real example: a
    /// 520 KB prompt that previously had to round-trip through `sh -c`).
    /// `OPENCODE_CONFIG` is delivered via `Command::env` rather than an
    /// inline shell assignment.
    fn build_argv_with_attachments(
        &self,
        image_paths: &[PathBuf],
        json_events: bool,
    ) -> Vec<String> {
        let mut argv = vec![
            self.config.command.clone(),
            "run".to_string(),
            "--agent".to_string(),
            self.config.agent_name.clone(),
        ];

        if let Some(ref variant) = self.config.variant {
            argv.push("--variant".to_string());
            argv.push(variant.clone());
        }

        if let Some(ref session_id) = self.config.session_id {
            argv.push("--session".to_string());
            argv.push(session_id.clone());
        }

        if json_events {
            argv.push("--format".to_string());
            argv.push("json".to_string());
        }

        // No positional prompt — opencode reads it from stdin. The earlier
        // bn-19r7 ordering hazard ("prompt must come before -f") is moot
        // when there's no positional in argv at all.
        for image in image_paths {
            argv.push("-f".to_string());
            argv.push(image.display().to_string());
        }

        argv
    }

    /// Build the env block (extra vars layered onto the inherited env)
    /// for an `opencode run` invocation. Today this just carries
    /// `OPENCODE_CONFIG`; pulled into a helper so the prompt path and the
    /// health-check path stay in sync.
    fn build_env(config_path: &Path) -> Vec<(String, String)> {
        vec![(
            "OPENCODE_CONFIG".to_string(),
            config_path.display().to_string(),
        )]
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

    /// Like [`run_prompt`](Self::run_prompt) but writes the per-call config
    /// with extra boolean tool entries merged in. bn-xt4o: the `impute_gap`
    /// call uses this to flip on `webfetch` so the agent can browse.
    fn run_prompt_with_extra_tools(
        &self,
        prompt: &str,
        extra_tools: &[(&str, bool)],
    ) -> Result<String, LlmAdapterError> {
        let (_temp_dir, config_path) = self.write_config_with_extra_tools(extra_tools)?;
        let argv = self.build_argv();
        let env = Self::build_env(&config_path);
        let result = run_command_with_stdin(
            &argv,
            prompt.as_bytes(),
            &env,
            self.config.timeout,
        );

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

    /// Like [`run_prompt`](Self::run_prompt) but forwards `image_paths` as
    /// `-f <path>` flags on the `opencode run` invocation.
    fn run_prompt_with_attachments(
        &self,
        prompt: &str,
        image_paths: &[PathBuf],
    ) -> Result<String, LlmAdapterError> {
        self.run_prompt_with_attachments_and_format(prompt, image_paths, false)
    }

    /// Core `opencode run` invocation. When `json_events` is `true`, passes
    /// `--format json` and extracts the final assistant message from the
    /// NDJSON event stream (bn-1ikn). Otherwise returns the ANSI-stripped
    /// plain-text transcript as before.
    fn run_prompt_with_attachments_and_format(
        &self,
        prompt: &str,
        image_paths: &[PathBuf],
        json_events: bool,
    ) -> Result<String, LlmAdapterError> {
        // `_temp_dir` is held for the duration of the subprocess; its Drop removes
        // the config directory even if we panic or unwind here.
        let (_temp_dir, config_path) = self.write_config()?;
        let argv = self.build_argv_with_attachments(image_paths, json_events);
        let env = Self::build_env(&config_path);
        let result = run_command_with_stdin(
            &argv,
            prompt.as_bytes(),
            &env,
            self.config.timeout,
        );

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

        if json_events {
            Ok(extract_final_answer_from_json_events(&output.stdout))
        } else {
            Ok(strip_ansi_header(&output.stdout))
        }
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

    fn generate_concept_from_candidate(
        &self,
        request: GenerateConceptFromCandidateRequest,
    ) -> Result<(GenerateConceptFromCandidateResponse, ProvenanceRecord), LlmAdapterError> {
        let template = Template::load(
            "generate_concept_from_candidate.md",
            self.config.project_root.as_deref(),
        )
        .map_err(|err| {
            LlmAdapterError::Other(format!(
                "load generate_concept_from_candidate template: {err}"
            ))
        })?;

        let mut context = HashMap::new();
        context.insert("candidate_name".to_string(), request.candidate_name.clone());
        context.insert(
            "source_snippets".to_string(),
            format_candidate_snippets_for_prompt(&request.source_snippets),
        );
        context.insert(
            "existing_categories".to_string(),
            format_existing_categories_for_prompt(&request.existing_categories),
        );

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!(
                "render generate_concept_from_candidate template: {err}"
            ))
        })?;

        let started_at = unix_time_ms()?;
        let raw = self.run_prompt(&rendered.content)?;
        let response = parse_generate_concept_from_candidate_json(&raw)?;
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

    fn impute_gap(
        &self,
        request: ImputeGapRequest,
    ) -> Result<(ImputeGapResponse, ProvenanceRecord), LlmAdapterError> {
        let template =
            Template::load("impute_gap.md", self.config.project_root.as_deref()).map_err(|err| {
                LlmAdapterError::Other(format!("load impute_gap template: {err}"))
            })?;

        let mut context = HashMap::new();
        context.insert("gap_kind".to_string(), request.gap_kind.prompt_description().to_string());
        context.insert("concept_name".to_string(), request.concept_name.clone());
        context.insert(
            "existing_body".to_string(),
            if request.existing_body.trim().is_empty() {
                "(none)".to_string()
            } else {
                request.existing_body.clone()
            },
        );
        context.insert(
            "local_snippets".to_string(),
            format_candidate_snippets_for_prompt(&request.local_snippets),
        );

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render impute_gap template: {err}"))
        })?;

        let started_at = unix_time_ms()?;
        // bn-xt4o: flip on opencode's `webfetch` tool for this call so the
        // model can browse external sources. All other adapter calls leave
        // it off (costly + unnecessary for KB-internal reasoning).
        let raw = self.run_prompt_with_extra_tools(
            &rendered.content,
            &[("webfetch", true)],
        )?;
        let response = parse_impute_gap_json(&raw)?;
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
        //
        // bn-1ikn: when the caller asks for structured output (chart path),
        // invoke `opencode run --format json` and pull just the final
        // assistant message out of the NDJSON event stream — the per-tool
        // narration opencode streams ahead of the real answer is filtered at
        // the source instead of hoping `strip_tool_narration` can recover it
        // from a run-together plain-text blob.
        let answer = self.run_prompt_with_attachments_and_format(
            &rendered.content,
            &request.image_paths,
            request.structured_output,
        )?;
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
        let argv = self.build_argv();
        let env = Self::build_env(&config_path);

        let started_at = unix_time_ms()?;
        let result = run_command_with_stdin(&argv, prompt.as_bytes(), &env, timeout);
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

/// Parse opencode's `--format json` NDJSON event stream and return the
/// concatenated text of the final assistant message.
///
/// # Event schema (opencode 1.4.3, observed on `openai/gpt-5.4`)
///
/// `opencode run --format json` emits one JSON object per line. Each has a
/// top-level `type` field. The events we care about are `"text"` — these
/// carry `part.text` (the streamed assistant text) and
/// `part.metadata.openai.phase`, which takes values like
/// `"response_reasoning"` for mid-stream tool narration and
/// `"final_answer"` for the actual answer the user should see.
///
/// Intermediate events of other `type`s (`step_start`, `step_finish`,
/// `tool`, …) are ignored. Text events without `phase == "final_answer"`
/// (reasoning narration, tool pre-call chatter) are also skipped.
///
/// # Fallback behaviour
///
/// If the stream has *no* `final_answer` text event (e.g. the runner
/// version is older / the model skipped the metadata, or the output isn't
/// NDJSON at all), this returns the concatenation of every `text` event's
/// `part.text` — or, failing that, the raw stdout with ANSI stripping — so
/// we never silently discard the model's reply. The caller still runs
/// `strip_tool_narration` on the result as a belt-and-suspenders fallback.
fn extract_final_answer_from_json_events(stdout: &str) -> String {
    let mut final_parts: Vec<String> = Vec::new();
    let mut any_text_parts: Vec<String> = Vec::new();
    let mut saw_json = false;

    for raw_line in stdout.lines() {
        let line = raw_line.trim();
        if line.is_empty() || !line.starts_with('{') {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        saw_json = true;

        if value.get("type").and_then(serde_json::Value::as_str) != Some("text") {
            continue;
        }
        let Some(text) = value
            .get("part")
            .and_then(|p| p.get("text"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };

        any_text_parts.push(text.to_string());

        // Walk `part.metadata.*.phase` looking for "final_answer". The
        // provider key ("openai", "anthropic", …) isn't fixed, so we scan
        // all direct children of `metadata` rather than hardcoding a key.
        let is_final = value
            .get("part")
            .and_then(|p| p.get("metadata"))
            .and_then(serde_json::Value::as_object)
            .is_some_and(|meta| {
                meta.values().any(|provider| {
                    provider
                        .get("phase")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|phase| phase == "final_answer")
                })
            });

        if is_final {
            final_parts.push(text.to_string());
        }
    }

    if !final_parts.is_empty() {
        return final_parts.join("").trim().to_string();
    }

    // No `final_answer`-tagged events. If we saw *any* text events, return
    // their concatenation — better than returning empty and losing the
    // model's reply. If the stream wasn't JSON at all, fall back to plain
    // ANSI-stripped stdout so existing error paths still surface output.
    if saw_json && !any_text_parts.is_empty() {
        return any_text_parts.join("").trim().to_string();
    }
    strip_ansi_header(stdout)
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

/// Format candidate snippets as bullet lines for the
/// `generate_concept_from_candidate.md` prompt. Empty list renders as
/// a single "(no snippets available)" line so the prompt still parses cleanly.
fn format_candidate_snippets_for_prompt(
    snippets: &[crate::adapter::CandidateSourceSnippet],
) -> String {
    if snippets.is_empty() {
        return "- (no snippets available)".to_string();
    }
    snippets
        .iter()
        .map(|s| format!("- [{}] {}", s.source_document_id, s.snippet.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format the list of existing category tags for the
/// `generate_concept_from_candidate.md` prompt. Empty list renders as
/// "(none)" so the prompt stays grammatical.
fn format_existing_categories_for_prompt(categories: &[String]) -> String {
    if categories.is_empty() {
        "(none)".to_string()
    } else {
        categories
            .iter()
            .map(|c| format!("- {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
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
        // bn-18xw: prompts now travel via stdin. Inject a `cat > /dev/null`
        // prologue if the caller's script doesn't already drain stdin, so
        // the writer thread isn't left dangling on a SIGPIPE-or-EOF race.
        // Most short scripts in this module don't read stdin; we add it
        // here uniformly to keep callers tidy. If a script needs to capture
        // stdin, it should not use this helper.
        let drained_script = if script.contains("cat ") {
            script.to_string()
        } else {
            // Insert the drain just after the shebang line.
            match script.split_once('\n') {
                Some((shebang, rest)) => {
                    format!("{shebang}\ncat > /dev/null\n{rest}")
                }
                None => format!("#!/bin/sh\ncat > /dev/null\n{script}"),
            }
        };

        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        fs::write(&script_path, drained_script).expect("write script");

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

    /// Build a fake-opencode shell script that captures argv to `args_path`,
    /// captures stdin to `stdin_path`, and prints `response_body` on stdout.
    /// bn-18xw: argv is now lean (no prompt) and the prompt is delivered on
    /// stdin — so tests verify both channels separately.
    fn write_fake_opencode_script(
        script_path: &Path,
        args_path: &Path,
        stdin_path: &Path,
        response_body: &str,
    ) {
        // The fake script must record stdin BEFORE printing on stdout because
        // the parent's writer thread closes stdin on EOF — `cat` blocks until
        // we close the handle, which the new run_command_with_stdin does
        // promptly. Using `cat > stdin.txt` is safe.
        // We use a heredoc-like single-quoted body so the response body can
        // contain backslash escapes without shell expansion mangling them.
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\ncat > '{stdin}'\nprintf '%s' '{body}'\n",
            args = args_path.display(),
            stdin = stdin_path.display(),
            body = response_body.replace('\'', "'\\''"),
        );
        fs::write(script_path, script).expect("write fake opencode script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(script_path).expect("metadata").permissions();
            perms.set_mode(0o755);
            fs::set_permissions(script_path, perms).expect("chmod");
        }
    }

    #[test]
    fn summarize_document_invokes_opencode_run_and_parses_output() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_opencode_script(
            &script_path,
            &args_path,
            &stdin_path,
            "Summary from opencode.",
        );

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
        assert!(args.contains("run"), "argv should contain 'run': {args}");
        assert!(
            args.contains("--agent"),
            "argv should contain '--agent': {args}"
        );
        assert!(args.contains("kb"), "argv should contain agent name: {args}");

        // bn-18xw: prompt body MUST land on stdin, not in argv.
        let stdin_bytes = fs::read_to_string(&stdin_path).expect("read captured stdin");
        assert!(
            stdin_bytes.contains("Example Source"),
            "stdin should contain source title: {stdin_bytes}"
        );
        assert!(
            stdin_bytes.contains("A long source document."),
            "stdin should contain document text"
        );
        assert!(
            stdin_bytes.contains("80 words"),
            "stdin should contain max word budget"
        );
        assert!(
            !args.contains("Example Source"),
            "prompt body MUST NOT appear in argv (bn-18xw): {args}"
        );
    }

    #[test]
    fn extract_concepts_invokes_opencode_run_and_parses_json_payload() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        // The response body contains literal single quotes in
        // "Rust's reference safety analysis." — write_fake_opencode_script
        // shell-escapes them, so we pass the raw string here.
        let response = r#"{"concepts":[{"name":"Borrow checker","aliases":["borrowck"],"definition_hint":"Rust's reference safety analysis.","source_anchors":[{"heading_anchor":"ownership","quote":"The borrow checker validates references."}]}]}"#;
        write_fake_opencode_script(&script_path, &args_path, &stdin_path, response);

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

        // bn-18xw: prompt template variables travel via stdin, not argv.
        let stdin_bytes = fs::read_to_string(&stdin_path).expect("read captured stdin");
        assert!(stdin_bytes.contains("Ownership Notes"));
        assert!(stdin_bytes.contains("Rust ownership overview"));
        assert!(stdin_bytes.contains('5'));
    }

    #[test]
    fn nonzero_exit_surfaces_stderr_in_error() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fail-opencode.sh");
        // Drain stdin so the writer thread doesn't block on EPIPE while we
        // tear down the child — the new run_command_with_stdin pipes the
        // prompt regardless of whether the child cares to read it.
        fs::write(
            &script_path,
            "#!/bin/sh\ncat > /dev/null\nprintf 'opencode-error-output' >&2\nexit 1\n",
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
    fn variant_and_session_flags_appear_in_argv() {
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: "opencode".to_string(),
            agent_name: "kb".to_string(),
            variant: Some("fast".to_string()),
            session_id: Some("sess-123".to_string()),
            ..Default::default()
        });

        let argv = adapter.build_argv();

        // Each flag/value should be its own argv entry — no shell quoting,
        // no concatenation. Use position-based assertions so we catch
        // ordering regressions (e.g. value separated from its flag).
        let variant_idx = argv
            .iter()
            .position(|s| s == "--variant")
            .expect("missing --variant in argv");
        assert_eq!(argv.get(variant_idx + 1).map(String::as_str), Some("fast"));
        let session_idx = argv
            .iter()
            .position(|s| s == "--session")
            .expect("missing --session in argv");
        assert_eq!(
            argv.get(session_idx + 1).map(String::as_str),
            Some("sess-123")
        );
    }

    #[test]
    fn build_argv_with_attachments_emits_f_flag_per_image() {
        // bn-3dkw: each image must appear as its own `-f <path>` pair so
        // opencode's yargs parser picks them up as an array.
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: "opencode".to_string(),
            agent_name: "kb".to_string(),
            ..Default::default()
        });

        let images = [PathBuf::from("/abs/a.png"), PathBuf::from("/abs/b.jpg")];
        let argv = adapter.build_argv_with_attachments(&images, false);

        let f_flag_count = argv.iter().filter(|s| s.as_str() == "-f").count();
        assert_eq!(f_flag_count, 2, "expected two -f flags in: {argv:?}");
        assert!(
            argv.iter().any(|s| s == "/abs/a.png"),
            "missing a.png in: {argv:?}"
        );
        assert!(
            argv.iter().any(|s| s == "/abs/b.jpg"),
            "missing b.jpg in: {argv:?}"
        );
    }

    #[test]
    fn build_argv_omits_f_flag_when_no_attachments() {
        let adapter = OpencodeAdapter::new(OpencodeConfig::default());
        let argv = adapter.build_argv_with_attachments(&[], false);
        assert!(
            !argv.iter().any(|s| s == "-f"),
            "no -f flag expected when image_paths empty: {argv:?}"
        );
    }

    #[test]
    fn build_argv_does_not_include_prompt() {
        // bn-18xw: the prompt MUST NOT appear in argv — it's piped to stdin
        // so we sidestep `ARG_MAX` and shell-quoting. The argv should stay
        // tiny regardless of how big the hypothetical prompt is.
        let adapter = OpencodeAdapter::new(OpencodeConfig::default());
        let argv = adapter.build_argv();

        // Sanity: argv contains the executable, "run", "--agent", and the
        // agent name — that's the full lean shape today.
        assert_eq!(argv.len(), 4, "unexpected argv length: {argv:?}");
        assert_eq!(argv[1], "run");
        assert_eq!(argv[2], "--agent");

        // Argv stays well under 1 KB by construction.
        let total_bytes: usize = argv.iter().map(String::len).sum();
        assert!(
            total_bytes < 1024,
            "argv ballooned past 1 KB; prompt may have leaked into argv: {argv:?}"
        );
    }

    #[test]
    fn build_env_carries_opencode_config_path() {
        // bn-18xw: OPENCODE_CONFIG used to be inlined into a shell command
        // string. It now flows through Command::env so we don't need to
        // shell-quote the path.
        let config_path = PathBuf::from("/tmp/opencode.json");
        let env = OpencodeAdapter::build_env(&config_path);
        assert_eq!(env.len(), 1);
        assert_eq!(env[0].0, "OPENCODE_CONFIG");
        assert_eq!(env[0].1, "/tmp/opencode.json");
    }

    #[test]
    fn build_argv_contains_no_shell_metacharacters_for_pathological_inputs() {
        // bn-18xw regression guard: even if the agent name or variant
        // contains shell-active characters (single quotes, backticks,
        // dollar signs), nothing in our argv should require quoting because
        // we hand each arg to Command::args verbatim. Validates the
        // argv-direct contract.
        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: "opencode".to_string(),
            agent_name: "agent'with`dangerous$chars".to_string(),
            variant: Some("'$(rm -rf /)'".to_string()),
            ..Default::default()
        });
        let argv = adapter.build_argv();

        // Pathological strings appear in argv exactly as written — no
        // wrapping in single quotes, no backslash escaping.
        assert!(argv.iter().any(|s| s == "agent'with`dangerous$chars"));
        assert!(argv.iter().any(|s| s == "'$(rm -rf /)'"));
    }

    #[test]
    fn answer_question_forwards_image_paths_to_opencode_cli() {
        // bn-3dkw integration: the image paths on AnswerQuestionRequest reach
        // the opencode subprocess invocation as `-f <path>` args. Uses a
        // fake-opencode script that captures argv to disk.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_opencode_script(&script_path, &args_path, &stdin_path, "answer from opencode");

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
                structured_output: false,
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
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_opencode_script(&script_path, &args_path, &stdin_path, "answer without images");

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
                structured_output: false,
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

    // ---------------------------------------------------------------
    // bn-1ikn: JSON event stream parsing for the chart call path.
    // ---------------------------------------------------------------

    #[test]
    fn extract_final_answer_picks_only_final_answer_phase_events() {
        // Two tool-narration events (phase: "response_reasoning") before a
        // single final_answer event. The extractor must drop the narration
        // and return only the final caption.
        let stream = concat!(
            r#"{"type":"step_start","timestamp":1,"part":{"type":"step-start"}}"#, "\n",
            r#"{"type":"text","timestamp":2,"part":{"type":"text","text":"Checking the output location, then I'll write a matplotlib script.","metadata":{"openai":{"phase":"response_reasoning"}}}}"#, "\n",
            r#"{"type":"text","timestamp":3,"part":{"type":"text","text":"Writing the chart script now, then executing it.","metadata":{"openai":{"phase":"response_reasoning"}}}}"#, "\n",
            r#"{"type":"text","timestamp":4,"part":{"type":"text","text":"The chart shows a simple horizontal comparison.","metadata":{"openai":{"phase":"final_answer"}}}}"#, "\n",
            r#"{"type":"step_finish","timestamp":5,"part":{"type":"step-finish"}}"#, "\n",
        );

        let got = extract_final_answer_from_json_events(stream);
        assert_eq!(got, "The chart shows a simple horizontal comparison.");
    }

    #[test]
    fn extract_final_answer_handles_q_db1_shape() {
        // bn-1ikn: the pass-15 q-db1 real-world raw shape — two narration
        // lines with no blank line separating them from the caption, which
        // defeats `strip_tool_narration`. With `--format json`, each line
        // arrives as its own event and we never see the run-together text.
        let stream = concat!(
            r#"{"type":"step_start","timestamp":1,"part":{"type":"step-start"}}"#, "\n",
            r#"{"type":"text","timestamp":2,"part":{"type":"text","text":"Checking the output location, then I'll write and run a minimal matplotlib script to generate the PNG.","metadata":{"openai":{"phase":"response_reasoning"}}}}"#, "\n",
            r#"{"type":"text","timestamp":3,"part":{"type":"text","text":"Writing the chart script now, then I'll execute it and verify the PNG was created.","metadata":{"openai":{"phase":"response_reasoning"}}}}"#, "\n",
            r#"{"type":"text","timestamp":4,"part":{"type":"text","text":"The chart shows a simple horizontal comparison with two categories, `hexagonal` and `clean`, each assigned an equal value of 1.[1][2]","metadata":{"openai":{"phase":"final_answer"}}}}"#, "\n",
            r#"{"type":"step_finish","timestamp":5,"part":{"type":"step-finish"}}"#, "\n",
        );

        let got = extract_final_answer_from_json_events(stream);
        assert!(
            !got.contains("Checking the output location"),
            "narration leaked into output: {got}"
        );
        assert!(
            !got.contains("Writing the chart script"),
            "narration leaked into output: {got}"
        );
        assert!(got.starts_with("The chart shows"), "final caption lost: {got}");
    }

    #[test]
    fn extract_final_answer_concatenates_multi_chunk_final_answer() {
        // Some providers emit the final answer across multiple `text` events
        // (streaming). The extractor must concatenate them in order.
        let stream = concat!(
            r#"{"type":"text","part":{"type":"text","text":"Part one. ","metadata":{"openai":{"phase":"final_answer"}}}}"#, "\n",
            r#"{"type":"text","part":{"type":"text","text":"Part two.","metadata":{"openai":{"phase":"final_answer"}}}}"#, "\n",
        );

        assert_eq!(
            extract_final_answer_from_json_events(stream),
            "Part one. Part two."
        );
    }

    #[test]
    fn extract_final_answer_falls_back_to_any_text_when_no_phase_marker() {
        // Older opencode / non-openai providers may not emit the
        // `phase: "final_answer"` metadata. Rather than drop the model's
        // reply on the floor, return the concatenation of all text events
        // so `strip_tool_narration` (run after this in the CLI) can still
        // have a go.
        let stream = concat!(
            r#"{"type":"step_start","part":{"type":"step-start"}}"#, "\n",
            r#"{"type":"text","part":{"type":"text","text":"Hello world.","metadata":{}}}"#, "\n",
        );

        assert_eq!(extract_final_answer_from_json_events(stream), "Hello world.");
    }

    #[test]
    fn extract_final_answer_falls_back_to_plain_text_for_non_json_stream() {
        // If the runner didn't actually emit NDJSON (e.g. the --format flag
        // was dropped), don't silently return empty — run the usual
        // ANSI-header strip so callers still surface the model's reply.
        let stream = "\x1b[0m\n> kb · openai/gpt-5.4\n\x1b[0m\nPlain text answer.";
        assert_eq!(
            extract_final_answer_from_json_events(stream),
            "Plain text answer."
        );
    }

    #[test]
    fn extract_final_answer_ignores_unparseable_lines_between_events() {
        // Tolerate stderr-ish lines or blank lines mixed into the stream.
        let stream = concat!(
            "\n",
            "warning: some log line\n",
            r#"{"type":"text","part":{"type":"text","text":"real answer","metadata":{"openai":{"phase":"final_answer"}}}}"#, "\n",
        );

        assert_eq!(extract_final_answer_from_json_events(stream), "real answer");
    }

    #[test]
    fn extract_final_answer_works_with_anthropic_style_phase_key() {
        // The `part.metadata` object is keyed by provider. Don't hard-code
        // "openai" — any direct child whose `.phase == "final_answer"`
        // counts.
        let stream = concat!(
            r#"{"type":"text","part":{"type":"text","text":"claude's answer","metadata":{"anthropic":{"phase":"final_answer"}}}}"#, "\n",
        );

        assert_eq!(
            extract_final_answer_from_json_events(stream),
            "claude's answer"
        );
    }

    #[test]
    fn build_argv_adds_format_json_flag_when_requested() {
        // bn-1ikn: `--format json` only appears in the argv when callers
        // explicitly ask for the structured event stream. Other calls must
        // NOT include the flag (it would disturb their plain-text parsing).
        let adapter = OpencodeAdapter::new(OpencodeConfig::default());

        let plain = adapter.build_argv_with_attachments(&[], false);
        assert!(
            !plain.iter().any(|s| s == "--format"),
            "plain-text path must not include --format flag: {plain:?}"
        );

        let json = adapter.build_argv_with_attachments(&[], true);
        let format_idx = json
            .iter()
            .position(|s| s == "--format")
            .expect("json path missing --format flag");
        assert_eq!(
            json.get(format_idx + 1).map(String::as_str),
            Some("json"),
            "--format flag must be followed by 'json': {json:?}"
        );
    }

    #[test]
    fn answer_question_chart_path_passes_format_json_flag_to_opencode() {
        // When `structured_output: true` is set on the request, the opencode
        // subprocess must be invoked with `--format json`. The fake-opencode
        // script captures argv to disk and also emits a valid event stream
        // so we can round-trip the final-answer extraction.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");

        // Emit two narration events followed by a final_answer event.
        let event_stream = concat!(
            r#"{"type":"text","part":{"type":"text","text":"I'll check the inputs.","metadata":{"openai":{"phase":"response_reasoning"}}}}"#, "\n",
            r#"{"type":"text","part":{"type":"text","text":"Running now.","metadata":{"openai":{"phase":"response_reasoning"}}}}"#, "\n",
            r#"{"type":"text","part":{"type":"text","text":"The caption is this.","metadata":{"openai":{"phase":"final_answer"}}}}"#, "\n",
        );
        // bn-18xw: drain stdin into stdin_path, capture argv to args_path,
        // then emit the heredoc'd event stream.
        fs::write(
            &script_path,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\ncat > '{}'\ncat <<'JSON_END'\n{}JSON_END\n",
                args_path.display(),
                stdin_path.display(),
                event_stream,
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

        let (response, _prov) = adapter
            .answer_question(AnswerQuestionRequest {
                question: "render a chart".to_string(),
                context: vec!["source".to_string()],
                format: Some(String::new()),
                template_name: None,
                output_path: None,
                image_paths: Vec::new(),
                structured_output: true,
            })
            .expect("answer");

        assert_eq!(
            response.answer, "The caption is this.",
            "only the final_answer event text should reach the CLI"
        );

        let args = fs::read_to_string(&args_path).expect("captured args");
        let arg_lines: Vec<&str> = args.lines().collect();
        let format_idx = arg_lines
            .iter()
            .position(|l| *l == "--format")
            .unwrap_or_else(|| panic!("--format flag missing in argv: {args}"));
        assert_eq!(
            arg_lines.get(format_idx + 1),
            Some(&"json"),
            "--format flag should be followed by 'json' in argv: {args}"
        );
    }

    #[test]
    fn answer_question_default_path_does_not_pass_format_json_flag() {
        // structured_output: false (the default) must preserve the existing
        // plain-text invocation. Chart is the only caller that flips this on.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_opencode_script(&script_path, &args_path, &stdin_path, "plain answer");

        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: script_path.display().to_string(),
            timeout: Duration::from_secs(5),
            ..Default::default()
        });

        let (response, _prov) = adapter
            .answer_question(AnswerQuestionRequest {
                question: "regular question".to_string(),
                context: vec!["source".to_string()],
                format: Some(String::new()),
                template_name: None,
                output_path: None,
                image_paths: Vec::new(),
                structured_output: false,
            })
            .expect("answer");

        assert_eq!(response.answer, "plain answer");

        let args = fs::read_to_string(&args_path).expect("captured args");
        assert!(
            !args.lines().any(|l| l == "--format"),
            "--format flag must NOT appear when structured_output is false: {args}"
        );
    }

    #[test]
    fn run_prompt_pipes_one_mb_prompt_through_stdin_without_arg_max_failure() {
        // bn-18xw acceptance: the original sh -c path failed unpredictably on
        // ~1 MB prompts (ARG_MAX is 4 MB on Linux but envp halves it in
        // practice). With stdin delivery, argv stays tiny and we should
        // round-trip a 1 MB prompt cleanly. The fake-opencode script captures
        // stdin to disk so we can verify byte-for-byte round-trip.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-opencode.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_opencode_script(
            &script_path,
            &args_path,
            &stdin_path,
            "answer for big prompt",
        );

        let adapter = OpencodeAdapter::new(OpencodeConfig {
            command: script_path.display().to_string(),
            timeout: Duration::from_secs(15),
            ..Default::default()
        });

        // 1 MB prompt — well past anything you'd safely cram into argv.
        let big_prompt = "x".repeat(1024 * 1024);
        let answer = adapter
            .run_prompt(&big_prompt)
            .expect("1 MB prompt via stdin should succeed");
        assert_eq!(answer, "answer for big prompt");

        // argv stays small — the prompt is NOT in it.
        let args_bytes = fs::metadata(&args_path).expect("args metadata").len();
        assert!(
            args_bytes < 1024,
            "argv ballooned past 1 KB ({args_bytes} bytes); prompt may have leaked into argv"
        );

        // stdin captured the full 1 MB.
        let captured_stdin = fs::read(&stdin_path).expect("read captured stdin");
        assert_eq!(
            captured_stdin.len(),
            big_prompt.len(),
            "stdin byte count mismatch"
        );
        assert!(
            captured_stdin.iter().all(|&b| b == b'x'),
            "stdin payload corrupted in flight"
        );
    }
}
