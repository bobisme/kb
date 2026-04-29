use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

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
use crate::provenance::{ProvenanceRecord, TokenUsage};
use crate::subprocess::{SubprocessError, run_command_with_stdin};
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
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

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

    fn build_argv(&self) -> Vec<String> {
        self.build_argv_with_extra_tools(&[])
    }

    /// Build a `claude -p` argv with an explicit set of additional tools
    /// appended via `--allowedTools`. bn-xt4o: the `impute_gap` call uses
    /// this to turn on `WebSearch` + `WebFetch` so the agent can browse.
    /// Claude's default `--tools` set is left alone when `extra_tools` is
    /// empty so other calls continue to work with whatever the user
    /// configured.
    ///
    /// bn-18xw: the prompt is no longer included in argv — it's piped to
    /// the child's stdin. `claude -p` reads stdin when no positional prompt
    /// is provided, so we sidestep the `ARG_MAX` hazard and the shell-quoting
    /// tax for large prompts. This also means a failed `claude` invocation
    /// no longer echoes the rendered prompt back through stderr (sh -c
    /// used to log the entire failed command line).
    fn build_argv_with_extra_tools(&self, extra_tools: &[&str]) -> Vec<String> {
        let mut argv = vec![self.config.command.clone(), "-p".to_string()];

        if let Some(model) = &self.config.model {
            argv.push("--model".to_string());
            argv.push(model.clone());
        }

        if let Some(permission_mode) = &self.config.permission_mode {
            argv.push("--permission-mode".to_string());
            argv.push(permission_mode.clone());
        }

        if !extra_tools.is_empty() {
            argv.push("--allowedTools".to_string());
            // Claude's `--allowedTools` takes a comma-or-space separated
            // list. Pass a single comma-joined arg via Command::args —
            // no shell quoting needed.
            argv.push(extra_tools.join(","));
        }

        argv.push("--output-format".to_string());
        argv.push("json".to_string());

        argv
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
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

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
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

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
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

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

        Ok((
            GenerateConceptBodyResponse {
                body: strip_plain_text_wrappers(&parsed.text),
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
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;
        let response = parse_generate_concept_from_candidate_json(&parsed.text)?;
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

        // bn-3dkw: `claude` CLI doesn't expose a local-image attachment flag
        // (`--file` expects an Anthropic-uploaded `file_id`, and `--add-dir`
        // is a tool-access flag, not an attachment). Fall back to embedding
        // each image as a base64 data URI inline in the prompt — Claude 4.X
        // is multimodal and accepts data-URI images in user turns.
        let content_with_images =
            append_image_attachments_as_data_uris(&rendered.content, &request.image_paths)?;

        let prompt = RenderedPrompt {
            template_name: template.name,
            template_hash: template.template_hash,
            content: content_with_images,
            render_hash: rendered.render_hash,
        };

        let started_at = unix_time_ms()?;
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            prompt.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

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
        let argv = self.build_argv();
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;
        let response = parse_detect_contradictions_json(&parsed.text)?;
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

    fn impute_gap(
        &self,
        request: ImputeGapRequest,
    ) -> Result<(ImputeGapResponse, ProvenanceRecord), LlmAdapterError> {
        let template = Template::load("impute_gap.md", self.config.project_root.as_deref())
            .map_err(|err| LlmAdapterError::Other(format!("load impute_gap template: {err}")))?;

        let mut context = HashMap::new();
        context.insert(
            "gap_kind".to_string(),
            request.gap_kind.prompt_description().to_string(),
        );
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
        // bn-xt4o: enable claude's `WebSearch` + `WebFetch` tools for this
        // call so the agent can browse external sources. Other calls leave
        // the allowed-tools set at the user's default.
        let argv = self.build_argv_with_extra_tools(&["WebSearch", "WebFetch"]);
        let output = run_command_with_stdin(
            &argv,
            rendered.content.as_bytes(),
            &[],
            self.config.timeout,
        )
        .map_err(map_subprocess_error)?;

        if output.exit_code != Some(0) {
            return Err(classify_nonzero_exit(
                output.exit_code,
                &output.stderr,
                &output.stdout,
            ));
        }

        let parsed = parse_claude_json(&output.stdout)?;
        let response = parse_impute_gap_json(&parsed.text)?;
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
        let timeout = health_check_timeout(self.config.timeout);
        let argv = self.build_argv();

        let started_at = unix_time_ms()?;
        let result = run_command_with_stdin(&argv, prompt.as_bytes(), &[], timeout);
        let ended_at = unix_time_ms()?;

        let output = result.map_err(|e| match e {
            SubprocessError::TimedOut { timeout, .. } => LlmAdapterError::Timeout(format!(
                "Claude CLI health check exceeded timeout of {}s \u{2014} this can mean the harness is slow to start (try again), the runner isn't authenticated, or the model isn't available.",
                timeout.as_secs()
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

/// Format an alias list for the `concept_body.md` prompt.
/// Empty list renders as "(none)" so the template line "Aliases: " stays grammatical.
fn format_aliases_for_prompt(aliases: &[String]) -> String {
    if aliases.is_empty() {
        "(none)".to_string()
    } else {
        aliases.join(", ")
    }
}

/// Format quotes for the `detect_contradictions.md` prompt. Each quote is
/// rendered as a numbered entry (zero-based index) with its source label,
/// matching the response contract (`conflicting_quotes` is a list of
/// zero-based indices into the request's quotes list).
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

/// Format candidate snippets as bullet lines for the
/// `generate_concept_from_candidate.md` prompt. Empty list renders as a
/// single "(no snippets available)" line so the prompt still parses cleanly.
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

/// Strip wrapping code fences, blockquote markers, and surrounding whitespace
/// from a plain-text LLM response. Defensive — the prompt tells the model to
/// return plain text, but models occasionally wrap short bodies in triple
/// backticks or a `> ` blockquote anyway.
fn strip_plain_text_wrappers(text: &str) -> String {
    let trimmed = text.trim();
    let without_fence = trimmed.strip_prefix("```").map_or(trimmed, |rest| {
        // Drop the optional language tag up to the first newline.
        let after_header = rest.split_once('\n').map_or(rest, |(_, body)| body);
        after_header
            .rsplit_once("```")
            .map_or(after_header, |(body, _)| body)
            .trim()
    });
    without_fence.to_string()
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

/// Clamp the configured runner timeout into `[30s, 60s]` for health checks.
///
/// `kb doctor` shouldn't hang for 900s on a misconfigured harness, but a cold
/// Claude CLI invocation can take 10-15s on first call. The `[30s, 60s]` range
/// keeps doctor snappy while tolerating cold starts.
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

/// Append a trailing "Attachments" section to `prompt` containing each image
/// from `image_paths` as a markdown data-URI image.
///
/// bn-3dkw: the `claude` CLI does not accept local image paths directly — its
/// `--file` flag wants a pre-uploaded Anthropic `file_id`, not a path on disk.
/// Base64 data URIs in the prompt text are the cheapest portable fallback and
/// Claude 4.X reads them as real image inputs.
///
/// A missing file is surfaced as a transport-style error so the caller can
/// degrade gracefully rather than silently drop the attachment.
fn append_image_attachments_as_data_uris(
    prompt: &str,
    image_paths: &[PathBuf],
) -> Result<String, LlmAdapterError> {
    if image_paths.is_empty() {
        return Ok(prompt.to_string());
    }

    let mut out = String::with_capacity(prompt.len());
    out.push_str(prompt);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("\n## Attachments\n\n");
    out.push_str(
        "The following images are referenced by the sources above. \
         Inspect them when answering if they're relevant.\n\n",
    );

    for path in image_paths {
        let bytes = std::fs::read(path).map_err(|err| {
            LlmAdapterError::Other(format!(
                "read image attachment {}: {err}",
                path.display()
            ))
        })?;
        let mime = guess_image_mime(path);
        let encoded = base64_encode(&bytes);
        let label = path
            .file_name()
            .map_or_else(|| "image".to_string(), |n| n.to_string_lossy().into_owned());
        // writeln! into a String never fails; ignore the fmt::Result.
        let _ = writeln!(out, "![{label}](data:{mime};base64,{encoded})");
    }

    Ok(out)
}

/// Minimal RFC 4648 base64 encoder (standard alphabet, with `=` padding).
///
/// bn-3dkw: the `base64` crate isn't in the workspace, and adding it for one
/// call site is overkill. Output matches `data:`-URI encoding expectations.
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut iter = input.chunks_exact(3);
    for chunk in iter.by_ref() {
        let b0 = chunk[0];
        let b1 = chunk[1];
        let b2 = chunk[2];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0b11) << 4) | (b1 >> 4)) as usize] as char);
        out.push(ALPHABET[(((b1 & 0b1111) << 2) | (b2 >> 6)) as usize] as char);
        out.push(ALPHABET[(b2 & 0b11_1111) as usize] as char);
    }
    let rem = iter.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let b0 = rem[0];
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[((b0 & 0b11) << 4) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let b0 = rem[0];
            let b1 = rem[1];
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[(((b0 & 0b11) << 4) | (b1 >> 4)) as usize] as char);
            out.push(ALPHABET[((b1 & 0b1111) << 2) as usize] as char);
            out.push('=');
        }
        _ => unreachable!("chunks_exact(3) remainder is 0, 1, or 2"),
    }
    out
}

/// Map a file extension to a MIME type for data URI embedding. Defaults to
/// `application/octet-stream` so unknown extensions still produce a valid
/// data URI (the model will just ignore them) rather than failing the call.
fn guess_image_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    }
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

    /// bn-18xw: build a fake-claude shell script that captures argv to
    /// `args_path`, captures stdin to `stdin_path`, and prints
    /// `response_body` on stdout. Mirrors the opencode-side helper because
    /// both adapters now deliver the prompt via stdin instead of a
    /// shell-quoted positional.
    fn write_fake_claude_script(
        script_path: &Path,
        args_path: &Path,
        stdin_path: &Path,
        response_body: &str,
    ) {
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\ncat > '{stdin}'\nprintf '%s' '{body}'\n",
            args = args_path.display(),
            stdin = stdin_path.display(),
            body = response_body.replace('\'', "'\\''"),
        );
        fs::write(script_path, script).expect("write fake claude script");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(script_path).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(script_path, permissions).expect("chmod");
        }
    }

    #[test]
    fn summarize_document_invokes_claude_cli_and_builds_provenance() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_claude_script(
            &script_path,
            &args_path,
            &stdin_path,
            r#"{"result":"summary from claude","model":"claude-haiku","usage":{"input_tokens":21,"output_tokens":6}}"#,
        );

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

        // bn-18xw: the rendered prompt is delivered on stdin, not in argv.
        let stdin_bytes = fs::read_to_string(&stdin_path).expect("read captured stdin");
        assert!(stdin_bytes.contains("Example Source"));
        assert!(stdin_bytes.contains("A long source document."));
        assert!(stdin_bytes.contains("80 words"));
        assert!(
            !args.contains("Example Source"),
            "prompt body MUST NOT appear in argv (bn-18xw): {args}"
        );
    }

    #[test]
    fn extract_concepts_invokes_claude_cli_and_parses_json_payload() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_claude_script(
            &script_path,
            &args_path,
            &stdin_path,
            r#"{"result":"{\"concepts\":[{\"name\":\"Borrow checker\",\"aliases\":[\"borrowck\"],\"definition_hint\":\"Rust's reference safety analysis.\",\"source_anchors\":[{\"heading_anchor\":\"ownership\",\"quote\":\"The borrow checker validates references.\"}]}]}","model":"claude-haiku","usage":{"input_tokens":30,"output_tokens":9}}"#,
        );

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

        // bn-18xw: prompt template variables travel via stdin.
        let stdin_bytes = fs::read_to_string(&stdin_path).expect("read captured stdin");
        assert!(stdin_bytes.contains("Ownership Notes"));
        assert!(stdin_bytes.contains("Rust ownership overview"));
        assert!(stdin_bytes.contains('5'));
    }

    #[test]
    fn run_health_check_returns_healthy_when_output_contains_ok() {
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        // bn-18xw: drain stdin so the parent's writer thread doesn't EPIPE
        // before the child can finish writing its OK envelope.
        fs::write(
            &script_path,
            "#!/bin/sh\ncat > /dev/null\nprintf '{\"result\":\"OK\"}'",
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
        fs::write(
            &script_path,
            "#!/bin/sh\ncat > /dev/null\nprintf '{\"result\":\"Error\"}'",
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
    fn base64_encode_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn append_image_attachments_embeds_data_uris_when_paths_present() {
        // bn-3dkw: claude CLI has no local-image attach flag, so we append
        // the images as base64 data URIs in the prompt. Verify the prompt
        // grows with a markdown image reference per path.
        let tmp = TempDir::new().expect("tmp");
        let img = tmp.path().join("fig.png");
        fs::write(&img, b"\x89PNG\r\n\x1a\n").expect("png");

        let out = append_image_attachments_as_data_uris("prompt body", std::slice::from_ref(&img))
            .expect("embed image");
        assert!(out.starts_with("prompt body"));
        assert!(out.contains("## Attachments"));
        assert!(out.contains("data:image/png;base64,"));
        assert!(out.contains("fig.png"));
    }

    #[test]
    fn append_image_attachments_is_noop_when_empty() {
        let out = append_image_attachments_as_data_uris("prompt body", &[]).expect("noop");
        assert_eq!(out, "prompt body");
    }

    #[test]
    fn answer_question_forwards_image_data_uris_to_claude_prompt() {
        // bn-3dkw: the prompt the fake-claude harness sees on stdin must
        // include the data-URI-embedded image so an actual Claude
        // invocation would see the image bytes.
        // bn-18xw: the prompt is now delivered on stdin (not argv), so we
        // assert against captured stdin instead of captured argv.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_claude_script(
            &script_path,
            &args_path,
            &stdin_path,
            r#"{"result":"ok"}"#,
        );

        let img = tmp.path().join("diagram.png");
        fs::write(&img, b"\x89PNGfake").expect("png");

        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: script_path.display().to_string(),
            model: None,
            permission_mode: Some("default".to_string()),
            timeout: Duration::from_secs(5),
            project_root: None,
        });

        let (response, _prov) = adapter
            .answer_question(AnswerQuestionRequest {
                question: "describe the diagram".to_string(),
                context: vec!["source".to_string()],
                format: Some(String::new()),
                template_name: None,
                output_path: None,
                image_paths: vec![img],
                structured_output: false,
            })
            .expect("answer");
        assert_eq!(response.answer, "ok");

        let stdin_bytes = fs::read_to_string(&stdin_path).expect("captured stdin");
        assert!(
            stdin_bytes.contains("data:image/png;base64,"),
            "claude stdin must include data URI"
        );
        assert!(
            stdin_bytes.contains("diagram.png"),
            "claude stdin must include image label"
        );

        // Argv MUST stay free of the rendered prompt — that's the whole
        // point of bn-18xw.
        let args = fs::read_to_string(&args_path).expect("captured args");
        assert!(
            !args.contains("data:image/png;base64,"),
            "data URI must NOT appear in argv (bn-18xw): {args}"
        );
    }

    #[test]
    fn build_argv_does_not_include_prompt() {
        // bn-18xw: the prompt MUST NOT appear in argv — it's piped to stdin
        // so we sidestep ARG_MAX and shell-quoting. Even with all flags
        // populated, argv stays small and tidy.
        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: "claude".to_string(),
            model: Some("claude-opus".to_string()),
            permission_mode: Some("default".to_string()),
            timeout: Duration::from_secs(5),
            project_root: None,
        });
        let argv = adapter.build_argv();

        // Locate flags by position so a regression that breaks pairing
        // (flag without value, value without flag) is caught.
        let model_idx = argv
            .iter()
            .position(|s| s == "--model")
            .expect("missing --model");
        assert_eq!(argv.get(model_idx + 1).map(String::as_str), Some("claude-opus"));
        let perm_idx = argv
            .iter()
            .position(|s| s == "--permission-mode")
            .expect("missing --permission-mode");
        assert_eq!(argv.get(perm_idx + 1).map(String::as_str), Some("default"));
        let format_idx = argv
            .iter()
            .position(|s| s == "--output-format")
            .expect("missing --output-format");
        assert_eq!(argv.get(format_idx + 1).map(String::as_str), Some("json"));

        // argv stays well under 1 KB regardless of how big a prompt the
        // caller hands to run_command_with_stdin.
        let total_bytes: usize = argv.iter().map(String::len).sum();
        assert!(
            total_bytes < 256,
            "claude argv ballooned past 256 bytes; prompt may have leaked: {argv:?}"
        );
    }

    #[test]
    fn build_argv_with_extra_tools_passes_comma_joined_list() {
        // bn-xt4o + bn-18xw: --allowedTools takes a comma-or-space joined
        // list. We pass a single comma-joined arg via Command::args (no
        // shell quoting); previously this went through shell_quote.
        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig::default());
        let argv = adapter.build_argv_with_extra_tools(&["WebSearch", "WebFetch"]);

        let tools_idx = argv
            .iter()
            .position(|s| s == "--allowedTools")
            .expect("missing --allowedTools");
        assert_eq!(
            argv.get(tools_idx + 1).map(String::as_str),
            Some("WebSearch,WebFetch")
        );
    }

    #[test]
    fn build_argv_contains_no_shell_metacharacters_for_pathological_inputs() {
        // bn-18xw regression guard: shell-active characters in config
        // values flow into argv verbatim — Command::args bypasses the
        // shell entirely so quoting is unnecessary.
        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: "claude".to_string(),
            model: Some("model'with`backticks$and$dollar".to_string()),
            permission_mode: Some("'$(rm -rf /)'".to_string()),
            timeout: Duration::from_secs(5),
            project_root: None,
        });
        let argv = adapter.build_argv();

        assert!(argv.iter().any(|s| s == "model'with`backticks$and$dollar"));
        assert!(argv.iter().any(|s| s == "'$(rm -rf /)'"));
    }

    #[test]
    fn run_prompt_pipes_one_mb_prompt_through_stdin_without_arg_max_failure() {
        // bn-18xw acceptance: the original sh -c path failed unpredictably
        // on ~1 MB prompts. With stdin delivery, argv stays tiny and a
        // 1 MB prompt round-trips cleanly. Captures stdin to disk so we
        // can verify byte-for-byte fidelity.
        let tmp = TempDir::new().expect("temp dir");
        let script_path = tmp.path().join("fake-claude.sh");
        let args_path = tmp.path().join("args.txt");
        let stdin_path = tmp.path().join("stdin.txt");
        write_fake_claude_script(
            &script_path,
            &args_path,
            &stdin_path,
            r#"{"result":"answered","model":"claude-haiku","usage":{"input_tokens":1,"output_tokens":1}}"#,
        );

        let adapter = ClaudeCliAdapter::new(ClaudeCliConfig {
            command: script_path.display().to_string(),
            model: Some("claude-haiku".to_string()),
            permission_mode: Some("default".to_string()),
            timeout: Duration::from_secs(15),
            project_root: None,
        });

        let big_prompt = "y".repeat(1024 * 1024);
        let (response, _prov) = adapter
            .summarize_document(SummarizeDocumentRequest {
                title: "big".to_string(),
                body: big_prompt.clone(),
                max_words: 80,
            })
            .expect("1 MB prompt via stdin should succeed");
        assert_eq!(response.summary, "answered");

        // argv stays tiny.
        let args_bytes = fs::metadata(&args_path).expect("args metadata").len();
        assert!(
            args_bytes < 1024,
            "claude argv ballooned past 1 KB ({args_bytes} bytes); prompt may have leaked"
        );

        // stdin captures the prompt body — exact equality fails because the
        // template wraps the body in a prompt scaffold, but the body MUST
        // appear there in full.
        let captured = fs::read_to_string(&stdin_path).expect("read captured stdin");
        assert!(
            captured.contains(&big_prompt),
            "1 MB body missing from captured stdin (length {})",
            captured.len()
        );
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

    #[test]
    fn health_check_timeout_clamps_to_thirty_sixty_range() {
        // Below-floor runner timeouts get lifted to the 30s floor so cold-start
        // Claude CLI invocations don't false-negative.
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
}
