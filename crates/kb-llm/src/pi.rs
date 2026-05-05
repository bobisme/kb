//! `pi` runner adapter (bn-1s8y).
//!
//! Pi is a multi-provider non-interactive agent CLI (see `pi --help`).
//! It exposes the same shape of output as `opencode run` — a single
//! response printed to stdout — but its process model is more
//! resilient to parallel invocation. Opencode initialises a per-process
//! `SQLite` database on first call and races on `PRAGMA journal_mode = WAL`
//! when multiple processes start at the same instant (observed
//! empirically in bn-17t0). Pi avoids that fragility, which is why kb
//! prefers it as the default runner for non-Claude models.
//!
//! ## Invocation contract
//!
//! Mirrors `~/src/chief/ws/default/src/runner/pi.rs`:
//!
//! ```text
//! pi --print --no-extensions --no-skills --no-prompt-templates \
//!    --system-prompt <inline text> \
//!    [--provider <p>] [--model <m>] \
//!    --tools <list>  (default: read,grep,find,ls) \
//!    [--thinking <level>] \
//!    --session-dir <dir> \
//!    [--session <file> | --no-session] \
//!    @<user-prompt-path>      # MUST be the last argument
//! ```
//!
//! kb's prompts are self-contained — the rendered template carries
//! everything the model needs — so we pass an empty `--system-prompt`
//! and place the rendered template as the user message via `@<tempfile>`
//! to sidestep `ARG_MAX` for large prompts. The three `--no-*` flags
//! suppress pi's default coding-assistant context so the model only
//! sees what kb sent.
//!
//! ## Claude routing invariant
//!
//! Anthropic's subscription terms forbid running Claude models through
//! third-party harnesses. The router (`crate::router`) refuses to pick
//! `Backend::Pi` for a Claude-family model, but [`PiAdapter::new`] also
//! enforces this defensively — a future router refactor that lets a
//! claude-* slug slip through cannot ship via this adapter. The check
//! returns [`LlmAdapterError::Other`] with a message pointing the user
//! at the `[llm.runners.claude]` block in `kb.toml`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

use crate::adapter::{
    AnswerQuestionRequest, AnswerQuestionResponse, DetectContradictionsRequest,
    DetectContradictionsResponse, ExtractConceptsRequest, ExtractConceptsResponse,
    FilterConceptSuggestionsRequest, GenerateConceptBodyRequest, GenerateConceptBodyResponse,
    GenerateConceptFromCandidateRequest, GenerateConceptFromCandidateResponse,
    GenerateSlidesRequest, GenerateSlidesResponse, ImputeGapRequest, ImputeGapResponse, LlmAdapter,
    LlmAdapterError, MergeConceptCandidatesRequest, MergeConceptCandidatesResponse,
    RunHealthCheckRequest, RunHealthCheckResponse, SummarizeDocumentRequest,
    SummarizeDocumentResponse, parse_detect_contradictions_json, parse_extract_concepts_json,
    parse_filter_concept_suggestions_json, parse_generate_concept_from_candidate_json,
    parse_impute_gap_json, parse_merge_concept_candidates_json,
};
use crate::opencode::{
    format_aliases_for_prompt, format_candidate_snippets_for_prompt,
    format_contradiction_quotes_for_prompt, format_existing_categories_for_prompt,
    format_quotes_for_prompt, health_check_timeout, strip_plain_text_wrappers, unix_time_ms,
};
use crate::provenance::ProvenanceRecord;
use crate::router::is_claude_model;
use crate::subprocess::{SubprocessError, run_command_with_stdin};
use crate::templates::Template;

const HARNESS_NAME: &str = "pi";
const DEFAULT_TOOLS: &str = "read,grep,find,ls";

/// Configuration for the `pi` runner backend.
#[derive(Debug, Clone)]
pub struct PiConfig {
    /// Path to the `pi` binary (or just `"pi"` if on PATH).
    pub command: String,
    /// Model identifier. Pi accepts `provider/id` slugs (e.g.
    /// `"openai/gpt-5.4"`, `"google/gemini-2.5-pro"`); the
    /// optional [`Self::provider`] field is rarely needed when the
    /// model already carries its provider prefix.
    pub model: String,
    /// Optional explicit `--provider` override. Use this only when the
    /// model id is unprefixed and pi can't infer the provider on its
    /// own.
    pub provider: Option<String>,
    /// Comma-separated tool allowlist. Defaults to read-only tools so
    /// kb's compile-time LLM calls never accidentally mutate the
    /// filesystem the way an opencode write+edit-enabled agent could.
    pub tools: String,
    /// Optional thinking level: `off`, `minimal`, `low`, `medium`,
    /// `high`, `xhigh`.
    pub thinking: Option<String>,
    /// Per-call session directory. Defaults to `<root>/.kb/pi-sessions/`
    /// when [`Self::project_root`] is set; otherwise pi's own default
    /// (`~/.pi/agent/sessions/...`) takes over.
    pub session_dir: Option<PathBuf>,
    /// Subprocess timeout. Mirrors the opencode adapter's contract.
    pub timeout: Duration,
    /// Project root for template loading (and for the default
    /// session-dir resolution).
    pub project_root: Option<PathBuf>,
}

impl Default for PiConfig {
    fn default() -> Self {
        Self {
            command: HARNESS_NAME.to_string(),
            model: "openai/gpt-5.4".to_string(),
            provider: None,
            tools: DEFAULT_TOOLS.to_string(),
            thinking: None,
            session_dir: None,
            timeout: Duration::from_secs(900),
            project_root: None,
        }
    }
}

/// LLM adapter that delegates to the `pi` CLI.
#[derive(Debug)]
pub struct PiAdapter {
    config: PiConfig,
}

impl PiAdapter {
    /// Construct a `PiAdapter`.
    ///
    /// # Errors
    ///
    /// Returns [`LlmAdapterError::Other`] when `config.model` is a
    /// Claude-family identifier — Anthropic's subscription terms
    /// forbid running Claude models through third-party harnesses, and
    /// pi must never carry such a request even if a future router
    /// refactor would let one slip through.
    pub fn new(config: PiConfig) -> Result<Self, LlmAdapterError> {
        if is_claude_model(&config.model) {
            return Err(LlmAdapterError::Other(format!(
                "PiAdapter cannot run Claude-family model '{}': Claude models must route through \
                 the ClaudeCode backend (Anthropic subscription terms). Configure \
                 [llm.runners.claude] in kb.toml or pick a non-Claude model.",
                config.model
            )));
        }
        Ok(Self { config })
    }

    /// Build the full argv for one `pi --print` invocation. The user
    /// prompt is delivered via `@<path>` so the renderer can be as
    /// large as it wants without hitting `ARG_MAX`.
    fn build_argv(&self, prompt_path: &Path, system_prompt: &str) -> Vec<String> {
        let mut argv = vec![
            self.config.command.clone(),
            "--print".to_string(),
            "--no-extensions".to_string(),
            "--no-skills".to_string(),
            "--no-prompt-templates".to_string(),
            "--system-prompt".to_string(),
            system_prompt.to_string(),
        ];

        if let Some(provider) = self.config.provider.as_deref() {
            argv.push("--provider".to_string());
            argv.push(provider.to_string());
        }
        argv.push("--model".to_string());
        argv.push(self.config.model.clone());

        argv.push("--tools".to_string());
        argv.push(self.config.tools.clone());

        if let Some(level) = self.config.thinking.as_deref() {
            argv.push("--thinking".to_string());
            argv.push(level.to_string());
        }

        if let Some(dir) = self.resolved_session_dir() {
            argv.push("--session-dir".to_string());
            argv.push(dir.display().to_string());
        }
        argv.push("--no-session".to_string());

        // Positional `@<path>` MUST be last — pi parses positional
        // messages after all flags.
        argv.push(format!("@{}", prompt_path.display()));
        argv
    }

    /// Resolve the session directory. Explicit configs win; otherwise
    /// fall back to `<project_root>/.kb/pi-sessions` so kb's session
    /// state lives alongside its other state files.
    fn resolved_session_dir(&self) -> Option<PathBuf> {
        if let Some(dir) = &self.config.session_dir {
            return Some(dir.clone());
        }
        self.config
            .project_root
            .as_ref()
            .map(|root| root.join(".kb/pi-sessions"))
    }

    /// Render the prompt to a tempfile and run pi. Returns the raw
    /// stdout with any wrapping `` ```json `` / `` ``` `` fences
    /// stripped — kb's prompts ask the model for plain output but
    /// some providers (notably Claude through pi, though we forbid
    /// that route) wrap responses in code fences.
    fn run_prompt(&self, prompt: &str) -> Result<String, LlmAdapterError> {
        let (_temp_dir, prompt_path) = Self::write_prompt(prompt)?;
        let argv = self.build_argv(&prompt_path, "");

        let result = run_command_with_stdin(&argv, b"", &[], self.config.timeout);
        let output = result.map_err(|e| match e {
            SubprocessError::TimedOut { timeout, .. } => {
                LlmAdapterError::Timeout(format!("pi exceeded timeout of {timeout:?}"))
            }
            SubprocessError::Other(err) => {
                LlmAdapterError::Transport(format!("failed to invoke pi: {err}"))
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
                "pi exited with error: {details}"
            )));
        }

        Ok(strip_plain_text_wrappers(&output.stdout))
    }

    /// Write `prompt` to a fresh tempfile and return the directory + path.
    /// The directory's RAII guard must be held for the lifetime of the
    /// subprocess; dropping it removes the tempfile.
    fn write_prompt(prompt: &str) -> Result<(TempDir, PathBuf), LlmAdapterError> {
        let dir = tempfile::Builder::new()
            .prefix("kb-pi-")
            .tempdir()
            .map_err(|e| LlmAdapterError::Other(format!("create pi prompt dir: {e}")))?;
        let path = dir.path().join("prompt.md");
        std::fs::write(&path, prompt)
            .map_err(|e| LlmAdapterError::Other(format!("write pi prompt file: {e}")))?;
        Ok((dir, path))
    }
}

impl LlmAdapter for PiAdapter {
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
            harness: HARNESS_NAME.to_string(),
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
                .map_or_else(|| "no limit".to_string(), |v| v.to_string()),
        );

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render extract concepts template: {err}"))
        })?;

        let started_at = unix_time_ms()?;
        let raw = self.run_prompt(&rendered.content)?;
        let response = parse_extract_concepts_json(&raw)?;
        let ended_at = unix_time_ms()?;

        let provenance = ProvenanceRecord {
            harness: HARNESS_NAME.to_string(),
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
            harness: HARNESS_NAME.to_string(),
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
            harness: HARNESS_NAME.to_string(),
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
            harness: HARNESS_NAME.to_string(),
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
        // Pi has no per-call tool override the way opencode does for
        // webfetch — kb's `tools` field is the master allowlist. Either
        // the user has enabled the right tools globally or the impute
        // call won't browse external sources. That's strictly less
        // capable than opencode for impute_gap, but kb still produces
        // useful output from local snippets alone.
        let template =
            Template::load("impute_gap.md", self.config.project_root.as_deref()).map_err(|err| {
                LlmAdapterError::Other(format!("load impute_gap template: {err}"))
            })?;

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
        let raw = self.run_prompt(&rendered.content)?;
        let response = parse_impute_gap_json(&raw)?;
        let ended_at = unix_time_ms()?;

        let provenance = ProvenanceRecord {
            harness: HARNESS_NAME.to_string(),
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
        // bn-1s8y: pi does not expose opencode's `-f <path>` image
        // attachment mechanism via its CLI flags. For multimodal asks
        // through pi the rendered prompt must reference images by
        // path; pi providers that support images will load them
        // server-side. Falling back to text-only is acceptable for the
        // first pass; a follow-up bone can wire `--attach` if pi
        // adds it.
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
        context.insert("conversation".to_string(), request.conversation.clone());

        let rendered = template
            .render(&context)
            .map_err(|err| LlmAdapterError::Other(format!("render ask template: {err}")))?;

        let started_at = unix_time_ms()?;
        let answer = self.run_prompt(&rendered.content)?;
        let ended_at = unix_time_ms()?;

        let provenance = ProvenanceRecord {
            harness: HARNESS_NAME.to_string(),
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
            harness: HARNESS_NAME.to_string(),
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

    fn filter_concept_suggestions(
        &self,
        request: &FilterConceptSuggestionsRequest,
    ) -> Result<Vec<String>, LlmAdapterError> {
        if request.candidates.is_empty() {
            return Ok(Vec::new());
        }

        let template = Template::load(
            "filter_concept_suggestions.md",
            self.config.project_root.as_deref(),
        )
        .map_err(|err| {
            LlmAdapterError::Other(format!(
                "load filter_concept_suggestions template: {err}"
            ))
        })?;

        let candidates_json = serde_json::to_string_pretty(&request.candidates).map_err(|err| {
            LlmAdapterError::Other(format!("serialize concept-suggestion candidates: {err}"))
        })?;

        let mut context = HashMap::new();
        context.insert("candidates_json".to_string(), candidates_json);

        let rendered = template.render(&context).map_err(|err| {
            LlmAdapterError::Other(format!("render filter_concept_suggestions template: {err}"))
        })?;

        let raw = self.run_prompt(&rendered.content)?;
        let parsed = parse_filter_concept_suggestions_json(&raw)?;
        let known: std::collections::BTreeSet<&str> =
            request.candidates.iter().map(|c| c.slug.as_str()).collect();
        let kept: Vec<String> = parsed
            .accepted
            .into_iter()
            .filter(|slug| known.contains(slug.as_str()))
            .collect();
        Ok(kept)
    }

    fn caption_image(
        &self,
        path: &Path,
        prompt: &str,
    ) -> Result<(String, ProvenanceRecord), LlmAdapterError> {
        // Pi lacks a direct image-attachment flag in the form opencode
        // exposes (`-f <path>`). For now caption_image goes through pi
        // with the image path embedded in the rendered prompt; image-
        // capable providers will load it from the path. When pi adds
        // first-class attachment support we'll wire it here.
        let started_at = unix_time_ms()?;
        let prompt_with_image = format!("{prompt}\n\nImage path: {}", path.display());
        let caption = self.run_prompt(&prompt_with_image)?;
        let ended_at = unix_time_ms()?;

        let provenance = ProvenanceRecord {
            harness: HARNESS_NAME.to_string(),
            harness_version: None,
            model: self.config.model.clone(),
            prompt_template_name: "caption_image".to_string(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::hash_bytes(prompt.as_bytes()),
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: None,
            cost_estimate: None,
        };

        Ok((caption.trim().to_string(), provenance))
    }

    fn generate_slides(
        &self,
        _request: GenerateSlidesRequest,
    ) -> Result<(GenerateSlidesResponse, ProvenanceRecord), LlmAdapterError> {
        Err(LlmAdapterError::Other(
            "pi adapter generate_slides is not implemented yet".to_string(),
        ))
    }

    fn run_health_check(
        &self,
        _request: RunHealthCheckRequest,
    ) -> Result<(RunHealthCheckResponse, ProvenanceRecord), LlmAdapterError> {
        let timeout = health_check_timeout(self.config.timeout);

        let started_at = unix_time_ms()?;
        let (_dir, prompt_path) = Self::write_prompt("respond with OK")?;
        let mut argv = self.build_argv(&prompt_path, "");
        // Override the call-site timeout for the health check.
        let result = run_command_with_stdin(&argv, b"", &[], timeout);
        let ended_at = unix_time_ms()?;
        // argv unused after invocation but kept mutable for future
        // health-check-specific tweaks (e.g. stripped tool list).
        argv.clear();

        let output = result.map_err(|e| match e {
            SubprocessError::TimedOut { timeout, .. } => LlmAdapterError::Timeout(format!(
                "pi health check exceeded timeout of {}s — verify pi is on PATH and configured \
                 with valid provider credentials.",
                timeout.as_secs()
            )),
            SubprocessError::Other(err) => {
                let msg = err.to_string();
                if msg.contains("not found") || msg.contains("No such file or directory") {
                    LlmAdapterError::Transport(format!("pi not on PATH: {msg}"))
                } else {
                    LlmAdapterError::Transport(format!("failed to invoke pi: {msg}"))
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
            return Err(LlmAdapterError::Other(format!(
                "pi health check failed: {details}"
            )));
        }

        let provenance = ProvenanceRecord {
            harness: HARNESS_NAME.to_string(),
            harness_version: None,
            model: self.config.model.clone(),
            prompt_template_name: "run_health_check".to_string(),
            prompt_template_hash: kb_core::Hash::from([0u8; 32]),
            prompt_render_hash: kb_core::Hash::from([0u8; 32]),
            started_at,
            ended_at,
            latency_ms: ended_at.saturating_sub(started_at),
            retries: 0,
            tokens: None,
            cost_estimate: None,
        };

        let response = RunHealthCheckResponse {
            status: "ok".to_string(),
            details: Some(strip_plain_text_wrappers(&output.stdout)),
        };

        Ok((response, provenance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PiConfig {
        PiConfig {
            command: "pi".to_string(),
            model: "openai/gpt-5.4".to_string(),
            provider: None,
            tools: DEFAULT_TOOLS.to_string(),
            thinking: None,
            session_dir: None,
            timeout: Duration::from_secs(60),
            project_root: None,
        }
    }

    #[test]
    fn build_argv_includes_required_pi_flags_in_order() {
        let adapter = PiAdapter::new(cfg()).expect("non-claude model");
        let argv = adapter.build_argv(Path::new("/tmp/p.md"), "sys");

        // Mandatory shape: command first, prompt @file last.
        assert_eq!(argv[0], "pi");
        assert!(
            argv.last()
                .expect("argv has at least the command and prompt")
                .starts_with('@')
        );

        // The three suppression flags must be present so pi does not
        // mix in its default coding-assistant prompt context.
        assert!(argv.iter().any(|a| a == "--print"));
        assert!(argv.iter().any(|a| a == "--no-extensions"));
        assert!(argv.iter().any(|a| a == "--no-skills"));
        assert!(argv.iter().any(|a| a == "--no-prompt-templates"));

        // Model + tools always pass through.
        assert!(argv.iter().any(|a| a == "--model"));
        assert!(argv.iter().any(|a| a == "openai/gpt-5.4"));
        assert!(argv.iter().any(|a| a == "--tools"));
        assert!(argv.iter().any(|a| a == DEFAULT_TOOLS));

        // No-session by default for ephemeral compile-time calls.
        assert!(argv.iter().any(|a| a == "--no-session"));
    }

    #[test]
    fn build_argv_passes_provider_and_thinking_when_set() {
        let adapter = PiAdapter::new(PiConfig {
            provider: Some("google".to_string()),
            thinking: Some("medium".to_string()),
            ..cfg()
        })
        .expect("non-claude");
        let argv = adapter.build_argv(Path::new("/tmp/p.md"), "");

        let provider_idx = argv
            .iter()
            .position(|a| a == "--provider")
            .expect("--provider present when configured");
        assert_eq!(argv[provider_idx + 1], "google");
        let thinking_idx = argv
            .iter()
            .position(|a| a == "--thinking")
            .expect("--thinking present when configured");
        assert_eq!(argv[thinking_idx + 1], "medium");
    }

    #[test]
    fn build_argv_resolves_session_dir_under_project_root() {
        let adapter = PiAdapter::new(PiConfig {
            project_root: Some(PathBuf::from("/work/kb")),
            ..cfg()
        })
        .expect("non-claude");
        let argv = adapter.build_argv(Path::new("/tmp/p.md"), "");
        let dir_idx = argv
            .iter()
            .position(|a| a == "--session-dir")
            .expect("--session-dir present when project_root is set");
        assert_eq!(argv[dir_idx + 1], "/work/kb/.kb/pi-sessions");
    }

    #[test]
    fn build_argv_omits_session_dir_when_no_project_root_or_explicit_dir() {
        let adapter = PiAdapter::new(cfg()).expect("non-claude");
        let argv = adapter.build_argv(Path::new("/tmp/p.md"), "");
        assert!(!argv.iter().any(|a| a == "--session-dir"));
    }

    #[test]
    fn new_refuses_claude_family_models() {
        for model in [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "anthropic/claude-haiku-3",
            "claude/sonnet-4",
        ] {
            let err = PiAdapter::new(PiConfig {
                model: model.to_string(),
                ..cfg()
            })
            .expect_err("must refuse claude-family model");
            let msg = format!("{err:#}");
            assert!(
                msg.contains("Claude") || msg.contains("claude"),
                "claude-routing error must mention Claude — got: {msg}"
            );
            assert!(
                msg.to_lowercase().contains("claude") && msg.contains(model),
                "error must include the rejected model id — got: {msg}"
            );
        }
    }

    #[test]
    fn new_accepts_non_claude_models() {
        for model in [
            "openai/gpt-5.4",
            "google/gemini-2.5-pro",
            "groq/llama-3.3",
            "openrouter/some-model",
        ] {
            PiAdapter::new(PiConfig {
                model: model.to_string(),
                ..cfg()
            })
            .unwrap_or_else(|err| panic!("non-claude model '{model}' rejected: {err:#}"));
        }
    }

    #[test]
    fn write_prompt_creates_tempfile_with_contents() {
        let _adapter = PiAdapter::new(cfg()).expect("non-claude");
        let (dir, path) = PiAdapter::write_prompt("hello pi").expect("write tempfile");
        assert!(path.exists(), "prompt tempfile must exist");
        let body = std::fs::read_to_string(&path).expect("read tempfile");
        assert_eq!(body, "hello pi");
        // dir guard goes out of scope here; ensure the path is gone after.
        drop(dir);
        assert!(
            !path.exists(),
            "prompt tempfile must be removed when TempDir is dropped"
        );
    }
}
