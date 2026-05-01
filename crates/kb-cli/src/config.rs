#![allow(dead_code)]

use std::{collections::HashMap, env, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

/// Top-level CLI configuration parsed from `kb.toml`.
///
/// `Eq` is intentionally not derived: the `[retrieval]` section carries
/// `f32` thresholds which have no total ordering. `PartialEq` is preserved
/// for tests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub llm: LlmConfig,
    pub compile: CompileConfig,
    pub ask: AskConfig,
    pub lint: LintConfig,
    pub publish: PublishConfig,
    pub lock: LockConfig,
    pub ingest: IngestConfig,
    pub retrieval: RetrievalConfig,
    pub semantic: SemanticConfig,
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

        if let Ok(raw) = env::var("KB_SEMANTIC") {
            self.retrieval.semantic = parse_bool_var("KB_SEMANTIC", &raw)?;
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
                permission_mode: Some("default".to_string()),
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
    /// `[compile.captions]` — bn-2qda. Vision-LLM auto-captions for
    /// undescribed images.
    pub captions: CaptionsConfig,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            token_budget: 25_000,
            captions: CaptionsConfig::default(),
        }
    }
}

/// `[compile.captions]` section of `kb.toml`. bn-2qda.
///
/// Controls the vision-LLM auto-caption pass that runs during `kb compile`,
/// generating 2-3 sentence descriptions for images whose alt-text is empty,
/// generic, or matches the filename stem. Captions are cached by content
/// hash (`<root>/.kb/cache/captions/<hash>.txt`) so a re-compile pays the
/// LLM cost zero times.
///
/// Default: `enabled = true`, runner = "claude" (uses
/// `caption_image` on the configured Claude adapter), model =
/// "claude-haiku-4-5", `allow_paths = ["wiki/", "sources/", "raw/"]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct CaptionsConfig {
    /// Master switch. When `false` the captions pass is a no-op.
    pub enabled: bool,
    /// Which `[llm.runners.<name>]` to use for vision input. The named
    /// runner must implement `LlmAdapter::caption_image` — claude does, the
    /// opencode adapter does too via the `-f <path>` flag, and any other
    /// adapter falls back to "vision unsupported" which the pass logs and
    /// skips.
    pub runner: String,
    /// Vision-capable model id passed through to the runner. Forwarded
    /// verbatim — kb does not validate against a hardcoded model list.
    pub model: String,
    /// Path prefixes (relative to the KB root) the pass is allowed to read
    /// images from. Privacy guard: an image outside any prefix is skipped
    /// silently. Empty list means "no images allowed", failing closed
    /// against a misconfigured TOML.
    pub allow_paths: Vec<String>,
}

impl Default for CaptionsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            runner: "claude".to_string(),
            model: "claude-haiku-4-5".to_string(),
            allow_paths: vec![
                "wiki/".to_string(),
                "sources/".to_string(),
                "raw/".to_string(),
            ],
        }
    }
}

impl CaptionsConfig {
    /// Translate the parsed TOML into the runtime
    /// [`kb_compile::captions::CaptionsConfig`] the captions pass consumes.
    /// The two types are kept separate so the pipeline doesn't depend on
    /// `toml`/`serde` plumbing.
    #[must_use]
    pub fn to_pipeline_config(&self) -> kb_compile::captions::CaptionsConfig {
        kb_compile::captions::CaptionsConfig {
            enabled: self.enabled,
            allow_paths: self.allow_paths.clone(),
            prompt: kb_compile::captions::DEFAULT_PROMPT.to_string(),
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
            // bn-2cs2: default to `auto` so the model can produce supporting
            // artifacts (Excalidraw, charts) when the question warrants
            // them, without requiring the caller to pre-guess the right
            // `--format`. Users who need the cheap text-only path pass
            // `--format=md` explicitly.
            artifact_default_format: "auto".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LintConfig {
    pub require_citations: bool,
    pub missing_citations_level: String,
    pub missing_concepts: LintMissingConceptsConfig,
    pub contradictions: LintContradictionsConfig,
    pub citation_verification: LintCitationVerificationConfig,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            require_citations: true,
            missing_citations_level: "warn".to_string(),
            missing_concepts: LintMissingConceptsConfig::default(),
            contradictions: LintContradictionsConfig::default(),
            citation_verification: LintCitationVerificationConfig::default(),
        }
    }
}

/// `[lint.missing_concepts]` section of `kb.toml`. Controls the
/// `missing_concepts` lint check: whether it runs by default and the
/// frequency thresholds used to filter out noise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LintMissingConceptsConfig {
    pub enabled: bool,
    pub min_sources: usize,
    pub min_mentions: usize,
}

impl Default for LintMissingConceptsConfig {
    fn default() -> Self {
        // Mirrors `kb_lint::MissingConceptsConfig::default()`. Kept in sync
        // so the TOML defaults and the library defaults agree.
        Self {
            enabled: true,
            min_sources: 3,
            min_mentions: 5,
        }
    }
}

/// `[lint.contradictions]` section of `kb.toml`. Controls the LLM-powered
/// cross-source contradiction check added in bn-3axp.
///
/// Default is `enabled = false` because each enabled concept triggers one
/// LLM round-trip; this check is opt-in per pass via
/// `kb lint --check contradictions` regardless of the TOML value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LintContradictionsConfig {
    pub enabled: bool,
    pub min_sources: usize,
}

impl Default for LintContradictionsConfig {
    fn default() -> Self {
        // Mirrors `kb_lint::ContradictionsConfig::default()`.
        Self {
            enabled: false,
            min_sources: 2,
        }
    }
}

/// `[lint.citation_verification]` section of `kb.toml`. Controls the
/// `unverified-quote` check added in bn-166d which inspects quoted
/// spans in compiled concept pages and verifies they appear in the
/// cited source.
///
/// Default: `enabled = true, level = "warn", fuzz_per_100_chars = 1`
/// — lined up with the bone spec so existing builds pick up the
/// check automatically without their compile pass turning red.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct LintCitationVerificationConfig {
    pub enabled: bool,
    pub level: String,
    pub fuzz_per_100_chars: u32,
}

impl Default for LintCitationVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: "warn".to_string(),
            fuzz_per_100_chars: kb_core::DEFAULT_FUZZ_PER_100_CHARS,
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
        // 10 minutes. A single LLM round-trip can take tens of seconds, and
        // `kb compile` over a large corpus can hold the lock for minutes, so a
        // short timeout (e.g. 5s) causes overlapping commands to fail
        // immediately. 10 minutes is long enough to ride out realistic
        // workloads without hanging forever on a truly stuck holder.
        Self {
            timeout_ms: 600_000,
        }
    }
}

/// `[retrieval]` section of `kb.toml`. Tunes the hybrid retrieval pipeline
/// added in bn-3qsj.
///
/// `semantic = false` short-circuits the semantic tier, producing
/// lexical-only results indistinguishable from the pre-hybrid behavior.
/// The `KB_SEMANTIC=0` env var overrides this at runtime — useful when
/// debugging fusion regressions without touching the file.
///
/// `min_semantic_score` and `min_semantic_top_score_no_lexical` are
/// optional (bn-2xbd) — when unset, the floors come from the active
/// semantic backend's calibrated defaults (see
/// [`kb_query::SemanticBackendKind::default_min_semantic_score`]). Hash and
/// `MiniLM` live in different cosine score regimes, so a single global floor
/// either lets noise through (low) or filters real matches (high).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct RetrievalConfig {
    pub semantic: bool,
    pub rrf_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_semantic_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_semantic_top_score_no_lexical: Option<f32>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            semantic: true,
            rrf_k: kb_query::RRF_K,
            // None = pick the active backend's calibrated floor at
            // hybrid-options build time (see `to_hybrid_options`).
            min_semantic_score: None,
            min_semantic_top_score_no_lexical: None,
        }
    }
}

impl RetrievalConfig {
    /// Translate the parsed config into the runtime [`HybridOptions`] the
    /// `kb-query` crate consumes, picking backend-tuned floors when the
    /// user hasn't pinned `min_semantic_score` /
    /// `min_semantic_top_score_no_lexical` in `kb.toml`. bn-2xbd.
    #[must_use]
    pub const fn to_hybrid_options(
        &self,
        backend_kind: kb_query::SemanticBackendKind,
    ) -> kb_query::HybridOptions {
        let min_score = match self.min_semantic_score {
            Some(value) => value,
            None => backend_kind.default_min_semantic_score(),
        };
        let min_top = match self.min_semantic_top_score_no_lexical {
            Some(value) => value,
            None => backend_kind.default_min_semantic_top_score_no_lexical(),
        };
        kb_query::HybridOptions {
            semantic_enabled: self.semantic,
            rrf_k: self.rrf_k,
            min_semantic_score: min_score,
            min_semantic_top_score_no_lexical: min_top,
        }
    }
}

/// `[semantic]` section of `kb.toml`. Selects the embedding backend that
/// `kb compile` writes into the corpus and that `kb ask` / `kb search` use
/// to embed queries. bn-1rww.
///
/// `backend = "hash"` (default) keeps the always-available zero-dependency
/// hash n-gram backend. `backend = "minilm"` switches to the ONNX-backed
/// MiniLM-L6-v2 sentence transformer (384-dim) and only works when the
/// binary was built with `--features semantic-ort`.
///
/// `model_path` and `tokenizer_path` are honored only by the `minilm`
/// backend and override the default cache convention
/// (`~/.cache/kb/models/`); leave them unset to let kb auto-download from
/// Hugging Face on first compile.
///
/// Backend changes are detected by `kb compile`'s embedding-sync pass —
/// the on-disk vector store is wiped and rebuilt automatically when the
/// stored `backend_id` no longer matches the active backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct SemanticConfig {
    pub backend: SemanticBackendKindConfig,
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

/// Mirror of [`kb_query::SemanticBackendKind`] kept here so `kb-query`
/// stays free of `serde`/`toml` plumbing. The two enums must agree —
/// see [`SemanticConfig::to_backend_config`].
///
/// `Default` is platform-aware (bn-2xbd): `Minilm` on non-Windows,
/// `Hash` on Windows. Mirrors [`kb_query::SemanticBackendKind`] so a
/// freshly-init'd `kb.toml` matches what the binary will actually use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticBackendKindConfig {
    #[cfg_attr(target_os = "windows", default)]
    Hash,
    #[cfg_attr(not(target_os = "windows"), default)]
    Minilm,
}

impl From<SemanticBackendKindConfig> for kb_query::SemanticBackendKind {
    fn from(value: SemanticBackendKindConfig) -> Self {
        match value {
            SemanticBackendKindConfig::Hash => Self::Hash,
            SemanticBackendKindConfig::Minilm => Self::Minilm,
        }
    }
}

impl SemanticConfig {
    /// Translate the parsed config into the runtime
    /// [`kb_query::SemanticBackendConfig`] the `kb-query` factory consumes.
    /// `~`-prefixed paths are expanded against the user's home directory so
    /// agent users don't have to hand-type absolute paths.
    #[must_use]
    pub fn to_backend_config(&self) -> kb_query::SemanticBackendConfig {
        kb_query::SemanticBackendConfig {
            kind: self.backend.into(),
            model_path: self.model_path.as_deref().map(expand_user_path),
            tokenizer_path: self.tokenizer_path.as_deref().map(expand_user_path),
        }
    }
}

fn expand_user_path(raw: &str) -> std::path::PathBuf {
    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let mut path = std::path::PathBuf::from(home);
            path.push(stripped);
            return path;
        }
    }
    std::path::PathBuf::from(raw)
}

/// `[ingest]` section of `kb.toml`. Bundles per-stage knobs: the markitdown
/// preprocessor and the OCR fallback that fires when markitdown's extraction
/// is empty/short for a scanned PDF.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct IngestConfig {
    pub markitdown: MarkitdownConfig,
    pub ocr: OcrConfig,
}

/// `[ingest.markitdown]` section of `kb.toml`. Controls whether `kb ingest`
/// routes non-markdown source files through Microsoft's `markitdown` CLI
/// before feeding them into the normalize pipeline. See
/// `kb_ingest::MarkitdownOptions` for the runtime counterpart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct MarkitdownConfig {
    /// Master switch. When false, no file is ever routed through markitdown
    /// regardless of its extension. Defaults on — the conversion is a no-op
    /// if the binary isn't on PATH (we warn once and skip).
    pub enabled: bool,
    /// Executable name (or absolute path) invoked per file. Defaults to the
    /// string `"markitdown"` which relies on `$PATH` lookup.
    pub command: String,
    /// Extensions (no leading dot) that should always go through markitdown
    /// when `enabled == true`.
    pub extensions: Vec<String>,
    /// Extensions advertised as opt-in (slower or cloud-backed conversions
    /// like OCR and audio transcription). Stored so users can uncomment
    /// entries from a rendered config without having to remember the list;
    /// ingest-time dispatch only consults `extensions`.
    pub optional_extensions: Vec<String>,
}

impl Default for MarkitdownConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "markitdown".to_string(),
            extensions: kb_ingest::markitdown_default_extensions(),
            optional_extensions: kb_ingest::markitdown_default_optional_extensions(),
        }
    }
}

impl MarkitdownConfig {
    /// Convert the CLI-facing config into the runtime options the
    /// `kb-ingest` crate consumes. The two types are kept separate so
    /// `kb-ingest` doesn't depend on `toml` / `serde` config plumbing.
    #[must_use]
    pub fn to_options(&self) -> kb_ingest::MarkitdownOptions {
        kb_ingest::MarkitdownOptions {
            enabled: self.enabled,
            command: self.command.clone(),
            extensions: self
                .extensions
                .iter()
                .map(|e| e.to_ascii_lowercase())
                .collect(),
            optional_extensions: self
                .optional_extensions
                .iter()
                .map(|e| e.to_ascii_lowercase())
                .collect(),
        }
    }
}

/// `[ingest.ocr]` section of `kb.toml`. Controls the OCR fallback that
/// kicks in when markitdown returns empty/short text for a PDF (typically
/// scan-only PDFs: photographed pages, image-only legal docs). See
/// `kb_ingest::OcrOptions` for the runtime counterpart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[serde(deny_unknown_fields)]
pub struct OcrConfig {
    /// Master switch. Defaults on — the conversion is a no-op (warn + skip)
    /// if either `pdftoppm` or `tesseract` is missing on PATH, so leaving
    /// it on is safe even on machines without OCR installed.
    pub enabled: bool,
    /// Tesseract executable. Whitespace-split into program + leading argv
    /// (so `"docker run --rm tesseract"` works as a wrapper).
    pub command: String,
    /// Pdftoppm executable. Whitespace-split, same convention as `command`.
    /// Exposed mainly so containerized / wrapped poppler installs can be
    /// pointed at without code changes.
    pub pdftoppm_command: String,
    /// Tesseract language code(s). Pass-through; tesseract accepts
    /// `+`-joined language codes (`eng+fra`) so we don't validate here.
    pub language: String,
    /// Trigger threshold (in chars). When markitdown's output trims to less
    /// than this many chars, we attempt OCR. `0` disables the threshold —
    /// only literally empty text triggers OCR.
    pub min_chars_threshold: usize,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            command: "tesseract".to_string(),
            pdftoppm_command: "pdftoppm".to_string(),
            language: "eng".to_string(),
            min_chars_threshold: kb_ingest::OCR_DEFAULT_MIN_CHARS_THRESHOLD,
        }
    }
}

impl OcrConfig {
    /// Translate the parsed config into the runtime `OcrOptions` the
    /// `kb-ingest` crate consumes.
    #[must_use]
    pub fn to_options(&self) -> kb_ingest::OcrOptions {
        kb_ingest::OcrOptions {
            enabled: self.enabled,
            command: self.command.clone(),
            pdftoppm_command: self.pdftoppm_command.clone(),
            language: self.language.clone(),
            min_chars_threshold: self.min_chars_threshold,
        }
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
    fn semantic_section_defaults_match_platform() {
        let cfg = Config::default();
        let expected_cfg_kind = if cfg!(target_os = "windows") {
            SemanticBackendKindConfig::Hash
        } else {
            SemanticBackendKindConfig::Minilm
        };
        let expected_runtime_kind = if cfg!(target_os = "windows") {
            kb_query::SemanticBackendKind::Hash
        } else {
            kb_query::SemanticBackendKind::Minilm
        };
        assert_eq!(cfg.semantic.backend, expected_cfg_kind);
        let backend = cfg.semantic.to_backend_config();
        assert_eq!(backend.kind, expected_runtime_kind);
    }

    #[test]
    fn semantic_minilm_with_paths_round_trips() -> Result<()> {
        let toml = r#"
[semantic]
backend = "minilm"
model_path = "~/.cache/kb/models/minilm-l6-v2-int8.onnx"
tokenizer_path = "~/.cache/kb/models/minilm-l6-v2-tokenizer.json"
"#;
        let parsed = Config::from_toml(toml)?;
        assert_eq!(parsed.semantic.backend, SemanticBackendKindConfig::Minilm);
        let backend = parsed.semantic.to_backend_config();
        assert_eq!(backend.kind, kb_query::SemanticBackendKind::Minilm);
        // ~/ should expand. We don't assert against $HOME because tests can
        // run on hosts where it's set to something exotic; just confirm the
        // tilde was stripped.
        assert!(
            backend
                .model_path
                .as_deref()
                .is_some_and(|p| !p.to_string_lossy().starts_with('~'))
        );
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

    // bn-2qda: `[compile.captions]` section.

    #[test]
    fn captions_section_defaults_are_sane() {
        let cfg = Config::default();
        assert!(cfg.compile.captions.enabled);
        assert_eq!(cfg.compile.captions.runner, "claude");
        assert_eq!(cfg.compile.captions.model, "claude-haiku-4-5");
        assert_eq!(
            cfg.compile.captions.allow_paths,
            vec![
                "wiki/".to_string(),
                "sources/".to_string(),
                "raw/".to_string()
            ],
        );
    }

    #[test]
    fn captions_section_round_trips_through_toml() -> Result<()> {
        let toml = r#"
[compile.captions]
enabled = true
runner = "claude"
model = "claude-sonnet-4-7"
allow_paths = ["wiki/", "private-sources/"]
"#;
        let parsed = Config::from_toml(toml)?;
        assert_eq!(parsed.compile.captions.model, "claude-sonnet-4-7");
        assert_eq!(
            parsed.compile.captions.allow_paths,
            vec!["wiki/".to_string(), "private-sources/".to_string()]
        );
        Ok(())
    }

    #[test]
    fn captions_to_pipeline_config_propagates_allow_paths() {
        let cfg = CaptionsConfig {
            allow_paths: vec!["only-here/".to_string()],
            ..CaptionsConfig::default()
        };
        let pipeline_cfg = cfg.to_pipeline_config();
        assert!(pipeline_cfg.enabled);
        assert_eq!(pipeline_cfg.allow_paths, vec!["only-here/".to_string()]);
        // Prompt is propagated from the captions module's default — kept
        // out of the kb.toml schema so the prompt text doesn't sprawl into
        // user config.
        assert!(!pipeline_cfg.prompt.is_empty());
    }
}
