# Configuration

kb is configured through a `kb.toml` file at the KB root (e.g., `~/kb/kb.toml`).

## Config Sections

### `[model]`
```toml
[model]
default = "openai/gpt-5.4"
fallback = "openai/gpt-4o"
```
Sets the default model for LLM calls and an optional fallback.

### `[ask]`
```toml
[ask]
artifact_default_format = "md"
token_budget = 20000
```
Controls the default output format for `kb ask` and the token budget for retrieval.

### `[ingest]`
```toml
[ingest]
respect_gitignore = true
```
Controls whether ingest respects `.gitignore` rules in source directories.

### `[compile]`
```toml
[compile]
summary_model = "openai/gpt-5.4"
concept_model = "openai/gpt-5.4"
```
Allows overriding the model used for specific compilation passes.

### `[lint]`
```toml
[lint]
missing_citations_level = "warn"
broken_links_level = "error"
```
Controls lint severity levels. Values: `"warn"` or `"error"`.

### `[publish]`
```toml
[[publish.targets]]
name = "chief-notes"
path = "../chief/notes"
include = ["wiki/concepts/*.md", "outputs/questions/**/*.md"]
```
Defines publish targets: named paths to sync selected KB artifacts into.

### `[retry]`
```toml
[retry]
max_retries = 3
base_delay_ms = 500
timeout_ms = 120000
```
Controls retry and timeout behavior for LLM backend calls.

## Config Discovery

`kb` always loads config from `kb.toml` at the resolved KB root. The `--root` flag
overrides the root discovery walk described in the architecture documentation.

## Defaults

All config fields have defaults so `kb init` produces a working `kb.toml` with just
the essential root structure. Users only need to override fields they want to change.
