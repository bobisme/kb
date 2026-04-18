# LLM Backends

kb delegates all LLM calls to external harnesses. Direct API clients (Anthropic SDK,
OpenAI SDK) are explicitly out of scope — kb targets subscription-based agent harnesses
to amortize existing subscriptions rather than incur per-token API costs.

## Supported Backends

Two backends are supported:

### opencode (default)

The default backend for kb. Invoked as:
```
opencode run --agent <name> [--session <id>] <prompt>
```

The model is selected by writing a per-call `opencode.json` config file (specifying
tools, model, and provider) and pointing opencode at it. This config approach avoids
CLI flag proliferation and mirrors the pattern already in production in the `chief` tool.

opencode captures stdout; kb strips ANSI escapes and header lines beginning with `>`
before returning the model's response. Token usage and cost are not available in opencode
stdout; provenance records note this gap rather than inventing numbers.

### claude CLI

Used for any Claude model. Anthropic's terms forbid running Claude on subscription plans
through third-party harnesses, so every Claude call must go through the `claude` CLI.
Invoked as:
```
claude --json <prompt>
```
The JSON output schema wraps the model's response in a `result` field.

## Routing Rule

If the requested model family is Claude (model name starts with `claude`), kb invokes
the `claude` CLI. Otherwise, kb invokes `opencode run`.

The default model is `openai/gpt-5.4` via opencode.

## Invocation Contract

- Prompts are passed as positional arguments, not via stdin.
- Each call is stateless by default (no session continuity in v1).
- Timeouts are the adapter's responsibility; kb polls the child process and sends
  `SIGKILL` on timeout.
- Retry counts and final latency are recorded in the `ProvenanceRecord`; token usage
  is recorded only when the harness provides it.

## Prompt Templates

Prompts are stored as markdown files under `<kb-root>/prompts/` with `{{variable}}`
placeholders rendered at call time. The template hash is part of each `BuildRecord`
so prompt edits invalidate only the affected outputs.
