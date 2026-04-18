# kb Architecture Overview

kb is a Rust CLI tool for managing a personal, file-based knowledge base. The codebase is
organized as a Cargo workspace with a clear split between stable domain logic and
model-specific execution.

## Crate Structure

The workspace contains seven crates:

- **kb-core** — domain types, file layout, manifests, hashing, provenance, incremental
  dependency graph, and artifact metadata. This is the shared foundation; no other crates
  depend on anything that imports kb-core.
- **kb-llm** — model adapter traits, prompt rendering, execution wrappers, retry logic,
  and token accounting hooks. Provides the `LlmAdapter` trait implemented by the opencode
  and claude backends.
- **kb-ingest** — source importers for local files and URLs, HTML-to-markdown conversion,
  metadata normalization, and raw asset storage.
- **kb-compile** — wiki compilation passes including source page generation, concept
  extraction, backlink generation, index page updates, and stale detection.
- **kb-query** — question planning, lexical retrieval, context assembly, citation
  manifests, artifact generation, and retrieval plan persistence.
- **kb-lint** — integrity checks for broken links, missing citations, duplicate concepts,
  and stale derived artifacts.
- **kb-cli** — the `kb` binary; wires the other crates into CLI subcommands.

## KB Root Types

kb supports two KB root locations:

- **Global KB** at `~/kb` — the default, used when no project KB is found walking up
  from the current directory.
- **Project KB** at `<project>/kb/` — lives inside a project directory alongside other
  project content.

Root discovery for every `kb` command except `kb init`:
1. If `--root <path>` is passed, use it.
2. Walk up from `$PWD` looking for a directory containing `kb/kb.toml`.
3. Fall back to `~/kb` if it exists.
4. Fail with an error directing the user to `kb init`.

Global and project KBs are independent in v1: no cross-KB references, no fallback reads.

## Filesystem Layout

A KB root has this layout:

```
<kb-root>/
  kb.toml
  raw/              # append-mostly, treated as immutable input
  normalized/       # cleaned, tool-friendly source representations
  wiki/             # compiled markdown knowledge artifacts
  outputs/          # question-driven artifacts
  reviews/          # items awaiting human approval
  prompts/          # prompt templates
  state/            # machine-managed build state (reproducible)
  cache/
  logs/
  trash/
```

## Concurrency Safety

Only one mutating job holds the KB root lock at a time. Every mutating command writes a
`JobRun` manifest before changing outputs. Interrupted jobs are detectable and recoverable
on the next `compile` or `doctor` run.
