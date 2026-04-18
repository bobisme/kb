# CLI Commands

The kb CLI feels like a build tool for knowledge. All commands accept `--root <path>`,
`--json` for machine-readable output, and `--dry-run` where relevant.

## Core Commands

### `kb init`
Initializes a KB root in the current directory (or `--root`). Writes a default `kb.toml`
and creates the standard directory structure (`raw/`, `normalized/`, `wiki/`, `outputs/`,
etc.).

### `kb ingest <path-or-url>...`
Imports sources into `raw/` and creates normalized representations. Supports local files,
directories (with gitignore), and URLs.

### `kb compile`
Runs incremental compilation passes over changed or dependent content. Updates source
summaries, source wiki pages, concept pages, backlinks, and the global lexical index.

### `kb ask <question>`
Plans retrieval, gathers relevant context, calls an LLM backend, and writes the answer
artifact to `outputs/questions/<id>/`. Supports `--format md|marp|json|png`,
`--model <name>`, and `--promote` to enqueue the artifact for wiki promotion.

### `kb lint`
Runs integrity checks: broken links, missing citations, duplicate concept titles,
weak summaries, orphaned pages, and stale derived artifacts. Severity levels are
configured per rule in `kb.toml`.

### `kb doctor`
Validates config, model backend access, filesystem permissions, prompt templates, and
external tool availability (opencode, claude). Reports actionable errors.

### `kb status`
Shows counts of sources, wiki pages, outputs, freshness, recent job runs, stale entries,
failed tasks, and changed inputs since the last compile.

### `kb publish <target>`
Syncs selected artifacts to a configured publish target (a project-local `notes/`
directory). Preserves backlinks to the main KB.

## Follow-on Commands

### `kb search <query>`
Fast lexical search over titles, aliases, summaries, and headings. Returns candidate
source and wiki IDs.

### `kb inspect <id>`
Prints metadata, dependency edges, citations, generating job records, and current
freshness for any tracked entity.

### `kb review`
Lists pending promotions, concept merges, alias cleanup items, and failed review items.
Allows the user to approve or reject each item.

## JSON Output

All commands support `--json`. Output follows a stable envelope schema:
```json
{"schema_version": 1, "command": "<name>", "data": {...}}
```
This schema is stable across releases to support agent harnesses and scripting.
