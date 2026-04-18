# Incremental Compilation

`kb compile` runs incremental compilation passes over the KB, updating wiki pages and
the lexical index only for sources that have changed since the last run.

## How Stale Detection Works

Compilation works like a lightweight build system:

1. Hash all relevant inputs (normalized document content, prompt templates, config).
2. Build a dependency graph from source docs → normalized docs → wiki pages → outputs.
3. Compute the minimal stale set: any node whose input hashes differ from what was last
   recorded is stale, and all nodes that depend on it transitively are also stale.
4. Re-run only the affected passes in dependency order.
5. Record the build result and provenance in `state/manifest.json`.

The dependency graph is stored at `state/graph.json`. Content hashes are stored at
`state/hashes.json`.

## Compilation Passes

Passes run in this order:

1. **Ingest/normalize** — ensure normalized representations are up to date.
2. **Source summaries** — call the LLM to produce a summary and key topics for each
   stale normalized document.
3. **Source wiki pages** — write or update `wiki/sources/<src-id>.md` with the summary,
   key topics, and source metadata.
4. **Concept extraction** — extract candidate concept terms from source summaries and key
   topics.
5. **Concept pages** — create or update `wiki/concepts/<slug>.md` for each distinct
   concept, merging aliases.
6. **Backlinks and index** — regenerate the global index page and per-concept backlink
   sections.
7. **Lint passes** — run integrity checks and emit warnings or errors.

## Prompt Invalidation

When a prompt template under `prompts/` changes, only wiki pages that depended on that
template are re-generated. Other pages are unaffected. The template hash is part of each
`BuildRecord` so the dependency is tracked precisely.

## Guard Rails

- Never silently discard prior generated content without recording why.
- Preserve stable page IDs even when titles change.
- Prefer overwrite-by-rebuild for generated sections; manual sections are preserved via
  managed-region markers.
- One mutating job holds the root lock at a time.

## Compile Output

After a successful compile, `wiki/` contains up-to-date source pages and concept pages,
`state/indexes/lexical.json` contains an updated lexical search index, and
`state/manifest.json` records the job run.
