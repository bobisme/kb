# kb

A personal knowledge-base compiler. You feed it markdown sources, it feeds you back
a navigable wiki: per-source summaries, extracted concepts with backlinks, a
lexical search index, and a grounded-answer `ask` command. The wiki is plain
markdown — open it in Obsidian, push it to a static site, grep it from the
terminal.

kb is a single Rust binary. The LLM work happens through an existing subscription
agent — **opencode** (default) or **Claude Code**. No API keys in your shell.

## Why

Most personal notes sit as inert markdown: easy to write, painful to query. kb
treats your notes as a compile target. The pipeline is deterministic where it
can be (hashes, locks, incremental state) and lets the LLM do only the parts
that need judgment (summaries, concept extraction, grounded answers).

## Install

```sh
cargo install --locked --path crates/kb-cli
```

Requires Rust 1.93+. Verify with `kb --version`.

You also need one of:

- [opencode](https://github.com/sst/opencode) on your PATH — default, routes `openai/gpt-5.4` and compatible models
- [Claude Code](https://claude.com/product/claude-code) on your PATH — required for any `claude-*` model

Run `kb doctor` to verify the backend is reachable.

## Quickstart

```sh
mkdir ~/notes/kb && cd ~/notes/kb
kb init                         # scaffold kb.toml + dirs
kb doctor                       # verify LLM backend
kb ingest ~/notes/sources       # walks a dir of markdown recursively
kb compile                      # ~1-2 min per source (LLM summaries + concepts)

kb ask "how does X work?"       # grounded memo with citations
kb chat                         # interactive session against the wiki
kb search "raft"                # lexical, explainable ranking
```

Open `~/notes/kb/wiki/` in Obsidian or your editor and browse the generated
concept + source + question pages.

## The loop

```
     ┌────────────┐      ┌─────────────────┐      ┌──────────────┐
 md  │ kb ingest  │ raw/ │   kb compile    │ wiki/│   kb ask     │ answer
────▶│            │─────▶│  (LLM pipeline) │─────▶│   kb chat    │──────▶
     └────────────┘      └─────────────────┘      │   kb search  │
           ▲                      │                └──────────────┘
           │                      ▼                       │
           │              review queue                    │
           │                      │                       │
           │                      ▼                       │
           │              kb review approve ◀─────────────┘
           │                      │
           │                      ▼
           │              wiki/questions/
           │                      │
           └──────────────────────┘
                 ingest back
```

1. **Ingest** copies sources into `raw/inbox/<src>/<rev>/` and normalizes them
   into `normalized/<src>/source.md`. Content-addressable revision IDs mean
   re-ingesting unchanged files is a no-op; re-ingesting a modified file
   produces a new revision.
2. **Compile** runs LLM passes: source summaries, concept extraction, concept
   merge, backlinks (both frontmatter-sourced and corpus mention-scanning with
   plural-aware matching), a lexical search index, and wiki index pages. Each
   pass is hash-gated — nothing re-runs if inputs didn't change.
3. **Ask** builds a retrieval plan from the lexical index, calls the LLM with
   the selected candidates, writes a grounded markdown memo under
   `outputs/questions/q-<id>/`. Citations point at real wiki pages.
4. **Review queue** collects things that need human judgment: duplicate
   concepts, promotion candidates, proposed merges. `kb review approve`
   applies them.
5. **Promote** files an interesting `kb ask` answer back into `wiki/questions/`
   so subsequent queries can cite it.

## Commands

```
kb init                      scaffold a new KB
kb doctor                    health check (LLM backend, config, templates)
kb ingest <path|url>...      add documents
kb compile [--dry-run]       build the wiki (incremental, progress bars on TTY)
kb status                    counts, stale inputs, recent jobs
kb ask [QUERY]               grounded Q&A (reedline editor if no arg on TTY)
kb ask --editor              compose in $VISUAL / $EDITOR / vi
kb ask --format={md,marp,json}
kb ask --promote             queue the answer for wiki/questions/
kb chat                      interactive opencode session with read-only KB agent
kb search <term>             lexical search with explainable scoring
kb inspect <id|path>         show metadata, citations, build records
kb lint [--strict]           broken-links, orphans, stale-revision, missing-citations, duplicate-concepts
kb review list|show|approve|reject
kb forget <src-id>           remove a source (cascade: concepts, indexes, graph, lexical)
kb jobs list|prune           manage state/jobs/ manifests
kb publish <target>          copy the wiki tree (+ referenced image assets) elsewhere
```

Prefix matching is supported on `inspect` and `forget`: `kb inspect src-a7`
resolves if unique, errors with candidates listed if ambiguous.

## File layout

```
<kb root>/
├── kb.toml                      # config
├── raw/inbox/<src>/<rev>/       # original ingested files
├── normalized/<src>/            # normalized markdown + metadata + referenced image assets
├── wiki/
│   ├── index.md                 # top-level nav
│   ├── sources/<src>.md         # per-source summary page
│   ├── sources/index.md
│   ├── concepts/<slug>.md       # extracted concept pages with backlinks
│   ├── concepts/index.md
│   ├── questions/<slug>.md      # promoted Q&A answers
│   └── questions/index.md
├── outputs/questions/q-<id>/    # raw ask artifacts (answer.md/json, metadata.json, retrieval_plan.json)
├── reviews/                     # pending review items (concept merges, promotions, dups)
├── state/
│   ├── jobs/                    # per-command job manifests
│   ├── indexes/lexical.json     # search index
│   ├── build_records/           # per-pass provenance
│   ├── graph.json               # dependency graph
│   └── hashes.json              # incremental compile state
├── trash/                       # kb forget moves files here (not rm)
└── logs/
```

IDs are [terseid](https://crates.io/crates/terseid) base-36 with adaptive
length. A small KB has `src-a7x`, `rev-a6d99a`, `q-k2m` — grows as the corpus
grows.

## Configuration

`kb.toml` (written by `kb init`):

```toml
[llm]
default_runner = "opencode"
default_model  = "openai/gpt-5.4"

[llm.runners.opencode]
command = "opencode run"
tools_read = true
tools_write = true
tools_bash = true
timeout_seconds = 900

[llm.runners.claude]
command = "claude"
permission_mode = "default"
tools_read = true
tools_edit = true

[compile]
token_budget = 25000

[ask]
token_budget = 20000
artifact_default_format = "markdown"

[lint]
require_citations = true
missing_citations_level = "warn"

[publish.targets.obsidian]
path = "~/obsidian/vault/kb"
filter = "wiki/**"

[lock]
timeout_ms = 600000
```

## Ignoring files

kb honors `.gitignore` and `.kbignore` (kb-specific, not committed) when walking
ingest paths. Same syntax as gitignore, nested files in subdirs respected.

## Obsidian compatibility

The generated wiki is Obsidian-compatible out of the box:

- `[[wiki/concepts/foo]]` backlinks use the Obsidian convention
- Index page links use file-relative paths so they resolve inside Obsidian's viewer
- Image references in source summaries point at copies under
  `normalized/<src>/assets/`
- Frontmatter is standard YAML

Open your KB root as an Obsidian vault. No plugin required.
(An Obsidian plugin with native `ask`/`ingest`/`promote` commands is on the roadmap.)

## Images

Referenced images in source markdown (`![alt](./figures/diagram.png)`) are
copied into `normalized/<src>/assets/` on ingest and preserved through compile
and publish. The LLM currently sees image references as text — multimodal
consumption is planned but not implemented.

Standalone image ingestion (`kb ingest photo.png`) is rejected; images come in
through their referring markdown file.

## Supported inputs

| Type              | Status |
|-------------------|--------|
| Markdown (.md, .txt, .rst) | ✅ |
| Plain text        | ✅ |
| URLs              | ✅ (text extracted) |
| Images referenced from markdown | ✅ (copied, multimodal consumption planned) |
| PDF, Word, Excel  | planned via [markitdown](https://github.com/microsoft/markitdown) preprocessing |
| Git repos (`kb ingest <git-url>`) | planned |
| Standalone images | ❌ (binary rejection) |

## LLM backends

kb talks to LLMs through existing subscription-based agents. No API keys go
in `kb.toml` or your shell.

- **opencode** (default): any model opencode supports. Default is
  `openai/gpt-5.4`.
- **Claude Code**: required routing for any Claude model (per Anthropic ToS).
  `kb ask --model claude-sonnet-4-6 "..."` routes through the `claude` CLI.

Both runners support read + bash + (optionally) edit/write tools; kb's default
configuration keeps them read-only for the ask/compile paths and gives `kb
chat` a read-only agent.

## State, locking, recovery

kb is single-writer. Mutating commands (ingest, compile, ask, forget, review
approve) acquire an advisory file lock at `state/locks/root.lock`. Read-only
commands (status, search, inspect, lint) are always concurrent-safe.

Every command records a job manifest under `state/jobs/`. `kb status` and `kb
doctor` surface failed or interrupted runs. `kb jobs prune --failed
--older-than 7` to clean up. The stale-manifest reaper automatically clears
entries where the recorded pid is dead.

`kb forget src-X --force` moves everything associated with a source (the raw
copy, the normalized dir, its wiki source page, orphaned concepts it solely
sources, matching build records) into `trash/<src>-<timestamp>/`. Nothing is
truly deleted.

## Limitations

- Single-writer only — no distributed ops.
- No multimodal ask yet (image references pass through as text).
- No repo / PDF / Word ingestion yet. Markdown and text only.
- No Obsidian plugin. The wiki tree is Obsidian-compatible but there's no
  editor integration.
- No chart/visualization output from queries.
- Concept linting covers structural issues (orphans, stale revisions,
  duplicates) but not semantic ones (contradictions, missing articles,
  web-search imputation) — those are on the roadmap.

## Roadmap (high-value, not yet built)

- Obsidian plugin that wraps `kb` commands (ask, ingest, compile, promote)
- markitdown-backed preprocessing for PDF / Word / Excel / images-with-OCR
- `kb ingest <git-url>` for repos (README, docs, source-code comments)
- Multimodal retrieval so the LLM can actually see images
- `--format=chart` that asks the LLM to produce matplotlib via its bash tool
- Concept categories / hierarchical index
- `kb lint --impute` that uses the LLM's web tool to backfill gaps
- Missing-concept and contradiction detection
- Optional `kb serve` local web UI

## Dev

Project uses `maw` for workspace management. See `ws/default/AGENTS.md`.

```sh
just check        # clippy + test build
just test         # full workspace tests
just install      # reinstall ~/.cargo/bin/kb (locked)
```

Tests in `crates/*/src/**` (unit) and `crates/kb-cli/tests/` (integration).
LLM-dependent integration tests use stub adapters; no network needed for `just
test`.

## License

TBD.
