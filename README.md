# kb

A personal knowledge-base compiler. You feed it markdown sources, it feeds you back
a navigable wiki: per-source summaries, extracted concepts with backlinks, a
lexical search index, and a grounded-answer `ask` command. The wiki is plain
markdown — open it in Obsidian, push it to a static site, grep it from the
terminal.

kb is a single Rust binary. The LLM work happens through an existing subscription
agent — **pi** (preferred default), **opencode**, or **Claude Code**. No API
keys go in `kb.toml` or your shell.

## Why

Most personal notes sit as inert markdown: easy to write, painful to query. kb
treats your notes as a compile target. The pipeline is deterministic where it
can be (hashes, locks, incremental state) and lets the LLM do only the parts
that need judgment (summaries, concept extraction, grounded answers).

## Install

```sh
cargo install --locked --path crates/kb-cli   # or `just install`
```

Requires Rust 1.93+. Verify with `kb --version`.

You also need a runner agent on your PATH. `kb init` auto-detects which is
installed and picks one as the default — preference order is
**pi → opencode → claude**:

- [pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent) — preferred default. Designed
  for non-interactive concurrent use; handles kb's parallel compile cleanly.
- [opencode](https://github.com/sst/opencode) — supported alternative.
- [Claude Code](https://claude.com/product/claude-code) — required for any
  `claude-*` model (per Anthropic ToS, Claude models must route through the
  `claude` CLI). kb routes Claude-family slugs to it automatically when a
  `[llm.runners.claude]` section is configured.

Run `kb doctor` to verify the backend is reachable.

## Quickstart

```sh
mkdir ~/notes/kb && cd ~/notes/kb
kb init                         # scaffold kb.toml + dirs (auto-picks pi/opencode/claude)
kb doctor                       # verify LLM backend
kb ingest ~/notes/sources       # walks a dir of markdown / PDFs / audio / repos recursively
kb compile                      # parallel LLM passes (summaries + concepts + merge)

kb ask "how does X work?"       # grounded memo with citations
kb ask --session work "..."     # multi-turn session: subsequent turns reuse retrieval context
kb search "raft"                # hybrid lexical + semantic + structural ranking
kb eval run                     # score retrieval against evals/golden.toml
kb serve                        # local read-only web UI
```

Open `~/notes/kb/wiki/` in Obsidian or your editor and browse the generated
concept + source + question pages.

## The loop

```

     ┌────────────┐      ┌─────────────────┐      ┌──────────────┐
 md  │ kb ingest  │ raw/ │   kb compile    │ wiki/│   kb ask     │ answer
────>│            │─────>│  (LLM pipeline) │─────>│   kb chat    │──────>
     └────────────┘      └─────────────────┘      │   kb search  │
           ^                      │               └──────────────┘
           │                      v                       │
           │              review queue                    │
           │                      │                       │
           │                      v                       │
           │              kb review approve <─────────────┘
           │                      │
           │                      v
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
   merge (with parallel body generation), captions for undescribed images,
   concept-suggestion candidates from RAKE keyphrases, backlinks (both
   frontmatter-sourced and corpus mention-scanning with plural-aware
   matching), per-chunk semantic embeddings, a lexical search index, a
   citation-graph for structural retrieval, and wiki index pages. Each pass
   is hash-gated — nothing re-runs if inputs didn't change. Per-doc LLM
   calls fan out across a worker pool for ~50% wallclock reduction on
   small kbs.
3. **Ask** builds a retrieval plan via three-tier hybrid retrieval —
   lexical + semantic (MiniLM embeddings) + structural (citation-graph
   personalized PageRank) — fused with reciprocal rank fusion and
   optionally re-ranked by a cross-encoder when the tiers disagree on the
   top hit. Calls the LLM with the selected candidates, writes a grounded
   markdown memo under `outputs/questions/q-<id>/`. Citations point at
   real wiki pages and source-page anchors. Use `--session <id>` to
   thread multi-turn conversations.
4. **Review queue** collects things that need human judgment: duplicate
   concepts, promotion candidates, proposed merges. `kb review approve`
   applies them.
5. **Promote** files an interesting `kb ask` answer back into `wiki/questions/`
   so subsequent queries can cite it.

## Commands

```
kb init                      scaffold a new KB (auto-picks pi/opencode/claude)
kb doctor                    health check (LLM backend, config, templates)
kb ingest <path|url|repo>... add documents (markdown, text, URLs, PDFs, audio, git repos)
kb compile [--dry-run]       build the wiki (incremental, parallel LLM passes, progress bars on TTY)
kb status                    counts, stale inputs, recent jobs
kb ls                        tree-list non-hidden files known to this KB
kb ask [QUERY]               grounded Q&A (reedline editor if no arg on TTY)
kb ask --editor              compose in $VISUAL / $EDITOR / vi
kb ask --session <id>        multi-turn session (transcripts under .kb/sessions/<id>.json)
kb ask --format={md,marp,json}
kb ask --promote             queue the answer for wiki/questions/
kb session new|list|show     manage multi-turn ask sessions
kb chat                      interactive runner session with read-only KB agent
kb search <term>             hybrid retrieval (lexical + semantic + structural) with explainable scoring
kb inspect <id|path>         show metadata, citations, build records
kb resolve <kb-uri>          resolve kb:// artifact references for tools
kb lint [--strict]           broken-links, orphans, stale-revision, missing-citations, duplicate-concepts,
                             missing-concepts, drift, citation-verification (quote in source)
kb review list|show|approve|reject
kb forget <src-id>           remove a source (cascade: concepts, indexes, graph, lexical)
kb jobs list|prune           manage state/jobs/ manifests
kb eval run|list             golden Q/A retrieval-eval harness (P@K, MRR, nDCG@K)
kb publish <target>          copy the wiki tree (+ referenced image assets) elsewhere
kb serve                     local read-only web UI
kb migrate                   migrate a pre-`.kb/` layout into the current layout
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

`kb.toml` (written by `kb init` — exact contents depend on which runners
were detected on PATH):

```toml
[llm]
default_runner = "pi"                        # auto-picked: pi > opencode > claude
default_model  = "openai-codex/gpt-5.5"      # pi-shaped slug; matches the chosen runner

[llm.runners.pi]
command = "pi"
tools_read = true                            # read-only by default for compile/ask
timeout_seconds = 900

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

[semantic]
backend = "minilm"                           # ONNX MiniLM-L6 (default on Linux/macOS); falls back to "hash" on Windows

[semantic.rerank]
enabled = true                               # cross-encoder rerank, gated on lex/sem disagreement
top_k = 30
keep = 8

[semantic.structural]
enabled = true                               # citation-graph personalized PageRank tier

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

| Type                              | Status                                                                                 |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| Markdown (.md, .txt, .rst)        | ✅                                                                                     |
| Plain text                        | ✅                                                                                     |
| URLs                              | ✅ (text extracted)                                                                    |
| Images referenced from markdown   | ✅ (copied, vision-LLM auto-captioned during compile)                                  |
| PDF                               | ✅ (text extracted; OCR fallback via tesseract for scan-only PDFs; per-page citations) |
| Word, Excel, PowerPoint, etc.     | ✅ via [markitdown](https://github.com/microsoft/markitdown) preprocessing             |
| Git repos (`kb ingest <git-url>`) | ✅ (clones, walks docs, supports `--branch` / `--include` / `--exclude`)               |
| Audio (`.m4a`/`.mp3`/`.wav`/...)  | ✅ (whisper transcription + pyannote speaker diarization → kbtx transcripts)           |
| Standalone images                 | ❌ (binary rejection)                                                                  |

## LLM backends

kb talks to LLMs through existing subscription-based agents. No API keys go
in `kb.toml` or your shell.

- **pi** (preferred default): multi-provider non-interactive agent CLI.
  Used for any non-Claude model. Better behaviour under parallel
  invocation than opencode (no SQLite WAL race when multiple processes
  start simultaneously), which lets kb fan out per-doc LLM calls
  cleanly. Default model slug `openai-codex/gpt-5.5`.
- **opencode**: alternative runner for non-Claude models. Same
  capability surface as pi for kb's purposes; older kbs configured
  around it keep working unchanged. Default model slug `openai/gpt-5.4`.
- **Claude Code**: required routing for any Claude model (per Anthropic
  ToS). `kb ask --model claude-sonnet-4-6 "..."` routes through the
  `claude` CLI. Both the router (`crate::router`) and `PiAdapter::new`
  refuse to send Claude-family slugs to anything except this runner.

All three runners support read + (optionally) edit/write/bash tools;
kb's default configuration keeps them read-only for the ask/compile
paths and gives `kb chat` a read-only agent.

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
- Multimodal ask is text-only — image references in retrieved
  context flow through as paths, not pixels. Vision-LLM auto-captions
  produced at compile time go into the source page, so the model can
  read what an image shows even though it can't see it directly.
- No Obsidian plugin yet. The wiki tree is Obsidian-compatible but
  there's no editor integration. (`plugin-obsidian/` is a stub.)
- No chart/visualization output from queries (`--format=marp` exists
  for slide decks).

## Roadmap (high-value, not yet built)

- Obsidian plugin that wraps `kb` commands (ask, ingest, compile, promote)
- True multimodal retrieval so the LLM can see images, not just their captions
- `--format=chart` that asks the LLM to produce matplotlib via its bash tool
- Hierarchical concept index / category tree
- More aggressive `kb lint --impute` that uses the LLM's web tool to
  backfill gaps in concept pages

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
