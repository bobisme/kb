# kb plan

## Goal

Build a Rust CLI-first system that manages a personal, file-based knowledge base in `~/kb`.
The tool should ingest source material, incrementally compile a markdown wiki, answer
questions by producing new artifacts, and keep strong provenance so the system stays
useful as it grows.

## Success criteria

The plan should optimize for a loop that is actually useful in daily research work, not
just technically interesting.

- ingest and compile a real corpus of roughly 100-300 sources without the workflow
  collapsing into full rebuilds
- answer non-trivial questions into artifacts that cite their source material clearly
- let the user inspect any page or artifact and see what inputs, prompts, models, and
  passes produced it
- degrade gracefully when evidence is weak by surfacing uncertainty instead of inventing
  unsupported grounding
- keep the resulting files understandable enough that a user can browse the KB directly
  in Obsidian or a normal editor without special tooling

## Product shape

The first version should be a local developer tool, not a hosted product and not an
Obsidian-first plugin. The core bets are:

- local-first, file-based, git-friendly workflow
- markdown and images as the primary human-readable artifacts
- deterministic pipelines where possible, LLM calls where they add real value
- reproducible outputs with explicit provenance
- agent-friendly CLI commands that can later be exposed through pi/opencode tools

## Operating principles

- treat the system more like a compiler than a chat app
- prefer inspectable files and explicit state over hidden automation
- keep read paths cheap and deterministic; spend model calls mainly on synthesis,
  extraction, and rewrite steps
- require explicit review boundaries before synthetic outputs become durable wiki facts
- make it easy to explain why a page exists, why it changed, and what would rebuild it

## Non-goals for v1

- no custom database
- no multi-user collaboration
- no cloud sync or hosted service
- no full Obsidian plugin as the primary interface
- no fine-tuning pipeline in the first release
- no attempt to make the LLM directly edit arbitrary old files without guardrails

## High-level architecture

Use a small Rust workspace with a clear split between stable logic and model-specific
execution.

Suggested crates:

- `crates/kb-core`: domain types, file layout, manifests, hashing, provenance,
  incremental dependency graph, artifact metadata
- `crates/kb-llm`: model adapter traits, prompt rendering, execution wrappers,
  retry logic, token/accounting hooks
- `crates/kb-ingest`: source importers for local files, URLs, repos, images, PDFs,
  and metadata normalization
- `crates/kb-compile`: wiki compilation passes, summarization, concept extraction,
  backlink generation, stale detection
- `crates/kb-query`: question planning, retrieval, artifact generation, answer sessions
- `crates/kb-lint`: integrity checks, citation validation, duplicate detection,
  broken-link checks, stale index checks
- `crates/kb-cli`: the `kb` binary

This can begin as a single crate and split later if velocity matters, but the internal
module boundaries should follow this shape from the start.

## KB scopes

A `kb` invocation operates on a single **KB root**, but that root can live in one of two
places:

- **Global KB** at `~/kb` — the default, used when no project KB is found.
- **Project KB** at `<project>/kb/` — lives inside a project like `~/chief/<name>/kb/`,
  sitting alongside other project content. Not hidden: Obsidian and normal editors need to
  open it directly.

Root discovery rules for any `kb` command other than `kb init`:

1. If `--root <path>` is passed, use it.
2. Otherwise walk up from `$PWD` looking for the nearest directory containing a
   `kb/kb.toml` file. If found, that `kb/` directory is the root.
3. Otherwise fall back to `~/kb` if it exists.
4. If none of the above resolve, fail with a clear error pointing to `kb init`.

`kb init` creates the root in the current directory (or `--root`) by writing `kb.toml`
and the standard subdirectories.

Global and project KBs are **independent** in v1. A project KB does not read from or fall
back to `~/kb`, and citations inside a project KB never reference the global KB. Cross-KB
references (shared concept pages, federated search) can be considered later but are out
of scope for the first release.

## Filesystem layout

Code root: `~/src/kb`.

A KB root (global or project) has this layout:

```text
<kb-root>/
  kb.toml
  raw/
    inbox/
    web/
    papers/
    repos/
    datasets/
    images/
  normalized/
    <doc-id>/
      source.md
      metadata.json
      assets/
  wiki/
    index.md
    concepts/
    sources/
    projects/
    timelines/
    people/
  outputs/
    questions/
    reports/
    slides/
    figures/
  reviews/
    promotions/
    merges/
  prompts/
  state/
    manifest.json
    hashes.json
    graph.json
    indexes/
    jobs/
    locks/
  cache/
  logs/
  trash/
```

Key rules:

- `raw/` is append-mostly and treated as immutable input
- `normalized/` contains cleaned, tool-friendly source representations
- `wiki/` contains compiled markdown knowledge artifacts
- `outputs/` contains question-driven artifacts that may later be promoted into `wiki/`
- `reviews/` contains machine-prepared items that need a human or explicit CLI approval
  step before promotion or merge
- `state/` contains machine-managed incremental build state and should be reproducible

Write mutating commands so they can recover cleanly from interruption:

- write new artifacts to temporary paths first, then atomically rename into place
- keep per-job logs and status under `state/jobs/`
- use a root lock for mutating commands so two concurrent `kb compile` runs do not
  corrupt manifests or partially overwrite outputs

## Core data model

Every input and generated artifact should have a stable ID and metadata record.

Suggested entities:

- `SourceDocument`: logical source identity for a file, URL, repo, image, or dataset
- `SourceRevision`: immutable fetched snapshot of a source at a point in time
- `NormalizedDocument`: extracted text/assets/metadata derived from a specific source revision
- `WikiPage`: compiled markdown page with section-level provenance
- `Concept`: canonical topic node with aliases and backlinks
- `Question`: user prompt plus execution context and retrieval plan
- `Artifact`: any generated output such as report, slide deck, figure, or answer note
- `Citation`: pointer from a claim or section to one or more source spans/pages/files
- `BuildRecord`: tracks which inputs produced which outputs and with which hashes
- `JobRun`: one execution of `compile`, `ask`, `lint`, or `publish` with logs and status
- `ReviewItem`: pending promotion, merge, or canonicalization change awaiting approval

Each record should include:

- stable ID
- created/updated timestamps
- source hashes
- model/tool version used
- prompt template or prompt hash
- dependency list
- output paths
- status (`fresh`, `stale`, `failed`, `needs_review`)

Identity and citation rules matter more than pretty file layout. The plan should make
these rules explicit:

- `SourceDocument` IDs should be stable across refetches of the same logical source
- `SourceRevision` IDs should change whenever fetched content changes materially
- citations should point at revision-level evidence, not floating URLs or mutable titles
- source locators should support at least heading anchors, line/character spans, page
  numbers, and asset references where applicable
- generated wiki pages can cite other wiki pages for navigation, but factual grounding
  should still resolve back to source revisions

## Managed markdown contract

Generated markdown should follow a strict file contract so humans and tools can both work
with it safely.

Suggested rules:

- every generated file has frontmatter with `id`, `type`, `title`, `status`,
  `source_document_ids`, `source_revision_ids`, `generated_at`, `generated_by`, and
  `build_record_id`
- generated sections use stable managed-region markers so rebuilds can replace the right
  content without clobbering unrelated text
- v1 should keep source pages and concept pages mostly or fully managed; manual notes can
  live separately until mixed-authoring rules are mature
- section IDs should be stable and machine-addressable so citations and backlinks do not
  break when headings are renamed
- page moves and title changes should preserve IDs and leave alias or redirect metadata
  behind where needed

## CLI surface

The CLI should feel like a build tool for knowledge.

Initial commands:

```text
kb init
kb ingest <path-or-url>...
kb compile
kb ask <question>
kb lint
kb doctor
kb status
kb publish <target>
```

Useful follow-on commands once the core loop exists:

```text
kb search <query>
kb inspect <id>
kb review
```

Recommended command details:

- `kb init`
  - initialize `~/kb`
  - write default `kb.toml`
  - create directory structure

- `kb ingest <path-or-url>...`
  - copy or register sources into `raw/`
  - fetch web pages and assets
  - optionally clone repos shallowly into `raw/repos/`
  - extract initial metadata and create source IDs

- `kb compile`
  - run incremental compilation passes over changed or dependent content
  - update summaries, source pages, concept pages, backlinks, and global indexes
  - mark stale pages when upstream inputs change

- `kb ask <question>`
  - plan a retrieval strategy
  - gather relevant wiki/source context
  - generate one or more artifacts into `outputs/`
  - optionally promote artifacts back into `wiki/` after review or explicit flag

- `kb lint`
  - detect broken links, missing citations, duplicate concepts, weak summaries,
    orphan pages, stale derived artifacts, and conflicting metadata

- `kb doctor`
  - validate config, model access, filesystem permissions, prompt templates,
    and external tools

- `kb status`
  - show counts, freshness, recent jobs, stale outputs, failed tasks, and changed inputs

- `kb publish <target>`
  - copy or sync selected artifacts into project-local `notes/` directories
  - preserve backlinks to the main KB when publishing out

- `kb search <query>`
  - run fast lexical search across titles, aliases, summaries, and selected body snippets
  - return candidate source IDs and wiki IDs for debugging retrieval

- `kb inspect <id>`
  - print metadata, dependency edges, citations, generating jobs, and current freshness

- `kb review`
  - list pending promotions, concept merges, alias merges, and failed review items

Useful flags to support early:

- `--root <path>`
- `--format md|marp|json|png`
- `--model <name>`
- `--since <rev-or-time>`
- `--dry-run`
- `--json`
- `--force`
- `--review`

## Incremental compilation design

This is the most important technical feature.

Compilation should work like a lightweight build system:

1. hash all relevant inputs
2. build a dependency graph from source docs to normalized docs to wiki pages to outputs
3. detect the minimal stale set when something changes
4. rerun only affected passes
5. record the build result and provenance

Suggested pass order:

1. ingest/normalize
2. extract metadata and source summaries
3. create/update source pages in `wiki/sources/`
4. extract candidate concepts and aliases
5. merge/update concept pages in `wiki/concepts/`
6. regenerate backlinks and index pages
7. run lint passes
8. optionally generate queued outputs

Important guardrails:

- never silently discard prior generated content without recording why
- preserve stable page IDs even if titles change
- separate source-derived facts from synthesized interpretation
- prefer overwrite-by-rebuild for generated sections instead of freeform append/edit
- allow pages to contain managed regions and manual regions later, but keep v1 mostly managed

Execution safety rules:

- only one mutating job should hold the KB root lock at a time
- every mutating command writes a `JobRun` manifest before changing outputs
- interrupted jobs should be detectable and recoverable on the next `compile` or `doctor`
- prompt-template changes should invalidate only the outputs that depend on those prompts

## Retrieval and indexing design

The project does not need full RAG infrastructure in v1, but it does need a concrete
retrieval story.

Use a simple layered approach:

1. maintain a lexical index over titles, aliases, tags, summaries, and key headings
2. rank candidates cheaply before reading full documents
3. expand into full source/wiki reads only for the top candidate set
4. save the retrieval plan, ranking reasons, and token budget with each `Question`

This keeps the system small while still giving the user a way to debug why a question read
certain files and ignored others.

## Question and artifact workflow

Questions should be first-class tracked objects, not ephemeral chats.

Proposed flow for `kb ask`:

1. create a `Question` record with timestamp and requested format
2. search indexes and summaries to identify candidate inputs
3. build a retrieval plan and save it
4. pull the most relevant source/wiki material
5. generate the requested artifact
6. write artifact plus metadata/provenance
7. optionally suggest follow-up questions or promotions into `wiki/`

Supported output types for early versions:

- markdown memo
- Marp slide deck
- structured JSON report
- matplotlib/plot data spec saved alongside an image

Later output types:

- HTML report
- graph visualization
- issue list / TODO export
- repo-local `notes/` sync bundles

## Review and promotion workflow

The plan should be explicit that generated output is not automatically trusted just because
it exists.

Suggested workflow:

1. `kb ask` writes the artifact and a sidecar metadata record
2. if promotion is requested, create a `ReviewItem` with the proposed destination,
   citations, and affected wiki pages
3. `kb review` or `kb publish --review` lets the user inspect and approve the change
4. approved promotions are written through the same managed-region contract as compile
5. rejected promotions remain in `outputs/` but are marked as rejected so the system does
   not keep re-suggesting them blindly

The same review queue can later handle concept merges, alias cleanup, and title
canonicalization.

## Provenance requirements

Trust is the central product problem, so provenance should be built in from day one.

Minimum provenance rules:

- every wiki page has frontmatter or sidecar metadata with source IDs
- every major claim section stores citations to source documents or source pages
- every generated artifact records the input set, prompt hash, model name, and timestamp
- every compilation pass records whether content is extractive, synthetic, or mixed
- lints should fail or warn when synthesized pages lose their source grounding

Nice v1.5 addition:

- section-level citation anchors or inline claim blocks with source references

Provenance should also be inspectable from the CLI, not only embedded in markdown. A user
should be able to run `kb inspect <id>` and see the full chain from output back to source
revisions and generating jobs.

## Config and model integration

Use a simple config file in `~/kb/kb.toml`.

Suggested config areas:

- data roots
- default model and fallback model
- token/cost limits
- enabled ingestion backends
- prompt template directory
- publish targets
- lint thresholds
- artifact defaults
- retry/backoff policy and timeouts
- per-command token or cost budgets
- review policy defaults for promotion or destructive rewrites

Model integration should be behind a trait so the rest of the app does not care which
agent harness actually executes a call.

Example trait responsibilities:

- `summarize_document`
- `extract_concepts`
- `merge_concept_candidates`
- `answer_question`
- `generate_slides`
- `run_health_check`

Backends should also expose enough metadata for provenance and debugging:

- provider/model name
- harness/runner version if available
- token usage and cost estimates (where the harness reports them)
- retry counts and final latency

### Execution backends

The project targets subscription-based agent harnesses, not direct API providers or local
model runners. The goal is to amortize existing agent subscriptions rather than pay
per-token API bills.

Approved harnesses:

- **opencode** (`opencode run`) — **default backend** for the first release.
- **Claude Code** for any Claude model. Mandatory: Anthropic's terms forbid running
  Claude models through third-party harnesses on subscription plans, so every Claude call
  must shell out to the `claude` CLI.
- **pi** for non-Claude models where it is a better fit.

Default model: `openai/gpt-5.4` via opencode. This matches the model used by the sibling
`chief` tool and lets the two share a mental model of "this is what opencode runs."

Explicitly out of scope for v1:

- direct Anthropic/OpenAI/etc. SDK or HTTP clients
- locally hosted model runners (llama.cpp, ollama, vLLM, etc.)

Routing rule: if the requested model family is Claude, invoke `claude`; otherwise invoke
`opencode run` (or `pi` when configured).

### opencode invocation contract

Mirror the contract already in production in `~/src/chief/ws/default/src/runner/`:

- Invoke as `opencode run --agent <name> [--session <id>] [--variant <v>] <prompt>`.
  Prompt is passed as a positional argument, not via stdin.
- Model is selected by writing a per-call `opencode.json` config file (tools, model,
  provider) and pointing opencode at it, rather than via CLI flags. See
  `chief/ws/default/src/runner/config.rs` for the JSON shape.
- Capture stdout; strip ANSI escapes and header lines starting with `>` before returning
  the response. See `chief/ws/default/src/runner/opencode.rs:121-136` for the parser.
- opencode does not return token usage or cost in stdout. Provenance records should note
  this gap rather than invent numbers; latency and retry counts can still be measured by
  the adapter.
- Each call is stateless by default. `--session <id>` is available if a future feature
  wants conversational continuity, but v1 should not rely on it.
- Timeouts are the adapter's responsibility: poll the child and `SIGKILL` on timeout
  (again, see chief's executor for a working pattern).

### Prompt templates

Store prompts as markdown files under `<kb-root>/prompts/`, with `{{variable}}`
placeholders rendered at call time. Follow chief's `runner/templates.rs` approach:

- a small regex-based renderer for `{{var}}` substitutions
- optional persona prefixes that get prepended with a `---` separator
- template hash becomes part of the `BuildRecord` so prompt edits invalidate the outputs
  that depended on them

Config (`kb.toml`) should specify, per task or default, which harness + model to use and
any harness-specific flags (e.g. Claude Code permission mode, opencode tool toggles for
read/write/edit/bash).

## Rust implementation notes

Suggested libraries:

- CLI: `clap`
- error handling: `anyhow`, `thiserror`
- serialization: `serde`, `serde_json`, `toml`
- hashing: `blake3`
- filesystem walking: `ignore`, `walkdir`
- markdown/frontmatter: `pulldown-cmark`, small custom frontmatter parser or serde-based approach
- async runtime if needed: `tokio`
- HTTP/fetching: `reqwest`
- HTML extraction: `readability`, `scraper`, or a small wrapper around external converters
- structured logs: `tracing`, `tracing-subscriber`
- tests: `insta`, `tempfile`, `assert_cmd`

Favor plain files over embedded databases until scaling pressure becomes real.

## Suggested repository structure

```text
~/src/kb/
  Cargo.toml
  crates/
    kb-core/
    kb-cli/
    kb-ingest/
    kb-compile/
    kb-query/
    kb-lint/
    kb-llm/
  notes/
    plan.md
  examples/
  tests/
```

## MVP scope

The MVP should prove that the workflow is real, not that every feature exists.

Ship when the tool can:

- initialize a KB root
- ingest local markdown files and web articles
- normalize them into stable source records
- compile source summaries and concept pages incrementally
- answer a question into a markdown file in `outputs/questions/`
- attach provenance metadata to every generated file
- lint for broken links and missing citations at a basic level

The MVP should stay narrow on input types even if the architecture anticipates more:

- fully support local markdown/text files and readable web articles
- allow PDFs, repos, datasets, and multimodal sources to remain explicit post-MVP work
- prefer a small number of well-supported source types over many shallowly supported ones

Target scale for the MVP:

- at least one real corpus, not only synthetic fixtures
- roughly 100k-500k words total indexed content
- compile and recompile fast enough that the user will actually keep the tool in their loop

Explicitly defer for later:

- repo ingestion with code-aware summarization
- image-heavy multimodal workflows
- advanced deduplication/ontology merging
- rich Obsidian plugin UI
- harness-specific plugins beyond a thin wrapper

## Implementation phases

### Phase 0: bootstrap

- create workspace and crate layout
- implement config loading and `kb init`
- define core structs and metadata schemas
- create filesystem helpers and path conventions
- define file locking, atomic write helpers, and `JobRun` skeletons

### Phase 1: ingest

- support local file ingest
- support URL ingest with HTML to markdown conversion
- create source IDs and metadata sidecars
- store raw assets and normalized markdown

### Phase 2: compile

- implement manifest/hashing/state tracking
- compile per-source summary pages
- generate wiki index pages
- generate simple concept pages from extracted tags/topics
- build the initial lexical index over titles, aliases, and summaries

### Phase 3: ask

- implement retrieval over summaries and wiki pages
- create question records and output directories
- generate markdown answer artifacts with provenance
- record retrieval plans and expose `kb search` / `kb inspect` for debugging

### Phase 4: lint and status

- add broken-link checks
- add missing-citation checks
- add duplicate title/concept warnings
- add freshness and stale reporting
- detect interrupted jobs and surface recovery guidance

### Phase 5: publish and integrations

- add a review queue for promotions and concept merges
- sync selected artifacts into repo-local `notes/`
- expose stable CLI JSON output for agent harnesses
- add optional thin Obsidian command integration later

### Phase 6: evaluation and hardening

- create a small benchmark set of gold questions against a real corpus
- measure compile time, incremental rebuild time, and answer grounding quality
- tune prompt templates and invalidation rules based on observed failures

## Testing strategy

The main risk is not syntax, it is state drift and low-trust outputs.

Tests should cover:

- source ID stability
- hash-driven stale detection
- reproducible path/layout generation
- provenance metadata presence on all generated artifacts
- compile pass behavior under changed inputs
- CLI snapshot tests for human-readable and JSON output
- end-to-end fixture tests with small synthetic corpora
- interrupted-run recovery and atomic-write behavior
- citation locator stability after title or heading rewrites
- retrieval-plan quality on a small gold question set
- selective invalidation when prompt templates or model settings change

Add a small golden test corpus under `tests/fixtures/` with a handful of markdown files,
HTML pages, and images so the incremental pipeline can be regression-tested.

Also keep one private real-world corpus for manual validation, because many of the product
risks here only show up once page counts, concept overlaps, and long-tail questions become
messy.

## Risks and mitigations

- provenance drift
  - mitigation: citations required on synthesized sections, lint aggressively
- over-coupling to a model provider
  - mitigation: isolate model adapter trait and provider config
- noisy concept explosion
  - mitigation: add canonicalization rules, aliases, and merge review mode
- repo bloat in `~/kb`
  - mitigation: separate raw/normalized/wiki/outputs/cache and support pruning
- hard-to-debug rebuild behavior
  - mitigation: explicit manifests, job logs, and `kb status`/`kb doctor`
- concurrent state corruption from overlapping runs
  - mitigation: root locks, atomic writes, and interrupted-job recovery
- weak answer quality that still looks polished
  - mitigation: saved retrieval plans, gold questions, provenance inspection, and review gates
- citation anchors drifting as pages evolve
  - mitigation: revision-level citations plus stable section IDs and locator validation

## Future extensions

- codebase-aware repo ingestion with language-specific summarizers
- graph exports for visualization
- scheduled health checks and auto-refresh jobs
- richer publish flows into multiple project `notes/` folders
- agent tool wrappers for pi/opencode
- optional Obsidian plugin for browse/trigger UX
- later evaluation of whether some artifacts should be promoted into training data

## Recommended next step

Start by implementing the narrowest viable path:

1. `kb init`
2. `kb ingest` for local markdown + URLs
3. `kb compile` for summaries + index pages
4. `kb ask` for markdown output with provenance
5. `kb lint` for broken links + missing citations

If that loop feels good on a real corpus in `~/kb`, then expand toward publishing,
repo-aware ingestion, and agent/editor integrations.
