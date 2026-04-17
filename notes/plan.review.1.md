# Review of `notes/plan.md`

## Executive Summary

The plan already has a strong core thesis: local-first, file-based, provenance-heavy, and
explicitly not over-designed around hosted infrastructure. It also does a good job of
framing the KB as a build system instead of a chat transcript.

The biggest gaps were around operational trust. The original plan did not yet pin down how
identity and citations survive refetches, how retrieval remains debuggable at medium
corpus sizes, how concurrent/interrupted jobs avoid corrupting state, or how generated
artifacts get reviewed before becoming durable wiki facts. Those are the areas with the
highest product risk, because they determine whether the KB remains believable and usable
after the first demo.

## Proposed Changes

### High Impact, Low Effort Change #1: Add Explicit Success Criteria And Operating Principles

**Current State:**

The plan defines a clear goal and product shape, but it does not yet state how success is
measured beyond feature presence.

**Proposed Change:**

Add a short `Success criteria` section and a small set of operating principles that make
the plan optimize for daily usefulness rather than feature breadth.

**Rationale:**

Without explicit success criteria, the implementation can drift toward building commands
instead of proving that the research loop works on a real corpus. Operating principles also
help resolve later trade-offs, for example whether to prefer a hidden optimization or an
inspectable but simpler file-based approach.

**Benefits:**

- Keeps implementation decisions aligned with actual user value
- Makes MVP acceptance more objective
- Helps prevent feature creep

**Trade-offs:**

- Adds a small amount of plan text up front

**Implementation Notes:**

Keep the criteria concrete: real corpus size, citation quality, inspectability, and
graceful handling of uncertainty.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
 ## Goal
 
 Build a Rust CLI-first system that manages a personal, file-based knowledge base in `~/kb`.
 The tool should ingest source material, incrementally compile a markdown wiki, answer
 questions by producing new artifacts, and keep strong provenance so the system stays
 useful as it grows.
+
+## Success criteria
+
+- ingest and compile a real corpus of roughly 100-300 sources without collapsing into
+  full rebuilds
+- answer non-trivial questions into artifacts with clear citations
+- let the user inspect any page or artifact and see what produced it
+- degrade gracefully when evidence is weak instead of fabricating grounding
+
+## Operating principles
+
+- treat the system more like a compiler than a chat app
+- prefer inspectable files and explicit state over hidden automation
+- require explicit review boundaries before synthetic outputs become durable wiki facts
```

---

### High Impact, High Effort Change #2: Separate Logical Source Identity From Source Revisions

**Current State:**

The plan defines `SourceDocument`, `NormalizedDocument`, and `Citation`, but it does not
yet describe how IDs behave when a URL is refetched, a repo advances, or a file changes.

**Proposed Change:**

Introduce `SourceRevision` as a first-class entity, and require citations to point at
revision-level evidence instead of mutable sources.

**Rationale:**

This is the core trust boundary. If the system only cites logical sources like URLs or file
paths, citations silently drift as content changes. Once that happens, the KB becomes hard
to trust and impossible to debug.

**Benefits:**

- Makes provenance durable across refetches and rebuilds
- Enables meaningful stale detection and selective invalidation
- Prevents citations from pointing at moving targets

**Trade-offs:**

- More metadata and slightly more complex schemas

**Implementation Notes:**

Keep `SourceDocument` stable across refetches, and let `SourceRevision` track immutable
snapshots. Add locator rules for headings, page numbers, text spans, and asset references.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
 Suggested entities:
 
-`SourceDocument`: original item ingested from file, URL, repo, image, or dataset
-`NormalizedDocument`: extracted text/assets/metadata derived from a source
+`SourceDocument`: logical source identity for a file, URL, repo, image, or dataset
+`SourceRevision`: immutable fetched snapshot of a source at a point in time
+`NormalizedDocument`: extracted text/assets/metadata derived from a specific source revision
@@
+Identity and citation rules matter more than pretty file layout:
+
+- `SourceDocument` IDs stay stable across refetches
+- `SourceRevision` IDs change whenever fetched content changes materially
+- citations point at revision-level evidence, not floating URLs or mutable titles
+- source locators support headings, spans, page numbers, and asset references
```

---

### High Impact, Medium Effort Change #3: Add A Concrete Retrieval And Inspection Layer

**Current State:**

`kb ask` mentions retrieval planning, but the plan does not define the actual retrieval
mechanism or how a user debugs bad retrieval.

**Proposed Change:**

Specify a simple lexical index for titles, aliases, summaries, and headings, and add
inspection-oriented commands like `kb search` and `kb inspect`.

**Rationale:**

The plan correctly avoids jumping straight to a vector database, but it still needs a real
retrieval story. At the corpus sizes described in the Karpathy tweet, the difference
between "works in practice" and "mysteriously misses key sources" is often just whether the
retrieval layer is explicit and debuggable.

**Benefits:**

- Keeps the MVP lightweight without hand-waving retrieval
- Gives users and agents a way to understand why sources were selected
- Improves answer quality without requiring full RAG infrastructure

**Trade-offs:**

- Adds index maintenance and a couple more CLI commands

**Implementation Notes:**

Start with lexical/BM25-style ranking over summaries and headings. Save ranking reasons and
token budget in the `Question` record.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
 kb status
 kb publish <target>
+kb search <query>
+kb inspect <id>
@@
+## Retrieval and indexing design
+
+1. maintain a lexical index over titles, aliases, tags, summaries, and key headings
+2. rank candidates cheaply before reading full documents
+3. expand into full reads only for the top candidate set
+4. save the retrieval plan, ranking reasons, and token budget with each `Question`
```

---

### High Impact, Medium Effort Change #4: Add Job Locking, Atomic Writes, And Recovery Rules

**Current State:**

The plan describes incremental compilation and job state, but it does not define how the
system behaves under interruption or concurrent mutating commands.

**Proposed Change:**

Define a root lock for mutating commands, require per-job manifests, and use temporary
writes plus atomic rename for artifacts.

**Rationale:**

This tool lives on the filesystem and is designed to be agent-friendly. That means it will
eventually be run by multiple shells, scripts, or agents. Without explicit locking and
recovery rules, the first interrupted `kb compile` can leave the KB in a partially updated
state that is hard to reason about.

**Benefits:**

- Prevents state corruption during overlapping runs
- Makes failures diagnosable
- Supports reliable incremental rebuilds

**Trade-offs:**

- Requires some up-front engineering in file/state management

**Implementation Notes:**

This should be part of the early foundation, not a later hardening pass.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
   state/
     manifest.json
     hashes.json
     graph.json
+    indexes/
     jobs/
+    locks/
@@
+Write mutating commands so they can recover cleanly from interruption:
+
+- write new artifacts to temporary paths first, then atomically rename into place
+- keep per-job logs and status under `state/jobs/`
+- use a root lock for mutating commands
```

---

### High Impact, Medium Effort Change #5: Add A Review And Promotion Workflow For Generated Knowledge

**Current State:**

The plan says artifacts may be promoted back into `wiki/`, but the approval workflow is not
yet defined.

**Proposed Change:**

Add `ReviewItem` as a first-class entity and define a review queue for promotions, merges,
and canonicalization changes.

**Rationale:**

This is where the product can either stay trustworthy or drift into self-reinforcing noise.
If synthetic outputs become wiki facts without an explicit review boundary, the KB can start
compounding its own mistakes.

**Benefits:**

- Preserves trust as the wiki grows
- Creates a natural place for human oversight without forcing manual editing everywhere
- Supports future merge/canonicalization workflows

**Trade-offs:**

- Adds some operational friction to promotion

**Implementation Notes:**

Keep v1 simple: create a sidecar record with destination path, affected pages, citations,
and approval state. Reuse the same mechanism later for concept merges.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
+- `ReviewItem`: pending promotion, merge, or canonicalization change awaiting approval
@@
+## Review and promotion workflow
+
+1. `kb ask` writes the artifact and a sidecar metadata record
+2. if promotion is requested, create a `ReviewItem`
+3. `kb review` lets the user inspect and approve the change
+4. approved promotions are written through the managed-region contract
+5. rejected promotions remain in `outputs/` but are marked as rejected
```

---

### Medium Impact, Low Effort Change #6: Tighten MVP Scope Around Supported Source Types

**Current State:**

The architecture names repos, images, PDFs, and datasets early, but the MVP section does
not sharply separate "planned later" from "fully supported now".

**Proposed Change:**

State clearly that MVP fully supports local markdown/text and readable web articles, while
other source types remain explicit post-MVP work.

**Rationale:**

The architecture should remain extensible, but the first shipped loop needs depth more than
breadth. Partial support for many source types often creates more product confusion than
value.

**Benefits:**

- Makes MVP implementation more realistic
- Reduces shallow ingestion work
- Improves the odds of a polished first loop

**Trade-offs:**

- Some anticipated source types are intentionally delayed

**Implementation Notes:**

It is fine for the CLI to reject unsupported types cleanly in v1.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
 Ship when the tool can:
 
 - initialize a KB root
 - ingest local markdown files and web articles
@@
+The MVP should stay narrow on input types even if the architecture anticipates more:
+
+- fully support local markdown/text files and readable web articles
+- allow PDFs, repos, datasets, and multimodal sources to remain explicit post-MVP work
+- prefer a small number of well-supported source types over many shallowly supported ones
```

---

### Medium Impact, Medium Effort Change #7: Add Evaluation And Gold Questions

**Current State:**

The testing section focuses on functional correctness and regression safety, but not on the
quality of answers or retrieval decisions.

**Proposed Change:**

Add a small benchmark set of real questions against a real corpus, and track compile time,
incremental rebuild time, and grounding quality.

**Rationale:**

This project's biggest failures will often be plausible-but-weak answers, not crashes. The
plan should therefore include quality evaluation as part of hardening, not as an optional
afterthought.

**Benefits:**

- Creates feedback loops for prompt and retrieval improvements
- Helps detect regressions in answer quality
- Makes performance trade-offs more evidence-based

**Trade-offs:**

- Requires a small amount of manual dataset creation

**Implementation Notes:**

Use a mix of synthetic fixture tests and one private real-world corpus.

**Git-Diff:**

```diff
--- plan.md
+++ plan.md
@@
 Tests should cover:
 
 - source ID stability
 - hash-driven stale detection
@@
+- retrieval-plan quality on a small gold question set
+- selective invalidation when prompt templates or model settings change
@@
+### Phase 6: evaluation and hardening
+
+- create a small benchmark set of gold questions against a real corpus
+- measure compile time, incremental rebuild time, and answer grounding quality
+- tune prompt templates and invalidation rules based on observed failures
```
