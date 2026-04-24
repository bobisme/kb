# Semantic retrieval for kb

Design notes for adding hybrid lexical + semantic retrieval. Pattern adapted
from `~/src/bones`'s `bones-search` + `bones-sqlite-vec` crates. Frankensearch
was evaluated and rejected as a dependency (asupersync runtime mismatch,
nightly toolchain, disproportionate scale for kb's per-vault corpus size).

## Problem

kb's retrieval is BM25-style lexical-only on tokens. It misses queries that
don't share vocabulary with the corpus — exactly what RAG should be best at.
Concrete failure mode (pass 16): asking about a "diagram" worked only because
the question contained the literal source-side word "USB". A paraphrased
query would have missed.

## Goal

Hybrid lexical + semantic retrieval with graceful degradation. Single model
tier to start. RRF fusion. sqlite-vec for indexed KNN, with an in-process
cosine fallback when the extension isn't loaded.

## Non-goals (v1)

- Cross-encoder reranking
- Two-tier (fast → quality) progressive delivery
- ANN / approximate KNN
- ML-quality embeddings — start with hash-embed; ORT/MiniLM is Phase 2
- Structural graph signal
- Streaming `kb ask` results

## Architecture

### New crate: `kb-sqlite-vec`

FFI shim. Same shape as `bones-sqlite-vec`:

- `register_auto_extension() -> Result<(), String>`
- Uses `OnceLock` to register exactly once per process via
  `sqlite3_auto_extension`
- Opt-out env var `KB_SQLITE_VEC_AUTO=0`
- Deps: `rusqlite`, `sqlite-vec`

Called once at process start (kb-cli `main`) before any DB connection.

### Module: `kb-query/src/semantic/`

Three files:

- `model.rs` — `EmbeddingBackend` trait + `HashEmbedBackend` (always
  available, zero deps)
- `embed.rs` — incremental embedding pipeline, schema management,
  content-hash dedup
- `search.rs` — KNN lookup with sqlite-vec fast path + Rust-side cosine
  fallback

### Module: `kb-query/src/hybrid.rs`

RRF-fuse lexical + semantic ranked lists. Public entry:

```rust
pub fn hybrid_search(
    root: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<HybridResult>>
```

`HybridResult` carries both `lexical_rank` and `semantic_rank` for
explanation in `kb search` and `--reasons` output.

### Storage

SQLite database at `.kb/state/embeddings.db`:

```sql
CREATE TABLE item_embeddings (
    item_id TEXT PRIMARY KEY,    -- e.g. "wiki/sources/src-1wz-foo.md"
    content_hash TEXT NOT NULL,  -- blake3 of body+backend_id
    embedding_json TEXT NOT NULL -- JSON array of f32 (sqlite-vec reads via vec_f32())
);

CREATE TABLE semantic_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    backend_id TEXT NOT NULL,           -- e.g. "hash-embed-256"
    embedding_dim INTEGER NOT NULL,
    last_compile_at_millis INTEGER NOT NULL DEFAULT 0
);
```

When `backend_id` changes (e.g. user enables Phase 2 ORT), we drop and rebuild
the table. `embedding_dim` is a sanity guard at read time — rows with the
wrong dim are silently skipped.

### What gets embedded

For v1: each compiled wiki page (sources + concepts) gets one embedding from
concatenating:

- Title (with weight: just included once near the top of the input string)
- Body without YAML frontmatter and managed regions
- Aliases (concepts only)

Total characters per embed capped at 4 KB; longer bodies truncated.

Note: this matches the existing `LexicalIndex` candidate set so the two
ranked lists fuse cleanly. Embedding the full `normalized/<id>/source.md` is
deliberately deferred to Phase 2 (would need a separate candidate-id space,
or a fan-out from the wiki source page back to its normalized body —
either is more work than the v1 win justifies).

### Sync triggers

After every `kb compile`, an `embedding_sync` pass:

- Reads all wiki page paths
- Computes content hash for each; if differs from stored row, re-embed
  and upsert
- For each stored embedding without a corresponding wiki page, delete

After `kb forget`, the cascade drops the source's wiki page → next compile's
sync drops the embedding row. (Or eagerly delete in forget — see open
questions.)

### Hybrid search flow

```rust
fn hybrid_search(query, root, limit) -> Vec<HybridResult> {
  let lexical = LexicalIndex::load(root)?.search(query, limit);

  let semantic = if semantic_enabled(root)? {
    let model = HashEmbedBackend::new();
    let qvec = model.embed(query)?;
    knn_search(&db, &qvec, limit)?
  } else {
    Vec::new()
  };

  let semantic_filtered = filter_by_thresholds(
    semantic,
    /*lexical_empty=*/ lexical.is_empty(),
  );

  rrf_fuse(lexical, semantic_filtered, RRF_K)
}
```

Threshold guards (lifted from bones — well-tuned):

- `MIN_SEMANTIC_SCORE = 0.15` — drop low-confidence semantic hits
- `MIN_SEMANTIC_TOP_SCORE_NO_LEXICAL = 0.20` — even higher floor when lexical
  returned nothing (avoids garbage on out-of-corpus queries)
- `RRF_K = 60` — standard

### Hash-embed backend

- 256-dim feature-hashed character n-grams (3–5 chars), inclusive
- Plus whitespace-token hashing for word-level signal
- L2-normalized output
- FNV-1a hashing — fast, simple, good distribution

Quality is intentionally weak. Catches morphological similarity
("authentication" / "authn") and exact-word-rephrased queries. Will NOT
match conceptual paraphrases ("how does login work?" → "auth service
token flow"). That's Phase 2.

The backend ships in ~80 lines of pure Rust, zero new deps beyond what's
already in the workspace.

### Phase 2 (later, feature-flagged, NOT in this bone)

- `semantic-ort` feature: ONNX Runtime + MiniLM-L6-v2-int8 (~30 MB model).
  Production-quality conceptual matching.
- `semantic-model2vec` feature: model2vec backend for Windows-friendly /
  no-ONNX systems.
- Backend selection at runtime in a `SemanticModel::load()` factory.
- Auto-download model on first use, cache in `~/.cache/kb/`.
- Bumping `backend_id` triggers a one-shot re-embed of the corpus.

## Wiring

### `kb ask`

In `crates/kb-cli/src/main.rs::run_ask`, replace `LexicalIndex::search` with
`hybrid_search`. Same `RetrievalPlan` output shape. Add `reasons` entries
like `"semantic match (score 0.62)"` alongside the existing lexical reasons.

### `kb search`

Same change. Human output prints both lexical and semantic reasons. JSON
output gains `lexical_score`, `semantic_score`, `lexical_rank`,
`semantic_rank` fields.

### `kb serve /search`

Trickles through automatically — it calls `kb_query::LexicalIndex::search`
today; switch to `hybrid_search`.

### `kb compile`

Add `embedding_sync` as a final pass after `index_pages`. Idempotent — when
no page hashes have changed, the pass is essentially free (one DB roundtrip
per page to compare hashes).

### `kb status`

Show `semantic index: 32 embeddings, 0 stale` alongside the existing
lexical index summary.

## Configuration

`kb.toml`:

```toml
[retrieval]
semantic = true              # default true
rrf_k = 60                   # standard
min_semantic_score = 0.15
```

Env override: `KB_SEMANTIC=0` for one-command opt-out.

## Migration

None needed. First `kb compile` after the upgrade builds the embedding
store from scratch. Estimated ~20 ms per wiki page with hash-embed on a
modern CPU; a 100-page corpus rebuilds in ~2 s.

`kb migrate` does NOT need extension — the embedding store self-heals on
the next compile.

## Tests

### Unit

- Hash-embed: deterministic for same input; correct dimension; L2-normalized
- RRF fusion: known fixture rankings produce expected fused order
- Cosine fallback path returns identical top-k to sqlite-vec path on
  small fixtures
- Threshold filter: drops below-MIN, double-strict when lexical empty

### Integration

- `kb compile` on a fresh kb populates `embeddings.db`; second compile is
  no-op for unchanged pages
- `kb ask "paraphrased question"` returns the right candidate even when
  lexical misses (use a fixture corpus where the relevant page uses
  different vocabulary from the query)
- `kb forget src-X` followed by `kb compile` removes the embedding row
- `KB_SEMANTIC=0` makes `kb ask` lexical-only (semantic_score = 0.0
  on every result)
- Disabling sqlite-vec via `KB_SQLITE_VEC_AUTO=0` falls back to Rust
  cosine; results identical

## Acceptance criteria

- `kb-sqlite-vec` crate compiles, registers the extension at process start
- `kb-query/src/semantic/{model,embed,search}.rs` + `hybrid.rs` land
- `kb compile` populates `.kb/state/embeddings.db`; idempotent on re-run
- `kb ask` and `kb search` use hybrid retrieval; existing tests still pass
- Semantic-disabled mode (env or config) preserves the old lexical-only
  behavior bit-for-bit
- Threshold guards prevent garbage on out-of-corpus queries (verified by
  a fixture test that asks something unrelated to a 5-page corpus and
  asserts no semantic ranks survive the filter)
- `just check` passes

## Open decisions

1. **Eager forget vs lazy sync.** Eagerly delete the embedding row in
   `kb forget` (one less stale row between forget and next compile),
   or let the next compile prune? — Recommend eager. Cheap, simpler
   reasoning for users.
2. **Embedding store location.** `.kb/state/embeddings.db` (current plan)
   vs a separate `.kb/state/semantic/` subtree? — Recommend single
   `embeddings.db` file. Easy to back up, easy to delete-and-rebuild.
3. **Should we embed normalized source bodies in v1?** — No. Wiki page
   bodies match the existing `LexicalIndex` candidate set; mixing
   normalized + wiki creates two namespaces and complicates fusion.
   Phase 2 if signal proves insufficient.
4. **Threshold tunability.** Expose `min_semantic_score` in `kb.toml` (yes,
   per the example above) but not `min_top_score_no_lexical` —
   the latter is an internal safety knob, not a tuning surface.

## Why not frankensearch

Evaluated; rejected as a direct dependency:

- `asupersync` runtime, no Tokio guarantee. kb is Tokio top-to-bottom
- Pinned nightly toolchain
- 9 crates, FSVI quantization, SIMD top-k, ANN, cross-encoder rerank,
  bundled `potion-128M` + `MiniLM` models — disproportionate scale for
  kb's typical corpus (tens to a few thousand pages per vault)
- "MIT + OpenAI/Anthropic Rider" license is non-standard

Borrowed concepts: RRF formula and `K=60` constant; query-classification
idea (deferred); cross-encoder rerank (Phase 3 idea, not now).

## Why bones-shaped instead

bones uses sqlite + sqlite-vec at exactly kb's scale. Tokio-compatible
(rusqlite is sync but works fine via `tokio::task::spawn_blocking`). Stable
Rust. MIT. Two-crate footprint. We can borrow the schema, the threshold
constants, the FFI shim pattern, and the hash-embed fallback — none of
which would be reusable from frankensearch given the runtime mismatch.
