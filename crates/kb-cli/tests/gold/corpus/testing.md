# Testing Strategy

The main risk in kb is not syntax errors but state drift and low-trust outputs. The test
suite is structured to catch those risks specifically.

## Test Types

### Unit Tests

Unit tests live alongside source code in `#[cfg(test)]` modules. They cover:
- Source ID stability (same file or URL always produces the same `src-` ID)
- Hash-driven stale detection (changed input → stale output, unchanged input → fresh)
- Slug and path generation (deterministic, collision-free)
- Citation extraction from artifact markdown

### Integration Tests

Integration tests live in `crates/kb-cli/tests/integration_tests.rs`. They use
`assert_cmd` to run the real `kb` binary against temp-directory KB roots, and `insta`
for snapshot assertions. Coverage includes:
- `kb init` creates the correct directory structure
- `kb ingest` handles local files, URLs (via HTTP mock), gitignore, and edge cases
- `kb compile` runs passes and updates the lexical index
- `kb ask` creates question records, retrieval plans, and artifacts
- `kb lint`, `kb status`, `kb inspect`, `kb review` produce correct JSON output
- Atomic write behavior (no partial files left on success)
- Interrupted-run recovery

### Snapshot Tests

Snapshot tests use `insta` to capture and compare the JSON envelope output of CLI
commands. Snapshots are stored in `tests/snapshots/`. Running `cargo insta review`
allows accepting updated snapshots.

### Gold Question Set

The gold question set is a small benchmark of ~20 questions with expected citation sets,
run against a real corpus of developer notes. Gold questions live in
`tests/gold/questions.toml` and the corresponding corpus lives in `tests/gold/corpus/`.

The gold harness (`tests/gold_harness.rs`) measures:
- **Retrieval coverage** — for each question, what fraction of the expected source
  documents appear in the retrieval plan candidates?
- **LLM citation coverage** — of the citations the LLM produced, what fraction refer
  to expected sources? (requires a real LLM backend)
- **Hallucination rate** — what fraction of citation references in the LLM response
  point to non-existent context items? (requires a real LLM backend)

Retrieval coverage runs in CI with a fake LLM. LLM citation metrics require a real
backend and are marked `#[ignore]`.

## Fixture Corpus

A small synthetic fixture corpus lives at `crates/kb-cli/tests/fixtures/corpus-tiny/`.
It contains edge-case files (unicode, anchors, images, long tokens) used in integration
tests that exercise ingest and compile behavior.

## Running Tests

```bash
just check   # clippy + compile tests (no run)
just test    # run all tests
cargo test --test gold_harness                    # retrieval coverage only
cargo test --test gold_harness -- --ignored       # add LLM citation metrics
GOLD_RECORD_BASELINE=1 cargo test --test gold_harness -- --ignored  # record baseline
```
