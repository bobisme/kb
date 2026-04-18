# Provenance and Citations

Trust is the central product concern for kb. Provenance is built in from day one so
users can inspect exactly what produced any page or artifact.

## Provenance Requirements

Every wiki page must have frontmatter with:
- `id` — stable page identifier
- `type` — `source`, `concept`, or `question_answer`
- `title` — human-readable title
- `status` — freshness status (`fresh`, `stale`, `failed`, `needs_review`)
- `source_document_ids` — list of `src-` IDs for the sources this page was derived from
- `source_revision_ids` — list of `rev-` IDs pinning the exact fetched content
- `generated_at` — Unix timestamp in milliseconds
- `generated_by` — `<harness>/<model>` string
- `build_record_id` — ID of the `BuildRecord` that produced this page

Every generated artifact records the input set, prompt hash, model name, and timestamp.
Every compilation pass records whether content is extractive, synthetic, or mixed.

## BuildRecord

A `BuildRecord` links outputs back to their inputs:
- `id` — stable build record identifier
- `input_ids` — list of source/wiki page IDs that were read
- `output_ids` — list of wiki page IDs or artifact IDs produced
- `input_hashes` — content hashes of all inputs at build time
- `prompt_template_name` — the template file used
- `prompt_template_hash` — BLAKE3 hash of the template content
- `model` — model name string
- `harness` — harness name (`opencode`, `claude`)
- `created_at_millis` — timestamp

## Citation Anchors

Citations in generated wiki pages reference source documents at the revision level,
not floating URLs or mutable titles. A citation points to a specific `rev-` ID so the
evidence remains pinnable even after a source is re-fetched with new content.

For question artifacts, the citation manifest numbers (like `[1]`, `[2]`) resolve to
specific wiki source page IDs recorded in `metadata.json`.

## Inspectability

Provenance is inspectable from the CLI via `kb inspect <id>`, which prints the full
chain from any artifact or wiki page back to the source revisions and generating jobs.

## Lint Enforcement

`kb lint` checks for missing citation metadata and will warn or error (configurable)
when a synthesized page lacks source grounding. The missing-citations lint level is set
via `lint.missing_citations_level` in `kb.toml`: `warn` (default) or `error`.
