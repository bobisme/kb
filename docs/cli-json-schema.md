# kb CLI JSON Output Schema

All `kb` commands support `--json` for machine-readable output. Every response is
wrapped in a stable top-level envelope so consumers can parse any command uniformly.

## Envelope (schema_version: 1)

```json
{
  "schema_version": 1,
  "command": "<subcommand>",
  "data": { ... },
  "warnings": [],
  "errors": []
}
```

| Field | Type | Description |
|---|---|---|
| `schema_version` | integer | Always `1` for this schema; bumped on breaking changes. |
| `command` | string | Subcommand that produced this output (e.g. `"ingest"`, `"compile"`). |
| `data` | object \| array | Command-specific payload (see per-command shapes below). |
| `warnings` | string[] | Non-fatal advisory messages. Currently always `[]`. |
| `errors` | string[] | Fatal error details (non-zero exit). Currently always `[]`. |

## Per-command `data` shapes

### `kb compile --json`

```json
{
  "total_sources": 12,
  "stale_sources": 3,
  "build_records_emitted": 3,
  "dry_run": false
}
```

| Field | Type | Description |
|---|---|---|
| `total_sources` | integer | Number of source nodes visited. |
| `stale_sources` | integer | Sources rebuilt during this run. |
| `build_records_emitted` | integer | Build record files written. |
| `dry_run` | boolean | Whether the run was a dry-run (no writes). |

### `kb ingest --json`

```json
{
  "dry_run": false,
  "results": [
    {
      "input": "path/or/url",
      "source_kind": "file",
      "outcome": "new_source",
      "source_document_id": "source-document-...",
      "source_revision_id": "source-revision-...",
      "content_path": "raw/inbox/...",
      "metadata_path": "raw/inbox/....json"
    }
  ],
  "summary": {
    "total": 1,
    "new_sources": 1,
    "new_revisions": 0,
    "skipped": 0
  }
}
```

`outcome` values: `"new_source"`, `"new_revision"`, `"skipped"`.

### `kb ask --json`

Normal run:

```json
{
  "question_id": "q-<short-suffix>",
  "question_path": "outputs/questions/<id>/question.json",
  "artifact_path": "outputs/questions/<id>/answer.md",
  "requested_format": "md"
}
```

`--dry-run` mode (no LLM call, returns retrieval plan):

```json
{
  "query": "...",
  "token_budget": 20000,
  "estimated_tokens": 4200,
  "candidates": [
    {
      "id": "wiki/concepts/example.md",
      "score": 42,
      "estimated_tokens": 800,
      "reasons": ["title matched 'example'"]
    }
  ]
}
```

### `kb status --json`

```json
{
  "total_sources": 24,
  "stale_sources": 2,
  "wiki_page_count": 90,
  "last_compile_at_millis": 1777000000000,
  "normalized_source_count": 24,
  "sources": {
    "total": 24,
    "by_kind": {
      "file": 24
    }
  },
  "wiki_pages": 24,
  "concepts": 66,
  "stale_count": 2,
  "semantic_index": {
    "embeddings": 90,
    "stale": 0
  },
  "recent_jobs": [],
  "failed_jobs": [],
  "failed_jobs_total": 0,
  "interrupted_jobs": [],
  "interrupted_jobs_total": 0,
  "changed_inputs_not_compiled": [],
  "sources_with_missing_origin": []
}
```

Chief-facing freshness aliases are intentionally redundant with older fields:
`total_sources == normalized_source_count`, `stale_sources == stale_count`, and
`wiki_page_count == wiki_pages + concepts`. `last_compile_at_millis` is `null`
when no successful compile job has been recorded.

### `kb search --json`

`data` is an array of result objects (empty array when no results or no index):

```json
[
  {
    "id": "wiki/concepts/example.md",
    "title": "Example Concept",
    "score": 37,
    "reasons": ["title matched 'example'"]
  }
]
```

### `kb inspect --json`

```json
{
  "target": "wiki/index.md",
  "resolved_id": "wiki/index.md",
  "kind": "wiki_page",
  "freshness": "fresh",
  "metadata": {
    "file_path": "wiki/index.md",
    "exists_on_disk": true,
    "size_bytes": 1234,
    "modified_at_millis": 1700000000000
  },
  "graph": {
    "direct_inputs": ["wiki/sources/example.md"],
    "direct_outputs": [],
    "upstream": ["raw/inbox/example.md", "wiki/sources/example.md"],
    "downstream": []
  },
  "citations": ["- [[wiki/sources/example.md]]"],
  "build_records": [...],
  "generating_jobs": [...],
  "trace": null
}
```

`kind` values: `"wiki_page"`, `"normalized_document"`, `"source_revision"`,
`"artifact"`, `"question"`, `"build_record"`, `"job_run"`, `"entity"`.

`freshness` values: `"fresh"`, `"stale"`, `"missing"`, `"unknown"`.

### `kb resolve <kb-uri> --json`

`resolve` accepts `kb://` artifact references and returns the current target
metadata without requiring callers to read KB internals. Broken references are
represented in the payload with `broken: true` and still use exit code 0 so
doctor-style consumers can report every broken citation in one pass.

```json
{
  "uri": "kb://wiki/concepts/example.md",
  "target": "wiki/concepts/example.md",
  "stable_id": "concept:example",
  "current_path": "wiki/concepts/example.md",
  "title": "Example",
  "content_hash": "b3...",
  "freshness": "fresh",
  "broken": false,
  "broken_reason": null,
  "kind": "wiki_page"
}
```

`stable_id`, `current_path`, `title`, `content_hash`, and `kind` are nullable
when the reference is broken or the KB has no stable identity for the artifact.
`content_hash` is the current BLAKE3 hash of the resolved on-disk file.

### `kb lint --json`

```json
{
  "checks": [
    {
      "check": "broken-links",
      "issue_count": 2,
      "warning_count": 0,
      "error_count": 2,
      "issues": [
        {
          "kind": "broken_link",
          "severity": "error",
          "referring_page": "wiki/sources/page.md",
          "line": 3,
          "target": "wiki/missing.md",
          "message": "broken wiki link",
          "suggested_fix": "wiki/concepts/existing.md"
        }
      ]
    }
  ],
  "checks_ran": 1,
  "total_issue_count": 2,
  "warning_count": 0,
  "error_count": 2
}
```

### `kb doctor --json`

```json
{
  "checks": [
    {
      "name": "config",
      "status": "ok",
      "summary": "Parsed kb.toml successfully.",
      "remediation": null,
      "details": null
    }
  ],
  "status": "ok",
  "warning_count": 0,
  "error_count": 0,
  "exit_code": 0
}
```

`status` values: `"ok"`, `"warn"`, `"error"`.

`exit_code` values: `0` (ok), `1` (warnings), `2` (errors).

## Schema version policy

`schema_version` is the compatibility contract for chief-facing JSON. Breaking
changes to the envelope or any documented `data` field (rename, type change,
required field removal, or semantic change that invalidates old parsers) bump
`schema_version`.

Adding new optional fields is **not** a breaking change. Consumers should treat
unknown fields as ignorable and accept the current schema version plus N-1 where
they have an adapter. New fields that duplicate older fields as aliases must stay
consistent with the older field for the lifetime of the schema version.
