# Source Ingestion

`kb ingest` imports sources into the KB. Sources can be local files, URLs, or
directories. The ingest step creates stable identity records and stores raw content.

## Source Identity

Every source is assigned a stable `SourceDocument` ID of the form `src-XXXXXXXX` where
the hex suffix is the first 8 characters of a BLAKE3 hash derived from the source kind
and a normalized stable location:

- **Files**: the stable location is the canonicalized absolute path. Re-ingesting the
  same file always yields the same `src-` ID regardless of symlinks or relative path
  representations.
- **URLs**: the stable location is a normalized URL with lowercase scheme and host,
  sorted query parameters, and no fragment. Re-fetching the same logical URL always
  produces the same `src-` ID.

In addition to the document ID, each fetch produces a `SourceRevision` with an ID of
the form `rev-XXXXXXXX` derived from a BLAKE3 hash of the fetched content bytes.
Unchanged content reuses the same revision ID; changed content mints a new one.

In short: a `SourceDocument` ID identifies the stable source location, while a
`SourceRevision` ID identifies one specific fetched version of that source's content.

## Raw Storage Layout

Ingested sources are stored under `raw/`:

```
raw/
  inbox/    # local files dropped in manually
  web/      # fetched web pages and assets
  papers/   # PDFs
  repos/    # shallow git clones
  datasets/
  images/
```

Each ingested source also produces a sidecar `metadata.json` under
`normalized/<doc-id>/` with the source ID, revision ID, kind, stable location, and
fetch timestamp.

## Normalization

The ingest step also runs normalization to produce a tool-friendly markdown representation:

- HTML pages are converted to markdown via HTML extraction.
- PDFs produce extracted text markdown (post-MVP).
- Images are stored as-is with a minimal description sidecar.
- Local markdown files are copied with frontmatter extracted and validated.

The normalized output lives at `normalized/<doc-id>/source.md` alongside
`normalized/<doc-id>/metadata.json`.

## Gitignore Handling

When ingesting a directory, kb respects `.gitignore` files found within it. Files
excluded by `.gitignore` rules are skipped during ingest.
