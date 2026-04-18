# Managed Regions — Spec

## Decision

Generated markdown sections use HTML-comment fences that are invisible in Obsidian and
parseable by the rewriter:

```
<!-- kb:begin id=summary -->
...generated content...
<!-- kb:end id=summary -->
```

The `id` attribute holds a stable slug that identifies the section across rebuilds. Only
the content between matching begin/end fences is replaced on rewrite; all surrounding
text is left untouched.

## Section ID scheme

Section IDs are stable slugs derived from the section title **at first-assignment time**
only. Once a slug is written into a marker it is never re-derived from the heading text —
renaming a heading does not change the ID.

Slug derivation rule (`slug_from_title`):

1. Lowercase all alphabetic characters.
2. Replace every run of non-alphanumeric characters with a single hyphen.
3. Strip leading and trailing hyphens.

Examples: `"Executive Summary"` → `"executive-summary"`,
`"Background & Context"` → `"background-context"`.

IDs live inline in the markers themselves (no external registry required). This means
IDs survive title renames automatically: the marker stays intact, only the heading prose
above or below it changes.

## Fence grammar

```
begin-fence ::= "<!-- kb:begin id=" SLUG " -->"
end-fence   ::= "<!-- kb:end id=" SLUG " -->"
SLUG        ::= [a-z0-9][a-z0-9-]*[a-z0-9]  (or single [a-z0-9])
```

Nested regions are not supported in v1. The parser treats the first matching `kb:end` as
the close of its `kb:begin`.

## Rewriter contract

- `extract_managed_regions(text)` returns all regions in document order, each with its
  byte offsets and ID. Unknown content outside any region is not reported.
- `rewrite_managed_region(text, id, new_content)` replaces only the content between the
  matching fences and returns the full updated string. Returns `None` if the ID is not
  found (safe no-op).
- The caller is responsible for writing the result back to disk atomically.

## Citations

Citations may reference a specific managed section as `<page-id>#<section-id>`, e.g.
`wiki/concepts/rust.md#summary`. The section slug in the citation remains stable even if
the heading text changes.

## Manual content

Content outside managed regions is preserved verbatim. In v1 all generated sections are
managed; manual author notes can coexist in the same file as long as they appear outside
the fences.
