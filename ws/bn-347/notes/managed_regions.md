# Managed Regions & Section IDs

## Context
When an LLM compiler or human edits a generated page, we want to isolate AI-managed sections from human-curated content. We use "managed regions" denoted by HTML comments.

## Design

### Managed-Region Marker Format
We use HTML-comment fences:
- Start: `<!-- kb:begin id=<slug> -->`
- End: `<!-- kb:end id=<slug> -->`

This is invisible in Obsidian reading view. The parser recognizes these fences to extract regions, and the rewriter updates only content inside matching markers. Anything outside managed regions is preserved.

### Section ID Scheme
- **Stable slugs:** The section ID (`id=<slug>`) is derived from the initial section title/purpose.
- Once assigned, the slug survives even if the title inside the section is renamed.
- Citations can reference a section via `<page>#<section-id>`.
- The slug is stored explicitly in the HTML comment marker, preventing loss on title change.

## Implementation Requirements
- **Parser:** Detects fences and returns isolated chunks (managed vs unmanaged).
- **Rewriter:** Accepts a file stream or AST, finds the matched `id`, and replaces only the interior text, keeping surrounding content.
