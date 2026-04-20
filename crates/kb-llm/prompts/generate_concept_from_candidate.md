You are drafting a canonical concept entry for a knowledge base from a term
flagged by the missing-concepts lint.

## Candidate term

{{candidate_name}}

## Source snippets

Each bullet below is a short excerpt from a source document that mentions the
candidate term. Use them to decide the canonical name, aliases, and scope.

{{source_snippets}}

## Existing category tags

The knowledge base already uses these category tags. Reuse one when it fits
the new concept; only invent a new tag when nothing here matches.

{{existing_categories}}

## Task

Return a JSON object with the following fields:

- `canonical_name` — the clearest, most general display form of the concept.
  Usually matches the candidate term, but tidy casing/spelling if the sources
  use a consistent alternative (e.g. "foobar" → "FooBar System").
- `aliases` — between 0 and 5 alternate labels observed in the snippets
  (abbreviations, shorthand, spelling variants). Do not repeat the canonical
  name. Empty list is allowed.
- `category` — a short lowercase 1-3 word tag (prefer kebab-case) that places
  the concept in a high-level bucket. Reuse an existing category from the
  list above when one fits. If nothing obvious applies, return `null` — do
  NOT invent a category just to avoid null. No slashes in category names.
- `definition` — a 2-3 sentence GENERAL-SCOPE definition.
  Rules:
    - The first sentence MUST start with the canonical name followed by a
      linking verb ("is", "are", "refers to", "describes").
    - Describe what the concept is at the broadest scope that still covers
      the snippets. Do NOT scope the definition to any single instance or
      variant.
    - Do NOT copy snippets verbatim; synthesize across them.
    - Maximum 3 sentences.

## Output

Return only valid JSON in this exact shape — no prose, no markdown, no code
fences:

{
  "canonical_name": "FooBar System",
  "aliases": ["FooBar"],
  "category": "storage",
  "definition": "FooBar System is a distributed key-value store that …"
}
