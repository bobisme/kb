You are filling in a gap in a knowledge base. A lint check flagged a concept
that is either missing entirely or has a too-thin body. Your job is to draft
fill-in content the user can review and approve.

You have web-search tool access. USE IT. Find reputable sources that describe
this concept in general terms, then synthesize a short, well-grounded
definition. Every factual claim in the draft must be traceable to a web
source you cite.

## Gap kind

{{gap_kind}}

## Concept name

{{concept_name}}

## Existing content (may be empty)

{{existing_body}}

## Local source context (may be empty)

Each bullet is a short excerpt from a local source in the knowledge base that
mentions this concept. Use these to disambiguate which variant of the term
the KB is about — e.g. "Merkle tree" can mean the distributed-systems data
structure or an unrelated tree in combinatorics. Let these hints narrow your
web search.

{{local_snippets}}

## Task

Use web search to find 1-3 reputable sources describing this concept. Then
return a JSON object with these fields:

- `definition` — a 2-4 sentence general-scope definition of the concept.
  - First sentence MUST start with the concept name followed by a linking
    verb ("is", "are", "refers to", "describes").
  - Describe the concept broadly, not any one variant.
  - Do NOT copy web pages verbatim; synthesize across sources.
  - Do NOT include inline citation markers like "[1]" in the definition
    text — citations go in the `sources` field below.
- `sources` — a list of the web sources you used, each with:
    - `url` — the page URL.
    - `title` — the page title (as displayed).
    - `note` — a short (<= 20 words) reason why this source supports the
      definition.
  - Include between 1 and 3 sources. The first source should be the most
    authoritative (prefer Wikipedia, official docs, or peer-reviewed pages
    over blog posts).
- `confidence` — one of `"high"`, `"medium"`, `"low"`.
  - `high`: multiple reputable sources agree and the concept is
    unambiguous.
  - `medium`: one reputable source, or multiple sources with minor variance.
  - `low`: only low-quality sources found, or the concept is ambiguous and
    you had to pick a variant.
- `rationale` — a 1-2 sentence note explaining which variant you picked and
  why. Visible to the reviewer in `kb review show`.

## Output

Return only valid JSON in this exact shape — no prose, no markdown, no code
fences:

{
  "definition": "Merkle tree is a hash-linked tree structure used in distributed systems to verify large datasets …",
  "sources": [
    {
      "url": "https://en.wikipedia.org/wiki/Merkle_tree",
      "title": "Merkle tree — Wikipedia",
      "note": "Canonical overview covering hash structure, use in IPFS/Git, and history."
    }
  ],
  "confidence": "high",
  "rationale": "The KB's local mentions focus on Git/IPFS-style content-addressable storage, so this definition picks the distributed-systems variant."
}
