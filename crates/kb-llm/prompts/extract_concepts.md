# Concept Extraction Request

You are extracting candidate concepts from a source document for a knowledge base workflow.

## Title
{{title}}

## Maximum concepts
{{max_concepts}}

## Summary
{{summary}}

## Document
{{body}}

## Task

Identify the most important concepts introduced, defined, or materially discussed in the
source. The goal is a small, durable vocabulary that can be linked to and cited from OTHER
documents — not a comprehensive index of every fact, list item, or turn of phrase.

### How to judge "is this a concept?"

Apply the **reusability test**: *would this concept reasonably be cited from a DIFFERENT
document than the one it was extracted from?* If the answer is no — if the "concept" only
makes sense in the context of this one document — do not extract it.

Good concepts:
- Name a thing, technique, system component, or well-defined idea with a stable referent.
- Could appear as a hyperlink target in multiple unrelated documents.
- Have at least a phrase-length worth of substantive discussion in the source, not a
  one-line mention.

### How many concepts to extract

- For a short source (~1,000 words or less): aim for **3 to 5 concepts**. Going above 5
  should be rare and only justified by unusually high-density material.
- For longer sources: scale up gradually; do not exceed roughly one concept per ~200 words
  of body text.
- If `max_concepts` is set, treat it as a hard ceiling but do not treat it as a target:
  emitting fewer is better than padding the list with weak candidates.

Err on the side of emitting FEWER concepts. A missed concept can be added later; a
wrongly-extracted one creates noise and duplicate pages that must be manually cleaned up.

### Do NOT extract

The following categories have been frequent false positives. Skip them:

1. **Specific numeric defaults or magic numbers.** E.g. "5-second busy timeout default",
   "retry 3 times", "port 8080". These are parameter values, not concepts. If the
   surrounding mechanism is concept-worthy (e.g. "SQLite busy timeout"), extract that
   instead.
2. **Sub-items of a list or enumeration.** If the document enumerates a set (e.g. git
   rebase actions: `pick`, `edit`, `squash`, `fixup`, `drop`, `reword`), extract the
   containing concept ("git rebase todo actions") ONCE — do not emit one concept per list
   member. Individual action names go in `aliases` or are mentioned in the definition.
3. **Single-sentence paraphrases.** If a "concept" is really just a rewording of one
   sentence in the source (e.g. "already-pushed branch", "NFS incompatibility of WAL",
   "long-running reader checkpoint starvation"), skip it. A concept needs at least a
   paragraph's worth of conceptual substance, not a one-line restatement.
4. **Code fragments or API names.** Function names, struct names, and CLI flags are not
   concepts unless they correspond to a named, reusable idea (e.g. `--no-ff` is not a
   concept; "fast-forward merges" is).
5. **Topic-style rephrasings.** Names of the form "a cross-language explanation of X",
   "how Y works", "the story of Z", or other meta-descriptions of a *discussion* rather
   than of an idea. Extract the underlying idea, not the discussion about it.
6. **Meta-observations about the corpus or tooling.** E.g. "ingest handling of
   frontmatter", "extractor behavior on empty files". These describe this knowledge
   base's implementation, not reusable concepts.
7. **Fact-level observations.** E.g. "empty directory removal", "idempotent directory
   creation". A standalone fact about a system call is not a concept; the system call or
   the higher-level pattern may be.

### Positive guidance

- Prefer durable concepts (named techniques, systems, patterns, data structures,
  protocols) over incidental details or examples.
- Include `aliases` when the source uses alternate names, abbreviations, or shorthand.
- Use `definition_hint` for a short gloss grounded in the source (one sentence).
- Use `source_anchors` to point back into the source with stable heading anchors when
  possible and short supporting quotes.

Return only valid JSON in this shape:
{
  "concepts": [
    {
      "name": "Concept name",
      "aliases": ["Alias"],
      "definition_hint": "Short definition",
      "source_anchors": [
        {
          "heading_anchor": "optional-heading-id",
          "quote": "Short supporting quote"
        }
      ]
    }
  ]
}
