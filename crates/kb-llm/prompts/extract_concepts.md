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

### Worked examples

These are real false positives observed on prior compile runs. Use them as a guide when
deciding whether to spawn a new concept or to fold material into an existing one.

**Example 1 — trade-offs are not concepts.**

Source text (from a `regularization.md`) reads:

> ## Trade-offs
>
> Regularizers cost some training-set fit in exchange for generalization.

- WRONG: extract a new concept named "regularization-generalization trade-off".
- RIGHT: this is a property of `regularization` and `overfitting`, not a reusable thing.
  Mention the trade-off inside the `definition_hint` of `regularization` (e.g. "a family
  of techniques that trade training-set fit for generalization"). Do not spawn a new
  concept for the trade-off itself.

**Example 2 — single bullets are not concepts.**

Source text (from a `dropout.md` "Why it works" section) reads:

> - Co-adaptation prevention — features can't rely on specific others being present.

- WRONG: extract a new concept named "feature co-adaptation".
- RIGHT: fold it into the `definition_hint` of `dropout` (e.g. "stochastic unit masking
  that prevents feature co-adaptation"). A single bullet point is not a paragraph of
  conceptual substance (rule #3 above).

**Example 3 — one-line mechanism notes are not concepts.**

Source text (from an `attention.md`) reads:

> For autoregressive generation, mask the upper triangle of the score matrix.

- WRONG: extract a new concept named "causal masking".
- RIGHT: mention causal masking in the `definition_hint` of `attention`, or add
  "causal masking" / "causal attention" to its `aliases`. A one-sentence mechanism note
  does not meet the reusability bar.

**Example 4 — "X of Y" observations are not concepts.**

Source text (from a `batch-norm.md`) reads:

> Batch normalization smooths the loss landscape, which is one hypothesis for why it
> accelerates training.

- WRONG: extract a new concept named "loss landscape smoothing".
- RIGHT: this phrase describes an *observed effect* of batch normalization, not a named,
  reusable idea that another document would link to. Mention it in the `definition_hint`
  of `batch normalization` if relevant; otherwise omit.

**Example 5 — named, reusable ideas SHOULD be extracted.**

Source text (from an `adam.md`):

> Adam is an adaptive-learning-rate optimizer combining momentum (first moment) with
> RMSProp-style second-moment scaling.

- RIGHT: extract `adam` (with aliases like `adam optimizer`). It is a named technique,
  discussed at paragraph length, and would plausibly be linked from unrelated documents.

The pattern: if the candidate's name reads like "X of Y", "X-handling", "X behavior",
"X implementation", "X considerations", "X trade-off", or "X approach", it is almost
certainly a meta-observation about the source — not a concept — and should be folded
into an existing concept instead.

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
