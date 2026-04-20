# Contradiction Detection

You are auditing a knowledge-base concept for factual inconsistencies between
the source quotes it cites. Your output feeds a lint check — be strict about
what counts as a real contradiction and strict about the response shape.

## Concept

{{concept_name}}

## Quotes

Each quote below is numbered (zero-based). The `source` label names the
document the quote came from. Quotes from the same source may repeat a claim;
quotes from different sources that disagree are what we're looking for.

{{quotes}}

## Task

Decide whether these quotes make **contradictory claims about the concept
above**. A contradiction means two (or more) quotes state things that cannot
both be true at the same time — not merely different emphasis, different
scope, or different level of detail.

Do NOT flag:

- Quotes that disagree about unrelated concepts that happen to appear in the
  same passage.
- Quotes that describe different variants, implementations, or historical
  eras of a concept without asserting that one is universal.
- Quotes that use different terminology for the same thing.
- Quotes that are merely more specific or more general than each other.

DO flag:

- Direct factual disagreements ("X is Y" vs. "X is not Y", "X was
  introduced in 2015" vs. "X was introduced in 2018").
- Mutually exclusive claims about defaults, limits, or behavior.
- Claims about cause and effect that contradict each other.

## Response shape

Return ONLY valid JSON in exactly this shape — no prose before or after, no
code fences:

```
{
  "contradiction": <true or false>,
  "explanation": "<one to three sentences>",
  "conflicting_quotes": [<zero-based indices of conflicting quotes>]
}
```

Rules:

- When `contradiction` is `false`, `conflicting_quotes` MUST be the empty
  array `[]`.
- When `contradiction` is `true`, `conflicting_quotes` MUST name at least two
  indices from the list above.
- Indices are integers, zero-based, and must each refer to a quote that was
  supplied in the request.
- Keep `explanation` short — one to three sentences naming the specific
  disagreement or noting there is none.
