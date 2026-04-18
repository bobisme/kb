# Concept Merge Request

You are clustering concept candidates extracted from multiple documents into canonical
concept entries for a knowledge base.

## Candidates

{{candidates_json}}

## Task

Group the candidates by semantic identity. Two candidates refer to the same concept when
they describe the same idea, even if they use different names, phrasing, or abbreviations.

For each group:
- Choose a `canonical_name` (the clearest, most specific, and most commonly used form).
- List all `aliases` — alternate names, abbreviations, and common shorthand found across
  the candidate set. Do not repeat the canonical name in aliases.
- Include every original candidate object that belongs to the group in `members`.
- Set `confident: true` when the grouping is unambiguous.
- Set `confident: false` when you are unsure — this routes the group to human review
  instead of being silently merged.
- Include a `rationale` string for uncertain groupings explaining the ambiguity.

Return only valid JSON in this exact shape — no other text before or after:
{
  "groups": [
    {
      "canonical_name": "Borrow checker",
      "aliases": ["borrowck"],
      "members": [],
      "confident": true,
      "rationale": null
    }
  ]
}
