You are filtering an automatically-generated list of concept-page candidates for a personal knowledge base. Each candidate is a phrase pulled from existing source documents — some are genuinely useful concepts that deserve their own page, others are noise (generic phrases, sentence fragments, common words, near-duplicates of existing concepts).

Keep a candidate when it satisfies all of:
- it names a single, well-defined topic (a thing, technique, mechanism, pattern, system, or domain term)
- a reader could write 2-4 sentences explaining what it is without restating the surrounding source
- the surface form is not a generic stem ("system", "service", "model"), a partial phrase, or a sentence fragment
- it is not synonymous with a phrase already in the candidate list — pick the cleanest variant and drop the rest

Drop candidates that are generic adjective phrases, throwaway boilerplate, or named entities with too little supporting context to write about.

## Candidates

{{candidates_json}}

## Output

Reply with a JSON object containing the slugs of the candidates worth keeping, ordered by descending usefulness. Output JSON only, no explanation.

Schema:
```json
{"accepted": ["slug-1", "slug-2", "..."]}
```

If none of the candidates is worth keeping, return `{"accepted": []}`. Do not invent slugs that are not in the input list.
