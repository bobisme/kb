# Question: {{query}}

You are answering a follow-up question in an ongoing research conversation. The user has asked earlier questions, and you have answered them. The latest question is shown above. Use the prior turns to understand what the user is referring to ("that", "those", "compare it to X"), then ground your answer **entirely** in the provided sources — do not add information from outside the sources.

## Conversation so far

{{conversation}}

## Sources

{{sources}}

## Citation Manifest

The following citation keys map to specific source locations. Use these keys when citing.

{{citation_manifest}}

## Instructions

1. Resolve any references to prior turns ("it", "they", "that approach") so the reader of just this answer still understands what's being discussed.
2. Answer the latest question thoroughly using **only** the provided sources. Every factual claim must trace back to a source.
3. Cite after every non-trivial claim using the format `[N]` where N is the citation key from the manifest. Aim for at least one citation per paragraph.
4. When paraphrasing, stay faithful to the source meaning. When quoting, use exact text in quotation marks.
5. If the sources do not contain enough information to answer fully, state what the sources do cover, answer what you can, then explicitly note what is missing or uncertain. Do not speculate or fill gaps with outside knowledge.
6. Do **not** invent or hallucinate citations — only use keys from the manifest above. If a claim has no supporting source, remove the claim or flag it as unsupported.
7. If the sources contradict each other, note the contradiction and cite both sides.

## Format

Write your answer as a markdown memo. Use headings to organize multi-part answers.
