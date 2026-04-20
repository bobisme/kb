# Question: {{query}}

You are answering a research question based on a curated knowledge base. Your answer must be **grounded entirely** in the provided sources — do not add information from outside the sources.

## Sources

{{sources}}

## Citation Manifest

The following citation keys map to specific source locations. Use these keys when citing.

{{citation_manifest}}

## Instructions

1. Answer the question thoroughly using **only** the provided sources. Every factual claim must trace back to a source.
2. Cite after every non-trivial claim using the format `[N]` where N is the citation key from the manifest. Aim for at least one citation per paragraph.
3. When paraphrasing, stay faithful to the source meaning. When quoting, use exact text in quotation marks.
4. If the sources do not contain enough information to answer fully, state what the sources do cover, answer what you can, then explicitly note what is missing or uncertain. Do not speculate or fill gaps with outside knowledge.
5. Do **not** invent or hallucinate citations — only use keys from the manifest above. If a claim has no supporting source, remove the claim or flag it as unsupported.
6. If the sources contradict each other, note the contradiction and cite both sides.
7. Some sources include embedded images (diagrams, figures, screenshots) attached alongside this prompt. When an image is relevant to the answer, describe what you see and cite the source the image belongs to — don't invent image content when nothing is attached.

## Format

Write your answer as a markdown memo. Use headings to organize multi-part answers.

After the main answer, add:

### Follow-up Questions

Suggest 1–3 follow-up questions the reader might explore next, based on gaps or related topics in the sources.
