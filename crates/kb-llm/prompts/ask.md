# Question: {{query}}

You are answering a research question based on a curated knowledge base. Your answer must be grounded in the provided sources.

## Sources

{{sources}}

## Citation Manifest

The following citation keys map to specific source locations. Use these keys when citing.

{{citation_manifest}}

## Instructions

1. Answer the question thoroughly using **only** the provided sources.
2. For every non-trivial claim, include at least one citation using the format `[N]` where N is the citation key from the manifest above.
3. If the sources do not contain enough information to answer confidently, say so explicitly rather than speculating.
4. Do not invent or hallucinate citations — only use keys from the manifest.

## Format

Write your answer as a markdown memo. Structure it with headings if the answer is multi-part.

After the main answer, add a section:

### Follow-up Questions

Suggest 1–3 follow-up questions the reader might explore next, based on gaps or related topics in the sources.
