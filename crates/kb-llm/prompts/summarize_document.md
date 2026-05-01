# Document Summary Request

You are summarizing a source document for a knowledge base workflow.

## Title
{{title}}

## Maximum length
{{max_words}} words

## Document
{{body}}

## Task
Produce a faithful summary of the document. Prefer concrete details over vague generalities.
Keep the response within the requested word budget.
When the document body contains `<!-- kb:page N -->` markers (PDF source) and you quote a specific page, append the page in the citation: `[src-id p.7]` for a single page or `[src-id pp.7-9]` for a span. The page numbers come from the surrounding `kb:page N` markers — do not invent them.
Return only the summary text.
