# Question and Retrieval

`kb ask <question>` answers a question by planning a retrieval strategy, assembling
context from the KB, calling an LLM backend, and writing the result as a tracked artifact.

## Retrieval Plan

When `kb ask` is called, it first builds a **retrieval plan**:

1. Tokenize the query into terms.
2. Search the lexical index (`state/indexes/lexical.json`) by scoring each wiki page
   against the query. Scoring weights are: title match (4), alias match (3), heading
   match (2), summary keyword match (1).
3. Rank candidates by score and select the top-K entries that fit within the token budget.
4. Persist the retrieval plan to `outputs/questions/<id>/retrieval_plan.json`.

The retrieval plan records each candidate's ID (wiki page path), title, score, estimated
token count, and the scoring reasons.

## Token Budget

The token budget (`ask.token_budget` in `kb.toml`, default 20,000 tokens) caps the total
context passed to the LLM. Each wiki page entry in the retrieval plan has an estimated
token count derived from its character count divided by 4. Pages are included in order of
score until the budget is exhausted.

## Citation Manifest

Before calling the LLM, kb builds a **citation manifest**: a numbered list of the
retrieved sources. The manifest is rendered into the prompt so the LLM can cite sources
by number (e.g., `[1]`, `[2]`). After the LLM responds, kb validates which citation
numbers refer to real sources (valid citations) and which are hallucinated numbers
(invalid citations).

## Question Record

Each `kb ask` call creates a `Question` record at
`outputs/questions/<id>/question.json` with:
- `raw_query` — the user's exact question text
- `requested_format` — the output format (md, marp, json, png)
- `retrieval_plan` — path to the retrieval plan JSON
- `token_budget` — the configured token budget
- `requesting_context` — `project_kb` or `global_kb`

## Output Artifact

The answer artifact is written to `outputs/questions/<id>/answer.md` with YAML
frontmatter containing the question ID, artifact type (`question_answer`), source
document IDs, model name, and timestamp. A sidecar `metadata.json` records
valid/invalid citation indices and the full provenance record.

## Dry Run

With `--dry-run`, `kb ask` prints the retrieval plan without calling the LLM or
writing any artifacts. This is useful for debugging retrieval quality.
