# Grounding Quality Metrics

Measures how well kb answers are supported by cited sources.

## Metrics

### Citation Precision

Fraction of citations that reference valid manifest entries.

    citation_precision = valid_citations / (valid_citations + invalid_citations)

A hallucinated citation (referencing a non-existent source) lowers precision. An answer with 3 valid and 1 invalid citation scores 0.75.

### Citation Recall

Fraction of expected source documents that the answer actually cites.

    citation_recall = expected_sources_cited / total_expected_sources

Measured against the gold question set's `expected_sources` field. An answer that cites 2 of 3 expected sources scores 0.67.

### Hallucination Rate

Fraction of all citations that are fabricated (inverse of precision).

    hallucination_rate = invalid_citations / (valid_citations + invalid_citations)

### Uncertainty Handling

Binary per-question metric: does the answer show an uncertainty banner when grounding is weak, and omit it when grounding is strong?

Weak grounding triggers: empty context, very low token utilization (<10% of budget), or zero valid citations despite available sources.

## Verdicts

Each answer receives a verdict based on its metrics:

| Verdict | Criteria |
|---------|----------|
| Strong | precision >= 90%, recall >= 60%, hallucination <= 25% |
| Acceptable | precision >= 70%, recall >= 40%, hallucination <= 25% |
| Weak | precision or recall below acceptable, hallucination <= 25% |
| Ungrounded | hallucination > 25% |

## Ship Thresholds

Aggregate thresholds that must be met before shipping answer quality:

| Metric | Threshold |
|--------|-----------|
| Mean citation precision | >= 80% |
| Mean citation recall | >= 50% |
| Mean hallucination rate | <= 10% |
| Uncertainty accuracy | >= 80% |

These are defined in `GroundingThresholds::ship()` in `crates/kb-query/src/grounding.rs`.

## Running the Evaluation

Offline scorer (runs in CI, no LLM needed):

    cargo test --test gold_harness gold_grounding_scorer_offline

Full grounding evaluation with real LLM:

    cargo test --test gold_harness -- --ignored gold_llm_grounding

Record results as new baseline:

    GOLD_RECORD_BASELINE=1 cargo test --test gold_harness -- --ignored gold_llm_grounding

## Baseline

The baseline is stored in `crates/kb-cli/tests/gold/baseline.json`. When recorded with `GOLD_RECORD_BASELINE=1`, it includes a `grounding` section with all metrics and the thresholds used.

Current retrieval coverage baseline: 96.875% (31/32 questions at 100%). LLM grounding metrics require a real backend run.
