# kb eval — golden Q/A retrieval harness (bn-3sco)

A read-only smoke test for `kb`'s hybrid retriever. Define a list of
golden queries with expected sources/concepts, and `kb eval run` will
score the top-10 hybrid ranking against them. Use it whenever you
change a retrieval knob (`[retrieval]` in `kb.toml`, the embedding
backend, RRF k, etc.) to make sure you didn't regress.

## Layout

```
<kb-root>/
└── evals/
    ├── golden.toml             # your queries (see golden.toml.example)
    └── results/
        ├── 2026-04-30T12-34-56Z.json
        ├── 2026-04-30T12-34-56Z.json   # one per `kb eval run`
        ├── <save_as>.json              # via `--save-as <name>`
        └── latest.md                   # rendered table for the most recent run
```

## Bootstrapping

`golden.toml` is **not** auto-created by `kb init` — it's per-kb content
that you author once.

```sh
cp <kb-repo>/crates/kb-cli/src/eval/golden.toml.example <kb-root>/evals/golden.toml
$EDITOR <kb-root>/evals/golden.toml
```

## Subcommands

```sh
kb eval run                        # run + write a result
kb eval run --save-as q1-tuning    # also stash a copy at evals/results/q1-tuning.json
kb eval run --baseline q1-tuning   # diff this run against a saved baseline
kb eval list                       # list past runs (newest first)
```

## Metrics

All four metrics are in `[0.0, 1.0]`. Higher is better.

- **P@K (precision at K)** — fraction of the top-`K` results that match
  any expected source/concept. `P@5 = 0.6` means 3 of the top 5 hits
  were relevant. The denominator is `K` even when the result list is
  shorter, so a 3-result list with 2 hits scores `0.4` against P@5
  (not `0.67`) — short lists are penalised, not prizes.
- **MRR (mean reciprocal rank)** — `1 / r` where `r` is the 1-indexed
  position of the first relevant hit. Top-1 hit -> `1.0`. Third-position
  hit -> `0.333`. No hit in top-10 -> `0.0`. Sensitive to the very top.
- **nDCG@10 (normalized discounted cumulative gain)** — weighted version
  of P@10 where lower-ranked hits count proportionally less
  (`1 / log2(rank + 1)`). With binary relevance the ideal ranking is
  "all relevant items at the top, then everything else"; `nDCG@10 = 1.0`
  means we matched that ideal in the top-10.

## Corpus pinning

Each result records a `corpus_hash` — BLAKE3 of
`state/indexes/lexical.json`. When a baseline's hash differs from the
current run, the diff still renders but a `WARNING` line is prepended
(metrics may have shifted because the corpus changed, not because the
retriever did). Re-run `kb compile` and re-snapshot the baseline if you
want a clean A/B.

## Rules of thumb

- Cheap and reproducible: the harness performs **no LLM call**, only
  retrieval. Sub-second on small kbs.
- Add new queries one at a time, run, and snapshot before you tune.
  Five hand-curated queries beat fifty noisy ones.
- A single golden query can pin both a positive expectation
  (`expected_sources = ["src-cred"]`) and confirm a paraphrase recovers
  it. Lexical-only retrievers fail those queries; hybrid passes.
