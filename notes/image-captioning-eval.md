# Image-captioning model evaluation (2026-04-26 / 2026-04-27)

Context for bone **bn-2qda — ingest: vision-LLM auto-captions for undescribed images**.

The bone proposes captioning images during `kb compile` so screenshots become first-class searchable. Original draft suggested Florence-2-large as the local backend candidate, with the harness path (frontier vision LLM via opencode/claude) as the default. This note records the empirical evaluation that updated the local-backend recommendation.

## TL;DR

- **Default backend (harness via vision-capable LLM): unchanged.** Highest quality, zero new infra, content-hash cache makes per-image cost negligible.
- **Local backend recommendation changed: Qwen3.5-0.8B**, not Florence-2 / SmolVLM.
  - Qwen3.5-0.8B captions on the kb test corpus are nearly identical to the much-larger Qwen3.5-2B and dramatically better than SmolVLM-256M / 500M.
  - SmolVLM at 256M / 500M is **not viable** — it hallucinates structural detail and loops on text-rich images, both of which would actively pollute retrieval.
- **CPU performance is hardware-dependent.** On a 2019-era AVX2-only CPU (Ryzen 3900X), Qwen3.5-0.8B fp32 runs at ~30–60 s/image with full quality preserved. bf16 on the same CPU is ~10× slower (no AVX-512_BF16). Modern CPUs with AVX-512_BF16 + VNNI should be much faster, but were not benchmarked here.
- **GGUF + llama.cpp on CPU did not finish** in the test window (>15 min/image on this CPU) — the vision encoder pass via `mmproj-F16.gguf` appears to be the bottleneck and needs more investigation before being recommended.

## Method

Six representative images were staged from the consulting kb corpus and a synthetic test pass:

1. `01-hex-arch.png` — text card on dark blue, "Hexagonal Architecture Diagram" + outer ring (HTTP/DB/CLI), inner hexagon (domain core), arrows inward
2. `02-chart.png` — horizontal bar chart "Hexagonal vs Clean", both bars = 1, x="Height", y="Architecture", legend "Value"
3. `03-screenshot-a.png` — ML pipeline DAG ("Scenario A: Normal (strict) — branches: reach_freq, lift" / "Scenario B — Normal Run with Parent Check Bypassed")
4. `04-screenshot-b.png` — Excalidraw architecture diagram with Control Plane and Data Plane boxes, ~30 small named services
5. `05-screenshot-c.png` — code editor showing deployment / Helm-chart configuration
6. `06-screenshot-d.png` — Garden build dashboard with cards for ~20 services (rich-data-table, moonraker, hank, forebitt, …)

Same caption prompt for every model: *"Describe this image in 2-3 sentences for a knowledge base. Focus on what is depicted, any text, diagrams, charts, or relationships shown. Plain prose, no markdown."*

Models tested:
- `HuggingFaceTB/SmolVLM-256M-Instruct`
- `HuggingFaceTB/SmolVLM-500M-Instruct`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-0.8B`

Hardware:
- GPU runs: NVIDIA RTX 3090, bf16, deterministic-ish sampling.
- CPU runs: AMD Ryzen 9 3900X (12 physical cores, 24 threads, AVX2 only — no AVX-512, no AVX-512_BF16, no VNNI). Various dtype/quant combinations.

Raw outputs and timings live under `/tmp/smolvlm-test/` (not checked in).

## Results — quality

Per-image qualitative judgment vs ground truth observed by reading each image:

| Image | SmolVLM-256M | SmolVLM-500M | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|---|---|
| 01 hex-arch | starts OK then loops with hallucinated database-adapter lines | **near-perfect** — outer ring + HTTP/DB/CLI + inner hexagon + arrows | **perfect** — incl. "ports" terminology | **perfect** |
| 02 bar chart | title right; invents "meters" units, color half wrong, then loops | confuses axes, claims one bar taller than the other (they're equal) | **best of all** — got x-axis = "Height" right | confuses x-axis with legend "Value" |
| 03 ML pipeline | calls it Scenario A, fabricates a navigation bar | concise + correct title, no false detail | **best of all** — enumerates every node with exact tag (`(no tags)`, `(tag: reach_freq)`, `(tag: lift)`, `(tag: attribution)`) | summarizes structure correctly but at higher level |
| 04 Excalidraw | pure JSON-shaped hallucination (Topgap, Flynn, Fidelity ×6) | identifies Excalidraw + Control Plane; hallucinates components | identifies Control Plane / Data Plane / Infrastructure; real names with minor misspellings | identifies Control Plane / Data Plane / Infrastructure; real names |
| 05 code editor | "Python code" (it isn't), loops | "document with text" — conservative, accurate | "Kubernetes deployment script" — generic but right | named specific stages `helm-repos`, `vault pg-setup`, `infra-001..003` |
| 06 Garden dashboard | **"chat conversation between hank and maverick"** — total miss | identifies "rich-data-table"; loops "built 10d ago" 15× | reads service names + `source` badge on hank + `built 10d ago` correctly | reads service names + package managers (Docker/Maven/Go/npm) + "dirty" flag |

### Pattern observations

1. **SmolVLM 256M/500M hallucinate heavily on detail-rich images.** They invent JSON structures, fabricate quoted text, and loop on token-rich screenshots. Captions from these models would actively mislead retrieval — e.g. SmolVLM 256M's "chat between hank and maverick" for image 06 would match irrelevant queries about chats.
2. **OCR is the dividing line.** SmolVLM cannot read small text in dense screenshots. Qwen3.5 (both sizes) can. This matters because most kb-relevant images are exactly this kind: dashboards, code, architecture diagrams labeled with real service names.
3. **Qwen3.5-0.8B is at least as good as Qwen3.5-2B for this corpus.** On charts and tagged pipeline diagrams the 0.8B is *more* useful for retrieval — it preserves more text-as-metadata. The 2B's prose is slightly more concise but less specific. A 2.5× increase in parameters bought essentially nothing here.
4. **"Sampling defaults matter."** Qwen3.5 produced clean prose without any repetition penalty. SmolVLM's looping problem might be partly mitigable with `repetition_penalty` / `no_repeat_ngram_size`, but the underlying detail-blindness wouldn't go away.

## Results — performance

GPU (RTX 3090):

| Model | Latency / image | Load time |
|---|---|---|
| SmolVLM-256M | 0.9–5 s | 13 s |
| SmolVLM-500M | 0.5–6 s | 16 s |
| Qwen3.5-0.8B | 5–11 s | 24 s |
| Qwen3.5-2B | 4–7 s | 47 s (flash-linear-attention dep not installed) |

CPU (Ryzen 9 3900X, 12 cores, AVX2 only) — Qwen3.5-0.8B only:

| Variant | image 01 | image 02 | Quality |
|---|---|---|---|
| transformers bf16 | 290 s | 257 s | preserved |
| transformers fp32 | 28 s | 61 s | preserved (matches GPU output) |
| transformers int8 dynamic (`torch.quantization.quantize_dynamic`) | 30 s | 18 s | **regressed — caption fabricated** |
| llama.cpp Q4_K_M GGUF + mmproj-F16 | killed at 12 min | killed at 26 min | n/a (didn't finish) |

### Why these results

- **bf16 on CPU without AVX-512_BF16 falls back to software emulation** — roughly 10× slower than fp32 with native SIMD. Modern Intel (Sapphire Rapids+, 2023+) and AMD (Zen 4+, 2022+) have native bf16 and would invert this; older CPUs do not.
- **int8 dynamic quant in PyTorch helps mainly on CPUs with VNNI** (AVX-512 era). On AVX2-only it offers no speedup and visibly degrades quality. On image 01 the int8 caption became "interconnected diagrams arranged in a grid-like structure" — generic and wrong, where fp32 was perfect. Conclusion: do not use torch dynamic quant for this on consumer AVX2 CPUs.
- **The llama.cpp GGUF result is unexplained.** Both Q4_K_M and Q8_0 with `mmproj-F16` were tested; neither finished in the time budget. Theories:
  - F16 vision-encoder ops are slow without F16 SIMD on this CPU.
  - Qwen3.5's hybrid attention (Gated DeltaNet, 3:1 linear-to-softmax ratio) may have a slow CPU path.
  - 1000+ visual tokens of prompt processing dominate.
  - Configuration error (e.g., flag interactions).
  Worth a deeper look before recommending GGUF as the CPU default. The transformers fp32 path was the fastest finishing CPU path tested.

## Recommendations for bn-2qda

Replace the Florence-2-large local-backend pick with a device-aware Qwen3.5-0.8B backend, with the harness still as the default.

**Two backends:**

1. `runner = "harness"` (default) — passes image bytes through to the configured vision-capable LLM via opencode / Claude Code. Highest quality. Cache keyed by image content hash means each image is captioned once, ever.

2. `runner = "qwen35-local"` (opt-in) — local Qwen3.5-0.8B via transformers (or eventually `ort` / `candle` once vision pipelines are mature). Auto-detect device and dtype:
   - GPU present → `dtype = bf16` on GPU
   - CPU only, modern (AVX-512_BF16 detected) → `dtype = bf16` on CPU
   - CPU only, AVX2 (no AVX-512_BF16) → `dtype = fp32` on CPU (do **not** silently use bf16; quality stays the same but throughput collapses ~10×)
   - Never use torch's int8 dynamic quant — quality regression observed.

**Suggested config skeleton:**

```toml
[compile.captions]
enabled = true
runner = "harness"
model = "claude-haiku-4-5"
allow_paths = ["wiki/", "sources/"]

[compile.captions.local]
backend = "qwen35-0.8b"
device = "auto"            # cuda → bf16; cpu → fp32 unless AVX-512_BF16 present
fallback_to_harness = true # if local fails or exceeds latency budget, escalate
max_seconds_per_image = 90 # bail out on slow CPUs
```

**Caveats to surface in bn-2qda's description:**

- Harness path remains the right default. The local backend exists for offline / privacy / cost-control scenarios, not as a quality replacement.
- Local on old CPUs is **marginal**: ~30–60 s/image. Tolerable for batch compile; painful for interactive captioning during ingest. The cache mitigates repeated cost but not first-time pain.
- Modern CPU performance (AVX-512_BF16, VNNI) is unmeasured here; expect substantially better but verify before recommending.
- GGUF path is **not yet recommended** — needs a working benchmark before being added as a third option. Either rerun on a CPU with AVX-512 or with a different mmproj precision (F32 instead of F16) and a more recent llama.cpp build.

## Files referenced

- `/tmp/smolvlm-test/caption.py` — SmolVLM driver (256M + 500M)
- `/tmp/smolvlm-test/qwen35.py` — Qwen3.5-2B GPU driver
- `/tmp/smolvlm-test/qwen35-08b.py` — Qwen3.5-0.8B GPU driver
- `/tmp/smolvlm-test/qwen35-cpu.py` — CPU bf16/fp32 driver
- `/tmp/smolvlm-test/qwen35-cpu-int8.py` — CPU int8-dynamic driver
- `/tmp/smolvlm-test/run.log`, `qwen35*.log` — verbatim outputs
- `/tmp/smolvlm-test/images/0[1-6]-*.png` — staged test images

These are not checked in; rerun the drivers from scratch if the corpus changes meaningfully or if a new candidate model arrives (e.g., a Qwen3.6-VL-small).
