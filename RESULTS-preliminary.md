# Preliminary results

**These are single-prompt smoke tests, not the benchmark.** One coding prompt
("Write a Python function that reverses a string"), 128 tokens, greedy, batch 1,
after one warm-up request. They exist to verify configs load and to sanity-check
magnitudes before committing ~30 h to the full grid. Treat every number as
provisional until Stage 1 replaces it with a 200-prompt mean.

Hardware: one DGX Spark (GB10), 273 GB/s, vLLM 0.28.0, `gpu-memory-utilization 0.55`.

---

## Batch-1 decode

### Qwen3.8-27B-FP8 — 30.87 GB, roofline ceiling `273/30.87` = **8.84 tok/s**

| config | tok/s | τ | speedup | % of ceiling |
|---|---|---|---|---|
| baseline | 8.00 | — | 1.00× | **90.5%** |
| MTP k=7 (native) | 18.48 | 5.03 | 2.31× | — |
| **DFlash 2 k=7** | **30.25** | 5.00 | **3.78×** | — |
| DSpark k=7 | *fails to load* | — | — | — |

### gemma-4-31B-it-NVFP4 — 31 GB, ceiling **8.80 tok/s**

| config | tok/s | τ | speedup | % of ceiling |
|---|---|---|---|---|
| baseline | 6.95 | — | 1.00× | 79% |
| **MTP k=4** (assistant) | **28.72** | 4.38 | **4.13×** | — |
| EAGLE3 k=3 | 19.27 | 2.65 | 2.77× | — |
| DFlash k=7 | 7.12 | 1.36 | 1.02× | — |
| DFlash k=15 (block-matched) | 7.08 | 1.30 | 1.02× | — |

---

## Three findings worth chasing

### 1. Same acceptance length, 1.64× the throughput

Qwen MTP and DFlash 2 both achieve **τ ≈ 5.0**, yet DFlash 2 delivers 30.25 tok/s
against MTP's 18.48. Acceptance is held fixed, so the entire difference is *drafting
cost*: MTP runs its head autoregressively 7 times per verify step, DFlash 2 drafts the
whole block in a single forward pass. This is the DFlash design claim isolated cleanly
by accident — and it argues that on a bandwidth-starved box, **how cheaply you draft
matters as much as how well you draft.**

### 2. Batch-1 decode is at 90% of roofline — so speculation is the only lever left

Qwen3.8-27B-FP8 reaches 90.5% of its pure bandwidth limit. The H200 in the DFlash 2
model card reaches only 44% of *its* limit (68.9 of 155.5 tok/s). At 273 GB/s the weight
read dominates each step so completely that kernel-launch, sampling and Python overhead
disappear into it; at 4.8 TB/s they are a large fraction of step time.

Consequences: the §1 roofline model is *more* predictive here than on a datacenter card,
and there is essentially nothing left to win from kernel tuning.

### 3. Does NVFP4 break hidden-state speculative heads? — **open**

| target | head | consumes | τ | outcome |
|---|---|---|---|---|
| Qwen FP8 | DFlash 2 | hidden states | 5.00 | works |
| gemma NVFP4 | MTP assistant | **tokens** | 4.38 | works |
| gemma NVFP4 | EAGLE3 (3 layers) | hidden states | 2.65 | degraded, 66% of max |
| gemma NVFP4 | DFlash (6 layers) | hidden states | 1.30 | **broken, 8% of max** |

The head is not misconfigured: it declares 60 target layers, hidden 5376, vocab 262144,
all of which match the NVFP4 target exactly, and block-matching k to its `block_size: 16`
changed nothing (τ 1.36 → 1.30).

The pattern that fits is that **NVFP4 perturbs the hidden states feature-level drafters
read, while token-level drafters are immune** — with damage scaling in how many layers
the head taps. If it holds, it is a practical warning: quantize your target to NVFP4 on a
Spark and your EAGLE3/DFlash head may quietly stop paying for itself while still
appearing to run.

**Test in flight:** re-run gemma DFlash against `RedHatAI/gemma-4-31B-it-FP8-block`
(same weights, gentler quantization). If τ recovers, the hypothesis holds. If it stays
~1.3, the cause is elsewhere and this section gets rewritten.

---

## Blocked

**DSpark on vLLM 0.28** — `ValueError: hf_overrides must be a dict for get_quant_config`.
`vllm/config/speculative.py:770` does `if not self.quantization: self.quantization =
self.target_model_config.quantization`, so a BF16 draft head inherits the target's FP8 and
then finds no quantization config in its own `config.json`. Passing `"quantization": null`
does not help — `null` is falsy, so it takes the same branch. DFlash 2 loads fine against
the identical target, so this is specific to DSpark's loader
(`spec_decode/dspark/utils.py` calls `get_draft_quant_config` directly).

Per the no-conversion policy (EXPERIMENTS.md §3.1), DSpark moves to **SGLang**, which is
the reference implementation for the `RadixArk` checkpoint anyway. Not yet attempted.
