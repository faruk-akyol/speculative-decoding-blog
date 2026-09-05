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

### gemma-4-31B-it-FP8-block — 32 GB, ceiling **8.53 tok/s**

| config | tok/s | τ | speedup |
|---|---|---|---|
| baseline | 5.99 | — | 1.00× (70% of ceiling) |
| MTP k=4 | 25.61 | 3.84 | 4.28× |
| EAGLE3 k=3 | 20.80 | 3.04 | 3.47× |
| DFlash k=7 | 11.14 | 2.26 | 1.86× |

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

### 3. Target quantization changes acceptance — but not the way I first guessed

Same gemma-4-31B weights, same heads, same prompt and k; only the target's quantization
differs (`nvidia/…-NVFP4` vs `RedHatAI/…-FP8-block`):

| head | consumes | NVFP4 τ | FP8 τ | Δ |
|---|---|---|---|---|
| DFlash (6 aux layers) | hidden states | 1.36 | 2.26 | **+66%** |
| EAGLE3 (3 aux layers) | hidden states | 2.65 | 3.04 | +15% |
| MTP assistant | **tokens** | 4.38 | 3.84 | **−12%** |

Throughput, for reference (NVFP4 baseline 6.95, FP8 baseline 5.99 tok/s):

| head | NVFP4 | FP8 |
|---|---|---|
| MTP k=4 | 28.72 (4.13×) | 25.61 (4.28×) |
| EAGLE3 k=3 | 19.27 (2.77×) | 20.80 (3.47×) |
| DFlash k=7 | 7.12 (1.02×) | 11.14 (1.86×) |

**The original hypothesis was that NVFP4 corrupts the hidden states feature-level heads
read, leaving token-level heads untouched. The MTP control refutes the clean version:**
MTP did not hold steady, it moved 12% the *other* way.

What can honestly be said right now:

- **DFlash's +66% is large enough to be real.** A drafter tapping six target layers is
  materially damaged by NVFP4, and the head is not misconfigured — it declares 60 target
  layers, hidden 5376, vocab 262144, all matching, and block-matching k to its
  `block_size: 16` changed nothing (τ 1.36 → 1.30).
- **EAGLE3's +15% and MTP's −12% are not yet distinguishable from noise.** One 128-token
  generation at k=4 is roughly 29 draft steps — far too small a sample to resolve effects
  that size.
- A plausible story for the MTP direction is that a more aggressively quantized target
  produces *more predictable* output, which flatters a token-level drafter even as it
  degrades a feature-level one. That is speculation, and it is not tested.

**Resolution:** Stage 1 runs 200 prompts per config, which settles all three effects with
real statistics. Until then this stays an open question, not a finding. It is worth the
extra runs either way — "your NVFP4 quantization may be silently costing you half your
speculative speedup" is exactly the kind of practical warning this post should carry, but
only if it survives a proper sample.

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
