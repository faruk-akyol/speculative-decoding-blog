# Speculative Decoding on a DGX Spark

A blog post, and the benchmark harness behind it, about **speculative decoding**: what the
methods are, how they differ, and — with real measurements on one NVIDIA DGX Spark (GB10) —
when each one is actually worth using.

> **Status:** experiment design complete, benchmarks not yet run.
> See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full plan.

---

## The short version

A decode step is memory-bandwidth-bound. Speculative decoding trades compute for bandwidth
by drafting `k` tokens cheaply and verifying them in a single batched pass. That trade is
free only while the verify pass stays memory-bound, which gives a crossover concurrency:

```
c* = B* / (k + 1)          B* = peak_compute / memory_bandwidth
```

The DGX Spark has unusually little bandwidth (273 GB/s) relative to its compute
(~125 TFLOPS dense BF16), so `B* ≈ 458` — against **≈ 206 for an H200**. Its memory-bound
plateau is **2.2× wider**.

This predicts something specific and checkable. The published `Qwen3.8-27B-DFlash2`
benchmarks show MTP and DSpark going *net-negative* on an H200 at concurrency 32
(0.77–0.95× — slower than not speculating at all). At k=7 that's an effective batch of 256,
past the H200's ridge point of 206. **On a Spark, 256 is still comfortably under 458** — so
the same configuration should still be winning.

That is the hypothesis this repo exists to test.

It matters more here than on a datacenter card. Qwen3.8-27B-FP8 is ~30.8 GB, so batch-1
decode on this box is capped at `273 / 30.8 ≈ 8.9 tok/s` regardless of how good the kernels
are. Speculative decoding isn't a micro-optimisation on a Spark — it's the difference
between a model you can use interactively and one you can't.

## What's measured

**Two model families**, both 27–31B, deliberately chosen for contrasting attention designs:

| | attention | speculative heads available |
|---|---|---|
| **Qwen3.8-27B** | hybrid: 48 linear (Mamba-style SSM) + 16 full, 3:1 | native MTP · DFlash 2 · DSpark |
| **gemma-4-31B-it** | sliding (window 1024) + full, 5:1 | MTP assistant · EAGLE3 · DFlash |

The pairing is the interesting part. Rejecting a speculated token in gemma means truncating
a KV cache; in Qwen3.8 it means rolling back recurrent SSM state across 48 of 64 layers.
Their cost curves should diverge with context length — gemma's KV cache grows and consumes
the very bandwidth speculation is trying to save, while Qwen3.8's linear layers hold
constant state.

**Seven method classes**, every one verified against the engine registry *and* confirmed to
have a real published checkpoint: `ngram`, `ngram_gpu`, `suffix`, `draft_model`, MTP,
EAGLE3, DFlash/DFlash2, DSpark. Methods with no published head for these targets (Medusa,
UNO) are reported as gaps rather than quietly omitted.

**Tasks:** Turkish (TR-MMLU, MLSUM-tr, Turkish instructions) and coding (HumanEval+, MBPP+,
RepoBench-py), chosen to span the copy-rate spectrum — which is what actually decides
whether a free method like `ngram` can compete with a trained head.

## Hardware

One NVIDIA DGX Spark. GB10 Grace Blackwell, 128 GB unified LPDDR5X @ 273 GB/s, 20-core Arm,
Ubuntu 24.04 aarch64, CUDA 13.0. Engines: vLLM 0.28.0, with SGLang as fallback for setups
vLLM can't load. No weight conversion is performed anywhere in this study.

Single node only — no multi-node, no second Spark.

## Layout

```
EXPERIMENTS.md    full experiment plan: roofline math, model/method matrix, stages, metrics
harness/          benchmark runner, server configs, dataset loaders   (not yet written)
results/          raw measurements and generated charts               (not yet populated)
blog/             the post itself                                     (not yet written)
```

## Reproducing

Not yet runnable — the harness isn't written. Model weights are expected under
`/home/spark/models` and are never committed to this repo.
