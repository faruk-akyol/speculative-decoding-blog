# Experiment Plan

Status: **draft for review**. Nothing here has been run yet.

---

## 1. Hardware, and why this box is the point

All measurements run on a single **NVIDIA DGX Spark (GB10 Grace Blackwell)**:

| | value |
|---|---|
| Unified memory | 128 GB LPDDR5X (119 GB usable) |
| **Memory bandwidth** | **273 GB/s** |
| Dense compute | ~500 TFLOPS FP4 · ~250 TFLOPS FP8 · ~125 TFLOPS BF16 |
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) |
| Software | Ubuntu 24.04 · aarch64 · CUDA 13.0 · driver 580.159.03 · vLLM 0.28.0 |

### 1.1 The ridge point

A decode step reads every weight once and does ~2 FLOPs per weight per sequence in the
batch. With `w` bytes per parameter, arithmetic intensity is

```
AI(B) = 2B / w        [FLOP per byte]
```

You stay memory-bound until `AI` reaches the hardware's ridge point `C_peak / BW`:

| accelerator | BW | dense BF16 | ridge point | saturating batch `B*` |
|---|---|---|---|---|
| **GB10 (Spark)** | 273 GB/s | 125 TF | **458 FLOP/B** | **458** |
| H100 SXM | 3.35 TB/s | 989 TF | 295 | 295 |
| **H200 SXM** | 4.8 TB/s | 989 TF | **206** | **206** |
| B200 | 8.0 TB/s | 2250 TF | 281 | 281 |

`B*` comes out the same for BF16, FP8 and FP4 on Blackwell, because quantization scales
compute and bytes-read by the same factor. Convenient: **one number characterises the box.**

### 1.2 The prediction this whole study tests

Speculative decoding replaces one batch-1 decode with one batch-`(k+1)` verify. So with
`c` concurrent requests and `k` draft tokens, the verify pass runs at effective batch
`c·(k+1)`. It is free exactly while that stays under `B*`. The crossover concurrency is:

```
c* = B* / (k+1)
```

At the k=7 used throughout the DFlash 2 report:

| | `B*` | predicted `c*` |
|---|---|---|
| H200 | 206 | **≈ 26 concurrent** |
| **GB10 (Spark)** | 458 | **≈ 57 concurrent** |

**This already explains published data.** The `z-lab/Qwen3.8-27B-DFlash2` model card
benchmarks MTP / DSpark / DFlash 2 on an H200 at k=7 and reports that at **concurrency 32**
MTP and DSpark go *net-negative* (0.77–0.95×) while DFlash 2 barely holds on
(1.01–1.45×). Concurrency 32 × 8 = 256 effective batch, which is past the H200's ridge
point of 206 — so the verify pass genuinely costs compute time. The model predicts the
sign flip that was measured.

On a Spark the same config gives 256 < 458, still memory-bound.

> **H1 — the headline hypothesis.** Speculative decoding should still be *clearly positive*
> at concurrency 32 on a DGX Spark, in the same configuration where it is a measured *loss*
> on an H200. Spark's plateau is 2.2× wider. The concurrency sweep in Stage 2 tests this
> directly, out to 64 so the predicted Spark crossover at ~57 is bracketed.

### 1.3 Why it matters more here

Qwen3.8-27B-FP8 is ~30.8 GB, so batch-1 decode on this box is capped at
`273 / 30.8 ≈ 8.9 tok/s` no matter how good the kernels are. The H200 reference measured
68.9 tok/s. Speculative decoding is not a micro-optimisation on a Spark — it is the
difference between a model you can use interactively and one you can't.

---

## 2. Models

Two families, both at the 27–31B class, both already on disk or downloading.

### Family A — Qwen3.8-27B (`Qwen/Qwen3.8-27B-FP8`, ~30.8 GB)

Hybrid linear attention: 64 layers, **48 `linear_attention` + 16 `full_attention`**
interleaved 3:1 (`full_attention_interval: 4`), Mamba-style with `linear_conv_kernel_dim: 4`
and an fp32 SSM state. `hidden_size` 5120, `head_dim` 256, vocab 248320,
`mtp_num_hidden_layers: 1`.

### Family B — gemma-4-31B-it (`nvidia/Gemma-4-31B-IT-NVFP4`, 31 GB, already local)

Sliding + full attention, 5:1, `sliding_window: 1024`. `hidden_size` 5376, vocab 262144.

### Why the pairing is the interesting part

Rejecting a speculated token in gemma means **truncating a KV cache**. In Qwen3.8 it means
**rolling back recurrent SSM state** across 48 of 64 layers. The cost curves diverge with
context length: gemma's KV cache grows and eats the very bandwidth speculation is trying to
save, while Qwen3.8's linear layers hold constant state. Stage 3 measures where they cross.

---

## 3. Method matrix

Everything below is verified present in the vLLM 0.28.0 registry and confirmed to have a
real published checkpoint. No speculative entries.

| Method | Class | Engine | Qwen3.8-27B | gemma-4-31B |
|---|---|---|---|---|
| baseline | — | vLLM | no spec | no spec |
| `ngram` | prompt lookup, no model | vLLM | ✅ | ✅ |
| `ngram_gpu` | GPU prompt lookup | vLLM | ✅ | ✅ |
| `suffix` | SuffixDecoding | **vLLM only** | ✅ | ✅ |
| `draft_model` / `STANDALONE` | classic draft–target | either | ✅ Qwen3.5-0.8B/2B/4B † | ✅ gemma-4-E2B/E4B/12B |
| MTP | native multi-token-prediction head | either | ✅ built into weights, 7-token | ✅ local `gemma-4-31b-it-assistant` |
| `eagle3` | feature-level autoregression | either | ❌ no head exists, any engine | ✅ `RedHatAI/…speculator.eagle3` |
| `dflash` | block-diffusion parallel drafting | either | — | ✅ `z-lab/gemma-4-31B-it-DFlash`, block 16 |
| `dflash` (v2) | + candidate-path selector | vLLM ≥0.28 / SGLang | ✅ `z-lab/Qwen3.8-27B-DFlash2`, block 8 | — |
| `dspark` | DFlash + sequential Markov head | either | ✅ `RadixArk/Qwen3.8-27B-DSpark`, block 7 | — |
| `medusa` | multi-head + tree attention | vLLM | ❌ no head published | ❌ no head published |
| `UNO` | LoRA-adapter draft on the target | **SGLang only** | ❌ no checkpoint | ❌ no checkpoint |

† **Cross-generation drafts.** Qwen3.8 ships only at 27B, but shares Qwen3.5's exact vocab
(248320) and special-token IDs, so Qwen3.5-0.8B/2B/4B load as drafts. That turns a gap into
a result: *what does acceptance rate cost you when the draft is a generation behind the
target?* Not previously published as far as I can find.

### 3.1 Engine policy

**No weight conversion, ever.** Where one engine can't run a setup, that setup runs on the
other engine and the run is labelled with the engine used. Concretely:

- **vLLM 0.28.0 is the default** — it covers the most methods and is what's already deployed
  on this box.
- **SGLang is the fallback** for anything vLLM can't load. Its algorithm set is
  `DFLASH · UNO · DSPARK · EAGLE · EAGLE3 · FROZEN_KV_MTP · STANDALONE · NGRAM`.
- **`suffix` is vLLM-only**; **`UNO` is SGLang-only**. Everything else overlaps.
- Cross-engine numbers are never compared head-to-head. Each engine gets its own
  no-spec baseline, and speedups are always computed *within* an engine.

### 3.2 Gaps, reported not patched

- **EAGLE3 for Qwen3.8-27B does not exist in any engine.** The nearest head
  (`VirVen/Qwen3.5-27B-EAGLE3-v2`) targets Qwen3.5-27B, a different model, and we are not
  converting weights. EAGLE3's real-world cost *is* that it needs a per-model trained head,
  and for a model this recent nobody has trained one. That is the finding.
- **Medusa** has no published head for either target.
- **UNO** has no published checkpoint for either target.

---

## 4. Datasets

Chosen to span the **copy-rate spectrum**, because that is what actually determines whether
a free method (`ngram`/`suffix`) can compete with a trained head.

### Turkish

| set | HF repo | shape | why |
|---|---|---|---|
| TR-MMLU | `AYueksel/TurkishMMLU` | short answers, low copy | worst case for spec dec; establishes the floor |
| MLSUM-tr | `reciTAL/mlsum` (`tr`) | long input, high copy | best case for `ngram`/`suffix` |
| Turkish chat | `merve/turkish_instructions` | long output, low copy | the realistic serving workload |

### Coding — standard benchmarks only

| set | HF repo | shape | why |
|---|---|---|---|
| HumanEval+ | `evalplus/humanevalplus` | short, scoreable | gives a real pass@1 quality gate |
| MBPP+ | `evalplus/mbppplus` | short, scoreable | second quality gate |
| RepoBench-py | `tianyang/repobench_python_v1.1` | long ctx, very high copy | where `ngram`/`suffix` should dominate |

### 4.1 What we can and can't claim about language

Six sets, as listed above. No cross-language control set is included.

That bounds one conclusion. Acceptance length will be reported per dataset, and the Turkish
sets will almost certainly differ from the coding sets — but **language and output domain
vary together across those two groups**, so a gap cannot be attributed to Turkish per se.
A Turkish head-to-head would need the same task in two languages (MLSUM ships `de/es/fr/ru/tr`
configs and would have served); that is deliberately out of scope here.

So the blog reports **per-dataset acceptance and the practical consequence for Turkish
serving** — which k and which method actually win on Turkish workloads on this box — without
claiming to have isolated language as the cause. Stated plainly rather than glossed, since
the honest version is still directly useful to anyone deploying Turkish on a Spark.

---

## 5. Stages

vLLM needs a server restart per `(method, k, draft)`, ~4 min each. Concurrency and context
length do **not** need a restart, so they're swept inside a single server.

| # | stage | varies | fixed | servers |
|---|---|---|---|---|
| 0 | Sanity + losslessness | method @ default k | c=1, 1K ctx, 50 prompts, temp 0 | reuses S1 |
| 1 | **k sweep** | method × k | c=1, ~1K ctx, 200 mixed prompts | ~50 |
| 2 | **Concurrency sweep** | c ∈ {1,2,4,8,16,32,64} | k\*, 1K ctx | 12 |
| 3 | **Context sweep** | ctx ∈ {1K,8K,32K,128K} × c ∈ {1,8} | k\* | 12 |
| 4 | Draft-size ablation | draft ∈ {3 sizes} × k | c=1 | 6 |
| 5 | Task + quality runs | 6 datasets × c ∈ {1,8} | k\* | 12 |

Stage 5's six sets: TR-MMLU · MLSUM-tr · Turkish chat · HumanEval+ · MBPP+ · RepoBench-py.

Block sizes cap `k`: DSpark 7, DFlash2 8, gemma DFlash 16, gemma EAGLE3 3 (default).

---

## 6. Metrics

**Per request:** TTFT, TPOT, end-to-end latency, output tok/s.

**Speculation:** acceptance length `τ = completion_tokens / verification_steps` — the same
definition the DFlash 2 card uses, so our numbers are directly comparable to their H200
column. Also per-position acceptance (how often draft token *i* survives), from
`vllm:spec_decode_num_accepted_tokens_total` / `num_draft_tokens_total` / `num_drafts_total`.

**Aggregate:** total output tok/s, speedup vs the no-spec baseline at the same concurrency.

**Quality:**
- *Losslessness* — at temp 0, token-for-token diff against the baseline. Reported as
  divergence rate and first-divergence index, not a boolean: a batched verify pass can
  differ from a batch-1 forward in the last bits of the logits at near-ties, and that
  distinction is worth showing rather than hiding.
- pass@1 on HumanEval+ / MBPP+; accuracy on TR-MMLU; ROUGE on MLSUM-tr.

---

## 7. Cost

Rough wall-clock at ~4 min load + measurement per server:

| stage | est. |
|---|---|
| 1 — k sweep | ~10 h |
| 2 — concurrency | ~5 h |
| 3 — context | ~6 h |
| 4 — draft ablation | ~1 h |
| 5 — task runs | ~8 h |
| **total** | **~30 h** |

Runs unattended overnight across two nights. **If that's too long, Stage 1 is the place to
cut** — trimming k to {2,4,7} per method roughly halves it.

---

## 8. Open items

1. **vLLM 0.28.0 upgrade.** Needed for `DFlash2DraftModel` (absent in 0.25.1/0.26/0.27).
   arm64 image exists (9.7 GB). Must re-verify gemma MTP still works after the upgrade —
   if it regresses, gemma MTP runs on the 0.25.1 image that works today, and the run is
   labelled accordingly (§3.1).
2. **DSpark routing.** `RadixArk/Qwen3.8-27B-DSpark` declares `DSparkDraftModel`, which the
   vLLM registry maps to a DeepSeek module, while its own `model_type` is `qwen3`. There is a
   separate `Qwen3DSparkModel` entry. Needs an empirical load test; SGLang is the fallback,
   and is likely the reference implementation for this checkpoint anyway (its DSPARK path has
   extra knobs: SPS cost table, per-position STS temperature table, ragged verify).
3. **Power.** Worth logging per-token energy if `tegrastats` exposes it on GB10 — "joules per
   token" is arguably the most honest metric for a desktop box.

## 8b. Measured gotchas (found during setup, before any benchmarking)

These are real findings from getting the engines running, and belong in the post.

1. **`--gpu-memory-utilization` means something different on unified memory.**
   On a discrete GPU it is a fraction of dedicated VRAM. On GB10 it is a fraction of
   *all 119 GB shared with the OS*. Setting `0.80` for a 31 GB model made vLLM claim
   ~95 GB and allocate a **63 GiB KV cache** (450k tokens) for a run needing a few
   hundred MB — which drove the box into swap, stalled server startup for ~9 minutes,
   and got the supervising shell script OOM-killed twice. `0.55` is ample for these
   targets; the 128K-context stage is the only one that needs it raised.
2. **vLLM 0.28 removed `--disable-log-requests`** (now `--enable-log-requests`, default
   off). Anything scripted against 0.25.1 — including the deployment currently on this
   box — breaks on upgrade.
3. **`vllm serve --help` needs `--gpus all`.** The parser is built from a live
   `VllmConfig`, so introspecting flags without a GPU raises rather than printing help.
4. **The Qwen3.8-27B-FP8 repo ships 66 per-layer shards**, not the 11 the Hub file
   listing suggests. Verify against `model.safetensors.index.json`, not file count.

### First measurement — and why the first attempt was wrong

Qwen3.8-27B-FP8, batch 1, greedy, 128 tokens:

| condition | tok/s | % of 8.84 tok/s ceiling |
|---|---|---|
| `gpu-memory-utilization 0.80`, swapping | 4.09 | 46% |
| **`0.55`, no swap** | **8.00** | **90.5%** |

The first number was memory-pressure artefact, not signal — worth recording because it is
exactly the trap gotcha #1 sets, and it very nearly became a published figure.

The clean number is the interesting one. **Batch-1 decode reaches 90% of the pure
bandwidth limit.** For contrast, the H200 in the DFlash 2 card hits 68.9 tok/s against its
own `4800 / 30.87 = 155.5` ceiling — only **44%**. The Spark sits far closer to its
roofline, because at 273 GB/s the weight read so dominates each step that fixed overheads
(kernel launch, sampling, Python) vanish into it, while at 4.8 TB/s those same overheads
are a large fraction of step time.

Two consequences, both of which sharpen the post's argument:

1. The roofline model in §1 is *more* predictive on this box than on a datacenter card.
2. There is essentially no headroom left in kernel optimisation — 90% of the ceiling is
   already reached. Speculative decoding is the only remaining lever, which is precisely
   the claim §1.3 makes.

At DFlash 2's reported 3.43×, this baseline projects to roughly **27 tok/s**.

## 9. Scope

Single node only. This box has a second DGX Spark attached and a multinode/Ray setup on
disk; **none of it is used.** Every number in this study comes from one GB10.
