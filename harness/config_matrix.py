"""The (family x method x k x draft) configuration matrix.

Mirrors the method table in EXPERIMENTS.md §3. Every entry here corresponds to a
checkpoint that exists on disk; methods with no published head for these targets
(Medusa, UNO, EAGLE3-for-Qwen) are absent by design and reported as gaps rather
than stubbed out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

M = "/models"  # container-side model root


@dataclass
class Config:
    label: str
    family: str          # "qwen" | "gemma"
    model: str           # container path to the target
    method: str | None   # None = no-speculation baseline
    k: int | None = None
    draft: str | None = None          # container path to the draft/head
    image_key: str = "vllm"
    max_model_len: int = 8192
    max_num_seqs: int = 64
    gpu_mem_util: float = 0.55
    quantization: str | None = None
    text_only: bool = True            # both targets are VLMs; we benchmark text
    spec_overrides: dict = field(default_factory=dict)

    def spec_config(self) -> dict | None:
        if self.method is None:
            return None
        cfg: dict = {"method": self.method}
        if self.draft:
            cfg["model"] = self.draft
        if self.k is not None:
            cfg["num_speculative_tokens"] = self.k
        cfg.update(self.spec_overrides)
        return cfg

    def extra_args(self) -> list[str]:
        args: list[str] = []
        if self.quantization:
            args += ["--quantization", self.quantization]
        if self.text_only:
            # Disabling the vision path frees memory and removes an irrelevant
            # source of variance from text-only latency measurements.
            args += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
        sc = self.spec_config()
        if sc:
            args += ["--speculative-config", json.dumps(sc)]
        return args


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

QWEN = dict(family="qwen", model=f"{M}/qwen3.8-27b-fp8")
GEMMA = dict(family="gemma", model=f"{M}/gemma-4-31b-it-nvfp4", quantization="modelopt")

# Block sizes come from each head's config.json and cap the usable k.
BLOCK_LIMITS = {
    "qwen_dspark": 7,     # RadixArk/Qwen3.8-27B-DSpark   block_size 7
    "qwen_dflash2": 8,    # z-lab/Qwen3.8-27B-DFlash2     block_size 8
    "gemma_dflash": 16,   # z-lab/gemma-4-31B-it-DFlash   block_size 16
    "qwen_mtp": 7,        # Qwen3.8 ships a 7-token MTP head
}


def _c(label: str, base: dict, **kw) -> Config:
    return Config(label=label, **base, **kw)


# --------------------------------------------------------------------------
# Stage 1 - k sweep. One entry per (method, k).
# --------------------------------------------------------------------------

K_FULL = [1, 2, 3, 4, 5, 6, 7, 8]


def stage1_configs() -> list[Config]:
    out: list[Config] = [
        _c("qwen-baseline", QWEN, method=None),
        _c("gemma-baseline", GEMMA, method=None),
    ]

    # -- Qwen3.8-27B --
    for k in [k for k in K_FULL if k <= BLOCK_LIMITS["qwen_mtp"]]:
        out.append(_c(f"qwen-mtp-k{k}", QWEN, method="qwen3_5_mtp", k=k))
    for k in [k for k in K_FULL if k <= BLOCK_LIMITS["qwen_dspark"]]:
        out.append(_c(f"qwen-dspark-k{k}", QWEN, method="dspark",
                      draft=f"{M}/qwen3.8-27b-dspark", k=k))
    for k in [k for k in K_FULL if k <= BLOCK_LIMITS["qwen_dflash2"]]:
        out.append(_c(f"qwen-dflash2-k{k}", QWEN, method="dflash",
                      draft=f"{M}/qwen3.8-27b-dflash2", k=k))
    for k in K_FULL:
        out.append(_c(f"qwen-ngram-k{k}", QWEN, method="ngram", k=k,
                      spec_overrides={"prompt_lookup_max": 4, "prompt_lookup_min": 2}))
    out.append(_c("qwen-suffix", QWEN, method="suffix"))
    for k in [2, 4, 6]:
        out.append(_c(f"qwen-draft0.8b-k{k}", QWEN, method="draft_model",
                      draft=f"{M}/qwen3.5-0.8b", k=k))

    # -- gemma-4-31B --
    for k in K_FULL:
        out.append(_c(f"gemma-mtp-k{k}", GEMMA, method="mtp",
                      draft=f"{M}/gemma-4-31b-it-assistant", k=k))
    for k in [1, 2, 3, 4, 5]:
        out.append(_c(f"gemma-eagle3-k{k}", GEMMA, method="eagle3",
                      draft=f"{M}/gemma-4-31b-eagle3", k=k))
    for k in [k for k in K_FULL if k <= BLOCK_LIMITS["gemma_dflash"]] + [11, 15]:
        out.append(_c(f"gemma-dflash-k{k}", GEMMA, method="dflash",
                      draft=f"{M}/gemma-4-31b-dflash", k=k))
    for k in K_FULL:
        out.append(_c(f"gemma-ngram-k{k}", GEMMA, method="ngram", k=k,
                      spec_overrides={"prompt_lookup_max": 4, "prompt_lookup_min": 2}))
    out.append(_c("gemma-suffix", GEMMA, method="suffix"))
    for k in [2, 4, 6]:
        out.append(_c(f"gemma-draftE2B-k{k}", GEMMA, method="draft_model",
                      draft=f"{M}/gemma-4-e2b-it", k=k))

    return out


# --------------------------------------------------------------------------
# Stage 4 - draft-size ablation, including the cross-generation question:
# Qwen3.8 ships only at 27B, so its drafts are Qwen3.5 models a generation
# behind the target (see EXPERIMENTS.md §3, footnote).
# --------------------------------------------------------------------------


def stage4_configs(k: int = 4) -> list[Config]:
    out = []
    for size in ["0.8b", "2b", "4b"]:
        out.append(_c(f"qwen-draft{size}-k{k}", QWEN, method="draft_model",
                      draft=f"{M}/qwen3.5-{size}", k=k))
    for name in ["e2b-it", "e4b-it", "12b-it"]:
        out.append(_c(f"gemma-draft{name}-k{k}", GEMMA, method="draft_model",
                      draft=f"{M}/gemma-4-{name}", k=k))
    return out


# --------------------------------------------------------------------------
# Stage 2 / 3 sweep axes. These vary inside a single loaded server, so they cost
# no extra restarts - which is why the grid can afford to be wide here.
# --------------------------------------------------------------------------

CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
CONTEXT_LENGTHS = [1024, 8192, 32768, 131072]
