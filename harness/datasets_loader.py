"""Dataset loading and prompt formatting for the speculative-decoding study.

Six sets, chosen in EXPERIMENTS.md §4 to span the *copy rate* spectrum - how much
of the output can be lifted verbatim from the prompt. That is what decides whether
a free method (ngram/suffix) can compete with a trained draft head, so it is the
axis the whole dataset choice is organised around.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from datasets import load_dataset

SEED = 20260905


@dataclass
class Prompt:
    """One benchmark request."""

    text: str
    dataset: str
    max_tokens: int
    # Free-form extras used by the scorers (canonical solution, answer key, ...).
    meta: dict = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    language: str  # "tr" | "en"
    # Qualitative copy rate: how much output is recoverable from the prompt.
    # Drives the expectation for ngram/suffix; see EXPERIMENTS.md §4.
    copy_rate: str  # "low" | "medium" | "high"
    max_tokens: int
    loader: str


SPECS: dict[str, DatasetSpec] = {
    "tr_mmlu": DatasetSpec("tr_mmlu", "tr", "low", 256, "_load_tr_mmlu"),
    "mlsum_tr": DatasetSpec("mlsum_tr", "tr", "high", 256, "_load_mlsum_tr"),
    "tr_chat": DatasetSpec("tr_chat", "tr", "low", 512, "_load_tr_chat"),
    "humaneval_plus": DatasetSpec("humaneval_plus", "en", "medium", 512, "_load_humaneval"),
    "mbpp_plus": DatasetSpec("mbpp_plus", "en", "medium", 512, "_load_mbpp"),
    "repobench_py": DatasetSpec("repobench_py", "en", "high", 256, "_load_repobench"),
}


# --------------------------------------------------------------------------
# Loaders. Each returns a list[Prompt]; `n` caps the sample, sampled with a
# fixed seed so every method sees byte-identical prompts.
# --------------------------------------------------------------------------


def _load_tr_mmlu(n: int) -> list[Prompt]:
    ds = load_dataset("AYueksel/TurkishMMLU", split="test")
    out = []
    for r in ds:
        choices = "\n".join(
            f"{letter}) {r[key]}"
            for letter, key in zip("ABCDE", ["choice_a", "choice_b", "choice_c", "choice_d", "choice_e"])
            if r.get(key)
        )
        out.append(
            Prompt(
                text=(
                    f"Aşağıdaki çoktan seçmeli soruyu cevapla. Önce kısaca gerekçelendir, "
                    f"sonra 'Cevap: X' biçiminde tek harfle bitir.\n\n"
                    f"Soru: {r['question']}\n{choices}\n\nCevap:"
                ),
                dataset="tr_mmlu",
                max_tokens=256,
                meta={"answer": r.get("answer")},
            )
        )
    return _sample(out, n)


def _load_mlsum_tr(n: int) -> list[Prompt]:
    ds = load_dataset("reciTAL/mlsum", "tu", split="test")
    out = []
    for r in ds:
        # Truncate the article so prompts stay in a comparable length band.
        article = " ".join(r["text"].split()[:600])
        out.append(
            Prompt(
                text=(
                    "Aşağıdaki haber metnini iki üç cümleyle özetle.\n\n"
                    f"Metin: {article}\n\nÖzet:"
                ),
                dataset="mlsum_tr",
                max_tokens=256,
                meta={"reference": r["summary"]},
            )
        )
    return _sample(out, n)


def _load_tr_chat(n: int) -> list[Prompt]:
    ds = load_dataset("merve/turkish_instructions", split="train")
    out = []
    for r in ds:
        instruction = (r.get("talimat") or "").strip()
        context = (r.get("giriş") or "").strip()
        if not instruction:
            continue
        text = instruction if not context else f"{instruction}\n\n{context}"
        out.append(Prompt(text=text, dataset="tr_chat", max_tokens=512))
    return _sample(out, n)


def _load_humaneval(n: int) -> list[Prompt]:
    ds = load_dataset("evalplus/humanevalplus", split="test")
    out = [
        Prompt(
            text=r["prompt"],
            dataset="humaneval_plus",
            max_tokens=512,
            meta={"task_id": r["task_id"], "test": r.get("test"), "entry_point": r.get("entry_point")},
        )
        for r in ds
    ]
    return _sample(out, n)


def _load_mbpp(n: int) -> list[Prompt]:
    ds = load_dataset("evalplus/mbppplus", split="test")
    out = [
        Prompt(
            text=(
                f"\"\"\"{r['prompt'].strip()}\n"
                f"{(r.get('assertion') or '').strip()}\n\"\"\"\n"
            ),
            dataset="mbpp_plus",
            max_tokens=512,
            meta={"task_id": r.get("task_id"), "test": r.get("test")},
        )
        for r in ds
    ]
    return _sample(out, n)


def _load_repobench(n: int) -> list[Prompt]:
    # cross_file_first exercises the high-copy regime: the completion often
    # repeats identifiers and call patterns that appear in the in-context files,
    # which is exactly what ngram/suffix exploit.
    ds = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first")
    out = [
        Prompt(
            text=f"{r['all_code']}",
            dataset="repobench_py",
            max_tokens=256,
            meta={"next_line": r.get("next_line")},
        )
        for r in ds
    ]
    return _sample(out, n)


def _sample(items: list[Prompt], n: int) -> list[Prompt]:
    rng = random.Random(SEED)
    if n >= len(items):
        return items
    return rng.sample(items, n)


def load(name: str, n: int = 200) -> list[Prompt]:
    """Load `n` prompts from dataset `name`, deterministically sampled."""
    spec = SPECS[name]
    return globals()[spec.loader](n)


def load_mixed_pool(n: int = 200) -> list[Prompt]:
    """Balanced pool across all six sets, used for the Stage 1 k sweep.

    Stage 1 finds a single k* per method on this pool and Stage 5 reuses it. That
    assumes acceptance does not vary wildly by dataset -- Stage 0 measures the
    variance and flags it if the assumption looks unsafe (see EXPERIMENTS.md §5).
    """
    per = max(1, n // len(SPECS))
    pool: list[Prompt] = []
    for name in SPECS:
        pool.extend(load(name, per))
    rng = random.Random(SEED)
    rng.shuffle(pool)
    return pool[:n]
