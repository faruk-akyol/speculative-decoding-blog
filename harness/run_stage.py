#!/usr/bin/env python3
"""Stage orchestrator for the speculative-decoding sweep.

    python run_stage.py 1            # k sweep
    python run_stage.py 2 --k-json results/k_star.json
    python run_stage.py 0 --quick    # sanity + losslessness only

Every run appends one JSON line to results/raw/stage<N>.jsonl, so a crashed or
killed sweep resumes without redoing completed configs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config_matrix as CM
from bench import RunResult, run_benchmark
from datasets_loader import SPECS, load, load_mixed_pool
from server import Server, preflight

RAW = "results/raw"


def _done_labels(path: str) -> set[str]:
    """Labels already recorded, so a resumed sweep skips them."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            try:
                out.add(json.loads(line)["run_label"])
            except Exception:
                continue
    return out


def _record(path: str, cfg, res: RunResult, extra: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        "run_label": extra.pop("run_label"),
        "config": cfg.label,
        "family": cfg.family,
        "method": cfg.method or "baseline",
        "k": cfg.k,
        "draft": cfg.draft,
        "image": cfg.image_key,
        "concurrency": res.concurrency,
        "wall_s": round(res.wall_s, 3),
        "n_ok": res.n_ok,
        "n_fail": res.n_fail,
        "total_output_tokens": res.total_output_tokens,
        "output_tok_per_s": round(res.output_tok_per_s, 2),
        "ttft_p50": res.ttft_p50,
        "ttft_p95": res.ttft_p95,
        "tpot_p50": res.tpot_p50,
        **res.spec,
        **extra,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    # Per-request detail lives beside the summary, not inside it.
    detail = os.path.join(RAW, "requests", f"{row['run_label']}.json")
    os.makedirs(os.path.dirname(detail), exist_ok=True)
    with open(detail, "w") as f:
        json.dump(res.requests, f)


# --------------------------------------------------------------------------


def stage1(args) -> None:
    """k sweep at concurrency 1 on the mixed pool -> finds k* per method."""
    out = os.path.join(RAW, "stage1.jsonl")
    done = _done_labels(out)
    prompts = load_mixed_pool(args.n)
    configs = CM.stage1_configs()
    print(f"stage 1: {len(configs)} configs, {len(prompts)} prompts, {len(done)} already done")

    for i, cfg in enumerate(configs, 1):
        label = f"s1::{cfg.label}::c1"
        if label in done:
            print(f"[{i}/{len(configs)}] skip {cfg.label} (done)")
            continue
        print(f"[{i}/{len(configs)}] {cfg.label} ...", flush=True)
        try:
            with Server(cfg) as s:
                res = asyncio.run(run_benchmark(s.base, "target", prompts, 1, cfg.label))
            _record(out, cfg, res, {"run_label": label, "stage": 1})
            print(f"    {res.output_tok_per_s:.1f} tok/s  tau={res.spec.get('acceptance_length')}")
        except Exception as e:  # noqa: BLE001 - one bad config must not kill the sweep
            print(f"    FAILED: {e}")
            with open(os.path.join(RAW, "failures.log"), "a") as f:
                f.write(f"{label}\t{e}\n")


def stage2(args) -> None:
    """Concurrency sweep at k*. Tests H1 - the c* = B*/(k+1) prediction."""
    out = os.path.join(RAW, "stage2.jsonl")
    done = _done_labels(out)
    prompts = load_mixed_pool(args.n)
    configs = _k_star_configs(args.k_json)
    print(f"stage 2: {len(configs)} configs x {len(CM.CONCURRENCIES)} concurrencies")

    for cfg in configs:
        if all(f"s2::{cfg.label}::c{c}" in done for c in CM.CONCURRENCIES):
            print(f"skip {cfg.label} (done)")
            continue
        print(f"{cfg.label} ...", flush=True)
        try:
            with Server(cfg) as s:
                for c in CM.CONCURRENCIES:
                    label = f"s2::{cfg.label}::c{c}"
                    if label in done:
                        continue
                    res = asyncio.run(run_benchmark(s.base, "target", prompts, c, cfg.label))
                    _record(out, cfg, res, {"run_label": label, "stage": 2})
                    print(f"    c={c:<3} {res.output_tok_per_s:8.1f} tok/s  "
                          f"tau={res.spec.get('acceptance_length')}")
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")
            with open(os.path.join(RAW, "failures.log"), "a") as f:
                f.write(f"s2::{cfg.label}\t{e}\n")


def stage5(args) -> None:
    """Per-dataset task runs at k*, for the Turkish-vs-coding comparison."""
    out = os.path.join(RAW, "stage5.jsonl")
    done = _done_labels(out)
    configs = _k_star_configs(args.k_json)
    per_ds = {name: load(name, args.n) for name in SPECS}

    for cfg in configs:
        print(f"{cfg.label} ...", flush=True)
        try:
            with Server(cfg) as s:
                for ds_name, prompts in per_ds.items():
                    for c in (1, 8):
                        label = f"s5::{cfg.label}::{ds_name}::c{c}"
                        if label in done:
                            continue
                        res = asyncio.run(
                            run_benchmark(s.base, "target", prompts, c, cfg.label, keep_text=True)
                        )
                        _record(out, cfg, res,
                                {"run_label": label, "stage": 5, "dataset": ds_name,
                                 "language": SPECS[ds_name].language,
                                 "copy_rate": SPECS[ds_name].copy_rate})
                        print(f"    {ds_name:<16} c={c} {res.output_tok_per_s:8.1f} tok/s  "
                              f"tau={res.spec.get('acceptance_length')}")
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")


def _k_star_configs(k_json: str) -> list:
    """Rebuild one Config per method using the k* chosen in Stage 1."""
    with open(k_json) as f:
        k_star = json.load(f)          # {"qwen-dspark": 7, "gemma-eagle3": 3, ...}
    by_label = {c.label: c for c in CM.stage1_configs()}
    out = []
    for method_label, k in k_star.items():
        want = method_label if k is None else f"{method_label}-k{k}"
        cfg = by_label.get(want) or by_label.get(method_label)
        if cfg is None:
            print(f"  warn: no config for {want}, skipping")
            continue
        out.append(cfg)
    return out


STAGES = {1: stage1, 2: stage2, 5: stage5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=int, choices=sorted(STAGES))
    ap.add_argument("-n", type=int, default=200, help="prompts per run")
    ap.add_argument("--k-json", default="results/k_star.json")
    ap.add_argument("--quick", action="store_true", help="tiny sample, for smoke-testing")
    args = ap.parse_args()
    if args.quick:
        args.n = 20
    preflight()
    os.makedirs(RAW, exist_ok=True)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
