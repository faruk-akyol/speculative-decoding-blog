"""Concurrency-controlled benchmark client + speculative-decoding metrics.

Measures what EXPERIMENTS.md §6 asks for: TTFT, TPOT, end-to-end latency,
output tok/s, and acceptance length. Streaming is always on, because TTFT is
otherwise unmeasurable and TPOT would silently absorb prefill time.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field

import aiohttp

from datasets_loader import Prompt


# --------------------------------------------------------------------------
# Metrics scraping
# --------------------------------------------------------------------------

_SPEC_RE = re.compile(r"^(vllm:spec_decode_[a-z_]+(?:_total)?)\s+([0-9.eE+-]+)$", re.M)


async def scrape_spec_metrics(session: aiohttp.ClientSession, base: str) -> dict[str, float]:
    """Pull the spec-decode counters off /metrics.

    Counters are cumulative for the life of the server, so callers must diff a
    before/after pair rather than reading absolute values.
    """
    try:
        async with session.get(f"{base}/metrics", timeout=aiohttp.ClientTimeout(total=30)) as r:
            body = await r.text()
    except Exception:
        return {}
    return {m.group(1): float(m.group(2)) for m in _SPEC_RE.finditer(body)}


def acceptance_length(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    """Acceptance length tau, computed two independent ways as a cross-check.

    vLLM proposes k tokens per draft step and accepts 0..k of them, then always
    emits one bonus token from the verify pass itself. So tokens emitted per
    verification step is (accepted + 1), and

        tau = num_accepted / num_drafts + 1

    This matches the definition used in the DFlash 2 model card ("completion
    tokens divided by verification steps"), which is what makes our numbers
    directly comparable to their H200 column.
    """
    def d(key: str) -> float | None:
        if key not in after:
            return None
        return after[key] - before.get(key, 0.0)

    accepted = d("vllm:spec_decode_num_accepted_tokens_total")
    drafted = d("vllm:spec_decode_num_draft_tokens_total")
    drafts = d("vllm:spec_decode_num_drafts_total")

    out: dict[str, float | None] = {
        "accepted_tokens": accepted,
        "draft_tokens": drafted,
        "draft_steps": drafts,
        "acceptance_length": None,
        "draft_token_accept_rate": None,
    }
    if accepted is not None and drafts:
        out["acceptance_length"] = accepted / drafts + 1.0
    if accepted is not None and drafted:
        # Fraction of *proposed* tokens that survived - the quantity that decides
        # whether a larger k is still paying for itself.
        out["draft_token_accept_rate"] = accepted / drafted
    return out


# --------------------------------------------------------------------------
# Request driver
# --------------------------------------------------------------------------


@dataclass
class RequestResult:
    dataset: str
    ok: bool
    ttft_s: float | None = None
    e2e_s: float | None = None
    output_tokens: int = 0
    text: str = ""
    error: str | None = None

    @property
    def tpot_s(self) -> float | None:
        """Time per output token, excluding prefill."""
        if not self.ok or self.ttft_s is None or self.e2e_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)


async def _one_request(
    session: aiohttp.ClientSession, base: str, model: str, p: Prompt, temperature: float
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": p.text,
        "max_tokens": p.max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    ttft: float | None = None
    ntok = 0
    chunks: list[str] = []
    try:
        async with session.post(
            f"{base}/v1/completions", json=payload, timeout=aiohttp.ClientTimeout(total=1800)
        ) as r:
            if r.status != 200:
                return RequestResult(p.dataset, False, error=f"HTTP {r.status}: {(await r.text())[:200]}")
            async for raw in r.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for ch in obj.get("choices") or []:
                    piece = ch.get("text") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        chunks.append(piece)
                # The final usage chunk is authoritative for token count; the
                # number of SSE chunks is not (a chunk can carry many tokens
                # under speculative decoding, which is the whole point).
                if obj.get("usage"):
                    ntok = obj["usage"].get("completion_tokens", 0) or 0
    except asyncio.TimeoutError:
        return RequestResult(p.dataset, False, error="timeout")
    except Exception as e:  # noqa: BLE001 - record and continue the sweep
        return RequestResult(p.dataset, False, error=f"{type(e).__name__}: {e}")

    e2e = time.perf_counter() - start
    text = "".join(chunks)
    return RequestResult(p.dataset, True, ttft, e2e, ntok or len(chunks), text)


@dataclass
class RunResult:
    label: str
    concurrency: int
    wall_s: float
    n_ok: int
    n_fail: int
    total_output_tokens: int
    output_tok_per_s: float
    ttft_p50: float | None
    ttft_p95: float | None
    tpot_p50: float | None
    spec: dict = field(default_factory=dict)
    requests: list[dict] = field(default_factory=list)


async def run_benchmark(
    base: str,
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    label: str,
    temperature: float = 0.0,
    keep_text: bool = False,
) -> RunResult:
    """Drive `prompts` through the server at fixed `concurrency`.

    Throughput is measured over the whole wall-clock window, so it reflects the
    sustained rate at that concurrency rather than a best-case single request.
    """
    conn = aiohttp.TCPConnector(limit=max(concurrency * 2, 16))
    async with aiohttp.ClientSession(connector=conn) as session:
        before = await scrape_spec_metrics(session, base)

        sem = asyncio.Semaphore(concurrency)

        async def worker(p: Prompt) -> RequestResult:
            async with sem:
                return await _one_request(session, base, model, p, temperature)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(p) for p in prompts))
        wall = time.perf_counter() - t0

        after = await scrape_spec_metrics(session, base)

    ok = [r for r in results if r.ok]
    ttfts = sorted(r.ttft_s for r in ok if r.ttft_s is not None)
    tpots = sorted(t for t in (r.tpot_s for r in ok) if t is not None)
    total_out = sum(r.output_tokens for r in ok)

    def pct(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    rows = []
    for r in results:
        d = asdict(r)
        d["tpot_s"] = r.tpot_s
        if not keep_text:
            d.pop("text", None)
        rows.append(d)

    return RunResult(
        label=label,
        concurrency=concurrency,
        wall_s=wall,
        n_ok=len(ok),
        n_fail=len(results) - len(ok),
        total_output_tokens=total_out,
        output_tok_per_s=total_out / wall if wall else 0.0,
        ttft_p50=statistics.median(ttfts) if ttfts else None,
        ttft_p95=pct(ttfts, 0.95),
        tpot_p50=statistics.median(tpots) if tpots else None,
        spec=acceptance_length(before, after),
        requests=rows,
    )
