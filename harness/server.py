"""Engine lifecycle: start a vLLM or SGLang server in Docker, wait for health, tear down.

Every benchmark config gets a fresh server. vLLM bakes the speculative config into
engine construction, so (method, k, draft) cannot be varied without a restart --
which is why the stage design in EXPERIMENTS.md §5 sweeps concurrency and context
*inside* one server and pays the restart cost only for method/k changes.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request

MODELS_HOST = "/home/spark/models"
MODELS_CONTAINER = "/models"
PORT = 8100
BASE = f"http://localhost:{PORT}"

IMAGES = {
    "vllm": "vllm/vllm-openai:v0.28.0",
    # Kept as the labelled fallback if 0.28 regresses a config that 0.25.1 serves
    # (EXPERIMENTS.md §3.1). Runs are always tagged with the image actually used.
    "vllm_025": "vllm/vllm-openai:v0.25.1",
}


class ServerError(RuntimeError):
    pass


class Server:
    """Context manager around one engine process.

    Usage:
        with Server(cfg) as s:
            ...  # s.base is ready to take requests
    """

    def __init__(self, cfg, name: str = "specbench", startup_timeout_s: int = 900):
        self.cfg = cfg
        self.name = name
        self.startup_timeout_s = startup_timeout_s
        self.base = BASE
        self.log_path: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Server":
        self._rm()
        cmd = self._docker_cmd()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self._wait_healthy()
        except Exception:
            self.dump_logs()
            self._rm()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.dump_logs()
        self._rm()
        # Unified memory needs a moment to actually come back before the next
        # 30 GB model starts loading; skipping this causes spurious OOMs.
        time.sleep(10)

    # -- internals ---------------------------------------------------------

    def _docker_cmd(self) -> list[str]:
        image = IMAGES[self.cfg.image_key]
        cmd = [
            "docker", "run", "-d", "--name", self.name,
            "--gpus", "all", "--ipc=host",
            "-v", f"{MODELS_HOST}:{MODELS_CONTAINER}",
            "-p", f"{PORT}:{PORT}",
            image,
            self.cfg.model,
            "--served-model-name", "target",
            "--port", str(PORT),
            "--max-model-len", str(self.cfg.max_model_len),
            "--max-num-seqs", str(self.cfg.max_num_seqs),
            "--gpu-memory-utilization", str(self.cfg.gpu_mem_util),
        ]
        cmd += self.cfg.extra_args()
        return cmd

    def _wait_healthy(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if not self._alive():
                raise ServerError(f"container exited during startup: {self._last_error()}")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                pass
            time.sleep(2)
        raise ServerError(f"timed out after {self.startup_timeout_s}s: {self._last_error()}")

    def _alive(self) -> bool:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        ).stdout
        return self.name in out.split()

    def _logs(self) -> str:
        return subprocess.run(
            ["docker", "logs", self.name], capture_output=True, text=True
        ).stdout + subprocess.run(
            ["docker", "logs", self.name], capture_output=True, text=True
        ).stderr

    def _last_error(self) -> str:
        lines = [
            ln for ln in self._logs().splitlines()
            if any(t in ln for t in ("Error", "error", "Traceback", "ValueError", "not supported", "Unrecognized"))
        ]
        return " | ".join(lines[-3:]) or "(no error lines in log)"

    def dump_logs(self, out_dir: str = "results/raw/serverlogs") -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, f"{self.cfg.label}.log")
        with open(self.log_path, "w") as f:
            f.write(self._logs())

    def _rm(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def preflight() -> None:
    """Fail loudly before a 30 h sweep rather than 6 h into it."""
    if not os.path.isdir(MODELS_HOST):
        raise ServerError(f"missing model root {MODELS_HOST}")
    have = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                          capture_output=True, text=True).stdout.split()
    for key, img in IMAGES.items():
        if img not in have:
            raise ServerError(f"missing docker image for '{key}': {img}")
