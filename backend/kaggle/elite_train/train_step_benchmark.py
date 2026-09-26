"""Kaggle GPU benchmark: seconds per training step on one GPU vs DataParallel over two.

Runs a short, non-saving training (400 steps from a 300k buffer) twice with the
production code, plus a CPU timing of the replay sampler at the real 3M buffer
size, so a long run's settings can be chosen from measurements. Nothing is
written to any dataset.
"""
from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
STEPS = 400


def find_code() -> Path:
    for hit in sorted(INPUT.rglob("board_encoding.py")):
        base = hit.parent.parent.parent
        if "chess-engine-code" in base.parts and (base / "scripts").exists():
            return base
    for archive in sorted(INPUT.rglob("chess_engine_code.zip")):
        target = WORK / "code"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return target
    raise FileNotFoundError("code dataset not attached")


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "chess", "PyYAML", "zstandard"], check=True)
    code = find_code()
    shards = sorted({p.parent for p in INPUT.rglob("*.npz")}, key=lambda p: -len(list(p.glob("*.npz"))))[0]
    model = sorted(INPUT.rglob("external_best_model.pth"))[0]
    print(f"[BENCH] code={code} shards={shards} model={model}", flush=True)

    config = WORK / "bench.yaml"
    config.write_text("\n".join([
        "external:",
        f"  samples_path: '{shards}'",
        "  max_samples: 300000",
        "  validation_split: 0.1",
        "  shuffle: true",
        "  dedup: true",
        "",
        "training:",
        "  buffer_size: 300000",
        "  batch_size: 128",
        "  epochs: 1",
        f"  train_steps_per_iter: {STEPS}",
        "  lr: 0.0002",
        "  use_amp: false",
        "",
        "replay:",
        "  capacity: 300000",
        "  recent_sample_fraction: 0.0",
        "",
    ]), encoding="utf-8")

    results = {}
    for label, extra_env in (("1 GPU", {"CUDA_VISIBLE_DEVICES": "0"}), ("2 GPUs", {})):
        env = dict(os.environ, PYTHONPATH=str(code), **extra_env)
        cmd = [sys.executable, "-m", "app.cli.train_external", "--config", str(config), "--device", "cuda",
               "--iterations", "1", "--save-dir", str(WORK / f"bench_{len(results)}"), "--base-model", str(model),
               "--no-save", "--max-val-samples", "2000"]
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True)
        seconds = [float(m) for m in re.findall(r"sec_per_step=([0-9.]+)", proc.stdout)][2:]  # skip warm-up
        for line in (proc.stdout + proc.stderr).splitlines():
            if "DataParallel" in line or "Traceback" in line or "Error" in line:
                print(f"[{label}] {line}", flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-3000:], flush=True)
        results[label] = statistics.median(seconds) if seconds else None
        print(f"[BENCH] {label}: exit={proc.returncode} median sec/step={results[label]} "
              f"(n={len(seconds)}) wall={time.perf_counter() - started:.0f}s", flush=True)

    sys.path.insert(0, str(code))
    import numpy as np
    from app.training.replay_buffer import _uniform_choice_without_replacement

    n, w = 3_000_000, 800_000
    active = np.arange(n, dtype=np.int64)

    def old():
        rc = active[n - w:][np.random.choice(w, size=102, replace=False)]
        pool = active[np.isin(active, rc, invert=True)]
        return pool[np.random.choice(pool.size, size=26, replace=False)]

    for name, fn in (("old sampler", old), ("new sampler", lambda: _uniform_choice_without_replacement(n, 128))):
        fn()
        t = time.perf_counter()
        for _ in range(10):
            fn()
        print(f"[BENCH] {name} at 3M: {(time.perf_counter() - t) / 10 * 1000:.1f} ms/step", flush=True)
    print(f"[BENCH] RESULT {results}", flush=True)


if __name__ == "__main__":
    main()
