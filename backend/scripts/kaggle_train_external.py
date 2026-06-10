from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SAMPLES_PATH = "auto"
DEFAULT_CHECKPOINT_INPUT_DIR = "auto"
DEFAULT_KAGGLE_DATASET_ID = os.environ.get("KAGGLE_CHECKPOINT_DATASET_ID", "jadakil/external-model-checkpoints")
DEFAULT_SAVE_DIR = "/kaggle/working/checkpoints"
DEFAULT_CONFIG_PATH = "/kaggle/working/external_training_kaggle.yaml"
DEFAULT_AUTOSAVE_DIR = "/kaggle/working/autosaves"


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _find_external_samples() -> str:
    input_root = Path("/kaggle/input")
    candidates = []
    if input_root.exists():
        for path in input_root.rglob("external_samples.npz"):
            if path.is_dir() or path.is_file():
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find external_samples.npz under /kaggle/input. "
            "Pass --samples-path explicitly."
        )
    candidates.sort(key=lambda p: (0 if p.is_dir() else 1, len(str(p)), str(p)))
    return str(candidates[0])


def _find_checkpoint_input_dir(dataset_id: str | None = None) -> str | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None

    candidates = []
    for path in input_root.rglob("external_latest_checkpoint.pth"):
        candidates.append(path.parent)
    for path in input_root.rglob("external_best_model.pth"):
        candidates.append(path.parent)

    if not candidates:
        return None
    preferred_slug = ""
    if dataset_id and "/" in dataset_id:
        preferred_slug = dataset_id.split("/", 1)[1].strip().lower()
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if preferred_slug and preferred_slug in str(p).lower() else 1,
            len(str(p)),
            str(p),
        ),
    )
    return str(candidates[0])


def _resolve_auto_paths(args: argparse.Namespace) -> None:
    if args.samples_path == "auto":
        args.samples_path = _find_external_samples()
        print(f"[AUTO] samples_path={args.samples_path}", flush=True)

    if args.checkpoint_input_dir == "auto":
        found = _find_checkpoint_input_dir(args.kaggle_dataset_id)
        args.checkpoint_input_dir = found
        if found:
            print(f"[AUTO] checkpoint_input_dir={found}", flush=True)
        else:
            print("[AUTO] no checkpoint input found; starting from scratch unless --base-model is set", flush=True)


def _write_config(args: argparse.Namespace) -> Path:
    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "external:",
        f"  samples_path: {_yaml_string(args.samples_path)}",
        f"  max_samples: {int(args.max_samples)}",
        f"  shuffle: {str(args.shuffle).lower()}",
        f"  validation_split: {float(args.validation_split)}",
        f"  seed: {int(args.seed)}",
        f"  dedup: {str(args.dedup).lower()}",
        f"  filter_invalid: {str(args.filter_invalid).lower()}",
        f"  drop_zero_states: {str(args.drop_zero_states).lower()}",
        "  checkpoint_prefix: external",
        f"  save_dir: {_yaml_string(args.save_dir)}",
        f"  benchmark_games: {int(args.benchmark_games)}",
        "",
        "training:",
        f"  buffer_size: {int(args.buffer_size)}",
        f"  batch_size: {int(args.batch_size)}",
        f"  epochs: {int(args.epochs)}",
        f"  train_steps_per_iter: {int(args.train_steps_per_iter)}",
        "",
        "system:",
        f"  checkpoint_path: {_yaml_string(Path(args.save_dir) / 'external_best_model.pth')}",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[CONFIG] wrote {config_path}", flush=True)
    return config_path


def _copy_checkpoint_inputs(args: argparse.Namespace) -> str | None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_model = args.base_model
    input_dir = Path(args.checkpoint_input_dir) if args.checkpoint_input_dir else None
    if input_dir and input_dir.exists():
        for name in ("external_latest_checkpoint.pth", "external_best_model.pth", "external_history.json"):
            src = input_dir / name
            if src.exists():
                dst = save_dir / name
                shutil.copy2(src, dst)
                print(f"[CHECKPOINT] copied {src} -> {dst}", flush=True)
        latest = save_dir / "external_latest_checkpoint.pth"
        if base_model is None and latest.exists():
            base_model = str(latest)

    if base_model and not Path(base_model).exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    return base_model


def _autosave_local(save_dir: Path, autosave_dir: Path, iteration: int) -> Path | None:
    if not save_dir.exists():
        print(f"[AUTOSAVE] skipped; save_dir does not exist: {save_dir}", flush=True)
        return None

    autosave_dir.mkdir(parents=True, exist_ok=True)
    archive_base = autosave_dir / f"external_checkpoints_iter_{iteration:04d}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=save_dir))
    print(f"[AUTOSAVE] local archive: {archive_path} size={archive_path.stat().st_size}", flush=True)
    return archive_path


def _prepare_kaggle_upload(save_dir: Path, upload_dir: Path, dataset_id: str, title: str) -> None:
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pattern in ("*.pth", "*.json"):
        for src in save_dir.glob(pattern):
            if src.name == "dataset-metadata.json":
                continue
            shutil.copy2(src, upload_dir / src.name)
            copied += 1

    if copied == 0:
        raise FileNotFoundError(f"No checkpoint/history files found in {save_dir}")

    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (upload_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _autosave_kaggle(
    save_dir: Path,
    autosave_dir: Path,
    dataset_id: str,
    title: str,
    iteration: int,
    *,
    delete_old_versions: bool,
    env: dict[str, str],
) -> None:
    upload_dir = autosave_dir / "kaggle_upload"
    _prepare_kaggle_upload(save_dir, upload_dir, dataset_id, title)

    status = subprocess.run(
        ["kaggle", "datasets", "status", dataset_id],
        cwd=str(upload_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    if status.returncode == 0:
        cmd = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(upload_dir),
            "-m",
            f"autosave external iteration {iteration}",
            "--dir-mode",
            "zip",
        ]
        if delete_old_versions:
            cmd.append("--delete-old-versions")
    else:
        cmd = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(upload_dir),
            "--dir-mode",
            "zip",
        ]

    result = subprocess.run(cmd, cwd=str(upload_dir), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print("[AUTOSAVE] Kaggle upload failed", flush=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        raise RuntimeError("Kaggle dataset autosave failed")
    print("[AUTOSAVE] Kaggle dataset updated", flush=True)
    if result.stdout:
        print(result.stdout, flush=True)


def _train_one_iteration(
    args: argparse.Namespace,
    config_path: Path,
    base_model: str | None,
    env: dict[str, str],
    iteration: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "app.cli.train_external",
        "--config",
        str(config_path),
        "--device",
        args.device,
        "--iterations",
        "1",
        "--save-dir",
        args.save_dir,
    ]

    if base_model and iteration == 1:
        cmd.extend(["--base-model", base_model])
    if args.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        cmd.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.train_steps is not None:
        cmd.extend(["--train-steps", str(args.train_steps)])

    _run(cmd, cwd=_backend_dir(), env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle wrapper for external chess-engine training")
    parser.add_argument("--samples-path", default=os.environ.get("KAGGLE_EXTERNAL_SAMPLES", DEFAULT_SAMPLES_PATH))
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--config-out", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-input-dir", default=DEFAULT_CHECKPOINT_INPUT_DIR)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--install-requirements", action="store_true")

    parser.add_argument("--max-samples", type=int, default=75_000_000)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-steps-per-iter", type=int, default=2000)
    parser.add_argument("--benchmark-games", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")
    parser.set_defaults(dedup=True)
    parser.add_argument("--no-filter-invalid", dest="filter_invalid", action="store_false")
    parser.set_defaults(filter_invalid=True)
    parser.add_argument("--keep-zero-states", dest="drop_zero_states", action="store_false")
    parser.set_defaults(drop_zero_states=True)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)

    parser.add_argument("--autosave", choices=("off", "local", "kaggle", "both"), default="kaggle")
    parser.add_argument("--autosave-every", type=int, default=1)
    parser.add_argument("--autosave-dir", default=DEFAULT_AUTOSAVE_DIR)
    parser.add_argument("--kaggle-dataset-id", default=DEFAULT_KAGGLE_DATASET_ID)
    parser.add_argument("--kaggle-dataset-title", default=None)
    parser.add_argument("--delete-old-versions", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _resolve_auto_paths(args)
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.autosave_every <= 0:
        raise ValueError("--autosave-every must be > 0")
    if args.autosave in {"kaggle", "both"} and not args.kaggle_dataset_id:
        raise ValueError("--kaggle-dataset-id is required for Kaggle autosave")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_backend_dir())
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if args.install_requirements:
        _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=_backend_dir(), env=env)

    config_path = _write_config(args)
    base_model = _copy_checkpoint_inputs(args)
    save_dir = Path(args.save_dir)
    autosave_dir = Path(args.autosave_dir)
    dataset_title = args.kaggle_dataset_title or (
        args.kaggle_dataset_id.split("/")[-1] if args.kaggle_dataset_id else "external-model-checkpoints"
    )

    print("[START] Kaggle external training", flush=True)
    print(f"[START] samples={args.samples_path}", flush=True)
    print(f"[START] save_dir={save_dir}", flush=True)
    print(f"[START] iterations={args.iterations}", flush=True)

    for iteration in range(1, int(args.iterations) + 1):
        print(f"[LOOP] iteration {iteration}/{args.iterations}", flush=True)
        _train_one_iteration(args, config_path, base_model, env, iteration)

        if args.autosave != "off" and iteration % int(args.autosave_every) == 0:
            if args.autosave in {"local", "both"}:
                _autosave_local(save_dir, autosave_dir, iteration)
            if args.autosave in {"kaggle", "both"}:
                _autosave_kaggle(
                    save_dir,
                    autosave_dir,
                    args.kaggle_dataset_id,
                    dataset_title,
                    iteration,
                    delete_old_versions=bool(args.delete_old_versions),
                    env=env,
                )

    print("[DONE] Kaggle external training finished", flush=True)


if __name__ == "__main__":
    main()
