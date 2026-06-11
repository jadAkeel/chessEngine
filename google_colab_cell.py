from pathlib import Path
import os
import shutil
import subprocess
import sys

# =========================
# EDIT THESE
# =========================

GITHUB_REPO_URL = "https://github.com/jadAkeel/chessEngine.git"
BRANCH_NAME = "main"

# Kaggle Dataset that contains external_samples.npz or shard_*.npz files.
KAGGLE_SAMPLES_DATASET_ID = "jadakil/chessengine"

# Optional Kaggle Dataset that contains previous checkpoints.
# Set to None if you want to start from scratch.
KAGGLE_CHECKPOINT_INPUT_DATASET_ID = "jadakil/external-model-checkpoints"

# Kaggle Dataset to create/update with new checkpoints.
KAGGLE_AUTOSAVE_DATASET_ID = "jadakil/external-model-checkpoints"
KAGGLE_AUTOSAVE_DATASET_TITLE = "external-model-checkpoints"
AUTOSAVE_MODE = "kaggle"
AUTOSAVE_EVERY = 1
DELETE_OLD_KAGGLE_VERSIONS = True

ITERATIONS = 10
DEVICE = "cuda"
TRAIN_STEPS_PER_ITER = 10000
BUFFER_SIZE = 5000000
MAX_SAMPLES = 75000000
MIN_FULLMOVE = 1
MAX_FULLMOVE = 12

# =========================
# PATHS
# =========================

CONTENT_DIR = Path("/content")
PROJECT_DIR = CONTENT_DIR / "chessEngine"
INPUT_DIR = CONTENT_DIR / "kaggle_input"
SAMPLES_INPUT_DIR = INPUT_DIR / "samples"
CHECKPOINT_INPUT_DIR = INPUT_DIR / "checkpoint_input"
CHECKPOINT_DIR = CONTENT_DIR / "checkpoints"
AUTOSAVE_DIR = CONTENT_DIR / "autosaves"


def run(cmd, cwd=None, env=None, check=True):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(map(str, cmd))}")
    return result


def setup_kaggle_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        os.chmod(kaggle_json, 0o600)
        print("[KAGGLE] using existing ~/.kaggle/kaggle.json", flush=True)
        return

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(
            '{"username": "%s", "key": "%s"}' % (env_user, env_key),
            encoding="utf-8",
        )
        os.chmod(kaggle_json, 0o600)
        print("[KAGGLE] wrote credentials from environment variables", flush=True)
        return

    try:
        from google.colab import files
    except Exception as exc:
        raise FileNotFoundError(
            "Kaggle credentials not found. Add ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY."
        ) from exc

    print("[KAGGLE] upload kaggle.json from your Kaggle Account settings", flush=True)
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("Uploaded file must be named kaggle.json")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.move("kaggle.json", kaggle_json)
    os.chmod(kaggle_json, 0o600)
    print("[KAGGLE] credentials installed", flush=True)


def download_kaggle_dataset(dataset_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(output_dir),
        "--unzip",
    ])


def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def find_samples_path(root):
    root = Path(root)
    shard_dirs = []
    for path in root.rglob("*.npz"):
        if path.name.startswith("shard_"):
            shard_dirs.append(path.parent)
    if shard_dirs:
        return sorted(set(shard_dirs), key=lambda p: (len(str(p)), str(p)))[0]

    sample_files = sorted(root.rglob("external_samples.npz"), key=lambda p: (len(str(p)), str(p)))
    if sample_files:
        return sample_files[0].parent

    npz_files = sorted(root.rglob("*.npz"), key=lambda p: (len(str(p)), str(p)))
    if npz_files:
        return npz_files[0].parent

    raise FileNotFoundError("No .npz samples found in the Kaggle samples dataset.")


def find_checkpoint_dir(root):
    root = Path(root)
    if not root.exists():
        return None
    candidates = []
    for name in ("external_latest_checkpoint.pth", "external_best_model.pth"):
        for path in root.rglob(name):
            candidates.append(path.parent)
    if not candidates:
        return None
    return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))[0]


def ensure_clean_project_dir():
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)


if AUTOSAVE_MODE != "kaggle":
    raise ValueError("AUTOSAVE_MODE must be 'kaggle' so checkpoints are versioned in the Kaggle Dataset.")
if AUTOSAVE_EVERY != 1:
    raise ValueError("AUTOSAVE_EVERY must be 1 to save after every completed iteration.")
if not KAGGLE_AUTOSAVE_DATASET_ID:
    raise ValueError("KAGGLE_AUTOSAVE_DATASET_ID is required for Kaggle autosave.")

setup_kaggle_credentials()
run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

if KAGGLE_SAMPLES_DATASET_ID:
    print("[DATA] downloading samples dataset:", KAGGLE_SAMPLES_DATASET_ID, flush=True)
    if SAMPLES_INPUT_DIR.exists():
        shutil.rmtree(SAMPLES_INPUT_DIR)
    download_kaggle_dataset(KAGGLE_SAMPLES_DATASET_ID, SAMPLES_INPUT_DIR)
else:
    raise ValueError("KAGGLE_SAMPLES_DATASET_ID is required")

if KAGGLE_CHECKPOINT_INPUT_DATASET_ID:
    print("[DATA] downloading checkpoint dataset:", KAGGLE_CHECKPOINT_INPUT_DATASET_ID, flush=True)
    if CHECKPOINT_INPUT_DIR.exists():
        shutil.rmtree(CHECKPOINT_INPUT_DIR)
    download_kaggle_dataset(KAGGLE_CHECKPOINT_INPUT_DATASET_ID, CHECKPOINT_INPUT_DIR)

SAMPLES_PATH = find_samples_path(SAMPLES_INPUT_DIR)
RESOLVED_CHECKPOINT_INPUT_DIR = find_checkpoint_dir(CHECKPOINT_INPUT_DIR)

print("GITHUB_REPO_URL =", GITHUB_REPO_URL)
print("BRANCH_NAME =", BRANCH_NAME)
print("PROJECT_DIR =", PROJECT_DIR)
print("SAMPLES_PATH =", SAMPLES_PATH)
print("CHECKPOINT_INPUT_DIR =", RESOLVED_CHECKPOINT_INPUT_DIR or "none")
print("CHECKPOINT_DIR =", CHECKPOINT_DIR)
print("AUTOSAVE_DIR =", AUTOSAVE_DIR)
print("AUTOSAVE_MODE =", AUTOSAVE_MODE)
print("KAGGLE_AUTOSAVE_DATASET_ID =", KAGGLE_AUTOSAVE_DATASET_ID)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)

ensure_clean_project_dir()
run(["git", "clone", "--branch", BRANCH_NAME, "--depth", "1", GITHUB_REPO_URL, str(PROJECT_DIR)])

BACKEND_DIR = PROJECT_DIR / "backend"
if not BACKEND_DIR.exists():
    raise FileNotFoundError(f"GitHub backend directory not found: {BACKEND_DIR}")

run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=BACKEND_DIR)

BASE_MODEL = None
if RESOLVED_CHECKPOINT_INPUT_DIR:
    latest = first_existing([
        RESOLVED_CHECKPOINT_INPUT_DIR / "external_latest_checkpoint.pth",
        RESOLVED_CHECKPOINT_INPUT_DIR / "latest_checkpoint.pth",
    ])
    best = first_existing([
        RESOLVED_CHECKPOINT_INPUT_DIR / "external_best_model.pth",
        RESOLVED_CHECKPOINT_INPUT_DIR / "best_model.pth",
    ])
    history = first_existing([
        RESOLVED_CHECKPOINT_INPUT_DIR / "external_history.json",
        RESOLVED_CHECKPOINT_INPUT_DIR / "history.json",
    ])
    for src, dst_name in (
        (latest, "external_latest_checkpoint.pth"),
        (best, "external_best_model.pth"),
        (history, "external_history.json"),
    ):
        if src and Path(src).exists():
            dst = CHECKPOINT_DIR / dst_name
            shutil.copy2(src, dst)
            print(f"[COPY] {src} -> {dst}", flush=True)
    if (CHECKPOINT_DIR / "external_latest_checkpoint.pth").exists():
        BASE_MODEL = CHECKPOINT_DIR / "external_latest_checkpoint.pth"
    elif (CHECKPOINT_DIR / "external_best_model.pth").exists():
        BASE_MODEL = CHECKPOINT_DIR / "external_best_model.pth"

print("BASE_MODEL =", BASE_MODEL if BASE_MODEL else "none; training starts from scratch")

env = os.environ.copy()
env["PYTHONPATH"] = str(BACKEND_DIR)
env.setdefault("PYTHONIOENCODING", "utf-8")

train_cmd = [
    sys.executable,
    "scripts/kaggle_train_external.py",
    "--samples-path", str(SAMPLES_PATH),
    "--checkpoint-input-dir", str(CHECKPOINT_DIR),
    "--save-dir", str(CHECKPOINT_DIR),
    "--config-out", str(CONTENT_DIR / "external_training_colab.yaml"),
    "--autosave", AUTOSAVE_MODE,
    "--autosave-every", str(AUTOSAVE_EVERY),
    "--autosave-dir", str(AUTOSAVE_DIR),
    "--kaggle-dataset-id", KAGGLE_AUTOSAVE_DATASET_ID,
    "--kaggle-dataset-title", KAGGLE_AUTOSAVE_DATASET_TITLE,
    "--iterations", str(ITERATIONS),
    "--device", DEVICE,
    "--train-steps-per-iter", str(TRAIN_STEPS_PER_ITER),
    "--buffer-size", str(BUFFER_SIZE),
    "--max-samples", str(MAX_SAMPLES),
    "--min-fullmove", str(MIN_FULLMOVE),
    "--max-fullmove", str(MAX_FULLMOVE),
]

if DELETE_OLD_KAGGLE_VERSIONS:
    train_cmd.append("--delete-old-versions")

if BASE_MODEL is not None:
    train_cmd.extend(["--base-model", str(BASE_MODEL)])

run(train_cmd, cwd=BACKEND_DIR, env=env)

print("Training finished.")
print("Latest GitHub code used from:", GITHUB_REPO_URL, "branch", BRANCH_NAME)
print("Kaggle samples dataset:", KAGGLE_SAMPLES_DATASET_ID)
print("Kaggle checkpoint input dataset:", KAGGLE_CHECKPOINT_INPUT_DATASET_ID)
print("New checkpoints saved in:", CHECKPOINT_DIR)
print("Kaggle autosave dataset:", KAGGLE_AUTOSAVE_DATASET_ID)
