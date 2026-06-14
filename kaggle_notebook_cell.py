from pathlib import Path
import os
import shutil
import subprocess
import sys

GITHUB_REPO_URL = "https://github.com/jadAkeel/chessEngine.git"
BRANCH_NAME = "main"

PROJECT_DIR = Path("/kaggle/working/chessEngine")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
OLD_KAGGLE_PROJECT_DIR = Path("/kaggle/input/datasets/jadakil/chessengine/chesEngineWithData")

DATA_DIR = OLD_KAGGLE_PROJECT_DIR / "backend" / "data"
MODEL_DIR = OLD_KAGGLE_PROJECT_DIR / "backend" / "models"
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
REPLAY_BUFFER_DIR = MODEL_DIR / "replay_buffer.d"
OUTPUT_DIR = Path("/kaggle/working")
AUTOSAVE_DIR = OUTPUT_DIR / "autosaves"
WORKING_MODEL_DIR = OUTPUT_DIR / "models"

ITERATIONS = 10
DEVICE = "cuda"
TRAIN_STEPS_PER_ITER = 10000
BUFFER_SIZE = 5000000
MAX_SAMPLES = 75000000
MIN_FULLMOVE = 1
MAX_FULLMOVE = 12
AUTOSAVE_MODE = "kaggle"
AUTOSAVE_EVERY = 1
DELETE_OLD_KAGGLE_VERSIONS = True
KAGGLE_CHECKPOINT_DATASET_ID = "jadakil/external-model-checkpoints"
KAGGLE_CHECKPOINT_DATASET_TITLE = "external-model-checkpoints"

SAMPLES_PATH = None
CHECKPOINT_INPUT_DIR = None
BASE_MODEL = None
LINK_REPLAY_BUFFER_FOR_REFERENCE = True

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

def find_old_project_dir():
    if (OLD_KAGGLE_PROJECT_DIR / "backend" / "scripts" / "kaggle_train_external.py").exists():
        return OLD_KAGGLE_PROJECT_DIR

    candidates = []
    if KAGGLE_INPUT_ROOT.exists():
        for backend_dir in KAGGLE_INPUT_ROOT.rglob("backend"):
            if (backend_dir / "data" / "external_samples.npz").exists() or (backend_dir / "models").exists():
                candidates.append(backend_dir.parent)
            elif (backend_dir / "scripts" / "kaggle_train_external.py").exists():
                candidates.append(backend_dir.parent)

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
    if not candidates:
        print("Available /kaggle/input directories:")
        run(["find", str(KAGGLE_INPUT_ROOT), "-maxdepth", "5", "-type", "d"], check=False)
        raise FileNotFoundError(
            "Old Kaggle project/data input was not found. TODO: add the old chessengine Kaggle Dataset as Notebook Input."
        )
    return candidates[0]

def first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None

def find_first_named(root, name):
    root = Path(root)
    if not root.exists():
        return None
    matches = sorted(root.rglob(name), key=lambda p: (len(str(p)), str(p)))
    return matches[0] if matches else None

def copy_file_if_available(src, dst):
    if src is None or not Path(src).exists():
        return False
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f"[KEEP] {dst} already matches {src}", flush=True)
        return True
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}", flush=True)
    return True

def link_path_if_available(src, dst):
    if src is None or not Path(src).exists():
        return False
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return True
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        print(f"[LINK] {dst} -> {src}", flush=True)
        return True
    except OSError as exc:
        print(f"[WARN] could not symlink {dst} -> {src}: {exc}", flush=True)
        return False

def verify_training_runtime():
    import torch

    print("[TORCH] version =", torch.__version__, flush=True)
    print("[TORCH] cuda build =", torch.version.cuda, flush=True)
    print("[TORCH] cuda available =", torch.cuda.is_available(), flush=True)
    try:
        run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("[WARN] nvidia-smi not found", flush=True)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE='cuda' but PyTorch CUDA is unavailable. "
            "Enable a Kaggle GPU accelerator and install requirements2_kaggle.txt, "
            "not requirements.txt."
        )
    if DEVICE == "cuda":
        print("[TORCH] gpu =", torch.cuda.get_device_name(0), flush=True)

if AUTOSAVE_MODE != "kaggle":
    raise ValueError("AUTOSAVE_MODE must be 'kaggle' so checkpoints are versioned in the Kaggle Dataset.")
if AUTOSAVE_EVERY != 1:
    raise ValueError("AUTOSAVE_EVERY must be 1 to save after every completed iteration.")
if not KAGGLE_CHECKPOINT_DATASET_ID:
    raise ValueError("KAGGLE_CHECKPOINT_DATASET_ID is required for Kaggle autosave.")

old_project = find_old_project_dir()
DATA_DIR = old_project / "backend" / "data"
MODEL_DIR = old_project / "backend" / "models"
REPLAY_BUFFER_DIR = first_existing([MODEL_DIR / "replay_buffer.d", MODEL_DIR / "replay_buffer"])

SAMPLES_PATH = first_existing([DATA_DIR / "external_samples.npz"])
if SAMPLES_PATH is None:
    SAMPLES_PATH = find_first_named(KAGGLE_INPUT_ROOT, "external_samples.npz")
if SAMPLES_PATH is None:
    raise FileNotFoundError("external_samples.npz was not found under the old Kaggle input. TODO: attach the dataset that contains backend/data/external_samples.npz.")

external_ckpt_dir = MODEL_DIR / "external"
if (external_ckpt_dir / "external_latest_checkpoint.pth").exists() or (external_ckpt_dir / "external_best_model.pth").exists():
    CHECKPOINT_INPUT_DIR = external_ckpt_dir
else:
    CHECKPOINT_INPUT_DIR = MODEL_DIR if MODEL_DIR.exists() else "auto"

print("GITHUB_REPO_URL =", GITHUB_REPO_URL)
print("BRANCH_NAME =", BRANCH_NAME)
print("PROJECT_DIR =", PROJECT_DIR)
print("OLD_KAGGLE_PROJECT_DIR =", old_project)
print("DATA_DIR =", DATA_DIR)
print("MODEL_DIR =", MODEL_DIR)
print("CHECKPOINT_DIR =", CHECKPOINT_DIR)
print("REPLAY_BUFFER_DIR =", REPLAY_BUFFER_DIR if REPLAY_BUFFER_DIR else "not found / not used by external training")
print("OUTPUT_DIR =", OUTPUT_DIR)
print("SAMPLES_PATH =", SAMPLES_PATH)
print("CHECKPOINT_INPUT_DIR =", CHECKPOINT_INPUT_DIR)
print("AUTOSAVE_MODE =", AUTOSAVE_MODE)
print("AUTOSAVE_EVERY =", AUTOSAVE_EVERY)
print("KAGGLE_CHECKPOINT_DATASET_ID =", KAGGLE_CHECKPOINT_DATASET_ID)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
WORKING_MODEL_DIR.mkdir(parents=True, exist_ok=True)

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)

# Try git clone, fall back to ZIP download if it fails
import urllib.request, zipfile
try:
    run(["git", "clone", "--branch", BRANCH_NAME, "--depth", "1", GITHUB_REPO_URL, str(PROJECT_DIR)])
except RuntimeError:
    print("[FALLBACK] git clone failed. Downloading ZIP instead...", flush=True)
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    zip_url = f"https://github.com/jadAkeel/chessEngine/archive/refs/heads/{BRANCH_NAME}.zip"
    zip_path = PROJECT_DIR.with_suffix(".zip")
    urllib.request.urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(PROJECT_DIR.parent)
    extracted = PROJECT_DIR.parent / f"chessEngine-{BRANCH_NAME}"
    if extracted.exists():
        shutil.move(str(extracted), str(PROJECT_DIR))
    zip_path.unlink()
    print("[FALLBACK] ZIP download and extraction complete.", flush=True)

BACKEND_DIR = PROJECT_DIR / "backend"
if not BACKEND_DIR.exists():
    raise FileNotFoundError(f"GitHub backend directory not found: {BACKEND_DIR}")

os.chdir(BACKEND_DIR)
print("CWD =", Path.cwd())

run([sys.executable, "-m", "pip", "install", "-r", "requirements2_kaggle.txt"], cwd=BACKEND_DIR)
verify_training_runtime()

link_path_if_available(SAMPLES_PATH, BACKEND_DIR / "data" / "external_samples.npz")

external_latest = first_existing([
    CHECKPOINT_INPUT_DIR / "external_latest_checkpoint.pth" if isinstance(CHECKPOINT_INPUT_DIR, Path) else Path("/__missing__"),
    MODEL_DIR / "external" / "external_latest_checkpoint.pth",
    MODEL_DIR / "external_latest_checkpoint.pth",
])
external_best = first_existing([
    CHECKPOINT_INPUT_DIR / "external_best_model.pth" if isinstance(CHECKPOINT_INPUT_DIR, Path) else Path("/__missing__"),
    MODEL_DIR / "external" / "external_best_model.pth",
    MODEL_DIR / "external_best_model.pth",
])
legacy_latest = first_existing([MODEL_DIR / "latest_checkpoint.pth"])
legacy_best = first_existing([MODEL_DIR / "best_model.pth"])
history_src = first_existing([
    MODEL_DIR / "external" / "external_history.json",
    MODEL_DIR / "external_history.json",
    MODEL_DIR / "history.json",
])

copy_file_if_available(external_latest or legacy_latest, CHECKPOINT_DIR / "external_latest_checkpoint.pth")
copy_file_if_available(external_best or legacy_best, CHECKPOINT_DIR / "external_best_model.pth")
copy_file_if_available(history_src, CHECKPOINT_DIR / "external_history.json")

if (CHECKPOINT_DIR / "external_latest_checkpoint.pth").exists():
    BASE_MODEL = CHECKPOINT_DIR / "external_latest_checkpoint.pth"
elif (CHECKPOINT_DIR / "external_best_model.pth").exists():
    BASE_MODEL = CHECKPOINT_DIR / "external_best_model.pth"
else:
    BASE_MODEL = None

if LINK_REPLAY_BUFFER_FOR_REFERENCE and REPLAY_BUFFER_DIR:
    link_path_if_available(REPLAY_BUFFER_DIR, BACKEND_DIR / "models" / Path(REPLAY_BUFFER_DIR).name)

print("BASE_MODEL =", BASE_MODEL if BASE_MODEL else "none; training starts from scratch if no checkpoint is auto-found")

env = os.environ.copy()
env["PYTHONPATH"] = str(BACKEND_DIR)
env.setdefault("PYTHONIOENCODING", "utf-8")

train_cmd = [
    sys.executable,
    "scripts/kaggle_train_external.py",
    "--samples-path", str(SAMPLES_PATH),
    "--checkpoint-input-dir", str(CHECKPOINT_INPUT_DIR),
    "--save-dir", str(CHECKPOINT_DIR),
    "--config-out", str(OUTPUT_DIR / "external_training_kaggle.yaml"),
    "--autosave", AUTOSAVE_MODE,
    "--autosave-every", str(AUTOSAVE_EVERY),
    "--autosave-dir", str(AUTOSAVE_DIR),
    "--kaggle-dataset-id", KAGGLE_CHECKPOINT_DATASET_ID,
    "--kaggle-dataset-title", KAGGLE_CHECKPOINT_DATASET_TITLE,
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
print("Kaggle samples used from:", SAMPLES_PATH)
print("Previous checkpoints staged from:", CHECKPOINT_INPUT_DIR)
print("New checkpoints saved in:", CHECKPOINT_DIR)
print("Kaggle autosave staging dir:", AUTOSAVE_DIR)
print("Kaggle autosave dataset:", KAGGLE_CHECKPOINT_DATASET_ID)
