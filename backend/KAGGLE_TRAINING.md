# Kaggle External Training

Use the normal `requirements.txt` for local/backend server hosting. It pins the CPU PyTorch wheel on purpose for predictable deployment.

Use `requirements2_kaggle.txt` for Kaggle/Colab training. It installs the project dependencies but intentionally leaves PyTorch alone so the notebook runtime keeps its CUDA-enabled torch build. Set the notebook accelerator to GPU before running from the top.

Use a copied config on Kaggle with absolute input and output paths:

```yaml
external:
  samples_path: /kaggle/input/YOUR_DATASET/external_samples.npz
  save_dir: /kaggle/working/models/external

system:
  checkpoint_path: /kaggle/working/models/external/external_best_model.pth
```

Run a tiny no-save smoke check before starting a long training job:

```bash
PYTHONPATH=. python -m app.cli.train_external \
  --config /kaggle/working/external_training_kaggle.yaml \
  --device cuda \
  --iterations 1 \
  --max-val-samples 2 \
  --max-train-samples 4 \
  --train-steps 1 \
  --no-save
```

Full training still uses the normal defaults when the smoke flags are omitted:

```bash
PYTHONPATH=. python -m app.cli.train_external \
  --config /kaggle/working/external_training_kaggle.yaml \
  --device cuda \
  --iterations 1
```

## One-script Kaggle runner

Use this script instead of the old notebook cells. It runs the full cycle:

1. auto-find external samples under `/kaggle/input`
2. auto-copy the latest checkpoint dataset from `/kaggle/input` to `/kaggle/working/checkpoints`
3. train one full iteration
4. update your Kaggle checkpoint Dataset
5. repeat for the next iteration

If your checkpoint Dataset id is `jadakil/external-model-checkpoints`, the simplest long run is:

```bash
cd /kaggle/working/chesEngineWithData/backend
python scripts/kaggle_train_external.py \
  --iterations 100 \
  --device cuda \
  --delete-old-versions
```

Add `--install-requirements` if the Kaggle image is missing backend Python dependencies and internet access is enabled. This installs `requirements2_kaggle.txt`, which intentionally does not reinstall PyTorch so Kaggle keeps its CUDA-enabled torch build.

For a first Kaggle smoke check:

```bash
python scripts/kaggle_train_external.py \
  --iterations 1 \
  --device cuda \
  --max-val-samples 2 \
  --max-train-samples 4 \
  --train-steps 1 \
  --delete-old-versions
```

If your checkpoint Dataset id is different, add:

```bash
  --kaggle-dataset-id YOUR_USERNAME/external-model-checkpoints \
  --kaggle-dataset-title external-model-checkpoints
```

Defaults:

- `--samples-path auto`
- `--checkpoint-input-dir auto`
- `--save-dir /kaggle/working/checkpoints`
- `--autosave kaggle`
- `--autosave-every 1`
- `--kaggle-dataset-id jadakil/external-model-checkpoints`
