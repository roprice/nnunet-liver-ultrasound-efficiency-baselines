# Experiment 06: external validation on RTX A6000

Run the prepared SMC-LUD Clean cohort through experiments 02, 03 and 04's 588-image, seed-42 models, folds 0–4 (15 separate models, not an ensemble). Use final 150-epoch checkpoints. No training is performed. Commands below assume an Ubuntu NVIDIA instance and run from the repository root unless stated otherwise.

## 1. Remote setup

On the **GPU host**, follow the existing GPU installation workflow:

```sh
nvidia-smi
python3 -c 'import sys; assert sys.version_info >= (3, 11)'
apt update
apt install python3-pip python3-venv python3-dev git tmux rsync -y
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git "$HOME/nnunet-liver-ultrasound-efficiency-baselines"
cd "$HOME/nnunet-liver-ultrasound-efficiency-baselines"
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1'
```

Python 3.11+ is needed here for `hashlib.file_digest`. Keep the environment outside the checkout. As in the training runbooks, if CUDA is unavailable, install a CUDA-enabled build of the same PyTorch version before proceeding; no alternate version or CUDA wheel index is prescribed here. nnU-Net supplies NumPy, Pillow and scikit-image.

## 2. Transfer inputs from local

On the **local computer**, set the actual SSH target and absolute remote checkout path (the example assumes root), then copy the runtime source and prepared inputs:

```sh
bash
GPU='root@<gpu-ip>'
REMOTE_REPO=/root/nnunet-liver-ultrasound-efficiency-baselines
rsync -avR experiments/06_external_validation/*.py \
  experiments/06_external_validation/*.md \
  experiments/06_external_validation/reference/ \
  experiments/02_data_efficiency/custom_trainers/ \
  experiments/03_augmentation_ablation/custom_trainers/ \
  experiments/04_gaussian_noise_ablation/custom_trainers/ \
  analysis/predictions_dataset/build_predictions_dataset.py \
  analysis/predictions_dataset/constants.py \
  "$GPU:$REMOTE_REPO/"
rsync -avR archives_staging/06_external_validation/smc_lud_prepared/ \
  "$GPU:$REMOTE_REPO/"
```

The prepared directory must contain `smc_lud_manifest.json` and `images/<case>_0000.png`; retain `conversion_report.json` for provenance. If it has not been prepared, run locally first:

```sh
.venv/bin/python -B experiments/06_external_validation/prepare_external_data.py
```

Preparation verifies the extracted source against `reference/smc_lud_reference.json` and writes grayscale PNGs without resizing. With prepared inputs, the source ZIP, extracted dataset, patient tables and AUL raw/preprocessed data need not be uploaded. The shared analysis module and its constants are required at runner import time; the three custom-trainer directories include the experiment-02 parent trainer used by both ablations.

Copy only the required model files, preserving archive layout:

```sh
for entry in \
  '02_data_efficiency nnUNetTrainer_dataSubsets_Seed42' \
  '03_augmentation_ablation nnUNetTrainer_brightnessAugmentation' \
  '04_gaussian_noise_ablation nnUNetTrainer_gaussianNoiseAugmentation'
do
  set -- $entry
  model="archives/$1/nnUNet_results/Dataset001_AUL/$2"'__nnUNetPlans__2d'
  rsync -avR "$model/plans.json" "$model/dataset.json" "$GPU:$REMOTE_REPO/"
  for fold in 0 1 2 3 4; do
    rsync -avR "$model/fold_$fold/checkpoint_final.pth" \
      "$model/fold_$fold/checkpoint_metadata.json" "$GPU:$REMOTE_REPO/"
  done
done
```

Keep using the **bash** session above so `set -- $entry` splits the pair. Each model folder needs `plans.json`, `dataset.json`, and all five folds' final checkpoint and metadata. The runner verifies checkpoint SHA256, size, trainer, fold and epoch 150 against metadata. `--checkpoint best` would also require all 15 `checkpoint_best.pth` files; final is the default.

To resume an existing local run, also transfer **only its predictions**, into the chosen remote output directory:

```sh
rsync -avR logs/06_external_validation/predictions/ "$GPU:$REMOTE_REPO/"
```

Use these masks only for the same cohort, inputs, checkpoint and settings. Existing MPS masks can be retained when continuing on CUDA, but that produces mixed-device provenance: preserve the original settings/logs separately and record the migration. Do not copy `merged_nnUNet_results/`: its absolute symlinks refer to the old machine. The runner creates remote links itself.

## 3. Verify and run remotely

On the **GPU host**, from the checkout with `$HOME/.venv` activated:

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/06_external_validation
python - <<'PY'
from importlib.metadata import version
import torch
for package, expected in [('nnunetv2', '2.8.1'), ('torch', '2.14.1')]:
    installed = version(package)
    print(package, installed)
    assert installed.split('+')[0] == expected, (package, installed)
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())
PY
python -m pip freeze > logs/06_external_validation/environment_packages.txt
nvidia-smi > logs/06_external_validation/nvidia_smi.txt
git rev-parse HEAD > logs/06_external_validation/source_revision.txt
git status --short >> logs/06_external_validation/source_revision.txt
python -B experiments/06_external_validation/external_validation_runner.py \
  --device cuda --check-only
```

Confirm the GPU is the rented RTX A6000. `EXPECTED_VERSIONS` in the runner is unused; the explicit check above accepts CUDA build suffixes such as `+cu…`. `--check-only` checks manifest/cohort filenames and all 15 checkpoint hashes, then prints prediction commands. It does **not** validate package versions, CUDA availability, prepared-image hashes, model JSON files or trainer loading. `PYTHONPATH` makes the shared `analysis` import available to the script.

Start a persistent session, then run **inside tmux**:

```sh
tmux new -s external_validation
. "$HOME/.venv/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail
python -u -B experiments/06_external_validation/external_validation_runner.py \
  --device cuda 2>&1 | tee -a logs/06_external_validation/runner.log
```

Detach with `Ctrl+b`, then `d`; reconnect with `tmux attach -t external_validation`. The runner sets the trainer paths, nnU-Net roots, `DATA_EFFICIENCY_EPOCHS=150` and `nnUNet_compile=false` itself. Prediction uses step size 0.5, one preprocessing/export worker, Gaussian weighting and mirroring defaults. Models run sequentially.

Rerun the same command after interruption: `--continue_prediction` is enabled by default and skips existing masks, then all cases are reevaluated. Summaries are written only after all models finish; `run_settings.json` is overwritten on each invocation. `--evaluate-only` recomputes reports from complete saved predictions. Use separate `--output-dir` paths for different cohorts/checkpoints; `--subset caliper` requires prepared `images_caliper/`. Optional `--proxy-masks <directory>` enables proxy-overlap metrics and requires transferring that directory as well.

## 4. Monitor and retrieve

From another **remote** shell in the checkout:

```sh
tail -f logs/06_external_validation/runner.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
```

Completion requires a successful process exit and `"status": "completed"` in `run_settings.json`, plus `per_image_results.csv` and `summary.json`. Failures record `status: failed` and the error when they occur inside the main run loop. Predictions live under `predictions/<experiment>_588images_seed42_fold<0-4>_final/`.

On the **local computer**, using the transfer variables above:

```sh
mkdir -p logs/06_external_validation_cuda
rsync -av --exclude='merged_nnUNet_results/' \
  "$GPU:$REMOTE_REPO/logs/06_external_validation/" \
  logs/06_external_validation_cuda/
rsync -avnc --exclude='merged_nnUNet_results/' \
  "$GPU:$REMOTE_REPO/logs/06_external_validation/" \
  logs/06_external_validation_cuda/
```

The second command checksum-checks the download; no file changes should be listed once the remote run has stopped. This retrieves masks, reports, settings, environment records and console log without machine-specific symlinks. Retry the same transfer after interruption. Verify completion and the retrieved files before releasing the rental.
