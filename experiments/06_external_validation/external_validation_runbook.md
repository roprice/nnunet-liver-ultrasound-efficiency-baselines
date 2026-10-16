# Experiment 06: external validation

Fixed models: experiments 02, 03 and 04, **588-image training pool, seed 42, folds 0–4, `checkpoint_final.pth` (150 epochs)**. Run 15 separate models on SMC-LUD Clean. No training is performed.

## Environment and inputs

Run commands from the repository root unless stated otherwise.

On the **GPU host** (Ubuntu, RTX A6000):

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
nvidia-smi
apt update
apt install python3-pip python3-venv python3-dev git tmux rsync -y
```

```sh
cd ~
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git
cd nnunet-liver-ultrasound-efficiency-baselines
```

```sh
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1'
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

If CUDA is unavailable, install a CUDA-enabled build of the same PyTorch version before proceeding.

Required inputs:

- `archives_staging/06_external_validation/smc_lud_prepared/{smc_lud_manifest.json,conversion_report.json,images/}`.
- Each experiment's `archives/<experiment>/nnUNet_results/Dataset001_AUL/<trainer>__nnUNetPlans__2d/{plans.json,dataset.json,fold_<0-4>/checkpoint_final.pth,fold_<0-4>/checkpoint_metadata.json}`.

On the **local computer**, prepare the cohort if needed:

```sh
.venv/bin/python -B experiments/06_external_validation/prepare_external_data.py
```

Transfer inputs from the **local computer**, using Bash. Replace the SSH target and remote path:

```bash
bash
GPU='root@<gpu-ip>'
REMOTE_REPO=/root/nnunet-liver-ultrasound-efficiency-baselines
rsync -avR archives_staging/06_external_validation/smc_lud_prepared/ \
  "$GPU:$REMOTE_REPO/"

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

## Check and run

On the **GPU host**, from the repository root:

```sh
. "$HOME/.venv/bin/activate"
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

`--check-only` checks cohort filenames and all 15 checkpoint hashes/metadata, then prints prediction commands.

## Full validation

On the **GPU host**:

```sh
tmux new -s external_validation
```

Inside tmux, from the repository root:

```bash
. "$HOME/.venv/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail
python -u -B experiments/06_external_validation/external_validation_runner.py \
  --device cuda 2>&1 | tee -a logs/06_external_validation/runner.log
```

Detach with `Ctrl+b`, then `d`; reattach with `tmux attach -t external_validation`.

Defaults: Clean cohort, final checkpoints, sequential models, step size 0.5, one preprocessing worker, one export worker, Gaussian weighting and mirroring enabled. The runner sets nnU-Net roots, custom-trainer paths, `DATA_EFFICIENCY_EPOCHS=150` and `nnUNet_compile=false`.

Resume by rerunning the same command: existing masks are skipped and all cases reevaluated. Keep inputs, checkpoints and settings identical. `run_settings.json` is overwritten on each invocation. `--evaluate-only` recomputes reports from saved predictions; use a separate `--output-dir` for a different run.

## Outputs and retrieval

`logs/06_external_validation/` contains:

- `run_settings.json`: device, cohort, checkpoint, Git revision, UTC boundaries and status.
- `per_image_results.csv` and `summary.json`: detection rates, patient-weighted rates, bootstrap intervals and fold-matched differences against experiment 02.
- `predictions/<experiment>_588images_seed42_fold<0-4>_final/`: masks and nnU-Net sidecars.
- `runner.log`, `environment_packages.txt`, `nvidia_smi.txt` and `source_revision.txt`.

After the command exits successfully, verify completion on the **GPU host**:

```sh
python - <<'PY'
import json
from pathlib import Path
output = Path('logs/06_external_validation')
assert json.loads((output / 'run_settings.json').read_text())['status'] == 'completed'
assert (output / 'per_image_results.csv').is_file()
assert (output / 'summary.json').is_file()
print('Completed')
PY
```

On the **local computer**, in the Bash session with the transfer variables:

```bash
mkdir -p logs/06_external_validation_cuda
rsync -av --exclude='merged_nnUNet_results/' \
  "$GPU:$REMOTE_REPO/logs/06_external_validation/" \
  logs/06_external_validation_cuda/
rsync -avnc --exclude='merged_nnUNet_results/' \
  "$GPU:$REMOTE_REPO/logs/06_external_validation/" \
  logs/06_external_validation_cuda/
```

The checksum dry run should list no file changes. Retry the same transfer after interruption.
