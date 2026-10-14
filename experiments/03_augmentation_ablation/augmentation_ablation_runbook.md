# Augmentation ablation: brightness variants

Test whether more frequent brightness augmentation improves malignant-mass detection. Trainer: `nnUNetTrainer_brightnessAugmentation`.


## Fixed conditions

| Setting | Value |
|---|---|
| Full dataset | AUL / `Dataset001_AUL` |
| Training-pool sizes | 588, 74 |
| Held-out test cases | 147 |
| Folds | 0–4 for each size |
| Test set | UltraBench AUL test set (Tupper & Gagné, 2025) |
| Initialization seed | 42 |
| Epoch budget | 150 |
| Brightness augmentation frequency |  p=0.5 |
| Saved checkpoints | `checkpoint_final.pth`, `checkpoint_best.pth` |

Uses `Dataset001_AUL` (ID 1) and `Dataset004_AUL_074` (ID 4), the canonical mapping and folds from `experiments/prepare_data/aul_splits.py`, and independent planning/preprocessing for each size. 

Increase brightness-augmentation probability from 0.15 to 0.5; retain the default transform and all other augmentations.

| Setting | nnU-Net default (v2.8.1) | This experiment |
|---|---|---|
| Multiplicative factor | 0.75–1.25 | 0.75–1.25 (unchanged) |
| Probability per training sample/patch | 0.15 | 0.5 (changed) |

## 1. Check Python and install system dependencies

Requires an Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and storage for source data, nnU-Net data and checkpoints, and logs.

```sh
 #
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
apt update
apt install python3-pip python3-venv python3-dev git unzip tmux rsync time -y

# Optional: color the shell prompt
cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
PROMPTEOF
source ~/.bashrc
```

## 2. Clone the repository

```sh
 #
cd ~
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git
cd nnunet-liver-ultrasound-efficiency-baselines
```

## 3. Start GPU usage and estimated cost recording

Set the GPU model and hourly rate in `gpu_rate.csv` below. Recording starts here and ends during completion in step 12; remaining setup time is included, archive download is excluded.

  ```sh
  #
  mkdir -p logs/03_augmentation_ablation
  printf 'gpu_model,hourly_rate_usd\nRTX 6000 Ada,1.226\n' > logs/03_augmentation_ablation/gpu_rate.csv
  printf 'event,utc\n' > logs/03_augmentation_ablation/gpu_events.csv
  cat > logs/03_augmentation_ablation/record_event.sh <<'EVENTEOF'
  #!/bin/sh
  printf '%s,%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/03_augmentation_ablation/gpu_events.csv
  EVENTEOF
  sh logs/03_augmentation_ablation/record_event.sh recording_started
  ```

  Sample the GPU in a detached tmux session so sampling continues through SSH disconnections.
    
  ```sh
#
  tmux new-session -d -s gpu_usage -c "$PWD" \
    'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop-ms=1000 > logs/03_augmentation_ablation/gpu_monitor_instance.csv'
  ```
  
  Per-fold timings are recorded separately from the total GPU rental estimate.

## 4. Create a Python environment

```sh
#
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
```

Keep `.venv` outside the checkout: the remote controller copies the entire repository and rejects symlinks in its backup trees.

## 5. Install Python dependencies

```sh
#
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' 'batchgeneratorsv2==0.3.6' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
python -m pip freeze > logs/03_augmentation_ablation/pip_freeze.txt
```

PyTorch 2.14.1 and batchgeneratorsv2 0.3.6 match `archives/02_data_efficiency/repo/logs/02_data_efficiency/pip_freeze.txt`. If the CUDA check fails, install a CUDA-enabled build of the same PyTorch version. Environment records are saved to `logs/03_augmentation_ablation/`.

## 6. Verify the ablation and run experiment checks

Run both checks on the GPU instance. Both commands must succeed.

### 6a. Verify brightness frequency change

Verify that the change in brightness frequency is made and that it's the only variation from nnU-Net augmentation defaults

```bash
#
set -o pipefail
python - <<'PY' 2>&1 | tee logs/03_augmentation_ablation/inspect_runtime_brightness.txt
from importlib.metadata import version
import sys
import batchgeneratorsv2.transforms.intensity.brightness as brightness
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

assert version('batchgeneratorsv2') == '0.3.6'
sys.path.insert(0, 'experiments/03_augmentation_ablation/custom_trainers')
from nnUNetTrainer_brightnessAugmentation import nnUNetTrainer_brightnessAugmentation
arguments = dict(
    patch_size=(32, 32), rotation_for_DA=(-0.2, 0.2),
    deep_supervision_scales=((1, 1), (0.5, 0.5)),
    mirror_axes=(0, 1), do_dummy_2d_data_aug=False)
def brightness_wrapper(pipeline):
    wrappers = [t for t in pipeline.transforms if isinstance(t, RandomTransform)
                and isinstance(t.transform, brightness.MultiplicativeBrightnessTransform)]
    assert len(wrappers) == 1
    return wrappers[0]

default = brightness_wrapper(nnUNetTrainer.get_training_transforms(**arguments))
wrapper = brightness_wrapper(nnUNetTrainer_brightnessAugmentation.get_training_transforms(**arguments))
assert wrapper.apply_probability == 0.5
assert default.apply_probability == 0.15
transform = wrapper.transform
assert type(transform) is type(default.transform)
assert type(transform.multiplier_range) is type(default.transform.multiplier_range) is BGContrast
assert transform.multiplier_range.contrast_range == default.transform.multiplier_range.contrast_range == (0.75, 1.25)
assert transform.synchronize_channels is default.transform.synchronize_channels is False
assert transform.p_per_channel == default.transform.p_per_channel == 1
print('Brightness transform matches default; only application probability differs: passed')
PY
```

### 6b. Run experiment checks

The suite checks the trainer, runner, and completion safeguards.

```bash
#
set -o pipefail
python -B -m unittest discover -s experiments/03_augmentation_ablation \
  -p test_augmentation_ablation.py -v 2>&1 | tee logs/03_augmentation_ablation/trainer_tests.txt
```

## 7. Configure nnU-Net directories

```sh

export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/03_augmentation_ablation/custom_trainers"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
test -f "$nnUNet_extTrainer/nnUNetTrainer_brightnessAugmentation.py"
cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/03_augmentation_ablation/custom_trainers"
ENVEOF
```

## 8. Download AUL from Zenodo

  Download record `7272660` and verify all three archive checksums before extracting:

```sh
#
mkdir -p data/source
(
  cd data/source || exit 1
  zenodo_get 7272660 || exit 1
  printf '%s\n' \
    'c37fef0cb2730236a79ef57e5315995e  Benign.zip' \
    '63894a9e5654a69c3b94bda84071dfb0  Malignant.zip' \
    'a7e16299b2cf12ca4a6c3468d2e4978f  Normal.zip' | md5sum -c - || exit 1
  mkdir -p AUL
  for archive in Benign.zip Malignant.zip Normal.zip; do
    unzip -q "$archive" -d AUL || exit 1
  done
)
```

## 9. Convert to nnU-Net format

```sh
#
python experiments/prepare_data/aul_conversion.py \
  --raw-data-dir data/source/AUL \
  --output-dir "${nnUNet_raw:?Run step 7 first}/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter checks assignments against the committed reference, checks for missing or unexpected annotations, and writes `conversion_report.json`. For Malignant images 229 and 306, it uses the liver polygon under `segmentation/outline/`. Malignant image 374 has no liver polygon; its label contains the mass only.

## 10. Verify input data and mark setup complete

```sh
#
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Set the fixed 150-epoch budget, then mark setup complete:

```sh
#
export DATA_EFFICIENCY_EPOCHS=150
sh logs/03_augmentation_ablation/record_event.sh setup_complete
```

  ## 11. Run the augmentation ablation
  
  ```sh
  #
  tmux new -s training
  ```
  
  Inside tmux, run from the repository root. The runner requires fresh outputs and does not resume.
  
  ```sh
  #
  . "$HOME/.venv/bin/activate"
  export DATA_EFFICIENCY_EPOCHS='150'
  rm -f augmentation_ablation_runner_exit_status.txt
  sh logs/03_augmentation_ablation/record_event.sh experiment_started
  if bash -o pipefail -c 'bash experiments/03_augmentation_ablation/augmentation_ablation_runner.sh 2>&1 | tee logs/augmentation_ablation_runner.log'; then
    RUN_STATUS=0
  else
    RUN_STATUS=$?
  fi
  printf '%s\n' "$RUN_STATUS" > augmentation_ablation_runner_exit_status.txt.tmp
  mv augmentation_ablation_runner_exit_status.txt.tmp augmentation_ablation_runner_exit_status.txt
  ```
  
  Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t training`. Check `logs/augmentation_ablation_runner.log` and per-stage logs if it fails.
  
  The runner preprocesses both dataset sizes and trains ten models. It runs final and best checkpoints on the 147 test cases: 20 prediction runs and 20 inference benchmarks. Under `logs/03_augmentation_ablation/`, masks go to `predictions/size{SIZE}/fold{FOLD}/` and `predictions_best/size{SIZE}/fold{FOLD}/`; benchmark reports go to `inference/size{SIZE}/fold{FOLD}/` and `inference_best/size{SIZE}/fold{FOLD}/`. Benchmarks measure warmed-up predictor-call latency and end-to-end batch throughput, saving separate masks.
  
  Training wall-clock in `training_times.csv` includes initialization, training, final held-out-fold validation, and softmax export (`--npz`). Only final and best checkpoints are saved; there are no milestone checkpoints or timings.
  
  
  ### Optionally monitor progress
  
  From a separate SSH session:
  
  ```sh
  #
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
  
  tail -f ~/nnunet-liver-ultrasound-efficiency-baselines/logs/augmentation_ablation_runner.log
  ```
  

  ## 12. Complete and archive the run
  
  ### Option A: Manual completion
  
  #### Verify completion
  
  On the GPU instance, from the repository root, check for exit status `0` and successful separate verification:
  
  ```sh
  #
  cat augmentation_ablation_runner_exit_status.txt  # must be 0
  python3 experiments/03_augmentation_ablation/augmentation_ablation_runner_remote_control.py verify-local
  ```
  
  #### Finish the GPU usage record
  
  ```sh
  #
  python3 experiments/03_augmentation_ablation/finish_recording.py
  ```
  
  Records missing completion events, stops the sampler, and writes the cost summary. Safe to rerun; download time is excluded.
  
  #### Download everything
  
  On your **local computer**, from the repository root (replace `<gpu-ip>`):
  
  ```sh
  #
  bash experiments/download_archive.sh 03_augmentation_ablation root@<gpu-ip>
  ```
  
  Downloads and verifies the repository, logs, and three nnU-Net directories, excluding the Python environment. The completed archive is published to `archives/03_augmentation_ablation/`. Retry interrupted downloads with the same command; partial copies remain in `archives_staging/03_augmentation_ablation/`. Published archives are immutable; later additions belong in `archives_amendments/`.
  
  #### Delete the GPU
  
  After `Archive verified`, delete the instance in Verda. You can delete the block volume simultaneously. Deleted block volums are recoverable for up to 96 hours..

### Option B: Automated completion and GPU deletion

 On a CPU, prepare this checkout, Python 3.10+, `rsync`, non-interactive SSH with a trusted GPU host key, and the configured [Verda CLI](https://docs.verda.com/cli/getting-started/). Keep the SSH private key and Cloud API credentials on the CPU. 
  
  After training starts, launch this once from the repository root **on the CPU**. Use the GPU's actual SSH address, instance ID, and absolute paths; no need to wait for training to finish:

```bash
#
python3 -B experiments/03_augmentation_ablation/augmentation_ablation_runner_remote_control.py finish \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800 \
  --instance-id '<gpu-instance-id>' \
  --destination archives/03_augmentation_ablation
```

  The controller waits for successful completion, finishes recording, verifies the outputs and downloaded archive, and deletes the instance without further approval. Any failed check prevents archive publication and instance deletion.
  
  The deletion command has no volume-retention option. However, as of October 2026 deleted volumes on Verda can be "recovered from the trash" within 96 hours.
