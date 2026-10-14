# Gaussian noise ablation: noise augmentation removed

Train models without Gaussian noise augmentation at training-pool size 588, for folds 0–4, seed 42, for 150 epochs. This yields 1 set of 5 models on the same 147 held-out AUL cases as experiments 01 and 02. Trainer: `nnUNetTrainer_gaussianNoiseAugmentation`.


## Fixed conditions

| Setting | Value |
|---|---|
| Full dataset | AUL / `Dataset001_AUL` |
| Training-pool size | 588 |
| Held-out test cases | 147 |
| Folds | 0–4 |
| Test set | UltraBench AUL test set (Tupper & Gagné, 2025) |
| Initialization seed | 42 |
| Epoch budget | 150 |
| Gaussian noise | p=0 (removed) |
| Saved checkpoints | `checkpoint_final.pth` and `checkpoint_best.pth` |

Uses `Dataset001_AUL` (ID 1), the canonical mapping and folds from `experiments/prepare_data/aul_splits.py`, and its own planning/preprocessing. 

Note that `p=0` represents the ablation test. It means that we modify the augmentation used in prior experiments by over-riding nnU-Net's default Gaussian noise augmentation so that it is never applied.

| Setting | nnU-Net default (v2.8.1) | This experiment |
|---|---|---|
| Noise variance range | 0–0.1 | 0–0.1 (never applied) |
| Probability per training sample/patch | 0.1 | 0 |

We don't override any other nnU-Net augmentation transforms.


## 1. Check Python and install system dependencies

These scripts are based on an Ubuntu GPU instance with Python 3.10+, working NVIDIA drivers/`nvidia-smi`, and space for the source data, three nnU-Net directories holding checkpoints and other data, and logs. 

```sh
# install python lest not be installed already
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
  # clone the study's repo from github
cd ~
git clone https://github.com/roprice/nnunet-liver-ultrasound-efficiency-baselines.git
cd nnunet-liver-ultrasound-efficiency-baselines
```

## 3. Start GPU usage and estimated cost recording

Edit the rate and model to match the GPU you're running on - that gets set in `logs/04_gaussian_noise_ablation/gpu_events.csv` (which you can also edit later, if you skip this step). Like many cloud GPU providers, Verda frequently updates pricing, so you will likely find this step necessary.

The estimate covers the recorded window, not time before or after it.

  ```sh
  # log the start of the experiment
  mkdir -p logs/04_gaussian_noise_ablation
  printf 'gpu_model,hourly_rate_usd\nRTX 6000 Ada,1.290\n' > logs/04_gaussian_noise_ablation/gpu_rate.csv
  printf 'event,utc\n' > logs/04_gaussian_noise_ablation/gpu_events.csv
  printf '%s\n' '#!/bin/sh' 'printf "%s,%s\n" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> logs/04_gaussian_noise_ablation/gpu_events.csv' > logs/04_gaussian_noise_ablation/record_event.sh
  sh logs/04_gaussian_noise_ablation/record_event.sh recording_started
  
  # Sample the GPU in a detached tmux session so sampling continues through SSH disconnections
  tmux new-session -d -s gpu_usage -c "$PWD" \
    'nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv --loop-ms=1000 > logs/04_gaussian_noise_ablation/gpu_monitor_instance.csv'
  ```
  
  The runner log remains the source for per-fold training and prediction times. The completion branches calculate elapsed hours and estimated cost from the events and `gpu_rate.csv`

## 4. Create a Python environment

```sh
# create and start the virtual environment
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
```

Keep `.venv` outside the checkout: the remote controller copies the entire repository and rejects symlinks in its backup trees.

## 5. Install Python dependencies

```sh
# install key dependences
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1' 'torch==2.14.1' zenodo-get
python -c 'import torch, numpy, PIL; assert torch.cuda.is_available()'
python -m pip freeze > logs/04_gaussian_noise_ablation/pip_freeze.txt
```

PyTorch is pinned to 2.14.1, the version an unpinned `nnunetv2==2.8.1` install resolved to on 2026-10-01. If the CUDA check fails, install a CUDA-enabled build of that same version before proceeding. The runner records the installed nnU-Net, PyTorch, CUDA and cuDNN versions, CPU and data-augmentation worker counts, the repository git SHA and working-tree status, and saves `pip freeze` to  `logs/04_gaussian_noise_ablation/`.

## 6. Verify the installed transforms

Verify the actual installed wrapper representation and trainer pipeline on this GPU environment, then run the unit tests:

```bash
python - <<'PY' 2>&1 | tee logs/04_gaussian_noise_ablation/inspect_runtime_gaussian_noise.txt
from importlib.metadata import version
import inspect
import sys
import torch
import batchgeneratorsv2.transforms.intensity.gaussian_noise as gaussian_noise
from batchgeneratorsv2.transforms.utils.random import RandomTransform

assert version('batchgeneratorsv2') == '0.3.6'
print('batchgeneratorsv2:', version('batchgeneratorsv2'))
print(inspect.getsource(RandomTransform))
sys.path.insert(0, 'experiments/04_gaussian_noise_ablation/custom_trainers')
from nnUNetTrainer_gaussianNoiseAugmentation import nnUNetTrainer_gaussianNoiseAugmentation
pipeline = nnUNetTrainer_gaussianNoiseAugmentation.get_training_transforms(
    patch_size=(32, 32), rotation_for_DA=(-0.2, 0.2),
    deep_supervision_scales=((1, 1), (0.5, 0.5)),
    mirror_axes=(0, 1), do_dummy_2d_data_aug=False)
wrappers = [t for t in pipeline.transforms if isinstance(t, RandomTransform)
            and isinstance(t.transform, gaussian_noise.GaussianNoiseTransform)]
assert len(wrappers) == 1
wrapper = wrappers[0]
print('Actual trainer wrapper:', repr(wrapper))
assert wrapper.apply_probability == 0.0
transform = wrapper.transform
assert transform.noise_variance == (0, 0.1)
assert transform.synchronize_channels is True and transform.p_per_channel == 1
image = torch.ones((3, 8, 8))
for _ in range(100):
    torch.testing.assert_close(wrapper(image=image)['image'], image)
print('Zero-probability wrapper leaves images unchanged: passed')
PY

python -B -m unittest discover -s experiments/04_gaussian_noise_ablation \
  -p test_gaussian_noise_ablation.py -v 2>&1 | tee logs/04_gaussian_noise_ablation/trainer_tests.txt
```

Require both commands to succeed with `pipefail` enabled and review the recorded source. Expected wrapper representation, ignoring whitespace: `RandomTransform(p=0.0, transform=GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True))`.

Tupper & Gagné (2025) found Gaussian noise to substantially reduce Dice for AUL mass segmentation. This experiment tests whether removing nnU-Net’s own Gaussian noise transform improves detection metrics. It is not a replication of Tupper & Gagné’s augmentation, which was additive and operated on [0, 1]-scaled images rather than nnU-Net’s z-score-normalized intensities

The unit tests have not yet been run locally against the experiment 02 parent trainer. The pinned GPU installation must pass the runtime checks and the same tests before launch.

## 7. Configure nnU-Net directories

```sh
# set up the environment
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/04_gaussian_noise_ablation/custom_trainers"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
test -f "$nnUNet_extTrainer/nnUNetTrainer_gaussianNoiseAugmentation.py"
cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/nnunet-liver-ultrasound-efficiency-baselines/experiments/04_gaussian_noise_ablation/custom_trainers"
ENVEOF
```

## 8. Download AUL from Zenodo

  Download record `7272660` and verify all three archive checksums before extracting:

```sh
  # get the AUL dataset
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
 # preprocess image dataset to nnU-net format
python experiments/prepare_data/aul_conversion.py \
  --raw-data-dir data/source/AUL \
  --output-dir "${nnUNet_raw:?Run step 7 first}/Dataset001_AUL" \
  --reference-mapping experiments/prepare_data/reference/case_mapping.json
```

Expect 588 training and 147 test image/label pairs. The converter stops if the case assignments differ from the committed reference or if annotation files are missing or unexpected, and writes `conversion_report.json`. For Malignant images 229 and 306, AUL files the liver polygon under `segmentation/outline/`, so the converter uses it as the liver. Malignant image 374 has no liver polygon; its label contains the mass only. The runner validates `case_mapping.json`.  Experiment 04 uses only the full 588-image dataset and creates no subsets.

## 10. Verify input data and mark setup complete

```sh
# verify split of AUL data
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 588
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 147
```

Check that `dataset.json` reports 588 training cases and `case_mapping.json` records all 735 cases. Set the fixed 150-epoch budget, then mark setup complete:

```sh
# Set the training length
export DATA_EFFICIENCY_EPOCHS=150
sh logs/04_gaussian_noise_ablation/record_event.sh setup_complete
```

  ## 11. Run the data-efficiency experiment
  
  ```sh
  tmux new -s training
  ```
  
  Inside tmux, run from the repository root with the fixed 150-epoch budget. The runner requires fresh outputs and does not resume. The launcher log stays outside `logs/04_gaussian_noise_ablation/`, and the exit-status file is written only after the runner exits:
  
  ```sh
  # Once in tmux, start the run
  . "$HOME/.venv/bin/activate"
  export DATA_EFFICIENCY_EPOCHS='150'
  rm -f gaussian_noise_ablation_runner_exit_status.txt
  sh logs/04_gaussian_noise_ablation/record_event.sh experiment_started
  if bash -o pipefail -c 'bash experiments/04_gaussian_noise_ablation/gaussian_noise_ablation_runner.sh 2>&1 | tee logs/gaussian_noise_ablation_runner.log'; then
    RUN_STATUS=0
  else
    RUN_STATUS=$?
  fi
  printf '%s\n' "$RUN_STATUS" > gaussian_noise_ablation_runner_exit_status.txt.tmp
  mv gaussian_noise_ablation_runner_exit_status.txt.tmp gaussian_noise_ablation_runner_exit_status.txt
  ```
  
  Detach with `Ctrl+b` then `d`; reattach with `tmux attach -t training`. Check `logs/gaussian_noise_ablation_runner.log` and per-stage logs if it fails.
  
  The runner preprocesses, trains, and predicts both checkpoints per fold on GPU (10 prediction sets). Best masks go to `predictions_best/size{SIZE}/fold{FOLD}/`; final paths are unchanged. It calls `experiments/04_gaussian_noise_ablation/benchmark_inference.py` for final and best only (10 benchmarks), writing best reports to `inference_best/size{SIZE}/fold{FOLD}/`. Both benchmarks record warmed-up predictor-call latency and end-to-end batch throughput, saving separate sets of masks.
  
  Each timed training step includes more than epochs: `nnUNetv2_train` finishes with a full validation of the held-out fold cases, and `--npz` adds exporting their softmax probabilities. Training wall-clock in `training_times.csv` therefore includes validation inference and file writing.
  
  The runner validates both checkpoints and saves their metadata and hashes for completion verification.
  
  
  ### Optionally monitor progress
  
  From a separate SSH session:
  
  ```sh
  # check status
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
  
  tail -f ~/nnunet-liver-ultrasound-efficiency-baselines/logs/gaussian_noise_ablation_runner.log
  ```
  
  Use `Ctrl+C` to close `tail`.
  
  The training will be complete when the final messages are as follows:
  
  ```
  End time: YYYY-MM-DDTXX:XX:XXZ
  Total wall clock: XXXXXXs
  ```

  ## 12. Complete and archive the run
  
  Option A takes you through manually completion (verifying, finishing, downloading). Option B  automatically completes the run and also automatically deletes the GPU instance on Verda.
  
  ### Option A: Manual completion
  
  #### Verify completion
  
  On the GPU instance, from the repository root, check for exit status `0` and successful separate verification:
  
  ```sh
  # Check that the run is complete
  cat gaussian_noise_ablation_runner_exit_status.txt  # must be 0
  python3 experiments/04_gaussian_noise_ablation/gaussian_noise_ablation_runner_remote_control.py verify-local
  ```
  
  #### Finish the GPU usage record
  
  ```sh
   # Mark the completion of the run
  python3 experiments/04_gaussian_noise_ablation/finish_recording.py
  ```
  
  Records missing completion events, stops the sampler, and writes the cost summary. Safe to rerun; download time is excluded.
  
  #### Download everything
  
  On your **local computer**, from the repository root (replace `<gpu-ip>`):
  
  ```sh
  bash experiments/download_archive.sh 04_gaussian_noise_ablation root@<gpu-ip>
  ```
  
  Copies the repository and three nnU-Net directories into `archives_staging/04_gaussian_noise_ablation/`, including final logs but not the Python environment. Checks file contents outside `.git/` and saves the matching Git commit in `git_commit.txt`, then publishes the verified copy to `archives/04_gaussian_noise_ablation/`. Retry interrupted staging downloads with the same command. Published archives are immutable; later additions belong in `archives_amendments/`.
  
  #### Delete the GPU
  
  After `Archive verified`, delete the instance in Verda. You can delete the block volume simultaneously. Deleted block volums are recoverable for up to 96 hours..

### Option B: Automated completion

 On a CPU, prepare this checkout, Python 3.10+, `rsync`, non-interactive SSH with a trusted GPU host key, and the configured [Verda CLI](https://docs.verda.com/cli/getting-started/). Keep the SSH private key and Cloud API credentials on the CPU. 
  
  After training starts, launch this once from the repository root **on the CPU**. Use the GPU's actual SSH address, instance ID, and absolute paths; no need to wait for training to finish:

```bash
python3 -B experiments/04_gaussian_noise_ablation/gaussian_noise_ablation_runner_remote_control.py finish \
  --ssh-target root@<gpu-ip> \
  --remote-home /root \
  --remote-repo /root/nnunet-liver-ultrasound-efficiency-baselines \
  --poll-max-seconds 604800 \
  --instance-id '<gpu-instance-id>' \
  --destination archives/04_gaussian_noise_ablation
```

  The controller waits for runner exit status `0`, verifies the experiment outputs, finishes recording, copies into `archives_staging/`, SHA-256-verifies the copy, confirms the GPU's identity, saves a completion manifest, publishes the immutable archive, and deletes the instance. No manual completion approval is required. Any failed check prevents publication and deletion.
  
  The deletion command has no volume-retention option. However, as of October 2026 deleted volumes on Verda can be "recovered from the trash" within 96 hours.
