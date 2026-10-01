# Prediction efficiency: Mac CPU

Predict and benchmark both `checkpoint_final.pth` and `checkpoint_best.pth` for experiment 02's 20 models on Mac CPU: four training-pool sizes × folds 0–4, seed 42, and the same 147 held-out AUL test cases. No training or new custom trainer is needed.

| Training-pool size | Model dataset |
|---:|---|
| 588 | `Dataset001_AUL` |
| 294 | `Dataset002_AUL_294` |
| 147 | `Dataset003_AUL_147` |
| 74 | `Dataset004_AUL_074` |

## 1. Copy the inputs to the Mac

Copy these experiment 02 outputs to the Mac:

- `nnUNet_raw/Dataset001_AUL/{imagesTs/,case_mapping.json}`.
- For each model dataset, `nnUNet_results/<name>/nnUNetTrainer_dataSubsets_Seed42__nnUNetPlans__2d/`, including `dataset.json`, `plans.json`, and `fold_<0..4>/{checkpoint_final.pth,checkpoint_best.pth}`.

Verify the transferred files against the source checksums. Training arrays and raw subset folders are not needed. The checkout must include experiment 02's custom trainer and `experiments/benchmark_inference.py`.

## 2. Create the Python environment and install dependencies

Use Python 3.10+ on macOS. Run from the repository root:

```sh
python3 -m venv "$HOME/.venv"
. "$HOME/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'nnunetv2==2.8.1'
python -c 'import torch, nnunetv2; print(torch.__version__)'
```

## 3. Configure paths and the epoch budget

Adjust the storage paths and replace the epoch-budget placeholder with the value used in experiment 02:

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export DATA_EFFICIENCY_EPOCHS='<chosen_positive_integer>'
```

## 4. Check the inputs and commands

```sh
python experiments/03_prediction_efficiency/prediction_efficiency_runner.py --dry-run
```

The runner sets `nnUNet_extTrainer` to experiment 02's trainer. The dry run checks the mapping, test-image names, model metadata, and all 40 checkpoints, then prints the commands without writing output.

## 5. Run the prediction efficiency experiment 

`logs/03_prediction_efficiency/` must be empty. The runner stops on failure and does not resume; archive incomplete outputs or use a new, empty `--output-dir` before rerunning.

```sh
caffeinate -i python experiments/03_prediction_efficiency/prediction_efficiency_runner.py
```

The runner predicts and calls `experiments/benchmark_inference.py` for both checkpoints of each fold and training-pool size on CPU (40 prediction and 40 benchmark runs). It uses experiment 02's benchmark settings: 2D `nnUNetPlans`, step size 0.5, mirroring enabled, three warm-up images, one batch repeat, and one preprocessing and export worker. The benchmark saves a separate set of masks.

## 6. Verify completion and retain results

Expect exit status `0` and `Completed 40 CPU checkpoint runs across 20 models`. The runner checks the standalone and benchmark masks for both checkpoints of all 20 models and requires four benchmark reports per checkpoint. Keep the complete output directory with the results.

## Output layout

```text
logs/03_prediction_efficiency/
  run_settings.json
  prediction_times.csv                         # 40 standalone prediction batches
  size<size>/fold<fold>/
    predict.log
    predictions_cpu/                           # 147 final masks
    predict_best.log
    predictions_best_cpu/                      # 147 best masks
    benchmark.log
    benchmark_best.log
    inference/
      inference_per_image_cpu_fold<fold>.csv    # 147 warmed-up predictor-call timings
      inference_summary_cpu_fold<fold>.csv      # distribution, loading, preprocessing, peak CPU RSS
      inference_throughput_cpu_fold<fold>.csv   # one 147-image end-to-end batch timing
      inference_settings_cpu_fold<fold>.json    # environment and inference options
      batch_cpu_seed42_fold<fold>_repeat1.log
      predictions_cpu_seed42_fold<fold>_repeat1/ # 147 final masks
    inference_best/
      inference_per_image_cpu_fold<fold>.csv
      inference_summary_cpu_fold<fold>.csv
      inference_throughput_cpu_fold<fold>.csv
      inference_settings_cpu_fold<fold>.json
      batch_cpu_seed42_fold<fold>_repeat1.log
      predictions_cpu_seed42_fold<fold>_repeat1/ # 147 best masks
```

The matching CUDA reports are under `logs/02_data_efficiency/inference/size<size>/fold<fold>/` for final and `logs/02_data_efficiency/inference_best/size<size>/fold<fold>/` for best. Per-image CSVs record warmed-up predictor-call latency; throughput CSVs record end-to-end batch time and average seconds per image.
