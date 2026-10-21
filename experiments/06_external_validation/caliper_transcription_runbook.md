# Experiment 06: HCC caliper transcription

Hand-transcribes the sonographers' caliper endpoints on HCC caliper frames, turns each pair into the same circular proxy used for hemangioma, and predicts those frames with the nnU-Net default-augmentation models. The goal is localization-grounded, multi-pass HCC data: only patients with at least 4 HCC caliper frames are queued (87 patients, 453 frames).

The transcriber records where the two marks are. They do not judge the lesion. Every frame that is not unambiguous is excluded.

Run commands from the repository root on the local computer.

## Requirements

- The extracted SMC-LUD source at `experiments/06_external_validation/SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)`.
- The existing proxy masks at `archives_staging/06_external_validation/proxy_masks`, from `extract_caliper_proxy.py`. Their hemangioma twins are predicted in the same run as the twin-pair test.
- matplotlib in the virtual environment:

```sh
.venv/bin/python -m pip install matplotlib
```

## 1. Transcribe

```sh
.venv/bin/python experiments/06_external_validation/transcribe_calipers.py transcribe
```

The tool shows one frame at a time and saves after every decision to `archives_staging/06_external_validation/caliper_transcriptions.json`. Quit at any time with `q`; rerunning resumes where you stopped.

| key | action |
|---|---|
| left click | mark a caliper center (two per frame) |
| `u` | undo the last click |
| Enter | save the two clicks as this frame's pair |
| `x` | exclude the frame |
| `q` | quit |

Rule: **save only when the frame has exactly one pair of caliper marks, on one lesion.** Click the center of each of the two marks. Press `x` for everything else:

- no caliper marks;
- more than one pair of marks, including two measurements of the same lesion or marks on separate lesions;
- a mark whose center you cannot locate;
- any doubt.

To zoom, use the toolbar magnifier, then click it again to switch it off before clicking marks. Clicks are ignored while zoom or pan is active.

Expect about 5 seconds per frame, roughly 40 minutes in total.

## 2. Write proxy masks

```sh
.venv/bin/python experiments/06_external_validation/transcribe_calipers.py masks
```

This writes one circle per saved pair (center at the midpoint of the two clicks, diameter equal to their distance) to `archives_staging/06_external_validation/proxy_masks_transcribed/`, with an `extraction_report.json` that counts saved and excluded frames.

## 3. Prepare the inference cohort

```sh
.venv/bin/python -I experiments/06_external_validation/prepare_twin_test.py \
  --proxy-masks archives_staging/06_external_validation/proxy_masks \
  --hcc-proxy-masks archives_staging/06_external_validation/proxy_masks_transcribed \
  --output-dir archives_staging/06_external_validation/caliper_transcribed_prepared
```

The cohort holds the 380 hemangioma caliper twins and every transcribed HCC frame.

## 4. Predict

On the GPU host, or locally with `--device mps`:

```sh
python experiments/06_external_validation/external_validation_runner.py \
  --subset caliper \
  --prepared archives_staging/06_external_validation/caliper_transcribed_prepared \
  --proxy-masks archives_staging/06_external_validation/proxy_masks_transcribed \
  --experiments data_efficiency \
  --output-dir logs/06_external_validation_caliper_transcribed \
  --device cuda
```

`logs/06_external_validation_caliper_transcribed/per_image_results.csv` then has overlap and centroid flags for each transcribed HCC frame.

## 5. Check that the marks do not change predictions

```sh
.venv/bin/python -m analysis.06_external_validation.twin_mark_effect.twin_mark_effect \
  --pairs archives_staging/06_external_validation/caliper_transcribed_prepared/twin_pairs.json \
  --predictions logs/06_external_validation_caliper_transcribed/predictions
```

The HCC frames carry visible marks, so their results stand only if the hemangioma twins show that marks barely change predictions. If any difference in `twin_mark_effect.md` reaches the study's 0.03 threshold, stop and remove the marks before predicting.

## 6. Project multi-pass detection

```sh
.venv/bin/python -m analysis.06_external_validation.multi_pass_projection.multi_pass_projection \
  --hcc-caliper-results logs/06_external_validation_caliper_transcribed/per_image_results.csv
```

With enough HCC patients with two or more transcribed frames, HCC overlap is fitted directly. The transferred-ρ row is reported either way.

## 7. Archive

Move `caliper_transcriptions.json`, `proxy_masks_transcribed/`, the prepared cohort's manifest and `twin_pairs.json`, and the run's `per_image_results.csv`, `run_settings.json`, `summary.json` and predictions under `archives/06_external_validation/`.
