# Experiment 06: HCC caliper transcription

Hand-transcribes the sonographers' caliper endpoints on HCC caliper frames, turns each pair into the same circular proxy used for hemangioma, and predicts those frames with the nnU-Net default-augmentation models. The goal is localization-grounded, multi-pass HCC data: only patients with at least 4 HCC caliper frames are queued (87 patients, 453 frames).

The transcriber records where the two marks are. They do not judge the lesion. Every frame that is not unambiguous is excluded.

Run commands from the repository root on the local computer.

## Requirements

- The extracted SMC-LUD source at `experiments/06_external_validation/SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)`.
- The automatic proxy masks at `archives/06_external_validation/proxy_masks/automatic/`, from `extract_caliper_proxy.py`. Their hemangioma twins are predicted in the same run, for the twin-pair test.
- matplotlib in the virtual environment:

```sh
.venv/bin/python -m pip install matplotlib
```

## 1. Transcribe

```sh
.venv/bin/python experiments/06_external_validation/transcribe_calipers.py transcribe
```

The tool shows one frame at a time and saves after every decision to `archives/06_external_validation/proxy_masks/hcc_transcribed/caliper_transcriptions.json`. Quit at any time with `q`; rerunning resumes where you stopped.

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

How to zoom:

- Click the magnifier button in the toolbar.
- On the image, press and drag a box around the mark, then release.
- Click the magnifier again to turn it off. Your clicks won’t register until you do, and a red message reminds you.
- Click the mark centers. You can zoom in on one mark, click it, press the home button (house icon) for the full view, then zoom in on the second mark. Your first click is kept.

Note that when you advance to the next frame, via Enter to save or x to exclude, the next frame opens at full view.



Expect about 5 seconds per frame, roughly 40 minutes in total.

## 2. Write proxy masks

```sh
.venv/bin/python experiments/06_external_validation/transcribe_calipers.py masks
```

This writes one circle per saved pair (center at the midpoint of the two clicks, diameter equal to their distance) to `archives/06_external_validation/proxy_masks/hcc_transcribed/`, next to the transcriptions, with an `extraction_report.json` that counts saved and excluded frames.

## 3. Prepare the inference cohort

```sh
.venv/bin/python -I experiments/06_external_validation/prepare_caliper_cohort.py
```

This writes `archives/06_external_validation/caliper/prepared/`: the 380 hemangioma caliper twins, every transcribed HCC frame, and `twin_pairs.json`.

## 4. Predict

Activate the virtual environment first: the runner calls `nnUNetv2_predict`, which is only on the path inside it. Locally, check the inputs, then predict:

```sh
source .venv/bin/activate
python experiments/06_external_validation/external_validation_runner.py \
  --subset caliper --experiments data_efficiency --device mps --check-only
mkdir -p archives/06_external_validation/caliper/logs
caffeinate -i python experiments/06_external_validation/external_validation_runner.py \
  --subset caliper --experiments data_efficiency --device mps \
  2>&1 | tee archives/06_external_validation/caliper/logs/runner.log
```

The check should report every prepared frame, 5 checkpoints and one proxy mask per transcribed frame. On the GPU host use `--device cuda` instead. `archives/06_external_validation/caliper/logs/per_image_results.csv` then has overlap and centroid flags for each transcribed HCC frame. Rerun the same command to resume after an interruption.

## 5. Check that the marks do not change predictions

```sh
.venv/bin/python -m analysis.06_external_validation.twin_mark_effect.twin_mark_effect
```

The HCC frames carry visible marks, so their results stand only if the hemangioma twins show that marks barely change predictions. If any difference in `twin_mark_effect.md` reaches the study's 0.03 threshold, stop and remove the marks before predicting.

## 6. Project multi-pass detection

```sh
.venv/bin/python -m analysis.06_external_validation.multi_pass_projection.multi_pass_projection
```

With enough HCC patients with two or more transcribed frames, HCC overlap is fitted directly. The transferred-ρ row is reported either way.
