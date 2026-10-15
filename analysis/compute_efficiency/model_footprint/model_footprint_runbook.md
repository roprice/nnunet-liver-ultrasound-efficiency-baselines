# Model-footprint verification and FP16 export

Run from the repository root with the existing environment:

```sh
.venv/bin/python -B -m unittest analysis.compute_efficiency.model_footprint.test_model_footprint
.venv/bin/python -B analysis/compute_efficiency/model_footprint/model_footprint.py
```

The default archive is `archives/02_data_efficiency`. The CLI verifies all 40 best/final checkpoints (four pools × five folds × two checkpoints) and writes `model_footprint.csv`, `.json`, and `.md` beside the script. Reports are regenerated only after verification; `--output-dir` selects another location. Outputs inside archives, linked report files, and existing retained-export files are rejected.

Every checkpoint must match its `checkpoint_metadata.json` SHA256, size, trainer, fold, and completed epoch. Embedded plans/dataset must match results metadata; `nnUNet_preprocessed/<dataset>/nnUNetPlans.json` must also match when present. Missing optional preprocessing plans are recorded. The real architecture is rebuilt with one input, three label-derived outputs, `allow_init=False`, and `deep_supervision=False`; strict loading retains every segmentation head.

Parameter totals use unique `network.parameters()`, with buffers counted separately. Weights-only FP32 and FP16 files are actually serialized to temporary disk and measured in bytes, including ZIP overhead. Conversion uses `network.half().state_dict()` to preserve shared storages. Floating tensors must be finite and have the expected dtype. FP16 files are safely reloaded and checked for exact tensor values, dtypes, aliases, and strict loading into a fresh FP16 network. Full checkpoints use `weights_only=True` with only NumPy scalar/dtype/Float64DType/Float32DType additions; there is no unsafe fallback.

By default all temporary weights files are deleted. To retain one verified artifact, defaulting to pool 588 / fold 4 / final:

```sh
.venv/bin/python -B analysis/compute_efficiency/model_footprint/model_footprint.py \
  --output-dir analysis/compute_efficiency/model_footprint/retained_report \
  --export-dir analysis/compute_efficiency/model_footprint/fp16_export \
  --export-pool 588 --export-fold 4 --export-checkpoint final
```

Export selection changes retention only; all 40 checkpoints are still verified. The retained `.pth` is only a state dict, accompanied by a JSON sidecar with complete plans/dataset, source and export checksums, versions, and loading settings. It is not a resumable training checkpoint. To reconstruct, pass the sidecar's `plans` and `dataset` to the CLI module's `build_network`, call `.half()`, and strictly load `torch.load(path, map_location="cpu", weights_only=True)`.

JSON/CSV retain fold/checkpoint completed epochs, SHA256, full checkpoint bytes, measured FP32/FP16 weights bytes, unique-storage tensor bytes, dtypes, and parameter/buffer counts. JSON also records plan/dataset checksums, optional archived training provenance, local versions, and the tool checksum. MiB means bytes / 2²⁰. Training-pool size precedes fold splitting. Disk footprint and strict loading do not establish FP16 prediction equivalence, accuracy, runtime memory, latency, or energy; no inference or training is run.
