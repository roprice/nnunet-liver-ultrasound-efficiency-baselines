# Experiment02 model footprint

Verified 40 best/final checkpoint files: four training pools × five folds × two checkpoints.
SHA256 and full-file size match checkpoint_metadata.json; embedded plans/dataset match results metadata.
Present preprocessed plans match results plans. Every source and FP16 roundtrip loads strictly, including all segmentation heads.

| Training pool | Stages | Patch | Unique parameters | Buffer elements | FP32 weights bytes | FP16 weights bytes | FP16 MiB | Verified files |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 74 | 8 | 640×768 | 46,326,581 | 0 | 185,409,943 | 92,756,823 | 88.460 | 10 |
| 147 | 8 | 512×768 | 46,326,581 | 0 | 185,409,943 | 92,756,823 | 88.460 | 10 |
| 294 | 8 | 640×768 | 46,326,581 | 0 | 185,409,943 | 92,756,823 | 88.460 | 10 |
| 588 | 8 | 640×768 | 46,326,581 | 0 | 185,409,943 | 92,756,823 | 88.460 | 10 |

Training pool is the available pool before fold splitting, not the per-fold training count.
Parameters are counted once through network.parameters(); buffers are separate. Each state dict has 396 keys and 148 unique FP16 storages.
Disk sizes are measured from torch.save weights-only files, including serialization overhead; MiB = bytes / 2²⁰. FP32 files are also measured, not estimated.
Full training checkpoint sizes range from 370,797,343 to 370,832,205 bytes and include training state; they are not weights-only sizes.
CSV/JSON contain every fold/checkpoint's completed epochs (current_epoch), checksums, full checkpoint bytes, tensor bytes, measured export bytes, dtypes, and separate buffer counts.

FP16 conversion uses the strictly loaded network.half().state_dict(), preserving aliases. Floating tensors must be finite FP32 before conversion and finite FP16 afterward; integer buffers retain their dtype.
Exports are safely reloaded with weights_only=True, checked for exact serialized tensor values/dtypes/aliases, and strictly loaded into a fresh FP16 architecture.

These measurements establish storage footprint and loading compatibility only. No inference, prediction-equivalence, accuracy, runtime-memory, latency, or energy measurement is performed.
Retained FP16 artifact: none (temporary measurement files removed).
Measurement versions: {"dynamic-network-architectures": "0.4.4", "nnunetv2": "2.8.1", "numpy": "2.5.3", "torch": "2.14.1"}.
Tool SHA256: `9489432201cbba6c3a6e763e28492df7bbcdcf16f3277dd1e9195ec189ef0eef`.
