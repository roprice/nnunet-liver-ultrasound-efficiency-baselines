This section contains the code used to run the training using nnUnet. Each experiment has its own set of code, including a markdown runbook.

These files are shared:
- benchmark_inference.py
- download_archive.sh (copies a finished run from the GPU server to `archives/<experiment>` and verifies it by checksum; the optional remote controllers write the same layout)
- prepare_data/aul_conversion.py (requires `--reference-mapping prepare_data/reference/case_mapping.json`, the committed case assignments, with UltraBench's AUL test set)
- prepare_data/aul_splits.py (shared train/test split, nested subsets, and five-fold assignments)
- prepare_data/aul_splits_verification.py (split tests; run with `python3 -m unittest discover -s experiments/prepare_data -p '*_verification.py'`)

The training-efficiency dry run is manual and lives in `01_training_efficiency/dry_run/`, with its own runbook. It is separate from the remote controller.
