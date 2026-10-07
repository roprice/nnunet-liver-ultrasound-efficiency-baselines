#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

DRYRUN_PREPROCESSED="${nnUNet_preprocessed%/}/dry_run"
DRYRUN_RESULTS="${nnUNet_results%/}/dry_run"
for path in "$DRYRUN_PREPROCESSED" "$DRYRUN_RESULTS"; do
    [[ ! -e "$path" && ! -L "$path" ]] || {
        echo "Dry-run output already exists: $path" >&2
        exit 1
    }
done
export nnUNet_preprocessed="$DRYRUN_PREPROCESSED"
export nnUNet_results="$DRYRUN_RESULTS"

# The dry run rehearses the real runner, so it calls the full run's script in dry-run mode.
bash "$SCRIPT_DIR/../training_efficiency_runner.sh" --dry-run
