#!/usr/bin/env bash
# Download an experiment's four source trees from the GPU server into archives/ and verify the copy.
# Run on your local computer: bash experiments/download_archive.sh ARCHIVE_NAME GPU_SSH [REMOTE_HOME] [REMOTE_REPO]
#   ARCHIVE_NAME  e.g. 01_training_efficiency, 01_training_efficiency/dry_run, 02_data_efficiency
#   GPU_SSH       e.g. root@203.0.113.7
#   REMOTE_HOME   default /root (holds nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
#   REMOTE_REPO   default $REMOTE_HOME/nnunet-liver-ultrasound-efficiency-baselines
# Stop any GPU sampler and finish the run first, so no file changes during the copy.
set -euo pipefail

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
[[ $# -ge 2 && $# -le 4 ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_NAME="$1"
GPU_SSH="$2"
REMOTE_HOME="${3:-/root}"
REMOTE_REPO="${4:-$REMOTE_HOME/nnunet-liver-ultrasound-efficiency-baselines}"

[[ "$ARCHIVE_NAME" =~ ^[A-Za-z0-9_][A-Za-z0-9_./-]*$ && "$ARCHIVE_NAME" != *..* ]] || {
    echo "Invalid archive name: $ARCHIVE_NAME" >&2
    exit 2
}
[[ "$GPU_SSH" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "GPU_SSH must look like user@host: $GPU_SSH" >&2
    exit 2
}
for path in "$REMOTE_HOME" "$REMOTE_REPO"; do
    [[ "$path" =~ ^/[A-Za-z0-9_./-]+$ && "$path" != *..* ]] || {
        echo "Remote paths must be absolute, without spaces or '..': $path" >&2
        exit 2
    }
done
command -v rsync >/dev/null || { echo 'rsync is required' >&2; exit 1; }

DEST="$(cd "$SCRIPT_DIR/.." && pwd)/archives/$ARCHIVE_NAME"
if [[ -e "$DEST" ]]; then
    # Allow a parent archive that so far holds only its dry_run subfolder.
    [[ -d "$DEST" ]] && [[ -z "$(ls -A "$DEST" | grep -vx dry_run || true)" ]] || {
        echo "Archive destination already exists: $DEST" >&2
        exit 1
    }
fi

TREES=(repo nnUNet_raw nnUNet_preprocessed nnUNet_results)
remote_path() {
    if [[ "$1" == repo ]]; then echo "$REMOTE_REPO"; else echo "$REMOTE_HOME/$1"; fi
}

for name in "${TREES[@]}"; do
    echo "Copying $name..."
    mkdir -p "$DEST/$name"
    rsync -a --exclude .DS_Store "$GPU_SSH:$(remote_path "$name")/" "$DEST/$name/"
done

# Second pass: compare file contents by checksum without changing anything. Any output is a difference.
echo 'Verifying copy by checksum...'
DIFFERENCES=0
for name in "${TREES[@]}"; do
    CHANGES="$(rsync -ac --delete --dry-run --itemize-changes --exclude .DS_Store \
        "$GPU_SSH:$(remote_path "$name")/" "$DEST/$name/")"
    if [[ -n "$CHANGES" ]]; then
        echo "Differences in $name:" >&2
        echo "$CHANGES" | head -20 >&2
        DIFFERENCES=1
    fi
done
[[ "$DIFFERENCES" == 0 ]] || { echo "Archive NOT verified: $DEST" >&2; exit 1; }
echo "Archive verified: $DEST"
