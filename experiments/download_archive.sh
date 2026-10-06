#!/usr/bin/env bash
# Download an experiment's four source trees from the GPU server into archives/ and verify the copy.
# Run locally: bash experiments/download_archive.sh [--live | --refresh] ARCHIVE_NAME GPU_SSH [REMOTE_HOME] [REMOTE_REPO]
#   ARCHIVE_NAME  e.g. 01_training_efficiency, 01_training_efficiency_dry_run, 02_data_efficiency
#   GPU_SSH       e.g. root@203.0.113.7
#   REMOTE_HOME   default /root (holds nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
#   REMOTE_REPO   default $REMOTE_HOME/nnunet-liver-ultrasound-efficiency-baselines
# Stop recording before downloading. Existing destinations can be updated by repeating the command.
# --live skips the running sampler; --refresh is a compatibility alias for the default final download.
# Git storage is copied but not checksum-compared; Git HEAD is checked and saved in git_commit.txt.
set -euo pipefail

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
LIVE=0
case "${1:-}" in
    --live) LIVE=1; shift ;;
    --refresh) shift ;;
esac
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
for program in rsync ssh git; do
    command -v "$program" >/dev/null || { echo "$program is required" >&2; exit 1; }
done

DEST="$(cd "$SCRIPT_DIR/.." && pwd)/archives/$ARCHIVE_NAME"
[[ ! -e "$DEST" || -d "$DEST" ]] || { echo "Archive destination is not a directory: $DEST" >&2; exit 1; }
EXCLUDES=(--exclude .DS_Store)
if [[ "$LIVE" == 1 ]]; then
    EXCLUDES+=(--exclude gpu_monitor_instance.csv)
fi
REMOTE_COMMIT="$(ssh "$GPU_SSH" "git -C '$REMOTE_REPO' rev-parse --verify HEAD")"
[[ "$REMOTE_COMMIT" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]] || { echo 'Invalid remote Git commit' >&2; exit 1; }

TREES=(repo nnUNet_raw nnUNet_preprocessed nnUNet_results)
remote_path() {
    if [[ "$1" == repo ]]; then echo "$REMOTE_REPO"; else echo "$REMOTE_HOME/$1"; fi
}

for name in "${TREES[@]}"; do
    echo "Copying $name..."
    mkdir -p "$DEST/$name"
    rsync -ac "${EXCLUDES[@]}" "$GPU_SSH:$(remote_path "$name")/" "$DEST/$name/"
done

# Git may repack objects without changing the recorded commit.
VERIFY_EXCLUDES=("${EXCLUDES[@]}" --exclude /.git/)
echo 'Verifying copy by checksum...'
DIFFERENCES=0
for name in "${TREES[@]}"; do
    CHANGES="$(rsync -ac --delete --dry-run --itemize-changes "${VERIFY_EXCLUDES[@]}" \
        "$GPU_SSH:$(remote_path "$name")/" "$DEST/$name/")"
    if [[ -n "$CHANGES" ]]; then
        echo "Differences in $name:" >&2
        printf '%s\n' "$CHANGES" >&2
        DIFFERENCES=1
    fi
done
LOCAL_COMMIT="$(git --no-optional-locks --git-dir="$DEST/repo/.git" rev-parse --verify HEAD)"
FINAL_REMOTE_COMMIT="$(ssh "$GPU_SSH" "git -C '$REMOTE_REPO' rev-parse --verify HEAD")"
if [[ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" || "$FINAL_REMOTE_COMMIT" != "$REMOTE_COMMIT" ]]; then
    echo 'Git HEAD differs or changed during the download' >&2
    DIFFERENCES=1
fi
[[ "$DIFFERENCES" == 0 ]] || { echo "Archive NOT verified: $DEST" >&2; exit 1; }
printf '%s\n' "$REMOTE_COMMIT" > "$DEST/git_commit.txt"
echo "Archive verified: $DEST"
if [[ "$LIVE" == 1 ]]; then
    echo 'Skipped the live sampler. Stop recording, then repeat without --live to include the final records.'
fi
