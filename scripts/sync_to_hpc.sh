#!/usr/bin/env bash
# Push the qmmmkit source tree to an HPC over SSH using rsync.
#
# Run from the repo root on a host with rsync available
# (Linux/macOS/WSL/Git Bash with rsync). For pure-Windows hosts without
# rsync, use sync_to_hpc.ps1 instead.
#
#   ./scripts/sync_to_hpc.sh user@hpc.example.edu:/home/user/qmmmkit
#   ./scripts/sync_to_hpc.sh -e "ssh -p 2222 -i ~/.ssh/id_ed25519" \
#                            user@hpc.example.edu:/scratch/user/qmmmkit
#
# Excludes: __pycache__, .pyc, .git/objects (we keep .git/HEAD so the
# provenance git-SHA still resolves on the HPC), build/, dist/,
# *.egg-info, .venv, .ruff_cache, .mypy_cache.

set -euo pipefail

SSH_CMD=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--ssh)    SSH_CMD="$2"; shift 2 ;;
        -n|--dry)    DRY_RUN="--dry-run"; shift ;;
        -h|--help)
            sed -n '2,16p' "$0" | sed 's/^# *//'
            exit 0 ;;
        --)          shift; break ;;
        *)           break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "usage: $0 [options] user@host:/remote/path" >&2
    exit 2
fi
DEST="$1"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

command -v rsync >/dev/null 2>&1 || {
    echo "rsync not found. On Windows: install via 'choco install rsync', use WSL, or use sync_to_hpc.ps1." >&2
    exit 1
}

echo "[sync] from: $repo_root"
echo "[sync] to:   $DEST"
[[ -n "$DRY_RUN" ]] && echo "[sync] DRY RUN — no files will be copied"

EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='.git/objects'
    --exclude='.git/lfs'
    --exclude='.venv'
    --exclude='build'
    --exclude='dist'
    --exclude='*.egg-info'
    --exclude='.ruff_cache'
    --exclude='.mypy_cache'
    --exclude='.pytest_cache'
    --exclude='qmmmkit_run'
    --exclude='handle.json'
)

RSYNC_FLAGS=(
    -av
    --delete
    --human-readable
    $DRY_RUN
)
[[ -n "$SSH_CMD" ]] && RSYNC_FLAGS+=(-e "$SSH_CMD")

rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" ./ "$DEST/"

cat <<EOF

[sync] DONE.

Next steps on the HPC:

  ssh ${DEST%%:*}
  cd ${DEST#*:}
  bash scripts/deploy_hpc.sh

EOF
