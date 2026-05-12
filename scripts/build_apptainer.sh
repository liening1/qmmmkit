#!/usr/bin/env bash
# Build a qmmmkit Apptainer image (qmmmkit.sif) from packaging/Apptainer.def.
#
# Run on a Linux host with apptainer (or singularity) installed. Most
# clusters expose `apptainer` on the login node; some restrict it to
# fakeroot-capable hosts. If neither is available, build with the
# remote builder via `--remote` (see fallback hints printed on failure).
#
#   ./scripts/build_apptainer.sh                  # writes ./qmmmkit.sif
#   ./scripts/build_apptainer.sh -o /scratch/me/qmmmkit.sif
#   ./scripts/build_apptainer.sh --remote         # use the Sylabs remote builder
#   ./scripts/build_apptainer.sh --fakeroot       # explicit fakeroot
#
# Once built, point a cluster profile at it::
#
#   apptainer_image: /scratch/me/qmmmkit.sif
#
# and the SLURM job script will run `apptainer exec <image> python -m
# qmmmkit.runner ...` automatically (see SlurmScheduler).

set -euo pipefail

OUT="qmmmkit.sif"
BUILD_FLAGS=()
USE_REMOTE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)   OUT="$2"; shift 2 ;;
        --remote)      USE_REMOTE=1; shift ;;
        --fakeroot)    BUILD_FLAGS+=(--fakeroot); shift ;;
        --force)       BUILD_FLAGS+=(--force); shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# *//'
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
RECIPE="$repo_root/packaging/Apptainer.def"
[[ -f "$RECIPE" ]] || { echo "[build] missing $RECIPE" >&2; exit 1; }

if command -v apptainer >/dev/null 2>&1; then
    BUILDER="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    BUILDER="singularity"
else
    cat >&2 <<'EOF'
[build] No apptainer/singularity on PATH.
        Options:
          1. Build remotely from a workstation that has Docker:
                docker build -f packaging/Dockerfile.builder -t qmmmkit .
                docker run --rm -v "$PWD:/out" qmmmkit \
                    apptainer build /out/qmmmkit.sif /qmmmkit/packaging/Apptainer.def
          2. Use the Sylabs Cloud builder:
                apptainer build --remote qmmmkit.sif packaging/Apptainer.def
          3. Ask the sysadmin to enable apptainer on the cluster.
EOF
    exit 1
fi
echo "[build] using $($BUILDER --version 2>&1 | head -n1)"

if [[ "$USE_REMOTE" -eq 1 ]]; then
    BUILD_FLAGS+=(--remote)
fi

# If we're not running as root and no fakeroot is requested, try fakeroot first.
if [[ "$EUID" -ne 0 && "$USE_REMOTE" -eq 0 ]]; then
    if ! printf '%s\n' "${BUILD_FLAGS[@]}" | grep -qx -- "--fakeroot"; then
        echo "[build] non-root build: adding --fakeroot (override with --remote if your cluster forbids it)"
        BUILD_FLAGS+=(--fakeroot)
    fi
fi

echo "[build] $BUILDER build ${BUILD_FLAGS[*]} $OUT $RECIPE"
$BUILDER build "${BUILD_FLAGS[@]}" "$OUT" "$RECIPE"

echo "[build] OK: $(ls -lh "$OUT" | awk '{print $5, $NF}')"
echo "[build] smoke test:"
$BUILDER run "$OUT" --help 2>&1 | head -n 5 || true

cat <<EOF

Wire it up by editing ~/.config/qmmmkit/clusters.yaml and adding:

  apptainer_image: $(realpath "$OUT")

Subsequent qmmmkit submit calls will run inside this image on the compute
nodes — no conda activation needed in the sbatch script.
EOF
