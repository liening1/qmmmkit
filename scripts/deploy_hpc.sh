#!/usr/bin/env bash
# Install qmmmkit on an HPC into a conda env. Run this on the HPC after
# transferring the source tree (git clone, scp, or rsync — see sync_to_hpc.sh).
#
#   chmod +x scripts/deploy_hpc.sh
#   ./scripts/deploy_hpc.sh            # default env name "qmmmkit"
#   ./scripts/deploy_hpc.sh -n my-env  # custom env name
#   ./scripts/deploy_hpc.sh --skip-validate
#
# Idempotent: safe to re-run (it `update`s the env in place).

set -euo pipefail

ENV_NAME="qmmmkit"
SKIP_VALIDATE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)        ENV_NAME="$2"; shift 2 ;;
        --skip-validate)  SKIP_VALIDATE=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# *//'
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
echo "[deploy] repo:        $repo_root"
echo "[deploy] env name:    $ENV_NAME"

# ---------------------------------------------------------------------------
# 1. Pick a conda flavour: prefer mamba > micromamba > conda.
# ---------------------------------------------------------------------------
if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
elif command -v micromamba >/dev/null 2>&1; then
    CONDA_BIN="micromamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="conda"
else
    cat >&2 <<'EOF'
[deploy] No conda/mamba/micromamba on PATH.
        Install one first, e.g.:
          curl -L micro.mamba.pm/install.sh | bash
        Re-source your shell, then re-run this script.
EOF
    exit 1
fi
echo "[deploy] using $($CONDA_BIN --version 2>&1 | head -n1)  ($(command -v "$CONDA_BIN"))"

# ---------------------------------------------------------------------------
# 2. Create / update the env.
# ---------------------------------------------------------------------------
ENV_FILE="$repo_root/environment.yml"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[deploy] missing $ENV_FILE — are you in the repo root?" >&2
    exit 1
fi

env_exists() {
    if [[ "$CONDA_BIN" == "micromamba" ]]; then
        micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"
    else
        $CONDA_BIN env list | awk '{print $1}' | grep -qx "$ENV_NAME"
    fi
}

if env_exists; then
    echo "[deploy] env '$ENV_NAME' exists — updating from environment.yml"
    $CONDA_BIN env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "[deploy] creating env '$ENV_NAME' from environment.yml"
    $CONDA_BIN env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# 3. Activate and `pip install -e .` so qmmmkit itself is importable.
# ---------------------------------------------------------------------------
echo "[deploy] activating env and installing qmmmkit (editable)"
# shellcheck disable=SC1091
if [[ "$CONDA_BIN" == "micromamba" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "$ENV_NAME"
else
    # conda's shell hook is the safest cross-distribution activation.
    eval "$($CONDA_BIN shell.bash hook)"
    $CONDA_BIN activate "$ENV_NAME"
fi

python -m pip install --upgrade pip
python -m pip install -e .

# Sanity import
python - <<'PY'
import qmmmkit
print(f"[deploy] qmmmkit {qmmmkit.__version__} installed at {qmmmkit.__file__}")
PY

# ---------------------------------------------------------------------------
# 4. Run the live-engine validator unless skipped.
# ---------------------------------------------------------------------------
if [[ "$SKIP_VALIDATE" -eq 1 ]]; then
    echo "[deploy] --skip-validate set; not running validator."
else
    echo "[deploy] running scripts/validate_ash_pyscf.py (this may take a couple of minutes)"
    if ! python "$repo_root/scripts/validate_ash_pyscf.py"; then
        echo "[deploy] validator reported failures. The env is installed; iterate from there." >&2
        exit 1
    fi
fi

cat <<EOF

[deploy] DONE.

  Activate later with:        $CONDA_BIN activate $ENV_NAME
  CLI is on PATH:              qmmmkit --help
  Run a manifest headlessly:   qmmmkit run path/to/manifest.yaml
  Submit to SLURM (after configuring ~/.config/qmmmkit/clusters.yaml):
                               qmmmkit submit manifest.yaml --cluster <name>

To wire up your workstation GUI to dispatch into THIS install, add a
cluster profile pointing at \$(hostname) with workdir on a shared filesystem,
and the worker will SSH back here and submit via sbatch.
EOF
