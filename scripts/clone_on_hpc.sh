#!/usr/bin/env bash
# Bootstrap qmmmkit on an HPC by cloning from GitHub. Run this on the HPC.
#
# First-time install (no source on the HPC yet):
#
#   bash <(curl -sSL https://raw.githubusercontent.com/<owner>/qmmmkit/main/scripts/clone_on_hpc.sh) \
#        --repo <owner>/qmmmkit
#
# OR, after manually cloning once:
#
#   git clone https://github.com/<owner>/qmmmkit.git ~/qmmmkit
#   bash ~/qmmmkit/scripts/clone_on_hpc.sh
#
# Subsequent updates (skips clone, runs `git pull`):
#
#   bash ~/qmmmkit/scripts/clone_on_hpc.sh --update
#
# Options:
#   --repo OWNER/NAME        GitHub slug, e.g. fddxwll/qmmmkit. Required on first install.
#   --dir PATH               Install directory (default: $HOME/qmmmkit).
#   --branch NAME            Git branch / tag to check out (default: main).
#   --ssh                    Use git@github.com:OWNER/NAME.git instead of HTTPS (needs SSH keys).
#   --update                 Skip the clone step; just pull + redeploy.
#   --skip-deploy            Stop after the clone/pull; don't run deploy_hpc.sh.
#   --skip-validate          Pass through to deploy_hpc.sh.
#   --env-name NAME          Conda env name (default: qmmmkit).
#
# Authentication for private repos:
#   - HTTPS + Personal Access Token: clone the first time manually with the
#     token in the URL (or use `git config credential.helper`), then this
#     script can update via plain `git pull`.
#   - SSH: add the HPC's pubkey as a deploy key on the repo, pass --ssh.

set -euo pipefail

REPO=""
INSTALL_DIR="${QMMMKIT_DIR:-$HOME/qmmmkit}"
BRANCH="main"
USE_SSH=0
UPDATE_ONLY=0
SKIP_DEPLOY=0
SKIP_VALIDATE=0
ENV_NAME="qmmmkit"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--repo)        REPO="$2"; shift 2 ;;
        -d|--dir)         INSTALL_DIR="$2"; shift 2 ;;
        -b|--branch)      BRANCH="$2"; shift 2 ;;
        --ssh)            USE_SSH=1; shift ;;
        --update)         UPDATE_ONLY=1; shift ;;
        --skip-deploy)    SKIP_DEPLOY=1; shift ;;
        --skip-validate)  SKIP_VALIDATE=1; shift ;;
        -n|--env-name)    ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# *//'
            exit 0 ;;
        *) echo "[bootstrap] unknown arg: $1"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Clone or update.
# ---------------------------------------------------------------------------
command -v git >/dev/null 2>&1 || {
    echo "[bootstrap] git is required but not on PATH." >&2
    exit 1
}

if [[ "$UPDATE_ONLY" -eq 1 ]]; then
    if [[ ! -d "$INSTALL_DIR/.git" ]]; then
        echo "[bootstrap] --update set but $INSTALL_DIR is not a git checkout." >&2
        exit 1
    fi
    cd "$INSTALL_DIR"
    echo "[bootstrap] pulling latest in $INSTALL_DIR (branch: $(git rev-parse --abbrev-ref HEAD))"
    git fetch --all --prune
    git pull --rebase --autostash
else
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        echo "[bootstrap] $INSTALL_DIR is already a git checkout."
        echo "            Use --update to pull, or remove the directory to re-clone."
        cd "$INSTALL_DIR"
    else
        if [[ -z "$REPO" ]]; then
            echo "[bootstrap] --repo OWNER/NAME is required for first install." >&2
            echo "            Example: --repo fddxwll/qmmmkit" >&2
            exit 2
        fi
        if [[ "$USE_SSH" -eq 1 ]]; then
            url="git@github.com:${REPO}.git"
        else
            url="https://github.com/${REPO}.git"
        fi
        mkdir -p "$(dirname "$INSTALL_DIR")"
        echo "[bootstrap] cloning $url -> $INSTALL_DIR (branch: $BRANCH)"
        git clone --branch "$BRANCH" --single-branch "$url" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
fi

echo "[bootstrap] at commit: $(git rev-parse --short HEAD)  ($(git log -1 --pretty=%s))"

# ---------------------------------------------------------------------------
# 2. Run the deploy script (env + pip install + validate).
# ---------------------------------------------------------------------------
if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
    echo "[bootstrap] --skip-deploy set; not running deploy_hpc.sh."
    exit 0
fi

DEPLOY="$INSTALL_DIR/scripts/deploy_hpc.sh"
[[ -x "$DEPLOY" ]] || chmod +x "$DEPLOY" 2>/dev/null || true
[[ -f "$DEPLOY" ]] || { echo "[bootstrap] missing $DEPLOY" >&2; exit 1; }

DEPLOY_ARGS=(--name "$ENV_NAME")
[[ "$SKIP_VALIDATE" -eq 1 ]] && DEPLOY_ARGS+=(--skip-validate)

bash "$DEPLOY" "${DEPLOY_ARGS[@]}"
