#!/usr/bin/env bash
set -euo pipefail

# Cleanup utility for src-model artifacts
# Safe by default: removes logs, results, __pycache__, and checkpoint_* files.
# Use flags to widen scope.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
RESULTS_DIR="$ROOT_DIR/results"
MODEL_DIR="$ROOT_DIR/models"
DATA_DIR="$ROOT_DIR/data"

DO_LOGS=1
DO_RESULTS=1
DO_PYCACHE=1
DO_CHECKPOINTS=1
DO_DATA=0
DO_ALL_MODELS=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --no-logs           Do not delete logs/*
  --no-results        Do not delete results/*
  --no-pycache        Do not delete __pycache__ folders
  --no-checkpoints    Do not delete models/checkpoint_*.pth
  --data              Also delete data/*.npz (datasets)
  --all-models        Also delete all models/*.pth and models/*.npz (keeps nothing)
  --dry-run           Show what would be deleted without deleting
  -h, --help          Show this help

Default behavior: remove logs, results, __pycache__, and checkpoint_*.pth only.
Best and final models are kept unless --all-models is used.
EOF
}

confirm() {
  read -r -p "$1 [y/N] " resp || true
  case "$resp" in
    [yY][eE][sS]|[yY]) return 0 ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-logs) DO_LOGS=0 ;;
    --no-results) DO_RESULTS=0 ;;
    --no-pycache) DO_PYCACHE=0 ;;
    --no-checkpoints) DO_CHECKPOINTS=0 ;;
    --data) DO_DATA=1 ;;
    --all-models) DO_ALL_MODELS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

delete_glob() {
  local pattern="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY-RUN rm -f $pattern"
  else
    rm -f $pattern 2>/dev/null || true
  fi
}

delete_dir_contents() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "DRY-RUN rm -rf ${dir}/*"
    else
      rm -rf "${dir}"/* 2>/dev/null || true
    fi
  fi
}

echo "Cleaning workspace at: $ROOT_DIR"

if [[ $DO_LOGS -eq 1 ]]; then
  echo "- Cleaning logs: $LOG_DIR"
  delete_dir_contents "$LOG_DIR"
fi

if [[ $DO_RESULTS -eq 1 ]]; then
  echo "- Cleaning results: $RESULTS_DIR"
  delete_dir_contents "$RESULTS_DIR"
fi

if [[ $DO_PYCACHE -eq 1 ]]; then
  echo "- Removing __pycache__ directories"
  if [[ $DRY_RUN -eq 1 ]]; then
    find "$ROOT_DIR" -type d -name __pycache__ -print
  else
    find "$ROOT_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
  fi
fi

if [[ $DO_CHECKPOINTS -eq 1 ]]; then
  echo "- Removing checkpoint files in models: checkpoint_*.pth"
  delete_glob "$MODEL_DIR/checkpoint_*.pth"
fi

if [[ $DO_ALL_MODELS -eq 1 ]]; then
  echo "! Will remove ALL model weights (*.pth, *.npz) from $MODEL_DIR"
  if confirm "Proceed deleting all model files?"; then
    delete_glob "$MODEL_DIR/*.pth"
    delete_glob "$MODEL_DIR/*.npz"
  else
    echo "  Skipped deleting all models."
  fi
fi

if [[ $DO_DATA -eq 1 ]]; then
  echo "! Will remove datasets (*.npz) from $DATA_DIR"
  if confirm "Proceed deleting datasets?"; then
    delete_glob "$DATA_DIR/*.npz"
  else
    echo "  Skipped deleting datasets."
  fi
fi

echo "Done."
