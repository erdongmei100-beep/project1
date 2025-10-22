#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "[ERROR] Virtual environment not found. Run setup_env.sh first."
  exit 1
fi
source "$ROOT/.venv/bin/activate"
python "$ROOT/run.py" --source "data/videos/ambulance.mp4" --roi "data/rois/ambulance.json" --save-video --clip --plate
