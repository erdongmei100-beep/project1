#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Removing existing virtual environment: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip || echo "[WARNING] Failed to upgrade pip."

if [ -n "${PIP_INDEX_URL:-}" ]; then
  echo "Using existing PIP_INDEX_URL=$PIP_INDEX_URL"
  python -m pip install -r "$ROOT/requirements.txt"
else
  python -m pip install -r "$ROOT/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo
echo "Virtual environment ready. Run:"
echo "    source .venv/bin/activate"
