#!/usr/bin/env bash
# ===========================================================
#  dna-predict - portable launcher (macOS / Linux)
#  Works no matter where the repo is cloned. Creates a venv +
#  installs deps on first run, then launches the bot.
# ===========================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- locate a Python 3 interpreter ---
PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
        PY="$c"
        break
    fi
done
if [ -z "$PY" ]; then
    echo
    echo "  [error] Python 3.9+ is required but was not found."
    echo "          Install it, then re-run predict.sh"
    echo
    exit 1
fi

VENV="$ROOT/venv"
VPY="$VENV/bin/python"

# --- create venv if missing ---
if [ ! -x "$VPY" ]; then
    echo
    echo "  [setup] Creating virtual environment..."
    "$PY" -m venv "$VENV"
fi

# --- auto-install dependencies on first run ---
if [ ! -f "$VENV/marker.lib.txt" ]; then
    echo
    echo "  [setup] Installing dependencies (first run - this may take a while)..."
    "$VPY" -m pip install --upgrade pip >/dev/null 2>&1 || true
    "$VPY" -m pip install -r "$ROOT/requirements.txt"
    echo "ok" > "$VENV/marker.lib.txt"
fi

# --- warn if model/data files are missing ---
missing=""
[ -f "$ROOT/data/scaler.pkl" ]  || missing="$missing  ! scaler.pkl (trained model)"
[ -f "$ROOT/data/variants.csv" ] || missing="$missing  ! variants.csv (variant index)"
if [ -n "$missing" ]; then
    echo
    echo "  [warn] Some data files are missing:$missing"
    echo "         Suggestions and gene info will be limited."
    echo "         See README.md for how to obtain the model."
    echo
fi

echo
echo "  Starting dna-predict CLI Bot..."
echo
exec "$VPY" -m src.cli