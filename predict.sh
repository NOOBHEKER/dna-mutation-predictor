#!/usr/bin/env bash
# ===========================================================
#  Portable launcher (macOS / Linux)
#  Creates a local venv, installs dependencies on first run,
#  then starts the interactive CLI. Works from any location.
# ===========================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo
    echo "  [error] Python 3.10+ was not found."
    echo "          Install it from https://www.python.org/downloads/"
    echo
    exit 1
fi

VENV="$ROOT/venv"
VPY="$VENV/bin/python"

if [ ! -x "$VPY" ]; then
    echo
    echo "  [setup] Creating virtual environment..."
    "$PY" -m venv "$VENV"
fi

if [ ! -f "$VENV/.dna-predict-installed" ]; then
    echo
    echo "  [setup] Installing dependencies (first run, this can take a while)..."
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -r "$ROOT/requirements.txt"
    echo "ok" > "$VENV/.dna-predict-installed"
fi

missing=""
[ -f "$ROOT/data/scaler.pkl" ]   || missing="$missing  ! scaler.pkl (trained head)"
[ -f "$ROOT/data/variants.csv" ] || missing="$missing  ! variants.csv (variant index)"
[ -f "$ROOT/data/tabular.pkl" ]  || missing="$missing  ! tabular.pkl (population head)"
if [ -n "$missing" ]; then
    echo
    echo "  [warn] Missing data files:"
    echo "$missing"
    echo "         Predictions will be limited. See data/README.md."
    echo
fi

echo
echo "  Starting DNA Mutation Predictor..."
echo
exec "$VPY" -m src.cli "$@"
