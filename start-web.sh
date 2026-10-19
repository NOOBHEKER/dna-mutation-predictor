#!/usr/bin/env bash
# ===========================================================
#  Portable Streamlit launcher (macOS / Linux)
#  Uses the same venv as predict.sh.
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
    echo "  [error] Python 3.10+ was not found."
    exit 1
fi

VENV="$ROOT/venv"
VPY="$VENV/bin/python"

if [ ! -x "$VPY" ]; then
    echo "  [setup] Creating virtual environment..."
    "$PY" -m venv "$VENV"
fi

if [ ! -f "$VENV/.dna-predict-web-installed" ]; then
    echo "  [setup] Installing web dependencies (first run)..."
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -r "$ROOT/requirements-web.txt"
    echo "ok" > "$VENV/.dna-predict-web-installed"
fi

echo
echo "  Starting dashboard at http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo
exec "$VPY" -m streamlit run "$ROOT/webapp/app.py" "$@"
