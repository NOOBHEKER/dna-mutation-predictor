"""Fill data/myv_cache.json for every mapped variant with an rsID.

Usage:
    venv\\Scripts\\python scripts/fill_myv.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.myvariant import _load_json, ensure_cached  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    cmap = _load_json(ROOT / "data" / "clinvar_map.json")
    pairs = [(m["rs"], m["alt"]) for m in cmap.values() if m.get("rs")]
    print(f"{len(pairs)} rsID variants to fill", flush=True)
    out = ensure_cached(pairs)
    print(f"DONE: {len(out)}/{len(pairs)} cached", flush=True)


if __name__ == "__main__":
    main()
