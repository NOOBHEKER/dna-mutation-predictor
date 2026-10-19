"""Fill data/vep_cache.json for every mapped ClinVar missense variant.

Reads data/clinvar_map.json, scores all coords via src.vep.ensure_cached
(batched, resumable -- rerun anytime to fill gaps).

Usage:
    venv\\Scripts\\python scripts/fill_vep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vep import _load_json, ensure_cached  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    cmap = _load_json(ROOT / "data" / "clinvar_map.json")
    coords = [(m["chrom"], m["pos"], m["ref"], m["alt"])
              for m in cmap.values()]
    print(f"{len(coords)} coords to fill", flush=True)
    out = ensure_cached(coords)
    print(f"DONE: {len(out)}/{len(coords)} cached", flush=True)


if __name__ == "__main__":
    main()
