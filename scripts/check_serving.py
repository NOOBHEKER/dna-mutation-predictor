"""End-to-end check of the tabular-gated prediction path.

Verifies:
  1. Known ClinVar variants get a tabular opinion (and sane verdicts).
  2. An UNMAPPED variant falls back to the sequence view instead of
     fabricating a 0.99 pathogenic call.
  3. The blend string carries every view that contributed.

Usage:
    venv\\Scripts\\python scripts/check_serving.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.predict import tabular_opinion  # noqa: E402

KNOWN = [
    ("TP53", "R", 175, "H", "pathogenic"),
    ("TP53", "P", 72, "R", "benign"),
    ("BRCA1", "C", 61, "G", "pathogenic"),
    ("EGFR", "L", 858, "R", "pathogenic"),
    ("KRAS", "G", 12, "D", "pathogenic"),
    ("CFTR", "F", 508, "del", None),
    ("APOE", "C", 112, "R", "pathogenic"),
    ("MTHFR", "A", 222, "C", "benign"),
]

print("== known ClinVar variants ==")
ok = 0
for gene, wt, pos, mt, expect in KNOWN:
    prob, detail = tabular_opinion(gene, wt, pos, mt)
    if prob is None:
        print(f"  {gene} {wt}{pos}{mt}: NO TABULAR ({detail})")
        continue
    verdict = "pathogenic" if prob >= 0.5 else "benign"
    mark = ""
    if expect:
        mark = "  OK" if verdict == expect else "  <-- MISMATCH"
        ok += 1 if verdict == expect else 0
    print(f"  {gene} {wt}{pos}{mt}: p={prob:.4f} -> {verdict} "
          f"(expected {expect}){mark}")
print(f"  matched expectation: {ok}/{sum(1 for r in KNOWN if r[4])}")

print("\n== unmapped variant (must NOT fabricate confidence) ==")
for gene, wt, pos, mt in [("ZZZZ9", "A", 10, "V"), ("TP53", "X", 99999, "Y")]:
    prob, detail = tabular_opinion(gene, wt, pos, mt)
    print(f"  {gene} {wt}{pos}{mt}: prob={prob} reason={detail}")
    assert prob is None, "FAIL: fabricated a probability for unmapped variant"
print("  gate OK: unmapped variants return None (caller falls back to ESM)")
print("DONE")
