"""Build gene|hgvsp -> genomic coordinates map from ClinVar variant_summary.

Reads data/variant_summary.txt.gz (GRCh38, single nucleotide variants)
and writes data/clinvar_map.json:
    {"TP53|R175H": {"rs": "28934578", "chrom": "17", "pos": 7673803,
                     "ref": "G", "alt": "A", "vid": "12345"}, ...}

Keys are normalized to 1-letter missense (same as src/preprocess AA_MAP).
Only exact single-ALT SNVs. Resumable/reporting via _map_stats.json.

Usage:
    venv\\Scripts\\python scripts/map_clinvar.py
"""
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import AA_MAP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "variant_summary.txt.gz")
VARIANTS = os.path.join(ROOT, "data", "variants.csv")
OUT = os.path.join(ROOT, "data", "clinvar_map.json")
STATS = os.path.join(ROOT, "data", "_map_stats.json")

P_HGVS = re.compile(r"\(p\.([A-Za-z]+)(\d+)([A-Za-z]+)\)")

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def norm_key(gene, wt3, pos, mt3):
    wt = AA_MAP.get(wt3)
    mt = AA_MAP.get(mt3)
    if not wt or not mt or wt not in STD_AA or mt not in STD_AA:
        return None
    if wt == mt:
        return None
    return f"{gene.upper()}|{wt}{pos}{mt}"


def main():
    import pandas as pd
    df = pd.read_csv(VARIANTS, usecols=["gene_symbol", "hgvsp"])
    want = set()
    for _, r in df.iterrows():
        m = re.match(r"p\.([A-Za-z]+)(\d+)([A-Za-z]+)$", str(r["hgvsp"]))
        if m:
            k = norm_key(r["gene_symbol"], m.group(1), m.group(2), m.group(3))
            if k:
                want.add(k)
    print(f"variants.csv: {len(df)} rows -> {len(want)} missense keys wanted")

    found = {}
    n_rows = n_snv38 = 0
    with gzip.open(SUMMARY, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(header)}
        need = ["GeneSymbol", "Name", "Type", "Assembly", "RS# (dbSNP)",
                "Chromosome", "PositionVCF", "ReferenceAlleleVCF",
                "AlternateAlleleVCF", "VariationID"]
        for c in need:
            assert c in ci, f"missing col {c}"
        for line in f:
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  scanned {n_rows:,} rows, matched {len(found):,}")
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            if parts[ci["Type"]] != "single nucleotide variant":
                continue
            if parts[ci["Assembly"]] != "GRCh38":
                continue
            n_snv38 += 1
            gene = parts[ci["GeneSymbol"]]
            if not gene or gene == "-1":
                continue
            m = P_HGVS.search(parts[ci["Name"]])
            if not m:
                continue
            key = norm_key(gene, m.group(1), m.group(2), m.group(3))
            if key is None or key not in want or key in found:
                continue
            alt = parts[ci["AlternateAlleleVCF"]]
            if "," in alt or alt in ("", "-", "na"):
                continue
            ref = parts[ci["ReferenceAlleleVCF"]]
            try:
                pos = int(parts[ci["PositionVCF"]])
            except ValueError:
                continue
            rs = parts[ci["RS# (dbSNP)"]]
            found[key] = {
                "rs": None if rs in ("-1", "") else rs,
                "chrom": parts[ci["Chromosome"]],
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "vid": parts[ci["VariationID"]],
            }
            if len(found) == len(want):
                break

    with open(OUT, "w") as f:
        json.dump(found, f)
    stats = {"wanted": len(want), "matched": len(found),
             "scanned_rows": n_rows, "snv38_rows": n_snv38,
             "coverage": len(found) / max(len(want), 1)}
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"matched {len(found)}/{len(want)} = {stats['coverage']:.3f}")
    print(f"scanned {n_rows:,} rows ({n_snv38:,} SNV/GRCh38)")


if __name__ == "__main__":
    main()
