"""
hgvsp.py -- HGVSp parsing and amino-acid maps.

Deliberately dependency-free: importing this module never pulls in
torch, transformers or pandas, so parsers and tests stay fast. Shared by
the data pipeline (preprocess), the predictor and the CLI.
"""
from __future__ import annotations

import re

# Mapping of 3-letter to 1-letter amino acids
AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Sec': 'U', 'Pyl': 'O', 'Xaa': 'X', 'Xle': 'J',
    'Asx': 'B', 'Glx': 'Z',
}

AA1_TO_3 = {}
for _t3, _a1 in AA_MAP.items():
    AA1_TO_3.setdefault(_a1, _t3)

CANONICAL_1 = set("ACDEFGHIKLMNPQRSTVWY")

# Missense HGVSp matcher for 1-letter key normalization (p.Arg175His style)
P_HGVS_KEY = re.compile(r"p\.([A-Za-z]+)(\d+)([A-Za-z]+)$")


def parse_hgvsp(hgvsp):
    """Parse an HGVSp substitution into (wt_res_3, position, mt_res_3).

    Accepts ``p.Arg175His``, ``Arg175His`` and other loose spellings;
    returns ``(None, None, None)`` when the notation is not a
    substitution (or is not a string).
    """
    if not isinstance(hgvsp, str):
        return None, None, None
    try:
        change = hgvsp[2:] if hgvsp.startswith('p.') else hgvsp
        match = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+|Ter|=|fs|del|ins)",
                         change)
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        match = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", change)
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
    except Exception:
        pass
    return None, None, None
