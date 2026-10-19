"""HGVSp parsing is dependency-free and shared by the pipeline and CLI."""
from __future__ import annotations

import pytest

from src.hgvsp import AA1_TO_3, AA_MAP, parse_hgvsp


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p.Arg175His", ("Arg", 175, "His")),
        ("Arg175His", ("Arg", 175, "His")),
        ("R175H", ("R", 175, "H")),
        ("p.Cys61Gly", ("Cys", 61, "Gly")),
        ("p.Gln301Ter", ("Gln", 301, "Ter")),
    ],
)
def test_parse_hgvsp(text, expected):
    assert parse_hgvsp(text) == expected


@pytest.mark.parametrize("text", [None, 42, "", "p.", "not-a-variant"])
def test_parse_hgvsp_rejects_non_substitutions(text):
    assert parse_hgvsp(text) == (None, None, None)


def test_amino_acid_tables_cover_canonical_residues():
    for one in "ACDEFGHIKLMNPQRSTVWY":
        assert one in AA1_TO_3
        assert AA_MAP[AA1_TO_3[one]] == one
