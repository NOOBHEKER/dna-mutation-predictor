"""The config agent parses loose input and rejects out-of-scope variants."""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from src.config_agent import analyze  # noqa: E402


def test_parse_three_letter_substitution():
    state = analyze("TP53 p.Arg175His")
    assert state.complete
    assert state.gene == "TP53"
    assert state.hgvsp == "p.Arg175His"
    assert state.card()["wt"] == "R"
    assert state.card()["mt"] == "H"


def test_parse_one_letter_substitution():
    state = analyze("TP53 R175H")
    assert state.complete
    assert state.gene == "TP53"
    assert state.hgvsp == "p.Arg175His"


def test_free_text_is_tolerated():
    state = analyze("what about BRCA1 K50Q?")
    assert state.gene == "BRCA1"
    assert state.complete


def test_nonsense_variant_is_flagged():
    state = analyze("TP53 p.Gln301Ter")
    assert not state.complete
    assert any(code == "unsupported_type" for code, _ in state.issues)
