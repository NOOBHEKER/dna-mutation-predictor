"""
config_agent.py — "Config Agent": the second AI of the dna-predict CLI.

A deterministic gap-analysis agent. Before every prediction it:
  1. Parses raw user input (loose formats accepted)
  2. Detects what is missing / inconsistent
  3. Interactively fills the gaps (with suggestions)
so the predictor always receives a complete, validated configuration card.

Accepted loose input formats (examples):
    TP53 p.Arg175His          -> gene + standard HGVSp
    TP53 Arg175His            -> gene + HGVSp without p. prefix
    TP53 R175H                -> gene + 1-letter mutation notation
    p.Cys61Gly                -> HGVSp only (gene asked later)
    C61G                      -> 1-letter mutation only
    "what about BRCA1 K50Q?"  -> free text, entities extracted
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from difflib import get_close_matches

import pandas as pd

from src.paths import data_path

VARIANTS_CSV = data_path("variants.csv")

# Mapping of 3-letter to 1-letter amino acids (kept local so importing this
# module never pulls in torch/transformers — that stays lazy in cli.py)
AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Sec': 'U', 'Pyl': 'O', 'Xaa': 'X', 'Xle': 'J',
    'Asx': 'B', 'Glx': 'Z',
}

AA3_TO_1 = dict(AA_MAP)
AA1_TO_3 = {}
for _t3, _a1 in AA_MAP.items():
    AA1_TO_3.setdefault(_a1, _t3)

CANONICAL_1 = set("ACDEFGHIKLMNPQRSTVWY")

# 3-letter notation: p.Arg175His / Arg175His / (Arg175His) / p.(Arg175His)
RE_VAR_3 = re.compile(
    r"([A-Z][a-z]{2})\s*\(?(\d+)\s*\)?([A-Z][a-z]{2}|Ter|=|fs|del|ins|\?)"
)
# 1-letter notation: R175H
RE_VAR_1 = re.compile(r"\b([A-Z])(\d+)\s*([A-Z])\b")

# verbs the user may prefix to free text
_LEAD_RE = re.compile(r"^(predict|run|analyze|check|do|bot|dna)\b[:\s]*", re.I)
# tokens that are noise when hunting for a gene symbol
_NOISE_TOKENS = {"p", "c", "g", "m", "fs", "ter", "del", "ins", "and", "the",
                 "what", "about", "of", "in", "for", "is", "a", "mutation",
                 "variant", "gene", "protein"}

_gene_lock = threading.Lock()
_gene_index = None  # (known_genes:set, variants_by_gene:dict)


# --------------------------------------------------------------------------
# Variant data index (loaded once from data/variants.csv)
# --------------------------------------------------------------------------
def gene_index():
    """Return (known_genes, variants_by_gene). variants_by_gene maps
    GENE -> list[(hgvsp, label)] containing only well-formed missense rows."""
    global _gene_index
    with _gene_lock:
        if _gene_index is None:
            try:
                df = pd.read_csv(VARIANTS_CSV)
            except Exception:
                _gene_index = (set(), {})
                return _gene_index
            known = set(df["gene_symbol"].dropna().str.upper())
            by_gene = {}
            for gene, hgvsp, label in df.itertuples(index=False):
                if not isinstance(hgvsp, str) or not isinstance(gene, str):
                    continue
                wt3, pos, mt3 = _parse_hgvsp_parts(hgvsp)
                if (wt3 and wt3 in AA3_TO_1 and mt3 and mt3 in AA3_TO_1
                        and pos):
                    by_gene.setdefault(gene.upper(), []).append(
                        (hgvsp, int(label)))
            _gene_index = (known, by_gene)
        return _gene_index


def known_genes():
    return gene_index()[0]


def variants_for_gene(gene: str):
    """Deduplicated known missense variants for a gene, pathogenic first."""
    _, by_gene = gene_index()
    seen, out = set(), []
    for hgvsp, label in by_gene.get(gene.upper(), []):
        if hgvsp not in seen:
            seen.add(hgvsp)
            out.append((hgvsp, label))
    out.sort(key=lambda t: -t[1])
    return out


def suggest_genes(text: str, n: int = 5):
    """Fuzzy 'did you mean' gene suggestions."""
    known = known_genes()
    if not known:
        return []
    up = text.upper()
    if up in known:
        return [up]
    return get_close_matches(up, known, n=n, cutoff=0.6)


def suggest_variants(gene: str, n: int = 8):
    return variants_for_gene(gene)[:n]


def nearby_variants(gene: str, pos: int, radius: int = 10, n: int = 5):
    """Known variants within +/- radius positions of `pos` in `gene`."""
    out = []
    for hgvsp, label in variants_for_gene(gene):
        parts = _parse_hgvsp_parts(hgvsp)
        if parts[1] and abs(parts[1] - pos) <= radius:
            out.append((hgvsp, label))
    out.sort(key=lambda t: -t[1])
    return out[:n]


# --------------------------------------------------------------------------
# HGVSp parsing helpers
# --------------------------------------------------------------------------
def _parse_hgvsp_parts(hgvsp: str):
    """Return (wt3, pos, mt3). Local re-implementation of preprocess.parse_hgvsp
    so this module stays free of heavy ML imports."""
    import re as _re
    try:
        change = hgvsp[2:] if hgvsp.startswith('p.') else hgvsp
        m = _re.match(
            r"([A-Za-z]+)([0-9]+)([A-Za-z]+|Ter|=|fs|del|ins)", change)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
    except Exception:
        pass
    return None, None, None


def _canonical_hgvsp(wt_1: str, pos: int, mt_1: str) -> str:
    return f"p.{AA1_TO_3.get(wt_1, wt_1)}{pos}{AA1_TO_3.get(mt_1, mt_1)}"


# --------------------------------------------------------------------------
# Config state
# --------------------------------------------------------------------------
@dataclass
class ConfigState:
    """Everything the predictor needs, plus what the agent figured out."""
    raw: str = ""
    gene: str | None = None
    hgvsp: str | None = None
    wt_1: str | None = None
    pos: int | None = None
    mt_1: str | None = None
    issues: list = field(default_factory=list)       # (code, message)
    suggestions: list = field(default_factory=list)  # (code, message)
    source: str = "parsed"                           # parsed|picked|manual

    @property
    def complete(self) -> bool:
        return bool(self.gene and self.hgvsp and self.wt_1
                    and self.pos and self.mt_1)

    def card(self) -> dict:
        return {
            "gene": self.gene,
            "hgvsp": self.hgvsp,
            "wt": self.wt_1,
            "pos": self.pos,
            "mt": self.mt_1,
            "source": self.source,
        }

    def copy(self) -> "ConfigState":
        return ConfigState(
            raw=self.raw, gene=self.gene, hgvsp=self.hgvsp, wt_1=self.wt_1,
            pos=self.pos, mt_1=self.mt_1, issues=list(self.issues),
            suggestions=list(self.suggestions), source=self.source)


def _set_variant(st: ConfigState, wt_1: str, pos: int, mt_1: str):
    st.wt_1, st.pos, st.mt_1 = wt_1, pos, mt_1
    st.hgvsp = _canonical_hgvsp(wt_1, pos, mt_1)


def _unsupported_note(mt3: str) -> str | None:
    if mt3 in ("Ter", "*", "\u2217"):
        return "nonsense (Ter) variant"
    if mt3 == "fs":
        return "frameshift variant"
    if mt3 in ("del", "ins"):
        return f"indel ({mt3}) variant"
    if mt3 == "=":
        return "synonymous (silent) variant"
    if mt3 == "?":
        return "unspecified variant"
    return None


def analyze(raw: str) -> ConfigState:
    """Parse raw free text into a ConfigState, flagging gaps and suggestions."""
    st = ConfigState(raw=raw)
    text = raw.strip()
    text = _LEAD_RE.sub("", text).strip()

    var_span = None

    # --- try 3-letter mutation notation first ---
    for m in RE_VAR_3.finditer(text):
        wt3, pos, mt3 = m.groups()
        if wt3 in AA3_TO_1:
            note = _unsupported_note(mt3)
            if mt3 in AA3_TO_1:
                _set_variant(st, AA3_TO_1[wt3], int(pos), AA3_TO_1[mt3])
            else:
                st.wt_1, st.pos = AA3_TO_1[wt3], int(pos)
                st.issues.append(
                    ("unsupported_type",
                     f"This looks like a {note} — the model only handles "
                     f"missense (amino-acid swap) variants."))
            var_span = m.span()
            break

    # --- else try 1-letter notation (R175H) ---
    if var_span is None:
        for m in RE_VAR_1.finditer(text):
            wt1, pos, mt1 = m.groups()
            if wt1 in CANONICAL_1 and mt1 in CANONICAL_1:
                _set_variant(st, wt1, int(pos), mt1)
                var_span = m.span()
                break

    # --- hunt for the gene symbol in the remaining text ---
    remainder = text
    if var_span:
        remainder = text[:var_span[0]] + " " + text[var_span[1]:]

    known = known_genes()
    found_genes = []
    fuzzy_hits = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", remainder):
        low = tok.lower()
        if low in _NOISE_TOKENS:
            continue
        up = tok.upper()
        if up in known:
            found_genes.append(up)
        elif len(up) >= 3:
            # tokens with digits (e.g. 'TP35' -> 'TP53') get a looser cutoff
            cutoff = 0.6 if any(ch.isdigit() for ch in tok) else 0.78
            hits = get_close_matches(up, known, n=5, cutoff=cutoff)
            if hits:
                # prefer the biologically most-studied gene (most ClinVar
                # variants on record) among equally-close candidates
                _, by_gene = gene_index()
                best = max(hits, key=lambda g: len(by_gene.get(g, ())))
                fuzzy_hits.append((up, best))

    if found_genes:
        st.gene = found_genes[0]
        if len(found_genes) > 1:
            st.issues.append(
                ("ambiguous_gene",
                 f"Multiple genes mentioned: "
                 f"{', '.join(sorted(set(found_genes)))} — using {st.gene}."))
    elif fuzzy_hits:
        st.suggestions.append(
            ("unknown_gene",
             f"'{fuzzy_hits[0][0]}' is not a known gene symbol — "
             f"did you mean {fuzzy_hits[0][1]}?"))
    elif not st.gene:
        if st.hgvsp:
            st.issues.append(("missing_gene", "No gene symbol found in input."))
        else:
            st.issues.append(
                ("missing_both",
                 "I couldn't find a gene symbol or a variant notation."))

    # --- consistency checks ---
    if st.gene and st.hgvsp:
        if st.mt_1 is not None and st.wt_1 == st.mt_1:
            st.issues.append(
                ("synonymous",
                 f"{st.wt_1}{st.pos}{st.mt_1} is a synonymous change — "
                 f"prediction would be meaningless."))
        st.suggestions.append(
            ("related_variants",
             f"{len(variants_for_gene(st.gene))} known variants on record "
             f"for {st.gene} — try 'all {st.gene}' to batch-predict them."))

    return st


# --------------------------------------------------------------------------
# Interactive gap filling
# --------------------------------------------------------------------------
def interactive_fix(st: ConfigState, console, max_rounds: int = 4) -> ConfigState:
    """Interactively ask the user for whatever is missing in `st`.

    `console` is a rich Console. Returns the (possibly completed) state.
    """
    from rich import box
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    for _ in range(max_rounds):
        if st.complete:
            return st

        # ---- missing gene ----
        if not st.gene:
            sug = suggest_genes(st.raw, n=3) if st.raw else []
            if sug:
                console.print(
                    f"  [cyan]\u2755[/cyan] Which gene? "
                    f"[dim]closest matches:[/dim] [bold]{', '.join(sug)}[/bold]")
                guess = Prompt.ask("      Gene symbol", default=sug[0],
                                   console=console)
            else:
                guess = Prompt.ask(
                    "  [cyan]\u2755[/cyan] Gene symbol (e.g. TP53)",
                    console=console)
            guess = (guess or "").strip().upper()
            if not guess:
                continue
            if guess in known_genes():
                st.gene, st.source = guess, "manual"
            elif suggest_genes(guess, 1):
                top = suggest_genes(guess, 1)[0]
                if Confirm.ask(
                        f"      [yellow]'{guess}' is unknown[/yellow] — did "
                        f"you mean [bold]{top}[/bold]?",
                        default=True, console=console):
                    st.gene, st.source = top, "manual"
                else:
                    st.gene, st.source = guess, "manual"
            else:
                st.gene, st.source = guess, "manual"
            continue

        # ---- missing variant ----
        if not st.hgvsp:
            variants = suggest_variants(st.gene, n=10)
            if variants:
                tbl = Table(box=box.SIMPLE_HEAD, header_style="bold cyan",
                            pad_edge=False, show_edge=False)
                tbl.add_column("#", justify="right", style="dim", width=3)
                tbl.add_column("Known variant", style="white")
                tbl.add_column("ClinVar", justify="center")
                for i, (hg, lab) in enumerate(variants, 1):
                    tbl.add_row(str(i), hg,
                                "[red]Pathogenic[/red]" if lab
                                else "[green]Benign[/green]")
                console.print(tbl)
                ans = Prompt.ask(
                    "  [cyan]\u2755[/cyan] Pick a # above, or type any HGVSp "
                    "(e.g. p.Arg175His / R175H)",
                    console=console).strip()
                if ans.isdigit() and 1 <= int(ans) <= len(variants):
                    hg = variants[int(ans) - 1][0]
                    wt3, pos, mt3 = _parse_hgvsp_parts(hg)
                    if wt3 and pos and mt3 and mt3 in AA3_TO_1:
                        _set_variant(st, AA3_TO_1[wt3], pos, AA3_TO_1[mt3])
                        st.source = "picked"
                    continue
            else:
                ans = Prompt.ask(
                    f"  [cyan]\u2755[/cyan] No local records for {st.gene} — "
                    f"enter an HGVSp (e.g. p.Arg175His / R175H)",
                    console=console).strip()
                if not ans:
                    continue
            parsed = analyze(f"{st.gene} {ans}")
            if parsed.hgvsp:
                st.hgvsp, st.wt_1, st.pos, st.mt_1 = (
                    parsed.hgvsp, parsed.wt_1, parsed.pos, parsed.mt_1)
            else:
                console.print(
                    "      [red]Could not parse that variant — try "
                    "p.Arg175His or R175H.[/red]")
            continue

        # ---- variant present but unsupported type ----
        if any(code == "unsupported_type" for code, _ in st.issues):
            mt = Prompt.ask(
                "  [cyan]\u2755[/cyan] Substitute amino acid (1-letter, "
                "e.g. H)", console=console).strip().upper()
            if len(mt) == 1 and mt in CANONICAL_1:
                _set_variant(st, st.wt_1, st.pos, mt)
                st.issues = [i for i in st.issues
                             if i[0] != "unsupported_type"]
            elif mt:
                console.print("      [red]Not a valid amino acid letter.[/red]")
            continue

        break  # nothing actionable

    return st
