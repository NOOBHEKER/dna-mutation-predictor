# Copyright (c) 2026 Agastyas Shii. MIT License.
"""
protein_graph.py — interactive protein visualization for the dna-predict CLI.

Renders an interactive matplotlib window:
  * residue map (position vs. property class), mutation highlighted
  * Kyte-Doolittle hydropathy profile with a sliding window
  * WT vs mutant sequence diff panel (generated mutant protein)
  * click any residue for a live info tooltip
  * optional overlay of batch-prediction results colored by pathogenicity
"""
from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AA_CLASS = {
    "A": "Hydrophobic", "V": "Hydrophobic", "I": "Hydrophobic",
    "L": "Hydrophobic", "M": "Hydrophobic", "F": "Hydrophobic",
    "W": "Hydrophobic",
    "S": "Polar", "T": "Polar", "N": "Polar", "Q": "Polar", "Y": "Polar",
    "K": "Basic (+)", "R": "Basic (+)", "H": "Basic (+)",
    "D": "Acidic (-)", "E": "Acidic (-)",
    "P": "Special", "G": "Special", "C": "Special",
}

CLASS_ORDER = ["Hydrophobic", "Polar", "Basic (+)", "Acidic (-)", "Special",
               "Other"]
CLASS_COLORS = {
    "Hydrophobic": "#f4a261",
    "Polar": "#2a9d8f",
    "Basic (+)": "#e76f51",
    "Acidic (-)": "#457b9d",
    "Special": "#9b5de5",
    "Other": "#8d99ae",
}

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}


def classify(aa: str) -> str:
    return AA_CLASS.get(aa.upper(), "Other")


def hydropathy(aa: str) -> float:
    return KYTE_DOOLITTLE.get(aa.upper(), 0.0)


def generate_mutant(seq: str, pos: int, mt: str) -> str:
    """Return the full mutant protein sequence (1-based pos)."""
    if not (1 <= pos <= len(seq)):
        return seq
    lst = list(seq)
    lst[pos - 1] = mt.upper()
    return "".join(lst)



def show_protein_graph(gene, seq, wt_1, pos, mt_1, results=None,
                       verdict=None, block=True):
    """Open the interactive protein graph window (blocks until closed)."""
    results = results or []
    L = len(seq)

    # position -> (prob, label) from batch results
    pred_by_pos = {}
    for r in results:
        wt3, p, mt3 = _quick_parse(r.get("hgvsp", ""))
        if p:
            pred_by_pos[p] = (r.get("prob", 0.0), r.get("label", 0))

    fig = plt.figure(figsize=(13, 9), facecolor="#14141e")
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.4, 1.15], hspace=0.35)
    ax_map = fig.add_subplot(gs[0])
    ax_hyd = fig.add_subplot(gs[1], sharex=ax_map)
    ax_seq = fig.add_subplot(gs[2])
    for ax in (ax_map, ax_hyd, ax_seq):
        ax.set_facecolor("#14141e")
        ax.tick_params(colors="#c8c8d8", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#3a3a55")

    # ---- Panel 1: residue map ------------------------------------------
    rng = np.random.RandomState(7)
    xs = np.arange(1, L + 1)
    cls_idx = np.array([CLASS_ORDER.index(classify(a)) for a in seq])
    ys = cls_idx + rng.uniform(-0.22, 0.22, size=L)

    ax_map.scatter(xs, ys, s=5, alpha=0.45, c="0.75", marker="o", zorder=2)

    if pred_by_pos:
        pxs = np.array(sorted(pred_by_pos))
        pys = cls_idx[pxs - 1] + rng.uniform(-0.22, 0.22, size=len(pxs))
        pprobs = np.array([pred_by_pos[p][0] for p in pxs])
        cmap = plt.get_cmap("RdYlGn_r")
        sc = ax_map.scatter(pxs, pys, s=34, c=pprobs, cmap=cmap,
                            vmin=0, vmax=1, edgecolors="black",
                            linewidths=0.4, marker="o", zorder=4)
        cbar = fig.colorbar(sc, ax=ax_map, pad=0.01, aspect=28)
        cbar.set_label("predicted pathogenicity", color="#c8c8d8", fontsize=8)
        cbar.ax.tick_params(colors="#c8c8d8", labelsize=7)
        cbar.outline.set_edgecolor("#3a3a55")

    ax_map.axvline(pos, color="#ff4d6d", lw=1.4, alpha=0.9, zorder=5)
    ax_map.scatter([pos], [CLASS_ORDER.index(classify(wt_1))], s=420,
                   marker="*", c="#ff4d6d", edgecolors="white",
                   linewidths=0.8, zorder=6, label=f"mutation {wt_1}{pos}{mt_1}")
    ax_map.set_yticks(range(len(CLASS_ORDER)))
    ax_map.set_yticklabels(CLASS_ORDER, fontsize=8)
    ax_map.set_ylim(-0.7, len(CLASS_ORDER) - 0.3)
    ax_map.set_ylabel("residue class", color="#c8c8d8", fontsize=9)
    ax_map.set_xlim(1, max(L, pos + 1))
    legend = ax_map.legend(loc="upper right", fontsize=8, framealpha=0.25)
    for t in legend.get_texts():
        t.set_color("#c8c8d8")
    ax_map.set_title(
        f"{gene}  \u00b7  {wt_1}{pos}{mt_1}"
        + (f"  \u00b7  {verdict}" if verdict else ""),
        color="#ffffff", fontsize=12, fontweight="bold", loc="left")

    tooltip = ax_map.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1e1e2e", ec="#4d4d7a",
                  alpha=0.95), color="#ffffff", fontsize=8.5, zorder=10,
        visible=False)

    # ---- Panel 2: Kyte-Doolittle hydropathy profile --------------------
    hyd = np.array([hydropathy(a) for a in seq])
    win = max(3, min(9, max(L // 40, 3) | 1))  # odd sliding window
    smooth = (np.convolve(hyd, np.ones(win) / win, mode="same")
              if L >= win else hyd)
    ax_hyd.plot(xs, smooth, color="#4cc9f0", lw=1.1, zorder=3)
    ax_hyd.axhline(0, color="#3a3a55", lw=0.7, zorder=1)
    ax_hyd.axvline(pos, color="#ff4d6d", lw=1.2, alpha=0.9, zorder=4)
    ax_hyd.scatter([pos], [smooth[min(pos - 1, L - 1)]], s=70, marker="o",
                   facecolors="none", edgecolors="#ff4d6d", linewidths=1.4,
                   zorder=5)
    ax_hyd.set_ylabel("hydropathy", color="#c8c8d8", fontsize=9)
    ax_hyd.set_xlabel("residue position", color="#c8c8d8", fontsize=9)

    # ---- Panel 3: WT vs mutant sequence diff (generated protein) -------
    ax_seq.axis("off")
    half = 17
    lo = max(0, pos - 1 - half)
    hi = min(L, pos + half)
    wt_win = seq[lo:hi]
    mt_seq = generate_mutant(seq, pos, mt_1)
    mt_win = mt_seq[lo:hi]

    ax_seq.text(0.01, 0.86, "WT :", color="#8d99ae", fontsize=8,
                family="monospace", transform=ax_seq.transAxes)
    ax_seq.text(0.01, 0.58, "MUT:", color="#8d99ae", fontsize=8,
                family="monospace", transform=ax_seq.transAxes)
    ax_seq.text(0.065, 0.86, wt_win, color="#e0e0ff", fontsize=9,
                family="monospace", transform=ax_seq.transAxes)
    for i, (a, b) in enumerate(zip(wt_win, mt_win)):
        color = "#ff4d6d" if a != b else "#7ee081"
        ax_seq.text(0.065 + i * 0.0115, 0.58, b, color=color,
                    fontweight="bold" if a != b else "normal",
                    fontsize=9, family="monospace",
                    transform=ax_seq.transAxes)
    ruler = "".join("|" if (lo + i + 1) % 10 == 0 else " "
                    for i in range(hi - lo))
    ax_seq.text(0.065, 0.30, ruler, color="#555577", fontsize=7,
                family="monospace", transform=ax_seq.transAxes)
    ax_seq.text(0.065, 0.06,
                f"position {lo + 1}\u2013{hi} of {L}   \u00b7   mutant "
                f"protein generated: {wt_1}{pos}{mt_1}",
                color="#8d99ae", fontsize=7.5, transform=ax_seq.transAxes)

    # ---- Interactive click tooltip --------------------------------------
    def on_click(event):
        if event.inaxes is not ax_map or event.xdata is None:
            return
        p = int(round(event.xdata))
        if not (1 <= p <= L):
            return
        aa = seq[p - 1]
        info = (f"{gene} residue {p}{aa}\n"
                f"class: {classify(aa)}   hydropathy: {hydropathy(aa):+.1f}")
        if p in pred_by_pos:
            pr, lb = pred_by_pos[p]
            info += (f"\npredicted: "
                     f"{'PATHOGENIC' if lb else 'benign'} ({pr:.2f})")
        if p == pos:
            info += f"\n\u2605 queried mutation -> {mt_1}"
        tooltip.set_text(info)
        tooltip.xy = (p, cls_idx[p - 1])
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    fig.suptitle("interactive protein graph \u2014 click any residue for info",
                 color="#555577", fontsize=8, y=0.995)
    if block:
        plt.show()
    return fig


def _quick_parse(hgvsp: str):
    """Tiny HGVSp parser (wt3, pos, mt3) avoiding heavy imports."""
    import re
    try:
        change = hgvsp[2:] if hgvsp.startswith("p.") else hgvsp
        m = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", change)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
    except Exception:
        pass
    return None, None, None


# ==========================================================================
# 3D interactive structure
# ==========================================================================
def fetch_alphafold(uniprot: str, timeout: float = 25.0):
    """Download the AlphaFold DB predicted structure for a UniProt accession.

    Returns (coords Nx3 ndarray, resnums, plddt) or None on failure.
    """
    import requests
    for ver in ("model_v4", "model_v3"):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-{ver}.pdb"
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and "ATOM" in r.text:
                got = _parse_pdb_ca(r.text)
                if got is not None:
                    return got
        except Exception:
            continue
    return None


def _parse_pdb_ca(pdb_text: str):
    """Parse C-alpha atoms + pLDDT (B-factor) from PDB text."""
    import io
    import warnings
    warnings.filterwarnings("ignore")
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af", io.StringIO(pdb_text))
    coords, resnums, plddt = [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " " or "CA" not in res:
                    continue
                ca = res["CA"]
                coords.append([float(v) for v in ca.coord])
                resnums.append(res.id[1])
                plddt.append(float(ca.bfactor))
        break
    if not coords:
        return None
    return np.array(coords), np.array(resnums), np.array(plddt)


def synthetic_structure(seq: str):
    """Deterministic pseudo-3D backbone from the sequence (offline fallback):
    helical segments connected by turns — good enough to explore interactively
    when AlphaFold DB is unreachable."""
    import zlib
    seed = zlib.crc32(seq.encode()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    coords = np.zeros((len(seq), 3))
    direction = np.array([1.0, 0.2, 0.1])
    direction /= np.linalg.norm(direction)
    p = np.zeros(3)
    for i in range(len(seq)):
        angle = i * 2.1
        p = p + direction * 1.5 + np.array(
            [0.55 * np.cos(angle), 0.55 * np.sin(angle), 0.0])
        if i % 21 == 20:  # kink into a new direction every ~21 residues
            theta = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta),
                                  rng.uniform(-0.4, 0.4)])
            direction /= np.linalg.norm(direction)
        coords[i] = p
    return coords, np.arange(1, len(seq) + 1), None


def show_protein_3d(gene, seq, wt_1, pos, mt_1, uniprot=None, results=None,
                    verdict=None, block=True):
    """Interactive 3D protein structure (drag to rotate, scroll to zoom).

    Uses the AlphaFold DB model when available (colored by pLDDT confidence),
    otherwise a synthetic backbone. Mutation site is a big star; batch
    prediction results overlay as green/red dots colored by probability.
    """
    results = results or []
    pred_by_pos = {}
    for r in results:
        _w, p, _m = _quick_parse(r.get("hgvsp", ""))
        if p:
            pred_by_pos[p] = (r.get("prob", 0.0), r.get("label", 0))

    got = fetch_alphafold(uniprot) if uniprot else None
    if got is not None:
        coords, _resnums, plddt = got
        source = f"AlphaFold DB ({uniprot}) \u00b7 colored by pLDDT"
    else:
        coords, _rn, plddt = synthetic_structure(seq)
        source = "synthetic backbone (AlphaFold DB unreachable)"

    L = min(len(seq), len(coords))
    xs, ys, zs = coords[:L, 0], coords[:L, 1], coords[:L, 2]

    fig = plt.figure(figsize=(11.5, 8.5), facecolor="#14141e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#14141e")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#1a1a28")
        pane.set_edgecolor("#3a3a55")
    ax.tick_params(colors="#c8c8d8", labelsize=7)

    # backbone trace
    ax.plot(xs, ys, zs, color="#4cc9f0", lw=0.9, alpha=0.85, zorder=2)

    # residue dots
    if plddt is not None:
        sc = ax.scatter(xs, ys, zs, c=plddt[:L], cmap="RdYlGn", s=14,
                        vmin=40, vmax=100, depthshade=True, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.7)
        cbar.set_label("pLDDT model confidence", color="#c8c8d8", fontsize=8)
        cbar.ax.tick_params(colors="#c8c8d8", labelsize=7)
        cbar.outline.set_edgecolor("#3a3a55")
    else:
        cols = [CLASS_COLORS[classify(a)] for a in seq[:L]]
        ax.scatter(xs, ys, zs, c=cols, s=14, depthshade=True, zorder=3)

    # batch prediction overlay (green -> red by pathogenic probability)
    if pred_by_pos:
        cmap = plt.get_cmap("RdYlGn_r")
        for p, (prob, label) in pred_by_pos.items():
            if 1 <= p <= L:
                ax.scatter([xs[p - 1]], [ys[p - 1]], [zs[p - 1]],
                           c=[cmap(prob)], s=110, marker="o",
                           edgecolors="black", linewidths=0.6, zorder=5)

    # mutation star
    if 1 <= pos <= L:
        ax.scatter([xs[pos - 1]], [ys[pos - 1]], [zs[pos - 1]],
                   c="#ff4d6d", s=430, marker="*", edgecolors="white",
                   linewidths=0.9, zorder=6,
                   label=f"mutation {wt_1}{pos}{mt_1}")
        leg = ax.legend(loc="upper left", fontsize=8, framealpha=0.25)
        for t in leg.get_texts():
            t.set_color("#c8c8d8")

    title = f"{gene}  \u00b7  {wt_1}{pos}{mt_1} in 3D"
    if verdict:
        title += f"  \u00b7  {verdict}"
    ax.set_title(title, color="#ffffff", fontsize=12, fontweight="bold")

    ax.text2D(0.01, 0.01,
              f"structure: {source}\n"
              f"drag = rotate \u00b7 scroll = zoom \u00b7 \u2605 = {wt_1}{pos}{mt_1}"
              + (" \u00b7 dots = batch predictions" if pred_by_pos else ""),
              transform=ax.transAxes, color="#8d99ae", fontsize=8)

    if block:
        plt.show()
    return fig
