"""Evo2 7B zero-shot variant-effect scoring via NVIDIA NIM (hosted).

Idea: ESM-2 sees PROTEIN sequence; Evo2 sees DNA with deep genomic
context. They capture complementary constraint signals, so Evo2's
zero-shot log-likelihood ratio (LLR) is a strong extra feature:

    LLR = log P(alt base | context) - log P(ref base | context)

LLR << 0  -> model finds the alternate unlikely  -> likely damaging
LLR ~= 0  -> alternate as likely as reference     -> likely benign

Method per missense variant:
  1. Fetch the gene's canonical CDS from Ensembl REST (disk-cached).
  2. Verify CDS codon at the protein position translates to the WT aa.
  3. Enumerate single-nucleotide changes ref_codon -> mutant AA codon.
  4. For each SNV: context = CDS up to (excluding) the variant base,
     one /generate call (num_tokens=1, enable_logits=True), read the
     first-token logits, softmax over A/C/G/T, take the LLR.
  5. Multi-SNV codons: chained sum (each step conditioned on prior
     alt bases; ref term uses the same conditioned distribution --
     an approximation, documented here).

All failures return None (graceful degradation -- the ESM pipeline
must keep working when Evo2 is unreachable).
"""

import json
import math
import os
import time

import requests

from src.paths import DATA_DIR

NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
EVO2_GENERATE_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-7b/generate"
ENSEMBL = "https://rest.ensembl.org"

CACHE_DIR = DATA_DIR / "evo2_cache"
CDS_CACHE = CACHE_DIR / "cds.json"
SCORE_CACHE = CACHE_DIR / "scores.json"

# Evo2 vocab indices for DNA bases (ASCII codes, per NVIDIA docs).
BASE_INDEX = {"A": 65, "C": 67, "T": 84, "G": 71}

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except Exception:
        pass


def evo2_generate(sequence, num_tokens=1, enable_logits=True,
                  retries=4, poll_seconds=120):
    """Single Evo2 /generate call. Returns parsed JSON dict or None."""
    if not NVIDIA_KEY:
        return None
    payload = {"sequence": sequence, "num_tokens": num_tokens,
               "enable_logits": enable_logits}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {NVIDIA_KEY}",
               "nvcf-poll-seconds": str(poll_seconds)}
    delay = 5
    for attempt in range(retries):
        try:
            r = requests.post(EVO2_GENERATE_URL, headers=headers,
                              json=payload, timeout=poll_seconds + 60)
            if r.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
    return None


# Max protein-coding transcripts tried per gene when the canonical
# CDS codon does not match (UniProt/Ensembl isoform differences).
EVO2_MAX_TRANSCRIPTS = 6


def _fetch_cds(tid):
    try:
        seq = requests.get(
            f"{ENSEMBL}/sequence/id/{tid}"
            "?content-type=text/plain;type=cds", timeout=30).text.strip()
        if seq and len(seq) % 3 == 0:
            return seq
    except Exception:
        pass
    return None


def get_cds(gene):
    """Canonical CDS for a human gene symbol (Ensembl REST, cached).

    Returns {"transcript": tid, "cds": seq}. When the canonical
    transcript's codon mismatches later, use get_candidate_cdss().
    """
    cands = get_candidate_cdss(gene)
    if not cands:
        return None
    tid, seq = cands[0]
    return {"transcript": tid, "cds": seq}


def get_candidate_cdss(gene):
    """All usable CDSs for a gene, canonical first (cached).

    Returns [(transcript_id, cds), ...]. Merges with any legacy
    single-record cache entry.
    """
    cache = _load_json(CDS_CACHE)
    entry = cache.get(gene)
    if isinstance(entry, dict) and entry.get("transcripts"):
        return [(tid, entry["transcripts"][tid])
                for tid in entry.get("order", entry["transcripts"])]
    # Legacy single-record entries are refetched once so the fallback
    # transcript list is available for every gene.
    try:
        r = requests.get(
            f"{ENSEMBL}/lookup/symbol/homo_sapiens/{gene}"
            "?content-type=application/json;expand=1", timeout=30)
        data = r.json()
        coding = [t for t in data.get("Transcript", [])
                  if t.get("biotype") == "protein_coding"]
        if not coding:
            return []
        coding.sort(key=lambda t: (t.get("is_canonical") != 1,
                                   -(t.get("length") or 0)))
        transcripts = {}
        order = []
        for t in coding[:EVO2_MAX_TRANSCRIPTS]:
            seq = _fetch_cds(t["id"])
            if seq:
                transcripts[t["id"]] = seq
                order.append(t["id"])
        if not transcripts:
            return []
        cache[gene] = {"transcript": order[0], "cds": transcripts[order[0]],
                       "transcripts": transcripts, "order": order}
        _save_json(CDS_CACHE, cache)
        return [(tid, transcripts[tid]) for tid in order]
    except Exception:
        return []


def _base_probs(logits):
    """Softmax over A/C/G/T logits -> dict base->prob."""
    vals = [logits[BASE_INDEX[b]] for b in "ACGT"]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return {b: e / s for b, e in zip("ACGT", exps)}


def _snv_paths(ref_codon, mt_aa):
    """All single-base changes of ref_codon translating to mt_aa.

    Returns list of (position_in_codon, alt_base). Empty if impossible.
    """
    out = []
    for i, ref_base in enumerate(ref_codon):
        for alt in "ACGT":
            if alt == ref_base:
                continue
            cand = ref_codon[:i] + alt + ref_codon[i + 1:]
            if GENETIC_CODE.get(cand) == mt_aa:
                out.append((i, alt))
    return out


def score_variant(gene, wt_res_1, pos, mt_res_1, use_cache=True):
    """Evo2 zero-shot LLR for a missense variant.

    Returns dict(llr, p_ref, p_alt, n_snvs, transcript, ref_codon,
    alt_codon) or None when scoring is impossible/failed.
    """
    gene = (gene or "").upper()
    key = f"{gene}|{wt_res_1}{pos}{mt_res_1}"
    if use_cache:
        hit = _load_json(SCORE_CACHE).get(key)
        if hit is not None:
            return hit

    result = _score_variant_live(gene, wt_res_1, pos, mt_res_1)

    if use_cache:
        cache = _load_json(SCORE_CACHE)
        cache[key] = result
        _save_json(SCORE_CACHE, cache)
    return result


def _score_variant_live(gene, wt_res_1, pos, mt_res_1):
    if wt_res_1 == mt_res_1:
        return None
    codon_start = (pos - 1) * 3
    match = None
    for tid, cds in get_candidate_cdss(gene):
        ref_codon = cds[codon_start:codon_start + 3]
        if len(ref_codon) == 3 and GENETIC_CODE.get(ref_codon) == wt_res_1:
            match = (tid, cds, ref_codon)
            break
    if match is None:
        # No transcript's codon matches (selenocysteine, bad mapping...).
        return None
    tid, cds, ref_codon = match
    rec = {"transcript": tid, "cds": cds}

    paths = _snv_paths(ref_codon, mt_res_1)
    if not paths:
        return None

    # Group candidate SNVs by codon position; pick the best single-base
    # path per position, then chain multi-SNV paths.
    by_pos = {}
    for i, alt in paths:
        by_pos.setdefault(i, []).append(alt)

    if len(by_pos) == 1:
        i = next(iter(by_pos))
        scored = []
        for alt in by_pos[i]:
            s = _score_snv(cds, codon_start + i, ref_codon[i], alt)
            if s is not None:
                scored.append((s["llr"], alt, s))
        if not scored:
            return None
        scored.sort(key=lambda t: t[0])
        _, alt, s = scored[0]
        s["ref_codon"] = ref_codon
        s["alt_codon"] = ref_codon[:i] + alt + ref_codon[i + 1:]
        s["n_snvs"] = 1
        return s

    # Multi-SNV codon (rare): chain positions in order.
    order = sorted(by_pos)
    context_lst = list(cds)
    total_llr, p_ref_first, p_alt_last = 0.0, None, None
    alt_codon = list(ref_codon)
    for step, i in enumerate(order):
        gpos = codon_start + i
        ref_base = ref_codon[i]
        # Score each candidate alt, keep the least-damaging path.
        cands = []
        for alt in by_pos[i]:
            s = _score_snv("".join(context_lst), gpos, ref_base, alt)
            if s is not None:
                cands.append((s["llr"], alt, s))
        if not cands:
            return None
        cands.sort(key=lambda t: t[0])
        _, alt, s = cands[0]
        total_llr += s["llr"]
        if step == 0:
            p_ref_first = s["p_ref"]
        p_alt_last = s["p_alt"]
        context_lst[gpos] = alt
        alt_codon[i] = alt
    return {"llr": total_llr, "p_ref": p_ref_first, "p_alt": p_alt_last,
            "n_snvs": len(order), "transcript": rec["transcript"],
            "ref_codon": ref_codon, "alt_codon": "".join(alt_codon)}


# Max upstream context (nt) sent to Evo2. Local constraint dominates the
# LLR and full-CDS contexts make hosted calls ~10x slower.
EVO2_CONTEXT_MAX = 4096


def _score_snv(sequence, gpos, ref_base, alt_base):
    """LLR for one nucleotide change at 0-based genomic/CDS position."""
    lo = max(0, gpos - EVO2_CONTEXT_MAX)
    context = sequence[lo:gpos]
    if not context:
        return None
    resp = evo2_generate(context, num_tokens=1, enable_logits=True)
    if not resp or not resp.get("logits"):
        return None
    probs = _base_probs(resp["logits"][0])
    p_ref, p_alt = probs.get(ref_base, 0.0), probs.get(alt_base, 0.0)
    if p_ref <= 0 or p_alt <= 0:
        return None
    return {"llr": math.log(p_alt / p_ref), "p_ref": p_ref, "p_alt": p_alt,
            "ref_codon": None, "alt_codon": None}


def llr_to_pathogenic_prob(llr, scale=2.0):
    """Map LLR (<=0 damaging) to a [0,1] pathogenicity pseudo-probability."""
    return 1.0 / (1.0 + math.exp(llr / scale + 1.0))


# Blend weight: ESM-ensemble vs Evo2. Tuned on 51 fresh ClinVar
# variants (see scripts/evo2_tune.py): ESM-only acc=0.804/auc=0.948,
# blend@0.4 acc=0.902/auc=0.964, blend@0.5 acc=0.922/auc=0.964.
# 0.0 = ESM only, 1.0 = Evo2 only. 0.4 chosen (conservative: favors
# the ensemble trained on 2200 samples over the zero-shot score).
EVO2_BLEND_WEIGHT = 0.40
EVO2_ENABLED = bool(NVIDIA_KEY) and os.environ.get("EVO2_DISABLE", "") != "1"


def evo2_opinion(gene, wt_res_1, pos, mt_res_1):
    """Evo2 second opinion for one variant.

    Returns (evo2_prob, llr, detail) or (None, None, reason) when
    Evo2 is disabled/unavailable -- callers must fall back to ESM.
    """
    if not EVO2_ENABLED:
        return None, None, "Evo2 disabled (no key / EVO2_DISABLE=1)"
    try:
        r = score_variant(gene, wt_res_1, pos, mt_res_1)
    except Exception as e:  # never break prediction on Evo2 errors
        return None, None, f"Evo2 error: {e}"
    if not r:
        return None, None, "Evo2 unscorable (codon/CDS/API)"
    prob = llr_to_pathogenic_prob(r["llr"])
    detail = (f"LLR={r['llr']:.2f} "
              f"{r.get('ref_codon') or '?'}->{r.get('alt_codon') or '?'}")
    return prob, r["llr"], detail


def blend_probs(esm_prob, evo2_prob, weight=EVO2_BLEND_WEIGHT):
    """Weighted blend; evo2_prob=None -> ESM only."""
    if evo2_prob is None:
        return esm_prob
    return (1.0 - weight) * esm_prob + weight * evo2_prob


if __name__ == "__main__":
    for gene, wt, pos, mt, expect in [
            ("TP53", "R", 175, "H", "pathogenic"),
            ("TP53", "P", 72, "R", "benign"),
            ("BRCA1", "C", 61, "G", "pathogenic")]:
        r = score_variant(gene, wt, pos, mt, use_cache=False)
        print(f"{gene} {wt}{pos}{mt} (expect {expect}): {r}")
