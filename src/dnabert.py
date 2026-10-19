"""DNABERT-2 zero-shot DNA embeddings for variant effect scoring.

Replaces/supplements the Evo2 NVIDIA NIM API with a local model.
Uses mean token embeddings from DNABERT-2 to compute a DNA-level
similarity score between reference and alternate sequences.

    DNABERT-2 embedding -> cosine similarity between ref and alt CDS
    low similarity at variant site -> likely damaging

All failures return None (graceful degradation).
"""

import json
import math
import os

import numpy as np
import requests

from src.paths import DATA_DIR

ENSEMBL = "https://rest.ensembl.org"
CACHE_DIR = DATA_DIR / "evo2_cache"
CDS_CACHE = CACHE_DIR / "cds.json"

DNABERT_MODEL = "zhihan1996/DNABERT-2"
DNABERT2_ENABLED = True  # set False to disable

_genetic_code = {
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


def get_cds(gene):
    """Fetch canonical CDS from Ensembl REST (cached)."""
    cache = _load_json(CDS_CACHE)
    entry = cache.get(gene)
    if entry:
        if isinstance(entry, dict):
            return entry.get("cds")
        return entry

    try:
        r = requests.get(
            f"{ENSEMBL}/lookup/symbol/homo_sapiens/{gene}"
            "?content-type=application/json;expand=1", timeout=30)
        data = r.json()
        coding = [t for t in data.get("Transcript", [])
                  if t.get("biotype") == "protein_coding"]
        if not coding:
            return None
        coding.sort(key=lambda t: (t.get("is_canonical") != 1,
                                    -(t.get("length") or 0)))
        tid = coding[0]["id"]
        seq = requests.get(
            f"{ENSEMBL}/sequence/id/{tid}"
            "?content-type=text/plain;type=cds", timeout=30).text.strip()
        if seq and len(seq) % 3 == 0:
            cache[gene] = {"transcript": tid, "cds": seq}
            _save_json(CDS_CACHE, cache)
            return seq
    except Exception:
        pass
    return None


def _snv_paths(ref_codon, mt_aa):
    """All single-base changes of ref_codon translating to mt_aa."""
    out = []
    for i, ref_base in enumerate(ref_codon):
        for alt in "ACGT":
            if alt == ref_base:
                continue
            cand = ref_codon[:i] + alt + ref_codon[i + 1:]
            if _genetic_code.get(cand) == mt_aa:
                out.append((i, alt))
    return out


class DNABERTScorer:
    """Lazy-loaded DNABERT-2 model for DNA sequence scoring."""

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        global DNABERT2_ENABLED
        if self._model is not None:
            return True
        if not DNABERT2_ENABLED:
            return False
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(
                DNABERT_MODEL, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                DNABERT_MODEL, trust_remote_code=True)
            self._model.eval()
            return True
        except Exception:
            DNABERT2_ENABLED = False
            return False

    def embed(self, dna_sequence):
        """Compute mean token embedding for a DNA sequence.
        Returns numpy array or None.
        """
        if not self._ensure_loaded():
            return None
        try:
            import torch
            inputs = self._tokenizer(
                dna_sequence, return_tensors="pt",
                padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Mean pooling over non-padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            return emb.squeeze(0).numpy()
        except Exception:
            return None


_scorer = DNABERTScorer()


def score_variant(gene, wt_res_1, pos, mt_res_1):
    """DNABERT-2 cosine similarity score for a missense variant.

    Returns dict(score, sim_ref, sim_alt, ref_codon, alt_codon, n_snvs)
    or None on failure.
    """
    if not DNABERT2_ENABLED:
        return None

    cds = get_cds(gene)
    if not cds:
        return None

    codon_start = (pos - 1) * 3
    ref_codon = cds[codon_start:codon_start + 3]
    if len(ref_codon) != 3 or _genetic_code.get(ref_codon) != wt_res_1:
        return None

    paths = _snv_paths(ref_codon, mt_res_1)
    if not paths:
        return None

    by_pos = {}
    for i, alt in paths:
        by_pos.setdefault(i, []).append(alt)

    if len(by_pos) == 1:
        i = next(iter(by_pos))
        scores = []
        for alt in by_pos[i]:
            s = _score_snv_dna(cds, codon_start + i, ref_codon[i], alt)
            if s is not None:
                scores.append((s["score"], alt, s))
        if not scores:
            return None
        scores.sort(key=lambda t: -t[0])  # higher sim = more similar = less damaging
        _, alt, s = scores[0]
        s["ref_codon"] = ref_codon
        s["alt_codon"] = ref_codon[:i] + alt + ref_codon[i + 1:]
        s["n_snvs"] = 1
        return s

    # Multi-SNV: chain
    order = sorted(by_pos)
    total_score = 0.0
    context_lst = list(cds)
    for i in order:
        gpos = codon_start + i
        ref_base = ref_codon[i]
        cands = []
        for alt in by_pos[i]:
            s = _score_snv_dna("".join(context_lst), gpos, ref_base, alt)
            if s is not None:
                cands.append((s["score"], alt, s))
        if not cands:
            return None
        cands.sort(key=lambda t: -t[0])
        _, alt, s = cands[0]
        total_score += s["score"]
        context_lst[gpos] = alt

    avg_score = total_score / len(order)
    return {"score": avg_score, "sim_ref": avg_score, "sim_alt": avg_score,
            "n_snvs": len(order), "ref_codon": ref_codon,
            "alt_codon": "".join(list(ref_codon)[:len(order)])}


SNV_CONTEXT = 512

def _score_snv_dna(sequence, gpos, ref_base, alt_base):
    """Cosine similarity score for one nucleotide change."""
    hi = min(len(sequence), gpos + SNV_CONTEXT + 1)

    ref_seq = sequence[:gpos] + ref_base + sequence[gpos + 1:hi]
    alt_seq = sequence[:gpos] + alt_base + sequence[gpos + 1:hi]

    ref_emb = _scorer.embed(ref_seq)
    alt_emb = _scorer.embed(alt_seq)

    if ref_emb is None or alt_emb is None:
        return None

    # Cosine similarity
    dot = np.dot(ref_emb, alt_emb)
    norm = np.linalg.norm(ref_emb) * np.linalg.norm(alt_emb)
    sim = float(dot / max(norm, 1e-8))

    return {"score": sim, "sim_ref": sim, "sim_alt": sim}


def score_to_pathogenic_prob(score, scale=0.5):
    """Map similarity score (high=similar=benign) to pathogenic probability."""
    if score is None:
        return None
    # low similarity -> high pathogenicity
    return 1.0 / (1.0 + math.exp((score - 0.5) / scale))


def dnabert_opinion(gene, wt_res_1, pos, mt_res_1):
    """DNABERT-2 second opinion for one variant.

    Returns (dnabert_prob, score, detail) or (None, None, reason).
    """
    if not DNABERT2_ENABLED:
        return None, None, "DNABERT-2 disabled"
    try:
        r = score_variant(gene, wt_res_1, pos, mt_res_1)
    except Exception as e:
        return None, None, f"DNABERT-2 error: {e}"
    if not r:
        return None, None, "DNABERT-2 unscorable"
    prob = score_to_pathogenic_prob(r["score"])
    detail = (f"cos_sim={r['score']:.3f} "
              f"{r.get('ref_codon') or '?'}->{r.get('alt_codon') or '?'}")
    return prob, r["score"], detail


if __name__ == "__main__":
    for gene, wt, pos, mt, expect in [
            ("TP53", "R", 175, "H", "pathogenic"),
            ("TP53", "P", 72, "R", "benign"),
            ("BRCA1", "C", 61, "G", "pathogenic")]:
        prob, score, detail = dnabert_opinion(gene, wt, pos, mt)
        print(f"{gene} {wt}{pos}{mt} (expect {expect}): prob={prob} {detail}")
