"""Population + conservation features via Ensembl VEP (REST, cached).

One batched VEP call per variant yields three classic pathogenicity
signals that ESM-2 embeddings lack:

  - gnomAD allele frequency (common => almost always benign)
  - SIFT score (0 = deleterious .. 1 = tolerated)
  - PolyPhen-2 score (0 = benign .. 1 = damaging)

Coordinates come from data/clinvar_map.json (see scripts/map_clinvar.py).
All results are disk-cached in data/vep_cache.json, so retraining and
prediction are free after the first fill. Failures return None --
callers must fall back to ESM-only features.

Feature layout (EXTRA_DIM = 6):
  [log10(AF + 1e-8), af_missing, sift, sift_missing, polyphen, pp_missing]
Missing numeric slots take neutral values (AF -> 1e-8, SIFT/PolyPhen ->
0.5) plus a 0/1 flag so trees can split on missingness.
"""

import json
import math
import os
import time
from pathlib import Path

import requests

ENSEMBL = "https://rest.ensembl.org"
VEP_URL = f"{ENSEMBL}/vep/human/region"

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "data" / "clinvar_map.json"
CACHE_PATH = ROOT / "data" / "vep_cache.json"

EXTRA_DIM = 6
EXTRA_NAMES = ["log_af", "af_missing", "sift", "sift_missing",
               "polyphen", "pp_missing"]

VEP_BATCH = 200
VEP_TIMEOUT = 60
VEP_SLEEP = 2.0  # politeness delay between batches (Ensembl throttles)


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


def _clinvar_map():
    return _load_json(MAP_PATH)


def vep_key(chrom, pos, ref, alt):
    return f"{chrom}:{pos}:{ref}:{alt}"


def fetch_vep_batch(variants):
    """POST a batch of 'chr pos pos ref/alt' strings. Returns list/None."""
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    delay = 10
    for _ in range(6):
        try:
            r = requests.post(VEP_URL, headers=headers,
                              json={"variants": variants},
                              timeout=VEP_TIMEOUT)
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


def _gnomad_max(sub):
    """Max gnomAD AF across populations in one frequencies sub-dict."""
    best = None
    for k, v in (sub or {}).items():
        if "gnomad" in k.lower() and isinstance(v, (int, float)):
            if best is None or v > best:
                best = float(v)
    return best


def _extract(rec, alt=None):
    """Pick AF / SIFT / PolyPhen out of one VEP response record."""
    af, sift, polyphen = None, None, None
    for cv in rec.get("colocated_variants") or []:
        freqs = cv.get("frequencies") or {}
        # Modern shape: frequencies = {ALT: {gnomade: .., gnomadg_nfe: ..}}
        subs = []
        if alt and isinstance(freqs.get(alt), dict):
            subs = [freqs[alt]]
        else:
            subs = [v for v in freqs.values() if isinstance(v, dict)]
        for sub in subs:
            v = _gnomad_max(sub)
            if v is not None and (af is None or v > af):
                af = v
        # Legacy shape: flat gnomAD_AF keys on the colocated variant
        for k, v in cv.items():
            kl = k.lower()
            if "gnomad" in kl and "af" in kl and isinstance(v, (int, float)):
                if af is None or v > af:
                    af = float(v)
    if af is None:
        maf = (rec.get("minor_allele_freq")
               or (rec.get("colocated_variants") or [{}])[0].get(
                   "minor_allele_freq"))
        try:
            af = float(maf) if maf is not None else None
        except (TypeError, ValueError):
            af = None
    sifts, polys = [], []
    for tc in rec.get("transcript_consequences") or []:
        try:
            if tc.get("sift_score") is not None:
                sifts.append(float(tc["sift_score"]))
            if tc.get("polyphen_score") is not None:
                polys.append(float(tc["polyphen_score"]))
        except (TypeError, ValueError):
            pass
    if sifts:
        sift = min(sifts)  # most deleterious
    if polys:
        polyphen = max(polys)  # most damaging
    return {"af": af, "sift": sift, "polyphen": polyphen}


def ensure_cached(coords_list):
    """Fill the disk cache for [(chrom, pos, ref, alt), ...].

    Returns {key: extracted_dict}. Network failures leave gaps --
    callers treat missing keys as missing features.
    """
    cache = _load_json(CACHE_PATH)
    todo = [(c, p, r, a) for c, p, r, a in coords_list
            if vep_key(c, p, r, a) not in cache]
    print(f"VEP fill: {len(todo)} uncached of {len(coords_list)} requested",
          flush=True)
    out = {}
    for i in range(0, len(todo), VEP_BATCH):
        batch = todo[i:i + VEP_BATCH]
        q = [f"{c} {p} {p} {r}/{a}" for c, p, r, a in batch]
        try:
            resp = fetch_vep_batch(q)
        except Exception:
            resp = None
        n_ok = 0
        if resp:
            for (c, p, r, a), rec in zip(batch, resp):
                if isinstance(rec, dict) and "error" not in rec:
                    cache[vep_key(c, p, r, a)] = _extract(rec, alt=a)
                    n_ok += 1
        _save_json(CACHE_PATH, cache)
        print(f"  batch {i // VEP_BATCH + 1}/"
              f"{(len(todo) + VEP_BATCH - 1) // VEP_BATCH}: "
              f"{n_ok}/{len(batch)} scored", flush=True)
        time.sleep(VEP_SLEEP)  # stay under Ensembl rate limits
    for c, p, r, a in coords_list:
        if vep_key(c, p, r, a) in cache:
            out[vep_key(c, p, r, a)] = cache[vep_key(c, p, r, a)]
    return out


def vep_single(chrom, pos, ref, alt):
    """Live single-variant VEP GET (updates cache). Returns rec or None."""
    try:
        r = requests.get(
            f"{VEP_URL}/{chrom}:{pos}-{pos}/{ref}/{alt}",
            headers={"Content-Type": "application/json"},
            timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        rec = data[0] if isinstance(data, list) and data else None
        if not isinstance(rec, dict) or "error" in rec:
            return None
        cache = _load_json(CACHE_PATH)
        cache[vep_key(chrom, pos, ref, alt)] = _extract(rec, alt=alt)
        _save_json(CACHE_PATH, cache)
        return cache[vep_key(chrom, pos, ref, alt)]
    except requests.RequestException:
        return None


def extra_for_variant(gene, wt_1, pos, mt_1, live=True):
    """6-dim extra feature vector for one variant.

    Cache first; optionally one live VEP GET on miss. Never raises.
    Returns (vector, source) with source in
    {"cache", "live", "neutral", "unmapped"}.
    """
    try:
        key = f"{(gene or '').upper()}|{wt_1}{pos}{mt_1}"
        m = _load_json(MAP_PATH).get(key)
        if not m:
            return features_from_rec({}), "unmapped"
        k = vep_key(m["chrom"], m["pos"], m["ref"], m["alt"])
        rec = _load_json(CACHE_PATH).get(k)
        if rec:
            return features_from_rec(rec), "cache"
        if live:
            rec = vep_single(m["chrom"], m["pos"], m["ref"], m["alt"])
            if rec:
                return features_from_rec(rec), "live"
        return features_from_rec({}), "neutral"
    except Exception:
        return features_from_rec({}), "neutral"


def features_from_rec(rec):
    """Raw VEP record -> 6-dim numpy vector (handles None fields)."""
    import numpy as np
    af = rec.get("af")
    sift = rec.get("sift")
    poly = rec.get("polyphen")
    return np.array([
        math.log10((af if af is not None else 0.0) + 1e-8),
        1.0 if af is None else 0.0,
        float(sift) if sift is not None else 0.5,
        1.0 if sift is None else 0.0,
        float(poly) if poly is not None else 0.5,
        1.0 if poly is None else 0.0,
    ], dtype=np.float32)


def extras_for_rows(keys):
    """Batch helper for training: [gene|wtposmt, ...] -> (n, 6) array.

    Unmapped rows get the neutral vector (missing flags = 1).
    """
    import numpy as np
    cmap = _load_json(MAP_PATH)
    cache = _load_json(CACHE_PATH)
    rows = []
    for key in keys:
        m = cmap.get(key)
        rec = cache.get(vep_key(m["chrom"], m["pos"], m["ref"], m["alt"])) \
            if m else None
        rows.append(features_from_rec(rec) if rec
                    else features_from_rec({}))
    return np.array(rows, dtype=np.float32)
