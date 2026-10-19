"""Population + conservation features via MyVariant.info (rsID batch API).

Same 6-dim EXTRA layout as src.vep (fully interchangeable):
  [log10(AF + 1e-8), af_missing, sift, sift_missing, polyphen, pp_missing]

  - AF: max(gnomad_exome.af, gnomad_genome.af) for the matching ALT allele
  - SIFT: min(dbnsfp.sift.score)  (0 = deleterious)
  - PolyPhen-2: max(dbnsfp.polyphen2.hdiv.score) (1 = damaging)

Why a second source: Ensembl VEP throttles aggressively on bulk fills;
MyVariant.info accepts 1000-id batches and keys everything by rsID,
which data/clinvar_map.json already stores. Variants without rsIDs
fall back to the VEP cache (src.vep).

Cache: data/myv_cache.json keyed "rsXXXX:ALT".
"""

import json
import os
import time
from pathlib import Path

import requests

MYV_URL = "https://myvariant.info/v1/variant"
FIELDS = ("dbsnp,vcf,dbnsfp.sift,dbnsfp.polyphen2,"
          "gnomad_exome.af,gnomad_genome.af")

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "data" / "clinvar_map.json"
CACHE_PATH = ROOT / "data" / "myv_cache.json"

BATCH = 1000
TIMEOUT = 60


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


def _num_list(x):
    """Scalar-or-list of scores -> flat list of floats."""
    if x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, str):
        try:
            return [float(x)]
        except ValueError:
            return []
    out = []
    for v in x:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _extract_doc(doc, alt):
    """One MyVariant doc -> {af, sift, polyphen} or None if ALT mismatches."""
    try:
        vcf = doc.get("vcf") or {}
        if str(vcf.get("alt", "")).upper() != str(alt).upper():
            return None
        af = None
        for src in ("gnomad_exome", "gnomad_genome"):
            g = doc.get(src) or {}
            # Shape varies: {af: 1e-5, ...} unfiltered vs
            # {af: {af: 1e-5, ...}} with a fields projection.
            v = g.get("af")
            if isinstance(v, dict):
                v = v.get("af")
            if isinstance(v, (int, float)) and (af is None or v > af):
                af = float(v)
        sift = None
        s = (doc.get("dbnsfp") or {}).get("sift") or {}
        scores = _num_list(s.get("score"))
        if scores:
            sift = min(scores)
        poly = None
        pp = (doc.get("dbnsfp") or {}).get("polyphen2") or {}
        for model in ("hdiv", "hvar"):
            scores = _num_list((pp.get(model) or {}).get("score"))
            if scores:
                v = max(scores)
                poly = v if poly is None or v > poly else poly
        return {"af": af, "sift": sift, "polyphen": poly}
    except Exception:
        return None


def fetch_batch(rsids):
    """POST one batch of rsIDs. Returns {rsid: [docs]} (missing -> absent)."""
    delay = 10
    for _ in range(5):
        try:
            r = requests.post(MYV_URL,
                              json={"ids": rsids, "fields": FIELDS},
                              timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                return None
            out = {}
            for doc in r.json():
                if isinstance(doc, dict) and "query" in doc:
                    out.setdefault(doc["query"], []).append(doc)
            return out
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
    return None


def ensure_cached(rs_alt_list):
    """Fill cache for [(rsid, alt), ...]. Returns {key: rec}. Resumable."""
    from src.vep import features_from_rec  # noqa (shared layout only)
    cache = _load_json(CACHE_PATH)
    todo = [(rs, alt) for rs, alt in rs_alt_list
            if f"{rs}:{alt}" not in cache]
    print(f"MyVariant fill: {len(todo)} uncached of {len(rs_alt_list)}",
          flush=True)
    out = {}
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            resp = fetch_batch([f"rs{rs}" for rs, _ in batch])
        except Exception:
            resp = None
        n_ok = 0
        if resp:
            for rs, alt in batch:
                docs = resp.get(f"rs{rs}", [])
                rec = None
                for doc in docs:
                    rec = _extract_doc(doc, alt)
                    if rec is not None:
                        break
                if rec is not None:
                    cache[f"{rs}:{alt}"] = rec
                    n_ok += 1
        _save_json(CACHE_PATH, cache)
        print(f"  batch {i // BATCH + 1}/"
              f"{(len(todo) + BATCH - 1) // BATCH}: {n_ok}/{len(batch)}",
              flush=True)
        time.sleep(1.0)
    for rs, alt in rs_alt_list:
        if f"{rs}:{alt}" in cache:
            out[f"{rs}:{alt}"] = cache[f"{rs}:{alt}"]
    return out


def features_from_myv(rec):
    """Shared 6-dim layout (delegates to src.vep for one definition)."""
    from src.vep import features_from_rec
    return features_from_rec(rec)


def extras_for_rows(keys):
    """Batch helper: [GENE|wtposmt, ...] -> (n, 6) array.

    Source priority per variant: MyVariant cache (rsID) -> VEP cache
    (coords) -> neutral vector. Same signature as src.vep.extras_for_rows.
    """
    import numpy as np
    from src.vep import features_from_rec, vep_key, _load_json as _vload
    from src.vep import CACHE_PATH as VEP_CACHE
    cmap = _load_json(MAP_PATH)
    myv = _load_json(CACHE_PATH)
    vepc = _vload(VEP_CACHE)
    rows = []
    for key in keys:
        rec = None
        m = cmap.get(key) if key else None
        if m:
            if m.get("rs"):
                rec = myv.get(f"{m['rs']}:{m['alt']}")
            if rec is None:
                rec = vepc.get(vep_key(m["chrom"], m["pos"],
                                       m["ref"], m["alt"]))
        rows.append(features_from_rec(rec) if rec else features_from_rec({}))
    return np.array(rows, dtype=np.float32)
