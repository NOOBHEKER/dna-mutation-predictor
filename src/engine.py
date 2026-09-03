"""
engine.py — torch-free helpers for the CLI bot.

Lightweight UniProt access used by the graph commands so the 2D/3D
visualizations work instantly without waking the ESM-2 engine. The
prediction engine itself still lives in src/predict.py (lazy-loaded).
"""
from __future__ import annotations

import threading
import time

import requests

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

_lock = threading.Lock()
_uniprot_cache = {}
_seq_cache = {}


def _retry_get(url: str):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(1.0 * (2 ** attempt))
    if last_exc:
        raise last_exc
    return None


def get_uniprot_id(gene: str):
    with _lock:
        if gene in _uniprot_cache:
            return _uniprot_cache[gene]
    url = (f"https://rest.uniprot.org/uniprotkb/search?"
           f"query=gene:{gene}+AND+organism_id:9606"
           f"&fields=accession&format=json")
    try:
        results = _retry_get(url).json().get("results", [])
        if results:
            acc = results[0]["primaryAccession"]
            with _lock:
                _uniprot_cache[gene] = acc
            return acc
    except Exception:
        pass
    return None


def get_protein_sequence(uniprot_id: str):
    with _lock:
        if uniprot_id in _seq_cache:
            return _seq_cache[uniprot_id]
    try:
        r = _retry_get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
        seq = "".join(r.text.splitlines()[1:])
        with _lock:
            _seq_cache[uniprot_id] = seq
        return seq
    except Exception:
        return None
