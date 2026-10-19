import os
import sys
# Allow `python src/preprocess.py` (script mode) to import sibling
# modules as `src.*` -- without this the repo root is not on sys.path
# and enrichment silently degrades to neutral features.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
import requests
import numpy as np
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.paths import data_path

# Configuration
VARIANT_DATA_PATH = data_path("variants.csv")
PROCESSED_DATA_PATH = data_path("processed_data.pt")
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
WINDOW_SIZE = 128
BATCH_SIZE = 32
MAX_VARIANTS = 5000
CONCURRENT_WORKERS = 20

# Thread-safe caching
UNIPROT_ID_CACHE = {}
SEQUENCE_CACHE = {}
_cache_lock = threading.Lock()

# UniProt API retry settings
MAX_RETRIES = 3
BASE_DELAY = 1.0
REQUEST_TIMEOUT = 15

def retry_request(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(BASE_DELAY * (2 ** attempt))
    return None

def fetch_variant_data(args):
    """Fetch UniProt ID + sequence for a single variant. Returns (index, wt, mt, rel_pos, label, gene, hgvsp) or None."""
    idx, gene, hgvsp, label = args

    wt_res_3, pos, mt_res_3 = parse_hgvsp(hgvsp)
    if wt_res_3 is None:
        return None

    wt_res = AA_MAP.get(wt_res_3)
    mt_res = AA_MAP.get(mt_res_3)
    if not wt_res or not mt_res:
        return None

    # Fetch UniProt ID (thread-safe)
    with _cache_lock:
        uniprot_id = UNIPROT_ID_CACHE.get(gene)
    if not uniprot_id:
        url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&fields=accession&format=json"
        try:
            response = retry_request(requests.get, url, timeout=REQUEST_TIMEOUT)
            data = response.json()
            results = data.get('results', [])
            if results:
                uniprot_id = results[0]['primaryAccession']
                with _cache_lock:
                    UNIPROT_ID_CACHE[gene] = uniprot_id
            else:
                return None
        except Exception:
            return None

    # Fetch sequence (thread-safe)
    with _cache_lock:
        seq = SEQUENCE_CACHE.get(uniprot_id)
    if not seq:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        try:
            response = retry_request(requests.get, url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                lines = response.text.splitlines()
                seq = "".join(lines[1:])
                with _cache_lock:
                    SEQUENCE_CACHE[uniprot_id] = seq
            else:
                return None
        except Exception:
            return None

    if pos > len(seq):
        return None

    start = max(0, pos - 1 - WINDOW_SIZE)
    end = min(len(seq), pos + WINDOW_SIZE)
    rel_pos = (pos - 1) - start

    wt_window = seq[start:end]
    mt_list = list(wt_window)
    mt_list[rel_pos] = mt_res
    mt_window = "".join(mt_list)

    return (idx, wt_window, mt_window, rel_pos, label, gene, hgvsp)

# Parsing + amino-acid tables live in src.hgvsp so they stay importable
# without torch. Re-exported here for backward compatibility.
from src.hgvsp import AA_MAP, P_HGVS_KEY, parse_hgvsp  # noqa: E402

def get_uniprot_id(gene):
    with _cache_lock:
        if gene in UNIPROT_ID_CACHE:
            return UNIPROT_ID_CACHE[gene]
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&fields=accession&format=json"
    try:
        response = retry_request(requests.get, url, timeout=REQUEST_TIMEOUT)
        results = response.json().get('results', [])
        if results:
            uniprot_id = results[0]['primaryAccession']
            with _cache_lock:
                UNIPROT_ID_CACHE[gene] = uniprot_id
            return uniprot_id
    except Exception:
        pass
    return None

def get_protein_sequence(uniprot_id):
    with _cache_lock:
        if uniprot_id in SEQUENCE_CACHE:
            return SEQUENCE_CACHE[uniprot_id]
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = retry_request(requests.get, url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            lines = response.text.splitlines()
            seq = "".join(lines[1:])
            with _cache_lock:
                SEQUENCE_CACHE[uniprot_id] = seq
            return seq
    except Exception:
        pass
    return None

def preprocess():
    df = pd.read_csv(VARIANT_DATA_PATH)

    if len(df) > MAX_VARIANTS:
        print(f"Limiting to {MAX_VARIANTS} variants (from {len(df)}).")
        # Over-sample benign since UniProt fetch fails ~60% for benign variants
        # 3500 benign * ~38% success = ~1330 benign
        # 1500 pathogenic * ~96% success = ~1440 pathogenic
        # Expected output: ~2770 samples, roughly balanced
        n_benign = min(3500, df[df['label'] == 0].shape[0])
        n_patho = min(1500, df[df['label'] == 1].shape[0])
        benign = df[df['label'] == 0].sample(n_benign, random_state=42)
        patho = df[df['label'] == 1].sample(n_patho, random_state=42)
        df = pd.concat([benign, patho]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Load ESM-2 -- but only when we must (re-)export ONNX. Phase 2 runs
    # entirely through onnxruntime, so keeping a second 600MB torch copy
    # resident just invites paging and a per-batch slowdown.
    import onnxruntime as ort
    onnx_path = data_path("esm_model.onnx")
    need_export = not (os.path.exists(onnx_path)
                       and os.path.getsize(onnx_path) > 1_000_000)
    if need_export:
        print("Loading ESM-2 (torch) for ONNX export...")
        torch.set_num_threads(os.cpu_count())
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        model = EsmModel.from_pretrained(ESM_MODEL_NAME)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Using {device}, threads={torch.get_num_threads()}")
    else:
        print("ONNX model present -- torch weights not loaded.")
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        device = torch.device('cpu')

    # Phase 1: concurrent fetch with INCREMENTAL checkpointing.
    # Dict keyed gene|hgvsp survives kills: reruns skip fetched variants.
    import pickle as _pk2
    _fetch_ckpt = data_path("_fetch_windows.pkl")
    _fetched = {}
    if os.path.exists(_fetch_ckpt):
        try:
            with open(_fetch_ckpt, "rb") as _f:
                _loaded = _pk2.load(_f)
            _fetched = dict(_loaded) if isinstance(_loaded, dict) \
                else {f"{_v[5]}|{_v[6]}": _v for _v in _loaded}
            print(f"\nPhase 1: resumed ({len(_fetched)} windows already "
                  f"fetched).")
        except Exception as e:
            print(f"\nPhase 1: checkpoint unreadable ({e}), refetching.")
            _fetched = {}
    print(f"\nPhase 1: Fetching sequences ({CONCURRENT_WORKERS} workers)...")
    tasks = [(i, row['gene_symbol'], row['hgvsp'], row['label'])
             for i, row in df.iterrows()
             if f"{row['gene_symbol']}|{row['hgvsp']}" not in _fetched]
    print(f"Processing {len(tasks)} variants "
          f"({len(_fetched)} skipped from checkpoint)...")
    if tasks:
        _since_save = 0
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = {executor.submit(fetch_variant_data, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching sequences"):
                result = future.result()
                if result:
                    _fetched[f"{result[5]}|{result[6]}"] = result
                    _since_save += 1
                    if _since_save >= 100:
                        with open(_fetch_ckpt, "wb") as _f:
                            _pk2.dump(_fetched, _f)
                        _since_save = 0
        with open(_fetch_ckpt, "wb") as _f:
            _pk2.dump(_fetched, _f)
    variant_windows = list(_fetched.values())

    print(f"\nPrepared {len(variant_windows)} variant windows.")

    if len(variant_windows) < 100:
        print("ERROR: Too few variants processed!")
        return

    print(f"Phase 1 checkpoint saved ({len(variant_windows)} windows).")

    # Phase 1b: enrichment coverage report (no network here -- caches are
    # filled beforehand by scripts/fill_vep.py and scripts/fill_myv.py).
    try:
        from src.vep import _load_json as _vep_load
        from pathlib import Path as _Path
        cmap = _vep_load(_Path(data_path("clinvar_map.json")))
        _n_map = 0
        for (_i, _wt, _mt, _rel, _lab, _gene, _hg) in variant_windows:
            _m = P_HGVS_KEY.match(str(_hg))
            _key = None
            if _m:
                _wt1, _mt1 = AA_MAP.get(_m.group(1)), AA_MAP.get(_m.group(3))
                if _wt1 and _mt1:
                    _key = f"{str(_gene).upper()}|{_wt1}{_m.group(2)}{_mt1}"
            if _key and cmap.get(_key):
                _n_map += 1
        print(f"Phase 1b: {_n_map}/{len(variant_windows)} variants "
              f"mapped to genomic coords.")
    except Exception as e:
        print(f"Phase 1b skipped ({type(e).__name__}: {e})")

    # Phase 2: ONNX-optimized ESM-2 inference
    print(f"\nPhase 2: ESM-2 inference ({len(variant_windows)} variants)...")

    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    if need_export:
        print("Exporting ESM-2 to ONNX...")
        dummy = tokenizer("ACDEFGHIKLMNPQRSTVWY", return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        torch.onnx.export(
            model,
            (dummy['input_ids'], dummy['attention_mask']),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'}, 'last_hidden_state': {0: 'batch', 1: 'seq'}},
            opset_version=18,
        )
        print("ONNX export done.")
        del model, dummy
        import gc
        gc.collect()
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = max(1, (os.cpu_count() or 4))
    _so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=_so,
                                providers=['CPUExecutionProvider'])

    # Phase 2 resume: embeddings checkpointed every 2 batches.
    # Windows are processed length-sorted (padded batches waste a lot of
    # compute on ESM's quadratic attention); the original order is
    # restored before saving so features/labels/metas/extra stay aligned.
    _emb_ckpt = data_path("_emb_ckpt.npz")
    _order = sorted(range(len(variant_windows)),
                    key=lambda i: len(variant_windows[i][1])
                    + len(variant_windows[i][2]))
    _pos = {orig: new for new, orig in enumerate(_order)}
    all_features, all_labels, all_metas = [], [], []
    _done_batches = 0
    if os.path.exists(_emb_ckpt):
        try:
            _z = np.load(_emb_ckpt, allow_pickle=True)
            if int(_z.get("order_sig", -1)) == len(_order):
                all_features = [r for r in _z["features"]]
                all_labels = [r for r in _z["labels"]]
                all_metas = [tuple(r) for r in _z["metas"]]
                _done_batches = int(_z["batches"])
                print(f"Phase 2: resumed at batch {_done_batches} "
                      f"({len(all_features)} embedded).")
        except Exception as e:
            print(f"Phase 2: ckpt unreadable ({e}), restarting inference.")
            all_features, all_labels, all_metas, _done_batches = [], [], [], 0

    _batch_ids = list(range(0, len(variant_windows), BATCH_SIZE))
    for _bi, i in enumerate(tqdm(_batch_ids, desc="ESM-2 inference")):
        if _bi < _done_batches:
            continue
        batch = [variant_windows[j] for j in _order[i:i + BATCH_SIZE]]
        wt_windows = [b[1] for b in batch]
        mt_windows = [b[2] for b in batch]
        rel_positions = [b[3] for b in batch]
        labels = [b[4] for b in batch]
        all_metas.extend([(b[5], b[6], b[4], b[0]) for b in batch])

        inputs_wt = tokenizer(wt_windows, return_tensors="np", padding=True, truncation=True, max_length=1024)
        inputs_mt = tokenizer(mt_windows, return_tensors="np", padding=True, truncation=True, max_length=1024)

        out_wt = sess.run(None, {'input_ids': inputs_wt['input_ids'], 'attention_mask': inputs_wt['attention_mask']})[0]
        out_mt = sess.run(None, {'input_ids': inputs_mt['input_ids'], 'attention_mask': inputs_mt['attention_mask']})[0]

        for j, rel_pos in enumerate(rel_positions):
            emb_wt = out_wt[j, rel_pos + 1]
            emb_mt = out_mt[j, rel_pos + 1]
            # Proven layout: [wt, mt] (1280 dims). Tabular extras
            # (gnomAD/SIFT/PolyPhen) ride alongside in 'extra', scaled
            # separately at train time -- NOT mixed into MI selection.
            all_features.append(np.concatenate([emb_wt, emb_mt]))
            all_labels.append(labels[j])

        if (_bi + 1) % 2 == 0:
            np.savez(_emb_ckpt,
                     features=np.array(all_features),
                     labels=np.array(all_labels),
                     metas=np.array(all_metas, dtype=object),
                     batches=np.array(_bi + 1),
                     order_sig=np.array(len(_order)))

    # Restore original window order (invert the length-sort permutation).
    _inv = [0] * len(_order)
    for _new, _orig in enumerate(_order):
        _inv[_orig] = _new
    all_features = [all_features[j] for j in _inv]
    all_labels = [all_labels[j] for j in _inv]
    all_metas = [all_metas[j] for j in _inv]

    final_features = torch.tensor(np.array(all_features), dtype=torch.float32)
    final_labels = torch.tensor(np.array(all_labels), dtype=torch.float32)

    # Tabular extras aligned row-for-row with features (neutral when unmapped)
    try:
        from src.myvariant import extras_for_rows  # MyV cache -> VEP -> neutral
        _keys = []
        for (_gene, _hg, _lab, _idx) in all_metas:
            _m = P_HGVS_KEY.match(str(_hg))
            if _m:
                _wt1, _mt1 = AA_MAP.get(_m.group(1)), AA_MAP.get(_m.group(3))
                _keys.append(f"{str(_gene).upper()}|{_wt1}{_m.group(2)}{_mt1}"
                             if _wt1 and _mt1 else None)
            else:
                _keys.append(None)
        extra = torch.tensor(extras_for_rows(_keys), dtype=torch.float32)
        n_mapped = int(sum(1 for k in _keys if k))
        print(f"Extras: {extra.shape} ({n_mapped}/{len(_keys)} keys mapped)")
    except Exception as e:
        print(f"Extras skipped ({type(e).__name__}: {e}) -- neutral fallback")
        import torch as _t
        extra = _t.zeros((len(all_features), 6), dtype=_t.float32)
        for _r in range(len(all_features)):
            extra[_r, 1] = extra[_r, 3] = extra[_r, 5] = 1.0

    torch.save({'features': final_features, 'labels': final_labels,
                'metas': all_metas, 'extra': extra}, PROCESSED_DATA_PATH)
    # Inference checkpoint now redundant -- remove so future runs with
    # different data can't resume stale embeddings. (Keep the Phase 1
    # fetch checkpoint: delete data/_fetch_windows.pkl to force refetch.)
    try:
        os.remove(_emb_ckpt)
    except OSError:
        pass
    print(f"\nSaved to {PROCESSED_DATA_PATH}")
    print(f"Features: {final_features.shape}, Labels: {final_labels.shape}, "
          f"Extra: {extra.shape}")
    print(f"Benign: {(final_labels==0).sum().item()}, Pathogenic: {(final_labels==1).sum().item()}")

if __name__ == "__main__":
    preprocess()