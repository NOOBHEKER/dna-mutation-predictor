import os
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
import requests
import numpy as np
from tqdm import tqdm
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
VARIANT_DATA_PATH = "data/variants.csv"
PROCESSED_DATA_PATH = "data/processed_data.pt"
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
        except requests.exceptions.RequestException as e:
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

def parse_hgvsp(hgvsp):
    try:
        change = hgvsp[2:] if hgvsp.startswith('p.') else hgvsp
        match = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+|Ter|=|fs|del|ins)", change)
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        match = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", change)
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
    except Exception:
        pass
    return None, None, None

# Mapping of 3-letter to 1-letter amino acids
AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Sec': 'U', 'Pyl': 'O', 'Xaa': 'X', 'Xle': 'J',
    'Asx': 'B', 'Glx': 'Z',
}

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

    # Load ESM-2
    print("Loading ESM-2 model...")
    import os
    torch.set_num_threads(os.cpu_count())
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using {device}, threads={torch.get_num_threads()}")

    # Phase 1: Concurrent fetching of all sequences
    print(f"\nPhase 1: Fetching sequences ({CONCURRENT_WORKERS} workers)...")
    tasks = [(i, row['gene_symbol'], row['hgvsp'], row['label'])
             for i, row in df.iterrows()]

    print(f"Processing {len(tasks)} variants...")
    variant_windows = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {executor.submit(fetch_variant_data, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching sequences"):
            result = future.result()
            if result:
                variant_windows.append(result)

    print(f"\nPrepared {len(variant_windows)} variant windows.")

    if len(variant_windows) < 100:
        print("ERROR: Too few variants processed!")
        return

    # Phase 2: ONNX-optimized ESM-2 inference
    print(f"\nPhase 2: ESM-2 inference ({len(variant_windows)} variants)...")
    import onnxruntime as ort

    # Export to ONNX for faster CPU inference
    print("Exporting ESM-2 to ONNX...")
    dummy = tokenizer("ACDEFGHIKLMNPQRSTVWY", return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    onnx_path = "data/esm_model.onnx"
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
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
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    all_features = []
    all_labels = []

    for i in tqdm(range(0, len(variant_windows), BATCH_SIZE), desc="ESM-2 inference"):
        batch = variant_windows[i:i+BATCH_SIZE]
        wt_windows = [b[1] for b in batch]
        mt_windows = [b[2] for b in batch]
        rel_positions = [b[3] for b in batch]
        labels = [b[4] for b in batch]

        inputs_wt = tokenizer(wt_windows, return_tensors="np", padding=True, truncation=True, max_length=1024)
        inputs_mt = tokenizer(mt_windows, return_tensors="np", padding=True, truncation=True, max_length=1024)

        out_wt = sess.run(None, {'input_ids': inputs_wt['input_ids'], 'attention_mask': inputs_wt['attention_mask']})[0]
        out_mt = sess.run(None, {'input_ids': inputs_mt['input_ids'], 'attention_mask': inputs_mt['attention_mask']})[0]

        for j, rel_pos in enumerate(rel_positions):
            emb_wt = out_wt[j, rel_pos + 1]
            emb_mt = out_mt[j, rel_pos + 1]
            all_features.append(np.concatenate([emb_wt, emb_mt]))
            all_labels.append(labels[j])

    final_features = torch.tensor(np.array(all_features), dtype=torch.float32)
    final_labels = torch.tensor(np.array(all_labels), dtype=torch.float32)

    torch.save({'features': final_features, 'labels': final_labels}, PROCESSED_DATA_PATH)
    print(f"\nSaved to {PROCESSED_DATA_PATH}")
    print(f"Features: {final_features.shape}, Labels: {final_labels.shape}")
    print(f"Benign: {(final_labels==0).sum().item()}, Pathogenic: {(final_labels==1).sum().item()}")

if __name__ == "__main__":
    preprocess()