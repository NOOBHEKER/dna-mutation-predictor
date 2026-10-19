# Data provenance

This project ships small, derived artifacts so that predictions and metrics can
be reproduced offline. It does **not** redistribute the raw ClinVar dump or the
ESM-2 weights.

## Committed data

| File | Rows / size | Source | Notes |
|---|---:|---|---|
| `data/variants.csv` | 66,210 rows, ~1.3 MB | ClinVar `variant_summary.txt.gz` | Balanced 33,105 pathogenic / 33,105 benign missense variants |
| `data/processed_data.pt` | 2,792 × 1,280, ~13.6 MB | Derived from ClinVar + ESM-2 | Frozen ESM-2 embeddings + labels |
| `data/test_indices.pt` | 559 indices, ~3.8 MB | Derived split | Frozen stratified hold-out |
| `data/scaler.pkl` | ~1.2 MB | Derived | XGBoost + SelectKBest + StandardScaler |
| `data/tabular.pkl` | ~131 KB | Derived | Population model + feature metadata |
| `data/clinvar_map.json` | 25,376 keys, ~2.6 MB | ClinVar coords | Gene|protein change → genomic coordinate |
| `data/myv_cache.json` | 25,335 keys, ~1.6 MB | MyVariant.info | gnomAD AF / SIFT / PolyPhen |
| `data/vep_cache.json` | 13,835 keys, ~0.9 MB | Ensembl VEP | Fallback annotations |
| `data/tabular_metrics.json` | small | Derived | Population model hold-out metrics |
| `data/train_metrics.json` | small | Derived | ESM-2 head training/test metrics |
| `data/metrics.json` | small | Derived | Metrics consumed by the dashboard and README |
| `data/split_tabular.npz`, `extras_matrix.npy`, `labels_tabular.npy` | small | Derived | Population-model reproducibility |
| `data/evo2_tune.jsonl`, `sweep_result.json` | small | Derived | Experimental records; not used for headline metrics |

## Not committed

| File | Size | Why |
|---|---:|---|
| `data/variant_summary.txt.gz` | ~422 MB | Raw ClinVar dump; download from NCBI |
| `data/esm_model.onnx` + `.data` | ~569 MB | Derived from Hugging Face ESM-2; regenerate with `scripts/export_onnx.py` |
| `data/_fetch_windows.pkl` | ~large | Intermediate sequence-window cache |
| `data/model.pt` | ~large | Legacy PyTorch checkpoint; not the shipped head |
| `data/_stack_oof_k*.npz`, `_emb_ckpt.npz` | scratch | Training intermediates |

## How the data was built

1. **Download ClinVar.** `src/download_data.py` fetches the NCBI
   `variant_summary.txt.gz` dump and filters to human missense variants with
   binary pathogenic/benign labels.
2. **Balance.** The published `variants.csv` is a balanced 33,105 / 33,105
   sample.
3. **Map coordinates.** `scripts/map_clinvar.py` joins variants to GRCh38
   coordinates and writes `data/clinvar_map.json`.
4. **Fetch population signals.** `scripts/fill_myv.py` queries MyVariant.info
   for gnomAD allele frequency, SIFT, and PolyPhen; `scripts/fill_vep.py`
   fills gaps from Ensembl VEP. Results are cached locally.
5. **Build protein windows.** `src/preprocess.py` resolves each gene to a
   UniProt canonical sequence, extracts a 128-aa window around the mutation,
   and embeds wild-type and mutant windows with ESM-2 150M.
6. **Freeze split + train.** `src/train.py` trains the ESM-2 head on a frozen
   stratified split and writes `data/scaler.pkl`. `scripts/train_tabular.py`
   trains the population head and writes `data/tabular.pkl`.

## Licensing and attribution

The derived artifacts are redistributed for reproducibility under the
repository's MIT license, to the extent permitted by the upstream terms.

- **ClinVar** data is public domain (NCBI).
- **UniProt** data is licensed under CC BY 4.0.
- **gnomAD** data is made available under the gnomAD terms; attribution is
  provided here and in the README.
- **ESM-2** weights are distributed by Meta AI under their model license. They
  are not redistributed in this repository; users download them from Hugging
  Face.
- **AlphaFold DB** structures are accessed via API and not redistributed.

If you redistribute derived artifacts, preserve this attribution.

## Reproducing the data pipeline

```bash
python src/download_data.py
python scripts/map_clinvar.py
python scripts/fill_myv.py
python scripts/fill_vep.py
python src/preprocess.py
python src/train.py
python scripts/train_tabular.py
python scripts/make_metrics.py
```

Network access is required for the fetch steps. The ONNX export is optional:

```bash
python scripts/export_onnx.py
```
