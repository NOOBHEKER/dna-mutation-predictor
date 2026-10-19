# Data directory

This folder contains **derived artifacts** needed to reproduce the published
metrics and run the predictor. Raw third-party dumps and large model weights are
not committed.

## Committed

| File | Purpose |
|---|---|
| `processed_data.pt` | 2,792 × 1,280 ESM-2 features + labels |
| `test_indices.pt` | Frozen 559-row test split |
| `scaler.pkl` | `(XGBClassifier, SelectKBest(k=400), StandardScaler)` |
| `tabular.pkl` | Population model dict + feature names |
| `variants.csv` | 66,210 balanced ClinVar missense variants (CLI index) |
| `clinvar_map.json` | `GENE|WTposMT` → genomic coordinate |
| `myv_cache.json` | MyVariant.info gnomAD/SIFT/PolyPhen cache |
| `vep_cache.json` | Ensembl VEP fallback cache |
| `train_metrics.json` | ESM-2 head metrics |
| `tabular_metrics.json` | Population head metrics |
| `metrics.json` | Metrics displayed by the README and dashboard |
| `split_tabular.npz` | Population-model split |
| `extras_matrix.npy` | Population features |
| `labels_tabular.npy` | Population labels |
| `evo2_tune.jsonl` | Experimental Evo2 tuning records (not headline metrics) |
| `sweep_result.json` | Experimental hyperparameter sweep |

## Not committed

- `esm_model.onnx` / `.onnx.data` — regenerate with
  `python scripts/export_onnx.py`.
- `variant_summary.txt.gz` — download from NCBI with
  `python src/download_data.py`.
- `model.pt`, `_fetch_windows.pkl`, `_emb_ckpt.npz`, `_stack_oof_k*.npz` —
  training scratch files.

## Pickle safety

`scaler.pkl`, `tabular.pkl`, and `processed_data.pt` are Python pickle/torch
archives. Only load copies from this repository (or another source you trust).
Do not load untrusted `.pkl` or `.pt` files.

## Regenerating

```bash
python scripts/make_metrics.py   # metrics.json from committed artifacts
python src/download_data.py      # raw ClinVar -> variants.csv
python src/preprocess.py         # embeddings -> processed_data.pt
python src/train.py              # scaler.pkl + test_indices.pt
python scripts/train_tabular.py  # tabular.pkl + tabular metrics
```

See [../docs/DATA.md](../docs/DATA.md) for provenance and licensing.
