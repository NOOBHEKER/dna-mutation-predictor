# Scripts

Run from the repository root with the project virtual environment active.

## Metrics and evaluation

| Script | Purpose |
|---|---|
| `make_metrics.py` | Recompute all published metrics from committed artifacts; rewrites `data/metrics.json` |
| `check_serving.py` | Sanity-check the serving pickle and prediction path |
| `audit_tabular.py` | Audit population-model robustness and coverage |

## Data / caches

| Script | Purpose |
|---|---|
| `map_clinvar.py` | Build `data/clinvar_map.json` from ClinVar coordinates |
| `fill_myv.py` | Fill the MyVariant.info cache (gnomAD/SIFT/PolyPhen) |
| `fill_vep.py` | Fill gaps from Ensembl VEP |
| `train_tabular.py` | Train the population model and write metrics |
| `fit_tabular_serving.py` | Fit the serving-compatible population artifact |
| `bench.py` | Benchmark inference paths |

## Models

| Script | Purpose |
|---|---|
| `export_onnx.py` | Export ESM-2 to ONNX (not committed; ~569 MB) |

## Notes

- Network-using scripts are rate-limited; re-runs reuse local caches.
- `train_tabular.py` expects `data/_fetch_windows.pkl`, a training
  intermediate that is not committed. Rebuild it with `src/preprocess.py`
  before retraining the population model.
- `make_metrics.py` is offline and is the canonical way to verify a clone.
