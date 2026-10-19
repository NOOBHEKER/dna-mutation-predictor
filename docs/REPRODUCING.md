# Reproducing the published results

Everything needed to reproduce the headline metrics is committed. The full
pipeline (raw ClinVar → embeddings → training) is also scripted, but requires
network access and substantially more time/disk.

## 1. Environment

Use **Python 3.10, 3.11, or 3.12** (3.12 recommended). Python 3.13+ may work but
is not covered by CI.

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-train.txt   # only for retraining
```

On Linux, install the CPU PyTorch wheel if you do not have CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. Recompute the headline metrics (fast, offline)

```bash
python scripts/make_metrics.py
```

Expected ESM-2 head values:

```text
n_test        559
accuracy      0.812165
balanced_acc  0.812433
auc           0.895083
precision     0.827839
recall        0.795775
f1            0.811490
specificity   0.829091
mcc           0.624954
confusion     TN=228 FP=47 FN=58 TP=226
```

The script rewrites `data/metrics.json`. The dashboard reads that file, so a
recomputation automatically updates the Performance page.

## 3. Run the test suite

```bash
pytest -q
```

Tests are offline. `tests/test_metrics.py` needs `torch`; it is skipped if torch
is unavailable. CI installs the CPU PyTorch wheel and runs the full suite.

## 4. Smoke-test the CLI

```bash
python -m src.cli --selftest
python -m src.predict --gene TP53 --hgvsp p.Arg175His
```

The first real prediction downloads ESM-2 (~600 MB) from Hugging Face and
fetches the TP53 sequence from UniProt. Set `HF_HOME` if you want the cache
elsewhere.

## 5. Optional ONNX export

The ONNX encoder is not committed (~569 MB). Export it once:

```bash
python scripts/export_onnx.py
```

This writes:

```text
data/esm_model.onnx
data/esm_model.onnx.data
```

`src/predict.py` and the CLI automatically prefer the ONNX file when present.
Delete both files to fall back to the transformers path.

## 6. Full pipeline from raw data (optional)

```bash
# 1. Download + filter ClinVar (writes data/variant_summary.txt.gz first)
python src/download_data.py

# 2. Build the coordinate map and population caches
python scripts/map_clinvar.py
python scripts/fill_myv.py
python scripts/fill_vep.py

# 3. Embed protein windows (GPU strongly recommended)
python src/preprocess.py

# 4. Train the ESM-2 head (freezes data/test_indices.pt)
python src/train.py

# 5. Evaluate
python src/evaluate.py

# 6. Train the population head
python scripts/train_tabular.py

# 7. Refresh dashboard metrics
python scripts/make_metrics.py
```

### Expected runtime

| Step | CPU | GPU |
|---|---|---|
| ClinVar download | 5–15 min (network) | — |
| Cache fills | 20–60 min (rate-limited APIs) | — |
| Embedding 2,792 variants | hours | ~10–30 min |
| ESM-2 head training | 5–15 min | 2–5 min |
| Population head | < 2 min | — |

## 7. Verify the published repository

```bash
git clone https://github.com/NOOBHEKER/dna-mutation-predictor.git
cd dna-mutation-predictor
python -m venv venv
# activate
pip install -r requirements-dev.txt
pytest -q
python scripts/make_metrics.py
```

If the metrics match the table in the README, the artifacts and split are
intact.

## Determinism notes

- The test split is frozen in `data/test_indices.pt`; do not regenerate it
  casually.
- XGBoost training uses `random_state=42`.
- Small floating-point differences across BLAS/CPU versions are expected in the
  embedding stage, but the committed `processed_data.pt` is fixed, so metric
  reproduction from artifacts is deterministic up to library versions.
- Do not run `quick_test.py` from old snapshots; it overwrote `scaler.pkl` with
  an incompatible format. It is not part of this release.
