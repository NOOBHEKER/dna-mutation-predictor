# Model / data files overview

The code is fully committed, but the **trained model** and the **large ESM-2
binary** are too big for Git and are intentionally excluded (see `.gitignore`).

## Required for the bot to run

| File                | What it is                          | Needed for |
|---------------------|-------------------------------------|------------|
| `data/scaler.pkl`   | Trained XGBoost + feature selector + scaler | real predictions |
| `data/variants.csv` | ClinVar missense variant index (1.3 MB)     | gene suggestions / `suggest` / `all` / `info` |

Both are **committed** in this repo, so predictions work out of the box.
If you ever replace the model, drop a new `scaler.pkl` (a
`(model, selector, scaler)` pickle) here.

## Optional heavy files (fast-path)

| File                  | Purpose                                   |
|-----------------------|-------------------------------------------|
| `data/esm_model.onnx` | ONNX-served ESM-2 encoder (~600 MB) — the "silent fast-path" wake |

If absent, the bot falls back to loading ESM-2 through transformers/torch
(needs internet on first download). You can export it with the ONNX export
step in `src/preprocess.py`.

## First-run behaviour

The launcher (`predict.bat` / `predict.sh`) warns you if
`scaler.pkl` or `variants.csv` are missing, but still starts the bot so you
can browse commands, suggestions, gene info and graphs. Predictions need
`scaler.pkl`.