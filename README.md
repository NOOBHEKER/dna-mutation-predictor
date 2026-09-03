<!-- python:markdown -->
# 🧬 DNA Mutation Effect Predictor — CLI Bot

Predicts whether a human **missense** mutation is **pathogenic** or **benign**
using embeddings from the **ESM-2** protein language model + a trained
**XGBoost** classifier, wrapped in an animated, easy-to-use terminal bot.

```
 ____  _   _    _    ____  _____   _   ____  _______
|  _ \| \ | |  / \  |  _ \|_   _| / \ |  _ \| ____|
| | | |  \| | / _ \ | |_) | | |  / _ \| |_) |  _|
| |_| | |\  |/ ___ \|  _ <  | | / ___ \  _ <| |___
|____/|_| \_/_/   \_\_| \_\ |_|/_/   \_\_| \_\_____|
```

## Quick start

```bat
:: Windows
predict.bat
```

```bash
# macOS / Linux
chmod +x predict.sh
./predict.sh
```

On first run the launcher creates a local `venv/`, installs dependencies
from `requirements.txt`, and starts the bot. No manual setup needed.

> The repo is small and fully committed, including the **trained model**
> (`data/scaler.pkl`) and the variant index — so predictions work out of
> the box. Only the **~600 MB ESM-2 ONNX fast-path binary** is not
> committed; if it is absent the bot falls back to loading ESM-2 through
> transformers/torch (needs internet on first download). See
> `data/README.md`.

## What the bot does

Type a mutation in a loose format and hit Enter. The "Config Agent" parses
it, fills in whatever is missing, then the model predicts.

| Try these               | Effect |
|-------------------------|--------|
| `TP53 R175H`            | Predict one mutation |
| `predict TP53 p.Arg175His` | Same, fully explicit |
| `p.Cys61Gly`            | Mutation only — gene asked next |
| `all TP53`              | Self-predict every known variant of a gene, ranked |
| `suggest BRCA1`         | List known ClinVar variants for a gene |
| `info TP53`             | Gene card: UniProt ID, length, variant stats |
| `genes`                 | Most-documented genes in the index |
| `graph TP53 R175H`      | Interactive 2D protein map (click residues) |
| `3d TP53 R175H`         | Interactive 3D structure (AlphaFold DB) |
| `summary`               | Session statistics |
| `help`, `exit`          | Help / quit |

Extras: fuzzy gene suggestions, tab autocomplete, live step-by-step
progress during inference, an animated bot mascot that tracks your cursor,
and a silent ONNX fast-path engine wake.

Self-test (no model load, no internet):
```bash
python -m src.cli --selftest
```

## Requirements

- Python **3.9+**
- Windows / macOS / Linux terminal (Windows Terminal or VS Code terminal
  recommended for full Unicode + colour)

Dependencies install automatically via the launcher. To install manually:

```bash
python -m pip install -r requirements.txt
```

## Project layout

```
predict.bat / predict.sh   portable launchers (create venv, install deps)
src/
  cli.py                   animated terminal bot (REPL, verdicts, demo)
  config_agent.py          "Config Agent" — parses loose input, gene index
  predict.py               ESM-2 + XGBoost prediction + ONNX fast path
  engine.py                torch-free UniProt access
  protein_graph.py         interactive 2D map + 3D structure
  download_data.py         download + filter ClinVar variants
  preprocess.py            extract ESM-2 embeddings (train data)
  train.py / evaluate.py   train & validate the classifier
data/                      model + variant index (see data/README.md)
```

## How the model works

1. You give a gene (e.g. `TP53`) and an HGVSp string (e.g. `p.Arg175His`).
2. The protein sequence is fetched from UniProt.
3. A ±128-residue window around the site is fed through **ESM-2** (150M
   params), producing 640-dim wild-type and mutant embeddings.
4. The concatenated 1280-dim vector is classified by a trained **XGBoost**
   model into **Pathogenic / Benign**.

Training data: ClinVar + UniProt human missense variants. Reported accuracy
≈ **82.5%** on a held-out test set.

Citation: *Lin et al.* "Evolutionary-scale prediction of atomic-level protein
structure with a language model." Science, 2023.

## Licence

MIT