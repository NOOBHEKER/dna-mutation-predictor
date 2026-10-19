# Architecture

## Pipeline

```text
gene + HGVSp
    │
    ▼
┌──────────────────────┐
│ src/hgvsp.py         │  parse p.Arg175His / R175H / Arg175His
└──────────┬───────────┘
           │  wt residue, position, mutant residue
           ▼
┌──────────────────────┐
│ src/engine.py        │  UniProt gene -> accession -> canonical sequence
└──────────┬───────────┘
           │
           ▼
   128-aa window centred on the mutation
           │
     ┌─────┴─────┐
     ▼           ▼
  WT window   mutant window
     │           │
     ▼           ▼
  ESM-2 150M  ESM-2 150M
     │           │
 640-d WT    640-d mutant
     └─────┬─────┘
           ▼
   concat 1,280 features
           │
   SelectKBest(k=400)
           │
   StandardScaler
           │
   XGBoost (500 trees, depth 6, lr 0.05)
           │
           ▼
  ESM-2 probability
           │
           ├──────────────┐
           ▼              ▼
  population model    optional DNA views
  (if coverage)       (Evo2 / DNABERT-2)
           │              │
           └──────┬───────┘
                  ▼
       weighted blend + label
```

## Modules

| Module | Responsibility |
|---|---|
| `src/cli.py` | Animated terminal bot, REPL, commands, progress UI |
| `src/config_agent.py` | Loose input parsing, gap detection, suggestions |
| `src/predict.py` | Model loading, embedding, prediction, optional ONNX path |
| `src/preprocess.py` | Dataset construction, sequence windows, ESM-2 embedding |
| `src/engine.py` | Torch-free UniProt access |
| `src/hgvsp.py` | Dependency-free HGVSp parsing and amino-acid tables |
| `src/paths.py` | Repository-relative data paths |
| `src/model.py` | PyTorch MLP baseline (not the shipped head) |
| `src/train.py` | ESM-2 head training + frozen split |
| `src/evaluate.py` | Hold-out evaluation |
| `src/protein_graph.py` | 2D residue map, hydropathy, AlphaFold 3D viewer |
| `src/myvariant.py` | gnomAD / SIFT / PolyPhen cache access |
| `src/vep.py` | Ensembl VEP fallback |
| `src/evo2.py` | Optional Evo2-7B zero-shot DNA score |
| `src/dnabert.py` | Optional DNABERT-2 embedding score |
| `webapp/app.py` | Streamlit dashboard |

## Model artifacts

`data/scaler.pkl` is a plain pickle containing a 3-tuple:

```python
(XGBClassifier, SelectKBest(k=400), StandardScaler)
```

`src/predict.py` also accepts legacy 4- and 5-tuples so old artifacts keep
working. `data/tabular.pkl` is a dict:

```python
{
  "model": GradientBoostingClassifier,
  "model_lr": LogisticRegression,
  "n": 2724,
  "gb_neutral_prob": 0.9918,
  "feature_names": ["log_af", "af_missing", "sift", "sift_missing",
                    "polyphen", "pp_missing"],
}
```

## Design decisions

- **Frozen split, committed.** Evaluation is only meaningful if the test rows do
  not change between runs. `data/test_indices.pt` is the source of truth.
- **Repo-relative paths.** `src/paths.py` resolves everything from the repository
  root, so launchers work from any working directory.
- **Lazy heavy imports.** `src/predict.py` imports torch/transformers only when
  the ONNX file is absent. `src/hgvsp.py` and `src/config_agent.py` stay
  torch-free.
- **Graceful degradation.** UniProt, Evo2, DNABERT-2, AlphaFold, and the
  population caches are all optional at inference time. Failures fall back to
  the ESM-2 head.
- **No fabricated dashboard numbers.** `webapp/app.py` reads
  `data/metrics.json`, which is generated from the artifacts.

## Data flow for the dashboard

```text
data/metrics.json ──► webapp/app.py ──► Performance page
data/scaler.pkl   ──► webapp/app.py ──► Predict / Batch pages
UniProt REST      ──► sequence lookup
AlphaFold API     ──► 3D structure page
```

The dashboard does **not** use the population model; its predictions are the
ESM-2 head plus optional Evo2. This is intentional: it keeps the app usable
without the local annotation caches. The CLI uses the full blend when coverage
is available.
