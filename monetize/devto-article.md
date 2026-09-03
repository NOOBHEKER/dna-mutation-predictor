# How I Built a DNA Mutation Predictor Using ESM-2 and XGBoost (Open Source)

**Tags:** `bioinformatics` `machinelearning` `python` `opensource`

---

I built a tool that tells you if a DNA mutation causes disease. No cloud API. No paid service. Just a Python CLI on your laptop.

Here's how it works — and how you can run it yourself.

## The Problem

When a single nucleotide changes in DNA (a missense mutation), it can alter a protein's function. Some changes cause disease. Most don't. ClinVar — the NIH's database of known variants — has ~2,800 classified missense mutations. I wanted to build a classifier that could predict pathogenicity from sequence alone.

## The Stack

- **ESM-2** (`facebook/esm2_t30_150M_UR50D`) — a 150M-parameter protein language model from Meta. It reads protein sequences and produces 640-dimensional embeddings that capture evolutionary and structural information.
- **XGBoost** — gradient boosted trees, trained on 40 selected features from the ESM-2 embeddings.
- **scikit-learn** — feature selection (`SelectKBest`) and scaling (`StandardScaler`).

No deep learning at inference time. The ESM-2 embeddings are computed once and fed into a lightweight XGBoost model. The result: **82.5% accuracy** on a 2,792-sample ClinVar dataset.

## How It Works

1. You input a protein symbol + mutation (e.g., `BRCA1 A1708E`)
2. The tool fetches the protein sequence from UniProt
3. ESM-2 produces a 640-dim embedding for the mutated sequence
4. XGBoost classifies: `PATHOGENIC` or `BENIGN`
5. You get a confidence score + a protein feature graph

## Running It

```bash
git clone https://github.com/NOOBHEKER/dna-mutation-predictor.git
cd dna-mutation-predictor
pip install -r requirements.txt
python -m src.cli
```

Or double-click `predict.bat` on Windows — it sets up everything automatically.

## What I Learned

- **ESM-2 embeddings contain enough signal** to classify pathogenicity without hand-crafted features
- **Feature selection matters:** 40 out of 640 dimensions outperformed the full embedding
- **CPU inference is slow** (~30s per prediction) but acceptable for research use
- **The hardest part was cleaning ClinVar data**, not training the model

## Try It

The project is open source under MIT license. Clone it, break it, improve it.

**GitHub:** https://github.com/NOOBHEKER/dna-mutation-predictor

If it's useful to your research, I'd appreciate a star or a coffee.
