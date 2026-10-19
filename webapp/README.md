# Streamlit dashboard

Interactive UI for the ESM-2 protein head: single predictions, batch CSV,
metrics, protein views, and a methods page.

## Run

From the repository root:

```bash
pip install -r requirements-web.txt
streamlit run webapp/app.py
```

Or use the portable launcher:

```bat
start-web.bat
```

```bash
./start-web.sh
```

The app opens at <http://localhost:8501>.

## What it uses

- `data/scaler.pkl` — ESM-2 head (XGBoost + selector + scaler)
- `data/metrics.json` — measured metrics shown on the Performance page
- ESM-2 from Hugging Face (downloaded on first prediction)
- UniProt / AlphaFold APIs for sequences and structures
- Optional Evo2 if `NVIDIA_API_KEY` is set

The dashboard intentionally shows **ESM-2-only** performance. The CLI applies
the population blend when local annotations are available.

## Notes

- No patient data is uploaded anywhere by this app; sequences are fetched from
  public APIs by gene symbol.
- This is a research tool, not a clinical device.
- The Performance page will show stale numbers only if `data/metrics.json` is
  stale; regenerate it with `python scripts/make_metrics.py`.
