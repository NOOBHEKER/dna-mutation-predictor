# Troubleshooting

## Python / installation

**`ModuleNotFoundError: No module named 'src'`**
Run commands from the repository root, or use the launchers. The package is not
installed globally by default.

**Torch installation fails or is huge**
Use the CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Python 3.13 / 3.14 issues**
Some dependencies lag behind the newest Python. Use 3.11 or 3.12 in a fresh venv.

**Windows: `python` opens the Microsoft Store**
Install Python from python.org and tick “Add Python to PATH”, or use `py -3.12`.

## Model / data

**First prediction hangs or downloads a lot**
ESM-2 150M (~600 MB) is being downloaded from Hugging Face. This is expected on
first use. Subsequent runs use the cache.

**`data/esm_model.onnx` is missing**
That is normal. The ONNX encoder is intentionally not committed. Either ignore
it (the transformers path is used) or run:

```bash
python scripts/export_onnx.py
```

**`UniProt ID not found` / `Could not fetch sequence`**
The CLI needs internet access to `rest.uniprot.org`. Check connectivity, VPN, or
proxy settings. Gene symbols are case-insensitive but must be current.

**`Could not parse HGVSp string`**
Use a missense substitution such as `p.Arg175His`, `Arg175His`, or `R175H`.
Nonsense (`Ter`), frameshift (`fs`), and indel (`del`/`ins`) variants are out of
scope.

**Population view seems missing**
The coordinate map covers the 66k-variant ClinVar sample, not all of ClinVar.
Variants outside it fall back to ESM-2 only. This is expected and safe.

## CLI / terminal

**Graph or 3D window does not open**
`src/protein_graph.py` uses matplotlib’s TkAgg backend. On a headless server,
use the Streamlit dashboard instead, or install Tk support for your Python.

**Garbled characters / emoji on Windows**
Use Windows Terminal and a UTF-8 capable font. The CLI reconfigures stdout to
UTF-8, but the terminal must render it.

**`--selftest` writes images where?**
To the system temporary directory. The paths are printed at the end.

## Streamlit

**Dashboard shows old metrics**
The app reads `data/metrics.json`. Re-run `python scripts/make_metrics.py` and
reload the page. Streamlit may cache modules; stop and restart the server.

**Dashboard prediction fails but CLI works**
The dashboard uses the transformers ESM-2 path (and optional Evo2). Ensure
`torch`, `transformers`, and internet access are available. The dashboard does
not use the population model by design.

**Port already in use**

```bash
streamlit run webapp/app.py --server.port 8502
```

## Tests

**`tests/test_metrics.py` is skipped**
Torch is not installed in that environment. This is intentional; the test is
skipped gracefully so lightweight environments can still run the suite. Install
CPU torch if you want to exercise it.

**Metrics differ slightly**
Tiny floating-point differences across BLAS/library versions are expected.
Large differences mean the artifacts or split were modified.

## Optional Evo2 / DNABERT-2

**Evo2 returns nothing**
Set `NVIDIA_API_KEY` in the environment. Free-tier keys work but are
rate-limited. `EVO2_DISABLE=1` disables the DNA API view cleanly.

**DNABERT-2 is slow or unavailable**
It downloads a separate model and is optional. Predictions fall back to ESM-2
without it.

## Still stuck?

Open an issue with:

- OS and Python version
- the exact command
- the full traceback
- whether `data/scaler.pkl`, `data/variants.csv`, and `data/processed_data.pt`
  are present
- the output of `python -m src.cli --selftest`

Do **not** paste API keys or patient data.
