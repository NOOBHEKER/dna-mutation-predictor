# Contributing

Thanks for considering a contribution. This project values reproducibility,
clear caveats, and small, reviewable changes.

## Setup

```bash
git clone https://github.com/NOOBHEKER/dna-mutation-predictor.git
cd dna-mutation-predictor
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
```

Run the offline checks:

```bash
pytest -q
ruff check .
python scripts/make_metrics.py
```

## What to contribute

Good first issues:

- Documentation fixes and clearer caveats.
- Tests for edge cases in HGVSp parsing or the config agent.
- Support for additional amino-acid notations.
- Improvements to the web dashboard that keep metrics honest.
- Better error messages for missing data or offline mode.

Please open an issue before large changes, especially anything that alters the
model, the frozen split, or the published metrics.

## Data and metric changes

- **Do not** regenerate `data/test_indices.pt` to improve a score. The frozen
  split is a contract with every previous result.
- If you add data, document provenance and licensing in `docs/DATA.md`.
- If metrics change, update `data/metrics.json`, the README table, the model
  card, and the dashboard (the dashboard reads JSON, so usually only the JSON
  and docs need changes).
- Never commit raw ClinVar dumps, ESM-2 ONNX weights, virtual environments, or
  API keys.

## Code style

- Python 3.10+ compatible.
- Keep heavy imports (`torch`, `transformers`, `streamlit`) lazy where practical.
- Use repository-relative paths via `src.paths.data_path`.
- Prefer clear names and small functions over clever one-liners.
- Run `ruff check .` before opening a PR.

## Tests

Add a test for every bug fix. Tests must be offline by default. If a test needs
a large optional dependency, guard it with `pytest.importorskip(...)`.

## Commits and pull requests

- Use a focused commit message, e.g. `fix: handle lowercase gene symbols`.
- Describe the motivation, the change, and how you tested it.
- Link the relevant issue.
- Keep the PR scoped; split unrelated changes.

## Code of conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

Do not report vulnerabilities in public issues. See [SECURITY.md](SECURITY.md).
