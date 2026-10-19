# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-10-19

### Added
- Portable, repo-relative data paths across the CLI, preprocessing, training,
  evaluation, and web dashboard.
- `src/hgvsp.py` — dependency-free HGVSp parsing shared by the pipeline, CLI,
  and tests.
- `src/paths.py`, `src/__main__.py` (`python -m src`).
- Streamlit dashboard (`webapp/app.py`) with real, measured metrics loaded from
  `data/metrics.json`.
- `scripts/make_metrics.py` — recomputes every published metric from committed
  artifacts.
- `scripts/export_onnx.py` — regenerates the optional ESM-2 ONNX fast path.
- Offline test suite (`tests/`).
- Documentation: model card, data provenance, architecture, reproduction guide,
  troubleshooting, contribution guide, security policy, and code of conduct.
- Portable `predict.bat` / `predict.sh` and `start-web.bat` / `start-web.sh`
  launchers.

### Changed
- Replaced hard-coded dashboard metrics and ROC/confusion curves with values
  measured from the shipped artifacts.
- Dashboard examples no longer include the unsupported `CFTR p.Phe508del`
  indel.
- CLI copy now describes the shipped XGBoost head instead of the unshipped
  stacking ensemble.
- Self-test plots are written to the system temp directory.
- ONNX export is documented and scripted instead of committing 569 MB of
  weights.

### Removed
- Stale deployment snapshot and training scratch files from the published tree.

### Fixed
- Prediction, training, evaluation, and dashboard paths no longer depend on the
  current working directory.
- `src.predict` imports torch/transformers lazily, so the ONNX path avoids the
  heavy import.
- `torch.load` compatibility across PyTorch versions.

[1.0.0]: https://github.com/NOOBHEKER/dna-mutation-predictor/releases/tag/v1.0.0
