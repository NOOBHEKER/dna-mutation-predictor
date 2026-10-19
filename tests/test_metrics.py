"""Recompute the headline metric from the committed artifacts.

This is the strongest reproducibility check in the suite. It intentionally
requires torch, because the feature matrix is a torch archive; CI runs it in a
separate job. The test is skipped elsewhere.
"""
from __future__ import annotations

import json
import pickle

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from src.paths import DATA_DIR  # noqa: E402


def _load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path)


def test_recomputed_esm_metrics_match_published_json():
    data = _load(str(DATA_DIR / "processed_data.pt"))
    features = data["features"].numpy()
    labels = data["labels"].numpy().astype(int)
    test_idx = _load(str(DATA_DIR / "test_indices.pt")).numpy()

    with open(DATA_DIR / "scaler.pkl", "rb") as handle:
        model, selector, scaler = pickle.load(handle)

    x_test = scaler.transform(selector.transform(features[test_idx]))
    y_test = labels[test_idx]
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = json.loads((DATA_DIR / "metrics.json").read_text(encoding="utf-8"))
    esm = metrics["esm_head"]

    assert len(y_test) == esm["n_test"] == 559
    assert accuracy_score(y_test, pred) == pytest.approx(esm["accuracy"], abs=1e-6)
    assert roc_auc_score(y_test, prob) == pytest.approx(esm["auc"], abs=1e-6)
    assert int(pred.sum()) == esm["confusion"]["tp"] + esm["confusion"]["fp"]
