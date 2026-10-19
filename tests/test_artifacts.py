"""Committed artifacts are present, loadable, and the expected shape."""
from __future__ import annotations

import json
import pickle

import pytest

from src.paths import DATA_DIR, ROOT, data_path


def test_repository_layout():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "src" / "predict.py").is_file()
    assert DATA_DIR.is_dir()


def test_small_artifacts_exist():
    for name in ("scaler.pkl", "tabular.pkl", "variants.csv",
                 "processed_data.pt", "test_indices.pt", "metrics.json"):
        path = DATA_DIR / name
        assert path.is_file(), f"missing committed artifact: {name}"
        assert path.stat().st_size > 0


def test_esm_head_pickle_shape():
    pytest.importorskip("xgboost")
    with open(data_path("scaler.pkl"), "rb") as handle:
        payload = pickle.load(handle)
    assert len(payload) == 3
    model, selector, scaler = payload
    assert hasattr(model, "predict_proba")
    assert getattr(selector, "k", None) == 400
    assert hasattr(scaler, "transform")


def test_population_metadata():
    with open(data_path("tabular.pkl"), "rb") as handle:
        payload = pickle.load(handle)
    assert set(payload) >= {"model", "model_lr", "feature_names", "n"}
    assert payload["n"] == 2724
    assert len(payload["feature_names"]) == 6


def test_metrics_json_schema():
    metrics = json.loads((DATA_DIR / "metrics.json").read_text(encoding="utf-8"))
    esm = metrics["esm_head"]
    assert esm["n_test"] == 559
    assert set(esm["confusion"]) == {"tn", "fp", "fn", "tp"}
    assert set(esm["roc"]) == {"fpr", "tpr"}
    assert len(esm["roc"]["fpr"]) == len(esm["roc"]["tpr"])
    assert "population_head" in metrics
    assert "blend" in metrics
