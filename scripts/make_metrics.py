"""Recompute the metrics that are shown in the README and the web app.

Reads the committed evaluation artifacts (``data/processed_data.pt``,
``data/test_indices.pt``, ``data/scaler.pkl``) and writes
``data/metrics.json``. Run from the repo root:

    python scripts/make_metrics.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.paths import data_path  # noqa: E402


def _load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path)


def _roc_points(y_true, y_prob, max_points=60):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int)
    return {
        "fpr": [round(float(v), 6) for v in fpr[idx]],
        "tpr": [round(float(v), 6) for v in tpr[idx]],
    }


def main():
    data = _load(data_path("processed_data.pt"))
    features = data["features"].numpy()
    labels = data["labels"].numpy().astype(int)
    test_idx = _load(data_path("test_indices.pt")).numpy()

    with open(data_path("scaler.pkl"), "rb") as f:
        payload = pickle.load(f)
    if len(payload) == 5:
        model, selector, scaler, _sc_extra, _base = payload
    elif len(payload) == 4:
        model, selector, scaler, _base = payload
    else:
        model, selector, scaler = payload

    x_test = scaler.transform(selector.transform(features[test_idx]))
    y_test = labels[test_idx]
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    importance = getattr(model, "feature_importances_", None)
    top = np.argsort(importance)[::-1][:20].tolist() if importance is not None else []

    metrics = {
        "generated_by": "scripts/make_metrics.py",
        "esm_head": {
            "description": "ESM-2 150M embeddings -> SelectKBest -> scaler -> XGBoost",
            "n_test": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 6),
            "auc": round(float(roc_auc_score(y_test, y_prob)), 6),
            "precision": round(float(precision_score(y_test, y_pred)), 6),
            "recall": round(float(recall_score(y_test, y_pred)), 6),
            "f1": round(float(f1_score(y_test, y_pred)), 6),
            "specificity": round(float(tn / (tn + fp)), 6),
            "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 6),
            "confusion": {"tn": int(tn), "fp": int(fp),
                          "fn": int(fn), "tp": int(tp)},
            "roc": _roc_points(y_test, y_prob),
            "model": {
                "type": type(model).__name__,
                "n_estimators": getattr(model, "n_estimators", None),
                "max_depth": getattr(model, "max_depth", None),
                "learning_rate": getattr(model, "learning_rate", None),
            },
            "features": {
                "selector": type(selector).__name__,
                "selected": int(getattr(selector, "k", 0) or 0),
                "input_dim": int(getattr(selector, "n_features_in_", 0) or 0),
            },
            "importance_top": [
                {"dim": int(i), "importance": round(float(importance[i]), 6)}
                for i in top
            ],
        },
        "blend": {
            "population_weight": 0.80,
            "esm_weight": 0.20,
            "note": ("The served CLI blends the population view with the ESM-2 "
                     "view when the variant has gnomAD/SIFT/PolyPhen data; the "
                     "blend itself has not been scored end-to-end on a hold-out."),
        },
    }

    tab_path = Path(data_path("tabular_metrics.json"))
    if tab_path.exists():
        tab = json.loads(tab_path.read_text(encoding="utf-8"))
        best = tab.get("best")
        metrics["population_head"] = {
            "description": "gnomAD AF / SIFT / PolyPhen tabular model",
            "best": best,
            "n": tab.get("n"),
            "n_test": tab.get("n_test"),
            "results": tab.get("results", {}),
        }

    out_path = Path(data_path("metrics.json"))
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(metrics["esm_head"], indent=2)[:1200])


if __name__ == "__main__":
    main()
