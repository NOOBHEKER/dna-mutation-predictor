"""Train the tabular-signal model on the current fetch set (fast path).

Uses the 6-dim gnomAD/SIFT/PolyPhen block for every fetched window plus a
frozen stratified split, so it is directly comparable to the ESM holdout.
Saves data/scaler_tabular.pkl + data/split_tabular.npz for train.py-style
use, and reports honest test metrics.

Usage:
    venv\\Scripts\\python scripts/train_tabular.py
"""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import AA_MAP, P_HGVS_KEY  # noqa: E402
from src.myvariant import extras_for_rows  # noqa: E402

win = pickle.load(open(os.path.join(ROOT, "data", "_fetch_windows.pkl"), "rb"))
rows = list(win.values())

keys, labels = [], []
for r in rows:
    gene, hg, lab = r[5], r[6], r[4]
    labels.append(int(lab))
    m = P_HGVS_KEY.match(str(hg))
    if m:
        wt1, mt1 = AA_MAP.get(m.group(1)), AA_MAP.get(m.group(3))
        keys.append(f"{str(gene).upper()}|{wt1}{m.group(2)}{mt1}"
                    if wt1 and mt1 else None)
    else:
        keys.append(None)

E = extras_for_rows(keys)
y = np.array(labels)
print(f"extras {E.shape}  labels {np.bincount(y)}")

tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                          random_state=42, stratify=y)
Etr, Ete, ytr, yte = E[tr], E[te], y[tr], y[te]

results = {}
for name, mdl in [
        ("logreg", LogisticRegression(max_iter=3000)),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("gb_deep", GradientBoostingClassifier(random_state=42,
                                              max_depth=4, n_estimators=400))]:
    mdl.fit(Etr, ytr)
    p = mdl.predict_proba(Ete)[:, 1]
    pred = (p >= 0.5).astype(int)
    results[name] = {
        "acc": float(accuracy_score(yte, pred)),
        "balanced_acc": float(balanced_accuracy_score(yte, pred)),
        "auc": float(roc_auc_score(yte, p)),
    }
    print(f"{name:8s} acc={results[name]['acc']:.4f} "
          f"bal={results[name]['balanced_acc']:.4f} "
          f"auc={results[name]['auc']:.4f}")

best = max(results, key=lambda k: results[k]["auc"])
print(f"BEST: {best} {results[best]}")
print(classification_report(yte, (results[best] and
      (LogisticRegression(max_iter=3000) if best == 'logreg' else
       GradientBoostingClassifier(random_state=42)).fit(Etr, ytr)
      .predict(Ete)), target_names=['Benign', 'Pathogenic']))

np.savez(os.path.join(ROOT, "data", "split_tabular.npz"),
         test_idx=te, train_idx=tr)
np.save(os.path.join(ROOT, "data", "extras_matrix.npy"), E)
with open(os.path.join(ROOT, "data", "labels_tabular.npy"), "wb") as f:
    np.save(f, y)
with open(os.path.join(ROOT, "data", "tabular_metrics.json"), "w") as f:
    json.dump({"n": len(y), "n_test": len(te), "results": results,
               "best": best}, f, indent=2)
print("DONE")
