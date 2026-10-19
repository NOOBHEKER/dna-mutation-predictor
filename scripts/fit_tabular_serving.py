"""Fit + persist the tabular (gnomAD/SIFT/PolyPhen) classifier for serving.

Saves data/tabular.pkl = {"model": fitted, "trained_on": n, "metrics": {...}}
The model is fit on ALL labelled rows (serving artifact); the honest
generalization numbers come from scripts/train_tabular.py's held-out split.

Usage:
    venv\\Scripts\\python scripts/fit_tabular_serving.py
"""
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

E = np.load(os.path.join(ROOT, "data", "extras_matrix.npy"))
y = np.load(os.path.join(ROOT, "data", "labels_tabular.npy"))

gb = GradientBoostingClassifier(random_state=42).fit(E, y)
lr = LogisticRegression(max_iter=3000).fit(E, y)

# What does the model say about a completely unknown variant (all-neutral
# extras)? Serving needs to know this to decide when to trust it.
neutral = np.zeros((1, E.shape[1]), dtype=np.float32)
neutral[0, 1] = neutral[0, 3] = neutral[0, 5] = 1.0  # all three "missing"
neutral[0, 0] = -8.0  # log10(0 + 1e-8) floor
gb_neutral = float(gb.predict_proba(neutral)[0, 1])

out = {
    "model": gb,
    "model_lr": lr,
    "n": int(len(y)),
    "gb_neutral_prob": gb_neutral,
    "feature_names": ["log_af", "af_missing", "sift", "sift_missing",
                      "polyphen", "pp_missing"],
}
with open(os.path.join(ROOT, "data", "tabular.pkl"), "wb") as f:
    pickle.dump(out, f)

print(f"fit on n={len(y)}  gb_neutral_prob={gb_neutral:.4f}")
print("saved data/tabular.pkl")
