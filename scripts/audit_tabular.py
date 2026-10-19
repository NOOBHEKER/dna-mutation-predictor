"""Robustness audit of the tabular signal.

The extras are strong largely because gnomAD OBSERVATION differs by class
(98% of benign vs 32% of pathogenic). That is real (ClinVar
ascertainment + BA1/BS1 clinical criteria) but it is partly a dataset
artifact. This script quantifies how much of the performance survives
when the "missing" indicator flags are removed, i.e. when the model must
use the actual AF / SIFT / PolyPhen values among variants that HAVE them.

Usage:
    venv\\Scripts\\python scripts/audit_tabular.py
"""
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = np.load(os.path.join(ROOT, "data", "extras_matrix.npy"))
y = np.load(os.path.join(ROOT, "data", "labels_tabular.npy"))

tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                          random_state=42, stratify=y)

VALUES = [0, 2, 4]          # log_af, sift, polyphen
FLAGS = [1, 3, 5]           # af_missing, sift_missing, pp_missing

print(f"n={len(y)} test={len(te)}\n")


def run(name, X):
    m = GradientBoostingClassifier(random_state=42).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    print(f"{name:34s} acc={accuracy_score(y[te], (p >= .5).astype(int)):.4f} "
          f"auc={roc_auc_score(y[te], p):.4f}")


run("all 6 (values + flags)", E)
run("values only (no missing flags)", E[:, VALUES])

# Restrict to rows where AF is actually observed -- the fair test of
# "given the variant is in gnomAD, can we rank by frequency/conservation?"
af_ok = E[:, 1] == 0
X = E[af_ok]
yy = y[af_ok]
sub_tr, sub_te = train_test_split(np.arange(len(yy)), test_size=0.2,
                                  random_state=42, stratify=yy)
m = GradientBoostingClassifier(random_state=42).fit(X[sub_tr], yy[sub_tr])
p = m.predict_proba(X[sub_te])[:, 1]
print(f"\nAF-observed subset only (n={len(yy)}, "
      f"{int((yy == 0).sum())} benign / {int((yy == 1).sum())} path):")
print(f"{'  values only, no flags':34s} "
      f"acc={accuracy_score(yy[sub_te], (p >= .5).astype(int)):.4f} "
      f"auc={roc_auc_score(yy[sub_te], p):.4f}")

# Single-feature reference points.
for i, nm in ((0, "log10(AF)"), (2, "SIFT"), (4, "PolyPhen")):
    v = E[:, i]
    ok = ~np.isnan(v)
    if ok.sum() == len(v):
        lr = LogisticRegression(max_iter=2000).fit(v[tr].reshape(-1, 1), y[tr])
        pp = lr.predict_proba(v[te].reshape(-1, 1))[:, 1]
        print(f"single {nm:34s} auc={roc_auc_score(y[te], pp):.4f}")
print("DONE")
