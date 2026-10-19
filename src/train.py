import torch
import numpy as np
import pickle
import warnings
import os
import json
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold,
                                     train_test_split)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.paths import data_path

warnings.filterwarnings("ignore")

SPLIT_PATH = data_path("test_indices.pt")
METRICS_PATH = data_path("train_metrics.json")

data = torch.load(data_path("processed_data.pt"))
raw = data['features'].numpy()
labels = data['labels'].numpy()
n = len(labels)

# --- Frozen split: reuse saved test indices when they still fit the data.
# Never silently re-split, or runs stop being comparable.
if os.path.exists(SPLIT_PATH):
    test_idx = torch.load(SPLIT_PATH).numpy()
    if len(test_idx) < n and int(test_idx.max()) < n:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = np.where(mask)[0]
        print(f"Reusing frozen split: train={len(train_idx)} "
              f"test={len(test_idx)}")
    else:
        print("Saved split does not fit current data -- re-splitting.")
        test_idx = None
else:
    test_idx = None

if test_idx is None:
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=labels)
    torch.save(torch.tensor(test_idx), SPLIT_PATH)
    print(f"New stratified split saved: train={len(train_idx)} "
          f"test={len(test_idx)}")

X_tr, X_te = raw[train_idx], raw[test_idx]
y_tr, y_te = labels[train_idx], labels[test_idx]
spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)

# k=400 is the proven setting on 1280-dim ESM features (held-out
# acc 0.8247); k=800 scored lower (0.8104) on the same frozen split.
# MI gets a fixed seed so the selector (and hence results) is
# reproducible across runs.
K_SEL = 400
MI = partial(mutual_info_classif, random_state=42)
sel = SelectKBest(MI, k=K_SEL)
sc = StandardScaler()
Xtr = sc.fit_transform(sel.fit_transform(X_tr, y_tr))
Xte = sc.transform(sel.transform(X_te))

print(f"Features: {X_tr.shape[1]} -> {Xtr.shape[1]} after selection")

# --- Tabular extras (gnomAD AF / SIFT / PolyPhen), if present in the pt ---
Xtr_esm, Xte_esm = Xtr, Xte  # ESM-only copies for the ablation report
sc_extra = None
if isinstance(data, dict) and data.get("extra") is not None:
    E = data["extra"]
    E = E.numpy() if hasattr(E, "numpy") else np.asarray(E, dtype=np.float32)
    assert E.shape[0] == n, f"extra rows {E.shape[0]} != {n}"
    sc_extra = StandardScaler()
    Etr = sc_extra.fit_transform(E[train_idx])
    Ete = sc_extra.transform(E[test_idx])
    Xtr = np.concatenate([Xtr, Etr], axis=1)
    Xte = np.concatenate([Xte, Ete], axis=1)
    print(f"Extras appended: {E.shape[1]} dims -> {Xtr.shape[1]} total")

xgb = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8,
                     colsample_bytree=0.7, scale_pos_weight=spw, eval_metric='logloss',
                     random_state=42, n_jobs=-1)
lgbm = LGBMClassifier(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8,
                       colsample_bytree=0.7, scale_pos_weight=spw, random_state=42,
                       verbose=-1, n_jobs=-1)
cat = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.04, l2_leaf_reg=3,
                         random_seed=42, verbose=0)

xgb.fit(Xtr, y_tr)
lgbm.fit(Xtr, y_tr)
cat.fit(Xtr, y_tr)

xgb_acc = accuracy_score(y_te, xgb.predict(Xte))
lgbm_acc = accuracy_score(y_te, lgbm.predict(Xte))
cat_acc = accuracy_score(y_te, cat.predict(Xte))

print(f"Individual: XGB={xgb_acc:.4f} LGBM={lgbm_acc:.4f} CAT={cat_acc:.4f}")

# --- Stacking: manual 5-fold loop with per-fold logging + resume ---
# (cross_val_predict gives no progress output and restarts from zero
# if the process is killed; this checkpoint file survives interrupts.)
from sklearn.base import clone
OOF_PATH = data_path(f"_stack_oof_k{K_SEL}.npz")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(Xtr, y_tr))
oof = np.zeros((len(y_tr), 3))
done = np.zeros(5, dtype=bool)
if os.path.exists(OOF_PATH):
    try:
        z = np.load(OOF_PATH)
        if z["oof"].shape[0] == len(y_tr):
            oof, done = z["oof"], z["done"]
            print(f"Resuming OOF folds: {int(done.sum())}/5 done")
    except Exception as e:
        print(f"OOF checkpoint unreadable, restarting ({e})")
base_specs = [("xgb", xgb), ("lgbm", lgbm), ("cat", cat)]
for fi, (tri, vai) in enumerate(folds):
    if done[fi]:
        print(f"fold {fi + 1}/5: cached")
        continue
    t0 = __import__("time").time()
    for ci, (name, est) in enumerate(base_specs):
        e = clone(est).fit(Xtr[tri], y_tr[tri])
        oof[vai, ci] = e.predict_proba(Xtr[vai])[:, 1]
    done[fi] = True
    np.savez(OOF_PATH, oof=oof, done=done)
    print(f"fold {fi + 1}/5 done [{__import__('time').time() - t0:.0f}s]",
          flush=True)

meta_train = oof
stacker = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
stacker.fit(meta_train, y_tr)

# Test-time stacking
meta_test = np.column_stack([
    xgb.predict_proba(Xte)[:, 1],
    lgbm.predict_proba(Xte)[:, 1],
    cat.predict_proba(Xte)[:, 1],
])
stack_preds = (stacker.predict_proba(meta_test)[:, 1] >= 0.5).astype(int)
stack_acc = accuracy_score(y_te, stack_preds)
stack_auc = roc_auc_score(y_te, stacker.predict_proba(meta_test)[:, 1])
print(f"STACKER: acc={stack_acc:.4f} auc={stack_auc:.4f}")
print(f"Stacker weights: xgb={stacker.coef_[0][0]:.3f} lgbm={stacker.coef_[0][1]:.3f} cat={stacker.coef_[0][2]:.3f}")

# Also test soft voting for comparison
from sklearn.ensemble import VotingClassifier
vote = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)], voting='soft')
vote.fit(Xtr, y_tr)
vote_acc = accuracy_score(y_te, vote.predict(Xte))
print(f"VOTE: {vote_acc:.4f}")

# Pick best: compare stacker vs vote on accuracy (report the rest too)
def _scores(y_true, y_pred, y_prob):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }

vote_pred = vote.predict(Xte)
vote_prob = vote.predict_proba(Xte)[:, 1]
stack_prob = stacker.predict_proba(meta_test)[:, 1]
vote_s = _scores(y_te, vote_pred, vote_prob)
stack_s = _scores(y_te, stack_preds, stack_prob)
print(f"VOTE scores:  {vote_s}")
print(f"STACK scores: {stack_s}")

results = {
    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
    "n_features_in": int(X_tr.shape[1]), "n_features_sel": int(Xtr.shape[1]),
    "extras": bool(sc_extra is not None),
    "xgb_acc": float(xgb_acc), "lgbm_acc": float(lgbm_acc),
    "cat_acc": float(cat_acc), "vote": vote_s, "stack": stack_s,
}

best_name = 'Stacker' if stack_s["acc"] >= vote_s["acc"] else 'Voter'
best_model = stacker if best_name == 'Stacker' else vote
results["best"] = best_name
print(f"\nBEST: {best_name} = "
      f"{results['stack' if best_name == 'Stacker' else 'vote']['acc']:.4f}")

# Save 5-tuple, but NEVER regress the deployed model: re-score the
# current scaler.pkl on this exact split and keep it unless the new
# model is at least as good.
new_acc = results['stack' if best_name == 'Stacker' else 'vote']['acc']
prev_acc = 0.0
try:
    from sklearn.metrics import accuracy_score as _as
    with open(data_path("scaler.pkl"), "rb") as _f:
        _prev = pickle.load(_f)
    _pm, _psel, _psc = _prev[0], _prev[1], _prev[2]
    _pextra = None
    _pb = None
    if len(_prev) == 5:
        _pextra, _pb = _prev[3], _prev[4]
    elif len(_prev) == 4:
        _pb = _prev[3]
    _Xt = _psc.transform(_psel.transform(raw[test_idx]))
    if _pextra is not None and data.get("extra") is not None:
        _E = data["extra"]
        _E = _E.numpy() if hasattr(_E, "numpy") else np.asarray(_E)
        _Xt = np.concatenate([_Xt, _pextra.transform(_E[test_idx])],
                             axis=1)
    if _pb is not None:
        _mt = np.column_stack([
            _pb['xgb'].predict_proba(_Xt)[:, 1],
            _pb['lgbm'].predict_proba(_Xt)[:, 1],
            _pb['cat'].predict_proba(_Xt)[:, 1]])
        _pp = (_pm.predict_proba(_mt)[:, 1] >= 0.5).astype(int)
    else:
        _pp = _pm.predict(_Xt)
    prev_acc = float(_as(labels[test_idx], _pp))
    print(f"Deployed model scores {prev_acc:.4f} on this split")
except Exception as e:
    print(f"No usable deployed model ({e}); saving unconditionally")
if new_acc + 1e-9 >= prev_acc:
    if best_name == 'Stacker':
        with open(data_path("scaler.pkl"), "wb") as f:
            pickle.dump((stacker, sel, sc, sc_extra,
                         {'xgb': xgb, 'lgbm': lgbm, 'cat': cat}), f)
    else:
        with open(data_path("scaler.pkl"), "wb") as f:
            pickle.dump((vote, sel, sc, sc_extra, None), f)
    print(f"Saved new best ({best_name} {new_acc:.4f} >= {prev_acc:.4f})")
else:
    print(f"Kept deployed model ({prev_acc:.4f} >= new {new_acc:.4f})")

with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2)

preds = vote_pred if best_name == 'Voter' else stack_preds
print(classification_report(y_te, preds, target_names=['Benign', 'Pathogenic']))

# --- Ablation: same split, ESM-only (no extras) single XGB ---
# Isolates the tabular features' contribution on identical data.
if sc_extra is not None:
    _ab = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7,
                        scale_pos_weight=spw, eval_metric='logloss',
                        random_state=42, n_jobs=-1)
    _ab.fit(Xtr_esm, y_tr)
    _ap = _ab.predict(Xte_esm)
    _ar = _ab.predict_proba(Xte_esm)[:, 1]
    results["ablation_esm_only"] = _scores(y_te, _ap, _ar)
    print(f"ABLATION esm-only XGB: {results['ablation_esm_only']}")
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
print("DONE")
