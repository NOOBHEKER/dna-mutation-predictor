# Copyright (c) 2026 Agastyas Shii. MIT License.
import torch, numpy as np, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

data = torch.load("data/processed_data.pt")
raw = data['features'].numpy()
labels = data['labels'].numpy()
n = len(labels)

rng = np.random.RandomState(42)
indices = rng.permutation(n)
si = int(0.8 * n)
train_idx, test_idx = indices[:si], indices[si:]
torch.save(torch.tensor(test_idx), "data/test_indices.pt")

X_tr, X_te = raw[train_idx], raw[test_idx]
y_tr, y_te = labels[train_idx], labels[test_idx]
spw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)

sel = SelectKBest(mutual_info_classif, k=400)
sc = StandardScaler()
Xtr = sc.fit_transform(sel.fit_transform(X_tr, y_tr))
Xte = sc.transform(sel.transform(X_te))

xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spw, eval_metric='logloss', random_state=42)
lgbm = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spw, random_state=42, verbose=-1)
cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0)

xgb.fit(Xtr, y_tr)
lgbm.fit(Xtr, y_tr)
cat.fit(Xtr, y_tr)

vote = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)], voting='soft')
vote.fit(Xtr, y_tr)

xgb_acc = accuracy_score(y_te, xgb.predict(Xte))
lgbm_acc = accuracy_score(y_te, lgbm.predict(Xte))
cat_acc = accuracy_score(y_te, cat.predict(Xte))
vote_acc = accuracy_score(y_te, vote.predict(Xte))

print(f"XGB={xgb_acc:.4f} LGBM={lgbm_acc:.4f} CAT={cat_acc:.4f} VOTE={vote_acc:.4f}")

best = max([(xgb_acc, xgb, 'XGB'), (lgbm_acc, lgbm, 'LGBM'), (cat_acc, cat, 'CAT'), (vote_acc, vote, 'Vote')], key=lambda x: x[0])
print(f"BEST: {best[2]} = {best[0]:.4f}")

with open("data/scaler.pkl", "wb") as f:
    pickle.dump((best[1], sel, sc), f)

preds = best[1].predict(Xte)
print(classification_report(y_te, preds, target_names=['Benign', 'Pathogenic']))
print("DONE")
