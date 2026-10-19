import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import MutationPredictor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, roc_auc_score)
import pickle
import numpy as np

from src.paths import data_path

PROCESSED_DATA_PATH = data_path("processed_data.pt")
MODEL_PATH = data_path("model.pt")
SCALER_PATH = data_path("scaler.pkl")
TEST_INDICES_PATH = data_path("test_indices.pt")
BATCH_SIZE = 32


def _load_checkpoint(path):
    """torch.load with the pre-2.6 default (full pickle) on every version."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path)

def compute_features(f):
    """Return features in the layout the saved model expects.

    - Old processed_data.pt is [wt, mt] (1280 dims): expand to
      [wt, mt, diff, product, abs_diff] (3200) for new models.
    - New processed_data.pt is already 3200 dims: return as-is.
    The final compatibility check happens against the selector's
    n_features_in_ below.
    """
    if f.shape[1] == 1280:
        h = 640
        emb_wt = f[:, :h]
        emb_mt = f[:, h:]
        diff = emb_wt - emb_mt
        prod = emb_wt * emb_mt
        abs_diff = np.abs(diff)
        return np.concatenate([emb_wt, emb_mt, diff, prod, abs_diff], axis=1)
    return f

def evaluate():
    data = _load_checkpoint(PROCESSED_DATA_PATH)
    raw = data['features'].numpy()
    labels = data['labels'].numpy()
    test_idx = _load_checkpoint(TEST_INDICES_PATH).numpy()

    # Build expanded features (same as preprocess.py now does)
    features = compute_features(raw)

    with open(SCALER_PATH, 'rb') as f:
        data_pkl = pickle.load(f)
    if len(data_pkl) == 5:
        model, selector, scaler, sc_extra, base_models = data_pkl
    elif len(data_pkl) == 4:
        model, selector, scaler, base_models = data_pkl
        sc_extra = None
    else:
        model, selector, scaler = data_pkl
        sc_extra, base_models = None, None

    # Backward compat: old selector trained on 1280-dim [wt, mt]
    expected = getattr(selector, "n_features_in_", None)
    if expected == 1280 and features.shape[1] == 3200:
        features = features[:, :1280]
    print(f"raw={raw.shape} features={features.shape} "
          f"selector_expects={expected} extras={sc_extra is not None}")

    X_test = selector.transform(features[test_idx])
    X_test_s = scaler.transform(X_test)
    if sc_extra is not None and isinstance(data, dict) \
            and data.get("extra") is not None:
        E = data["extra"]
        E = E.numpy() if hasattr(E, "numpy") else np.asarray(E)
        X_test_s = np.concatenate(
            [X_test_s, sc_extra.transform(E[test_idx])], axis=1)
    y_test = labels[test_idx]

    # Sklearn ensemble (stacker or voter)
    if base_models is not None:
        meta_test = np.column_stack([
            base_models['xgb'].predict_proba(X_test_s)[:, 1],
            base_models['lgbm'].predict_proba(X_test_s)[:, 1],
            base_models['cat'].predict_proba(X_test_s)[:, 1],
        ])
        preds = (model.predict_proba(meta_test)[:, 1] >= 0.5).astype(int)
        prob = model.predict_proba(meta_test)[:, 1]
    else:
        preds = model.predict(X_test_s)
        prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, prob)
    print(f"Ensemble: acc={acc:.4f} balanced_acc={bal:.4f} auc={auc:.4f} "
          f"(n={len(y_test)})")
    print(classification_report(y_test, preds, target_names=['Benign', 'Pathogenic']))

    # NN (optional: model.pt may be stale vs the selector — skip cleanly)
    nn_acc = None
    try:
        X_t = torch.tensor(X_test_s, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE)

        nn_model = MutationPredictor(X_test_s.shape[1])
        nn_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        nn_model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in loader:
                preds.extend(torch.sigmoid(nn_model(bx)).round().numpy())
                targets.extend(by.numpy())
        nn_acc = accuracy_score(targets, preds)
        print(f"\nNeural Net: {nn_acc:.4f}")
        print(classification_report(targets, preds,
                                    target_names=['Benign', 'Pathogenic']))
    except Exception as e:
        print(f"\nNeural Net: SKIPPED ({type(e).__name__}: {e})")

    best = max([a for a in (acc, nn_acc) if a is not None])
    print(f"=== BEST: {best:.4f} ===")

if __name__ == "__main__":
    evaluate()
