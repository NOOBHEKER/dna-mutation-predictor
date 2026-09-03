# Copyright (c) 2026 Agastyas Shii. MIT License.
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import MutationPredictor
from sklearn.metrics import accuracy_score, classification_report
import os, pickle, numpy as np

PROCESSED_DATA_PATH = "data/processed_data.pt"
MODEL_PATH = "data/model.pt"
SCALER_PATH = "data/scaler.pkl"
TEST_INDICES_PATH = "data/test_indices.pt"
BATCH_SIZE = 32

def compute_features(f):
    h = f.shape[1] // 2
    return f[:, h:] - f[:, :h]

def evaluate():
    data = torch.load(PROCESSED_DATA_PATH)
    features = compute_features(data['features'].numpy())
    labels = data['labels'].numpy()
    test_idx = torch.load(TEST_INDICES_PATH).numpy()

    with open(SCALER_PATH, 'rb') as f:
        model, selector, scaler = pickle.load(f)

    X_test = selector.transform(features[test_idx])
    X_test_s = scaler.transform(X_test)
    y_test = labels[test_idx]

    # Sklearn ensemble
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"Ensemble: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['Benign', 'Pathogenic']))

    # NN
    X_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE)

    nn_model = MutationPredictor(X_test_s.shape[1])
    nn_model.load_state_dict(torch.load(MODEL_PATH))
    nn_model.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in loader:
            p.extend(torch.sigmoid(nn_model(bx)).round().numpy())
            l.extend(by.numpy())
    nn_acc = accuracy_score(l, p)
    print(f"\nNeural Net: {nn_acc:.4f}")
    print(classification_report(l, p, target_names=['Benign', 'Pathogenic']))

    best = max(acc, nn_acc)
    print(f"=== BEST: {best:.4f} ===")

if __name__ == "__main__":
    evaluate()
