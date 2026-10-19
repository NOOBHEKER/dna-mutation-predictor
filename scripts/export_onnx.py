"""Export the ESM-2 encoder to ONNX for the optional fast path.

The exported pair is large (~570 MB total) and is intentionally NOT
committed to git. Run this once after cloning if you want the CLI to
skip the transformers/torch model load:

    python scripts/export_onnx.py

Outputs:
    data/esm_model.onnx
    data/esm_model.onnx.data   (external weights)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer, EsmModel  # noqa: E402

from src.paths import data_path  # noqa: E402

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"


def main():
    out = data_path("esm_model.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Loading {ESM_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    model.eval()

    dummy = tokenizer("ACDEFGHIKLMNPQRSTVWY", return_tensors="pt",
                      padding=True, truncation=True, max_length=1024)
    print(f"Exporting to {out} ...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        out,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )
    size = os.path.getsize(out) / 1e6
    print(f"done: {out} ({size:.1f} MB + external .data file)")


if __name__ == "__main__":
    main()
