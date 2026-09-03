# Copyright (c) 2026 Agastyas Shii. MIT License.
import os
import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from src.preprocess import get_uniprot_id, get_protein_sequence, parse_hgvsp, AA_MAP
import argparse, pickle

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
SCALER_PATH = "data/scaler.pkl"
ONNX_PATH = "data/esm_model.onnx"
WINDOW_SIZE = 128


class OnnxEsm:
    """ESM-2 encoder served by onnxruntime — fast wake without the full
    torch weight loader. Duck-typed for predict_with_steps via onnx_mode."""

    onnx_mode = True

    def __init__(self, path=ONNX_PATH, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or (os.cpu_count() or 4)
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.sess = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"])

    def __call__(self, input_ids, attention_mask):
        feed = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
        }
        return self.sess.run(None, feed)[0]

def load_model():
    with open(SCALER_PATH, 'rb') as f:
        model, selector, scaler = pickle.load(f)
    return model, selector, scaler

def predict_mutation(gene, wt_res_3, pos, mt_res_3, model, selector, scaler, tokenizer, esm_model, device):
    wt_res = AA_MAP.get(wt_res_3)
    mt_res = AA_MAP.get(mt_res_3)
    if not wt_res or not mt_res:
        return None, 0.0, "Invalid amino acid"

    uniprot_id = get_uniprot_id(gene)
    if not uniprot_id:
        return None, 0.0, f"Could not find UniProt ID for gene {gene}"

    seq = get_protein_sequence(uniprot_id)
    if not seq or pos > len(seq):
        return None, 0.0, f"Could not fetch sequence or position {pos} out of bounds"

    start = max(0, pos - 1 - WINDOW_SIZE)
    end = min(len(seq), pos + WINDOW_SIZE)
    rel_pos = (pos - 1) - start

    wt_window = seq[start:end]
    mt_list = list(wt_window)
    mt_list[rel_pos] = mt_res
    mt_window = "".join(mt_list)

    with torch.no_grad():
        inputs_wt = tokenizer(wt_window, return_tensors="pt", padding=True, truncation=True, max_length=514).to(device)
        inputs_mt = tokenizer(mt_window, return_tensors="pt", padding=True, truncation=True, max_length=514).to(device)

        out_wt = esm_model(**inputs_wt).last_hidden_state[0]
        out_mt = esm_model(**inputs_mt).last_hidden_state[0]

        emb_wt = out_wt[rel_pos + 1].cpu().numpy()
        emb_mt = out_mt[rel_pos + 1].cpu().numpy()

        feature = np.concatenate([emb_wt, emb_mt]).reshape(1, -1)
        feature_selected = selector.transform(feature)
        feature_scaled = scaler.transform(feature_selected)

        label = model.predict(feature_scaled)[0]
        prob = model.predict_proba(feature_scaled)[0]

    return int(label), float(prob[1]), "OK"

def parse_hgvsp_input(hgvsp_str):
    result = parse_hgvsp(hgvsp_str)
    if result[0] is None:
        return None, None, None
    return result

def predict_from_string(gene, hgvsp_str, model=None, selector=None, scaler=None, tokenizer=None, esm_model=None, device=None):
    if model is None:
        model, selector, scaler = load_model()
    if tokenizer is None:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
        esm_model.eval()

    wt_res_3, pos, mt_res_3 = parse_hgvsp(hgvsp_str)
    if wt_res_3 is None:
        return None, 0.0, "Error parsing HGVSp"

    return predict_mutation(gene, wt_res_3, pos, mt_res_3, model, selector, scaler, tokenizer, esm_model, device)


def predict_with_steps(gene, hgvsp_str, model=None, selector=None, scaler=None,
                       tokenizer=None, esm_model=None, device=None):
    """Generator version of predict_from_string that yields progress events.

    Yields (step_name, detail) tuples while working; returns
    (label, prob, message) via StopIteration.value when finished.
    """
    yield "Attach prediction engine", (
        "ensemble + ESM-2 ready" if model is not None and tokenizer is not None
        else "cold start")
    if model is None:
        yield "Load prediction engine", "XGB + LGBM + CatBoost ensemble"
        model, selector, scaler = load_model()

    if tokenizer is None:
        yield "Load ESM-2 language model", (
            "ONNX fast path" if os.path.exists(ONNX_PATH) else ESM_MODEL_NAME)
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        if esm_model is None and os.path.exists(ONNX_PATH):
            esm_model = OnnxEsm(ONNX_PATH)
        elif esm_model is None:
            esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
            esm_model.eval()

    yield "Parse variant notation", hgvsp_str
    wt_res_3, pos, mt_res_3 = parse_hgvsp(hgvsp_str)
    if wt_res_3 is None:
        return None, 0.0, f"Error parsing HGVSp '{hgvsp_str}'"

    wt_res = AA_MAP.get(wt_res_3)
    mt_res = AA_MAP.get(mt_res_3)
    if not wt_res or not mt_res:
        return None, 0.0, "Invalid amino acid"

    yield "Resolve gene -> UniProt", gene
    uniprot_id = get_uniprot_id(gene)
    if not uniprot_id:
        return None, 0.0, f"Could not find UniProt ID for gene {gene}"

    yield "Fetch reference protein sequence", uniprot_id
    seq = get_protein_sequence(uniprot_id)
    if not seq:
        return None, 0.0, f"Could not fetch sequence for {uniprot_id}"
    if pos > len(seq):
        return None, 0.0, (f"Position {pos} out of bounds "
                           f"(protein length {len(seq)})")

    yield "Build mutation window", f"{len(seq)} aa reference, p.{pos}"
    start = max(0, pos - 1 - WINDOW_SIZE)
    end = min(len(seq), pos + WINDOW_SIZE)
    rel_pos = (pos - 1) - start

    wt_window = seq[start:end]
    mt_list = list(wt_window)
    mt_list[rel_pos] = mt_res
    mt_window = "".join(mt_list)

    onnx_mode = getattr(esm_model, "onnx_mode", False)
    if onnx_mode:
        yield "ESM-2 embed (wild-type)", f"{end - start} residues \u00b7 ONNX"
        inputs_wt = tokenizer(wt_window, return_tensors="np", padding=True,
                              truncation=True, max_length=514)
        out_wt = esm_model(input_ids=inputs_wt["input_ids"],
                           attention_mask=inputs_wt["attention_mask"])
        emb_wt = np.asarray(out_wt[0, rel_pos + 1])

        yield "ESM-2 embed (mutant)", f"{wt_res_3}{pos}{mt_res_3} \u00b7 ONNX"
        inputs_mt = tokenizer(mt_window, return_tensors="np", padding=True,
                              truncation=True, max_length=514)
        out_mt = esm_model(input_ids=inputs_mt["input_ids"],
                           attention_mask=inputs_mt["attention_mask"])
        emb_mt = np.asarray(out_mt[0, rel_pos + 1])
    else:
        with torch.no_grad():
            yield "ESM-2 embed (wild-type)", f"{end - start} residues"
            inputs_wt = tokenizer(wt_window, return_tensors="pt",
                                  padding=True, truncation=True,
                                  max_length=514).to(device)
            out_wt = esm_model(**inputs_wt).last_hidden_state[0]
            emb_wt = out_wt[rel_pos + 1].cpu().numpy()

            yield "ESM-2 embed (mutant)", f"{wt_res_3}{pos}{mt_res_3}"
            inputs_mt = tokenizer(mt_window, return_tensors="pt",
                                  padding=True, truncation=True,
                                  max_length=514).to(device)
            out_mt = esm_model(**inputs_mt).last_hidden_state[0]
            emb_mt = out_mt[rel_pos + 1].cpu().numpy()

    yield "Feature select + scale", f"{emb_wt.size} dims -> selector"
    feature = np.concatenate([emb_wt, emb_mt]).reshape(1, -1)
    feature_selected = selector.transform(feature)
    feature_scaled = scaler.transform(feature_selected)

    yield "Ensemble vote", "XGB \u00b7 LGBM \u00b7 CAT"
    label = model.predict(feature_scaled)[0]
    prob = model.predict_proba(feature_scaled)[0]

    return int(label), float(prob[1]), "OK"


def main():
    parser = argparse.ArgumentParser(description="Predict the effect of a DNA mutation.")
    parser.add_argument("--gene", type=str, required=True, help="Gene symbol (e.g., TP53)")
    parser.add_argument("--hgvsp", type=str, required=True, help="HGVSp string (e.g., p.Cys61Gly)")
    args = parser.parse_args()

    print("Loading models...")
    model, selector, scaler = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    if os.path.exists(ONNX_PATH):
        esm_model = OnnxEsm(ONNX_PATH)
    else:
        esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
        esm_model.eval()

    label, prob, msg = predict_from_string(args.gene, args.hgvsp, model, selector, scaler, tokenizer, esm_model, device)

    if label is None:
        print(f"Error: {msg}")
    else:
        status = "Pathogenic" if label == 1 else "Benign"
        print(f"Mutation {args.hgvsp} in gene {args.gene}: {status} (Probability: {prob:.4f})")

if __name__ == "__main__":
    main()
