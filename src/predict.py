import os
import pickle
import argparse

import numpy as np

from src.engine import get_uniprot_id, get_protein_sequence
from src.hgvsp import AA_MAP, parse_hgvsp
from src.paths import data_path

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
SCALER_PATH = data_path("scaler.pkl")
ONNX_PATH = data_path("esm_model.onnx")
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
        data = pickle.load(f)
    # Normalize every historical format to the 5-tuple:
    # (model, selector, scaler, sc_extra, base_models).
    if len(data) == 5:
        return tuple(data)
    if len(data) == 4:
        model, selector, scaler, base_models = data
        return model, selector, scaler, None, base_models
    model, selector, scaler = data
    return model, selector, scaler, None, None


def _attach_extras(feature_scaled, sc_extra, gene, wt_1, pos, mt_1):
    """Append scaled tabular extras (or skip when model predates them)."""
    if sc_extra is None:
        return feature_scaled, "esm-only"
    try:
        from src.vep import extra_for_variant
        evec, src = extra_for_variant(gene, wt_1, pos, mt_1)
        esc = sc_extra.transform(evec.reshape(1, -1))
        return np.concatenate([feature_scaled, esc], axis=1), f"extras:{src}"
    except Exception:
        return feature_scaled, "extras:unavailable"


def build_feature(emb_wt, emb_mt, selector=None):
    """Build feature vector matching what the selector was trained on.

    New models (retrained after the v3 upgrade): [wt, mt, diff,
    product, abs_diff] = 5 x 640 = 3200 dims.
    Old models (current scaler.pkl, 1280-dim): [wt, mt] only.
    The selector's n_features_in_ decides, so old saved models keep
    working until you retrain.
    """
    expected = getattr(selector, "n_features_in_", None) if selector is not None else None
    if expected is not None and expected <= 1280:
        return np.concatenate([emb_wt, emb_mt])
    diff = emb_wt - emb_mt
    prod = emb_wt * emb_mt
    abs_diff = np.abs(diff)
    return np.concatenate([emb_wt, emb_mt, diff, prod, abs_diff])


def _stack_predict(model, base_models, feature_scaled):
    """Predict using stacking meta-model with base model probabilities."""
    if base_models is None:
        # Not a stacking model — direct predict
        return model.predict_proba(feature_scaled)[0]
    meta_features = np.column_stack([
        base_models['xgb'].predict_proba(feature_scaled)[:, 1],
        base_models['lgbm'].predict_proba(feature_scaled)[:, 1],
        base_models['cat'].predict_proba(feature_scaled)[:, 1],
    ])
    return model.predict_proba(meta_features)[0]


TABULAR_PATH = data_path("tabular.pkl")
_TABULAR = None
# Weight on the tabular (gnomAD AF / SIFT / PolyPhen) view when it has
# real data. That model scores 0.969 acc / 0.993 AUC held-out versus
# ~0.81 for the ESM-2 head, so it leads; ESM still contributes an
# orthogonal opinion.
TABULAR_BLEND_WEIGHT = 0.80


def _tabular_model():
    """Lazy-load the tabular classifier (None when unavailable)."""
    global _TABULAR
    if _TABULAR is None and os.path.exists(TABULAR_PATH):
        try:
            with open(TABULAR_PATH, "rb") as f:
                _TABULAR = pickle.load(f)
        except Exception:
            _TABULAR = False
    return _TABULAR or None


def tabular_opinion(gene, wt_1, pos, mt_1, live=True):
    """Population/conservation view for one variant.

    Returns (prob, detail) or (None, reason). `prob` is None whenever the
    variant has no observed annotation -- the tabular model treats
    "not in gnomAD" as strongly pathogenic, so applying it to an
    unannotated variant would fabricate confidence. Callers must fall
    back to the sequence-based views in that case.
    """
    tab = _tabular_model()
    if tab is None:
        return None, "tabular model not loaded"
    try:
        from src.vep import _load_json as _vload
        from src.vep import MAP_PATH, CACHE_PATH, vep_key
        from src.myvariant import CACHE_PATH as MYV_CACHE
        from src.vep import features_from_rec
        key = f"{(gene or '').upper()}|{wt_1}{pos}{mt_1}"
        m = _vload(MAP_PATH).get(key)
        if not m:
            return None, "variant not in ClinVar map"
        rec = None
        if m.get("rs"):
            rec = _vload(MYV_CACHE).get(f"{m['rs']}:{m['alt']}")
        if rec is None:
            rec = _vload(CACHE_PATH).get(
                vep_key(m["chrom"], m["pos"], m["ref"], m["alt"]))
        if live and rec is None:
            from src.myvariant import fetch_batch, _extract_doc
            try:
                docs = fetch_batch([f"rs{m['rs']}"]) if m.get("rs") else None
                if docs:
                    for doc in docs.get(f"rs{m['rs']}", []):
                        rec = _extract_doc(doc, m["alt"])
                        if rec is not None:
                            break
            except Exception:
                rec = None
        if not rec:
            return None, "no annotation cached"
        # Coverage gate: require at least one OBSERVED signal.
        observed = [k for k in ("af", "sift", "polyphen")
                    if rec.get(k) is not None]
        if not observed:
            return None, "no observed signal (af/sift/polyphen all missing)"
        vec = features_from_rec(rec).reshape(1, -1)
        prob = float(tab["model"].predict_proba(vec)[0, 1])
        detail = (f"gnomAD/SIFT/PolyPhen via {','.join(observed)} "
                  f"(af={rec.get('af')})")
        return prob, detail
    except Exception as e:
        return None, f"tabular error: {type(e).__name__}"


def _apply_evo2(gene, wt_1, pos, mt_1, esm_prob):
    """Blend the sequence + population views into one probability.

    Views, in order of measured strength on the held-out split:
      tabular (gnomAD AF / SIFT / PolyPhen) 0.969 acc  -- gated on coverage
      ESM-2 head                             ~0.81 acc
      Evo2-7B (NVIDIA NIM, optional)         DNA LLR
      DNABERT-2 (local, optional)            DNA cosine sim

    Never raises; any view that is missing or unannotated is simply
    dropped and the remaining weights renormalise.
    """
    evo2_prob = dnabert_prob = tab_prob = None
    parts = [f"esm={esm_prob:.4f}"]

    tab_prob, tab_detail = tabular_opinion(gene, wt_1, pos, mt_1)
    if tab_prob is not None:
        parts.append(f"tab={tab_prob:.4f}")
    else:
        parts.append(f"tab=off({tab_detail})")

    try:
        from src.evo2 import evo2_opinion, EVO2_ENABLED
        if EVO2_ENABLED:
            evo2_prob, llr, _d = evo2_opinion(gene, wt_1, pos, mt_1)
            if evo2_prob is not None:
                parts.append(f"evo2={evo2_prob:.4f}|llr={llr:.2f}")
    except Exception:
        pass

    try:
        from src.dnabert import dnabert_opinion, DNABERT2_ENABLED
        if DNABERT2_ENABLED:
            dnabert_prob, _s, _d = dnabert_opinion(gene, wt_1, pos, mt_1)
            if dnabert_prob is not None:
                parts.append(f"dnabert={dnabert_prob:.4f}")
    except Exception:
        pass

    weights, probs = [], []
    if tab_prob is not None:
        weights.append(TABULAR_BLEND_WEIGHT)
        probs.append(tab_prob)
        weights.append(1.0 - TABULAR_BLEND_WEIGHT)
        probs.append(esm_prob)
    else:
        weights.append(1.0)
        probs.append(esm_prob)
    if evo2_prob is not None:
        weights.append(0.25)
        probs.append(evo2_prob)
    if dnabert_prob is not None:
        weights.append(0.15)
        probs.append(dnabert_prob)

    total_w = sum(weights)
    final_prob = sum(w * p for w, p in zip(weights, probs)) / total_w
    return final_prob, "|" + "|".join(parts)

def predict_mutation(gene, wt_res_3, pos, mt_res_3, model, selector, scaler,
                     tokenizer, esm_model, device, base_models=None,
                     sc_extra=None):
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

    onnx_mode = getattr(esm_model, "onnx_mode", False)
    if onnx_mode:
        inputs_wt = tokenizer(wt_window, return_tensors="np", padding=True,
                              truncation=True, max_length=514)
        out_wt = esm_model(input_ids=inputs_wt["input_ids"],
                           attention_mask=inputs_wt["attention_mask"])
        emb_wt = np.asarray(out_wt[0, rel_pos + 1])

        inputs_mt = tokenizer(mt_window, return_tensors="np", padding=True,
                              truncation=True, max_length=514)
        out_mt = esm_model(input_ids=inputs_mt["input_ids"],
                           attention_mask=inputs_mt["attention_mask"])
        emb_mt = np.asarray(out_mt[0, rel_pos + 1])
    else:
        import torch
        with torch.no_grad():
            inputs_wt = tokenizer(wt_window, return_tensors="pt", padding=True, truncation=True, max_length=514).to(device)
            inputs_mt = tokenizer(mt_window, return_tensors="pt", padding=True, truncation=True, max_length=514).to(device)

            out_wt = esm_model(**inputs_wt).last_hidden_state[0]
            out_mt = esm_model(**inputs_mt).last_hidden_state[0]

            emb_wt = out_wt[rel_pos + 1].cpu().numpy()
            emb_mt = out_mt[rel_pos + 1].cpu().numpy()

    feature = build_feature(emb_wt, emb_mt, selector).reshape(1, -1)
    feature_selected = selector.transform(feature)
    feature_scaled = scaler.transform(feature_selected)
    feature_scaled, _extra_src = _attach_extras(
        feature_scaled, sc_extra, gene, wt_res, pos, mt_res)

    prob = _stack_predict(model, base_models, feature_scaled)

    esm_prob = float(prob[1])
    final_prob, evo_seg = _apply_evo2(gene, wt_res, pos, mt_res, esm_prob)
    final_label = int(final_prob >= 0.5)
    return final_label, final_prob, "OK" + evo_seg

def parse_hgvsp_input(hgvsp_str):
    result = parse_hgvsp(hgvsp_str)
    if result[0] is None:
        return None, None, None
    return result

def load_esm_runtime(device=None, esm_model=None):
    """Load the ESM-2 tokenizer + encoder, preferring the ONNX fast path.

    torch/transformers are imported only when the ONNX file is absent, so
    an ONNX-only install never pays the torch import cost.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    if esm_model is None and os.path.exists(ONNX_PATH):
        esm_model = OnnxEsm(ONNX_PATH)
    if esm_model is None:
        import torch
        from transformers import EsmModel

        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
        esm_model.eval()
    return tokenizer, esm_model, device


def predict_from_string(gene, hgvsp_str, model=None, selector=None, scaler=None,
                        tokenizer=None, esm_model=None, device=None,
                        base_models=None, sc_extra=None):
    if model is None:
        model, selector, scaler, sc_extra, base_models = load_model()
    if tokenizer is None:
        tokenizer, esm_model, device = load_esm_runtime(device, esm_model)

    wt_res_3, pos, mt_res_3 = parse_hgvsp(hgvsp_str)
    if wt_res_3 is None:
        return None, 0.0, "Error parsing HGVSp"

    return predict_mutation(gene, wt_res_3, pos, mt_res_3, model, selector, scaler,
                            tokenizer, esm_model, device,
                            base_models=base_models, sc_extra=sc_extra)


def predict_with_steps(gene, hgvsp_str, model=None, selector=None, scaler=None,
                       tokenizer=None, esm_model=None, device=None,
                       base_models=None, sc_extra=None):
    """Generator version of predict_from_string that yields progress events.

    Yields (step_name, detail) tuples while working; returns
    (label, prob, message) via StopIteration.value when finished.
    """
    yield "Attach prediction engine", (
        "classifier + ESM-2 ready" if model is not None and tokenizer is not None
        else "cold start")
    if model is None:
        yield "Load prediction engine", "XGBoost head + population caches"
        model, selector, scaler, sc_extra, base_models = load_model()

    if tokenizer is None:
        yield "Load ESM-2 language model", (
            "ONNX fast path" if os.path.exists(ONNX_PATH) else ESM_MODEL_NAME)
        tokenizer, esm_model, device = load_esm_runtime(device, esm_model)

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
        import torch
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

    yield "Build features", f"matching selector ({getattr(selector, 'n_features_in_', '?')} dims in)"
    feature = build_feature(emb_wt, emb_mt, selector).reshape(1, -1)
    feature_selected = selector.transform(feature)
    feature_scaled = scaler.transform(feature_selected)
    feature_scaled, extra_src = _attach_extras(
        feature_scaled, sc_extra, gene, wt_res, pos, mt_res)
    yield "Tabular extras", f"gnomAD/SIFT/PolyPhen ({extra_src})"

    yield "Classifier", ("stacking ensemble" if base_models else "XGBoost head")
    prob = _stack_predict(model, base_models, feature_scaled)
    esm_prob = float(prob[1])

    yield "Evo2-7B second opinion", "DNA-level LLR via NVIDIA NIM"
    final_prob, evo_seg = _apply_evo2(gene, wt_res, pos, mt_res, esm_prob)
    if evo_seg:
        yield "Blend verdict", (
            f"ESM {esm_prob:.3f} + Evo2 -> {final_prob:.3f}")
    final_label = int(final_prob >= 0.5)

    return final_label, final_prob, "OK" + evo_seg


def main():
    parser = argparse.ArgumentParser(description="Predict the effect of a DNA mutation.")
    parser.add_argument("--gene", type=str, required=True, help="Gene symbol (e.g., TP53)")
    parser.add_argument("--hgvsp", type=str, required=True, help="HGVSp string (e.g., p.Cys61Gly)")
    args = parser.parse_args()

    print("Loading models...")
    model, selector, scaler, sc_extra, base_models = load_model()
    tokenizer, esm_model, device = load_esm_runtime()

    label, prob, msg = predict_from_string(args.gene, args.hgvsp, model, selector, scaler,
                                           tokenizer, esm_model, device,
                                           base_models=base_models,
                                           sc_extra=sc_extra)

    if label is None:
        print(f"Error: {msg}")
    else:
        status = "Pathogenic" if label == 1 else "Benign"
        print(f"Mutation {args.hgvsp} in gene {args.gene}: {status} "
              f"(Probability: {prob:.4f})")
        parts = dict(seg.split("=", 1) for seg in str(msg).split("|")
                     if "=" in seg)
        if "esm" in parts:
            print(f"  ESM-2 head     : {float(parts['esm']):.4f}")
        if "tab" in parts:
            print(f"  gnomAD/SIFT/PP : {parts['tab']}")
        if "evo2" in parts:
            print(f"  Evo2-7B        : {float(parts['evo2']):.4f} "
                  f"(DNA LLR {float(parts.get('llr', 0.0)):+.2f})")
        if "dnabert" in parts:
            print(f"  DNABERT-2      : {float(parts['dnabert']):.4f}")

if __name__ == "__main__":
    main()
