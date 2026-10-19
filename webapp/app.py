import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch
import pickle
import requests
import pandas as pd
import sys
import re
import io
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="DNA Mutation Predictor", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}
REVERSE_AA = {v: k for k, v in AA_MAP.items()}
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
SCALER_PATH = str(ROOT_DIR / "data" / "scaler.pkl")
METRICS_PATH = ROOT_DIR / "data" / "metrics.json"
try:
    METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    METRICS = {}
ESM_METRICS = METRICS.get("esm_head", {})
WINDOW_SIZE = 128
MAX_UPLOAD_MB = 10
MAX_BATCH_ROWS = 500

EXAMPLES = [
    ("TP53", "p.Cys61Gly", "Tumor suppressor — DNA binding domain", "pathogenic"),
    ("TP53", "p.Arg175His", "Hotspot mutation in DNA-binding domain", "pathogenic"),
    ("TP53", "p.Arg50Gln", "Likely benign polymorphism", "benign"),
    ("BRCA1", "p.Gly1738Arg", "DNA repair — RING domain", "pathogenic"),
    ("KRAS", "p.Gly12Val", "Oncogene — GTPase domain", "pathogenic"),
    ("EGFR", "p.Leu858Arg", "Lung cancer — kinase domain", "pathogenic"),
    ("BRAF", "p.Val600Glu", "Melanoma — kinase domain", "pathogenic"),
]

GENE_COLORS = {
    "TP53": "#ef476f", "BRCA1": "#7b2ff7", "BRCA2": "#7b2ff7",
    "KRAS": "#ffd166", "EGFR": "#4cc9f0", "BRAF": "#f72585",
    "CFTR": "#06d6a0", "MYC": "#ff007f", "PTEN": "#2b66ff",
    "PIK3CA": "#00bb7f", "RB1": "#fb2c36", "APC": "#ffd166",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Anton&display=swap');

:root {
    --bg: #000;
    --bg-card: #0a0a0f;
    --bg-card-hover: #111118;
    --blue: #2b66ff;
    --blue-glow: rgba(43,102,255,0.25);
    --pink: #ff007f;
    --green: #00bb7f;
    --red: #fb2c36;
    --orange: #ffd166;
    --cyan: #4cc9f0;
    --purple: #7b2ff7;
    --text: #fff;
    --text-dim: #888;
    --text-muted: #555;
    --border: #1a1a2e;
    --border-hover: #2b66ff;
}

* { box-sizing: border-box; }

.stApp {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
}
.stApp > header { background: transparent !important; }
.block-container { padding-top: 1rem !important; max-width: 1200px !important; }

h1, h2, h3, h4, h5, h6, p, span, label, div, li, td, th {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
code, pre, .stCode {
    font-family: 'IBM Plex Mono', monospace !important;
    background: #0a0a12 !important;
    border-radius: 4px !important;
}

/* ─── ANIMATIONS ─── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px var(--blue-glow); }
    50% { box-shadow: 0 0 40px var(--blue-glow), 0 0 60px rgba(43,102,255,0.1); }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}
.fade-in { animation: fadeInUp 0.6s ease-out both; }
.fade-in-d1 { animation-delay: 0.1s; }
.fade-in-d2 { animation-delay: 0.2s; }
.fade-in-d3 { animation-delay: 0.3s; }
.fade-in-d4 { animation-delay: 0.4s; }

/* ─── HERO ─── */
.hero-banner {
    background: linear-gradient(135deg, #000 0%, #0a0a2e 30%, #0a1a1a 70%, #000 100%);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 3.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(43,102,255,0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(255,0,127,0.04) 0%, transparent 50%);
    animation: pulseGlow 4s ease-in-out infinite;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(43,102,255,0.015) 2px, rgba(43,102,255,0.015) 4px);
    pointer-events: none;
}
.hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--blue);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    position: relative;
    z-index: 1;
}
.hero-title {
    font-family: 'Anton', sans-serif !important;
    font-size: clamp(2rem, 5vw, 3.5rem) !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em;
    line-height: 0.95;
    background: linear-gradient(135deg, #fff 0%, #2b66ff 50%, #ff007f 100%);
    background-size: 200% 200%;
    animation: gradientShift 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem !important;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    font-size: 1.05rem !important;
    color: var(--text-dim) !important;
    font-weight: 300 !important;
    position: relative;
    z-index: 1;
    max-width: 600px;
}
.hero-dna {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 4px;
    position: relative;
    z-index: 1;
    margin-top: 1rem;
    opacity: 0.5;
}

/* ─── SECTION HEADERS (playarc // style) ─── */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--blue);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 1rem !important;
}

/* ─── STAT CARDS ─── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.15s cubic-bezier(.4,0,.2,1);
    position: relative;
    overflow: hidden;
}
.stat-card:hover {
    border-color: var(--blue);
    box-shadow: inset 0 0 0 1px var(--blue-glow), 0 0 30px var(--blue-glow);
    transform: translateY(-2px);
}
.stat-card .stat-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    display: block;
    animation: float 3s ease-in-out infinite;
}
.stat-card .stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--blue);
    line-height: 1.2;
}
.stat-card .stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.3rem;
}

/* ─── PIPELINE STEPS ─── */
.pipeline-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    position: relative;
    transition: all 0.15s cubic-bezier(.4,0,.2,1);
    display: flex;
    align-items: center;
    gap: 1rem;
}
.pipeline-step:hover {
    border-color: var(--blue);
    box-shadow: inset 0 0 0 1px var(--blue-glow);
    background: var(--bg-card-hover);
}
.pipeline-step .step-num {
    font-family: 'IBM Plex Mono', monospace;
    min-width: 28px;
    height: 28px;
    background: var(--blue);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    color: #000;
    flex-shrink: 0;
}
.pipeline-step .step-content { flex: 1; }
.pipeline-step .step-title {
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.15rem;
}
.pipeline-step .step-desc {
    font-size: 0.75rem;
    color: var(--text-dim);
}

/* ─── EXAMPLE CARDS ─── */
.example-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    transition: all 0.15s cubic-bezier(.4,0,.2,1);
    cursor: pointer;
    height: 100%;
}
.example-card:hover {
    border-color: var(--border-hover);
    box-shadow: inset 0 0 0 1px var(--blue-glow);
    transform: translateY(-1px);
}
.example-card .gene-name {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
}
.example-card .mutation {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
}
.example-card .desc {
    font-size: 0.7rem;
    color: var(--text-muted);
}

/* ─── RESULT CARD ─── */
.result-card {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out;
}
.result-card.pathogenic {
    background: linear-gradient(145deg, #0a0008 0%, #1a0010 50%, #0a0008 100%);
    border: 1px solid rgba(251,44,54,0.3);
    box-shadow: inset 0 0 0 1px rgba(251,44,54,0.1), 0 0 40px rgba(251,44,54,0.08);
}
.result-card.benign {
    background: linear-gradient(145deg, #000a08 0%, #001a10 50%, #000a08 100%);
    border: 1px solid rgba(0,187,127,0.3);
    box-shadow: inset 0 0 0 1px rgba(0,187,127,0.1), 0 0 40px rgba(0,187,127,0.08);
}
.result-card .result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-card.pathogenic .result-label { color: var(--red); }
.result-card.benign .result-label { color: var(--green); }
.result-card .result-gene {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-dim);
}
.result-card .result-prob {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-card.pathogenic .result-prob { color: var(--red); }
.result-card.benign .result-prob { color: var(--green); }

/* ─── SECTION DIVIDER ─── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), var(--blue), var(--border), transparent);
    margin: 2rem 0;
    opacity: 0.5;
}

/* ─── CORNER MARKERS (blueprint aesthetic) ─── */
.blueprint-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    position: relative;
}
.blueprint-card::before {
    content: '';
    position: absolute;
    top: 8px; left: 8px;
    width: 10px; height: 10px;
    border-top: 1px solid var(--blue);
    border-left: 1px solid var(--blue);
    opacity: 0.6;
}
.blueprint-card::after {
    content: '';
    position: absolute;
    bottom: 8px; right: 8px;
    width: 10px; height: 10px;
    border-bottom: 1px solid var(--blue);
    border-right: 1px solid var(--blue);
    opacity: 0.6;
}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050508 0%, #0a0a12 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem !important; }
section[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.15s ease !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(43,102,255,0.08) !important;
    border-color: rgba(43,102,255,0.2) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: rgba(43,102,255,0.12) !important;
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    box-shadow: inset 0 0 0 1px var(--blue-glow) !important;
}

/* ─── METRIC OVERRIDE ─── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--blue) !important;
    box-shadow: inset 0 0 0 1px var(--blue-glow) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-dim) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800 !important;
}

/* ─── INPUTS ─── */
.stTextInput > div > div > input,
.stSelectbox > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    transition: border-color 0.15s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--blue) !important;
    box-shadow: inset 0 0 0 1px var(--blue-glow) !important;
}

/* ─── FILE UPLOADER ─── */
.stFileUploader {
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
.stFileUploader:hover {
    border-color: var(--blue) !important;
}

/* ─── BUTTONS ─── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.08em !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
    border: 1px solid var(--border) !important;
}
.stButton > button:hover {
    border-color: var(--blue) !important;
    box-shadow: inset 0 0 0 1px var(--blue-glow) !important;
}
.stButton > button[kind="primary"] {
    background: var(--blue) !important;
    color: #000 !important;
    border-color: var(--blue) !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #3a75ff !important;
    box-shadow: 0 0 20px var(--blue-glow) !important;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue); }

/* ─── FOOTER ─── */
.footer-section {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.footer-badges {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.tech-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--text-dim);
    background: var(--bg-card);
}
.tech-badge:hover {
    border-color: var(--blue);
    color: var(--blue);
}
.footer-credits {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


def _go_predict(gene: str, hgvsp: str):
    """on_click callback: prefill Predict inputs and navigate there.

    Must be a callback (not inline button-body code): the sidebar radio
    (key="nav_radio") is instantiated before page content runs, so setting
    its session-state value in the script body raises StreamlitAPIException.
    Callbacks run before any widget instantiates, so this is safe.
    Button clicks already trigger a rerun, so no st.rerun() needed.
    """
    st.session_state["gene_input"] = gene
    st.session_state["hgvsp_input"] = hgvsp
    st.session_state["nav_radio"] = "Predict"


def _fill_predict(gene: str, hgvsp: str):
    """on_click callback for the Predict page's own quick-example buttons."""
    st.session_state["gene_input"] = gene
    st.session_state["hgvsp_input"] = hgvsp


def _push_history(gene: str, hgvsp: str, status: str, prob: float):
    hist = st.session_state.get("history", [])
    hist.insert(0, {"gene": gene, "hgvsp": hgvsp, "status": status, "prob": round(prob, 4)})
    st.session_state["history"] = hist[:10]


def parse_hgvsp(hgvsp: str):
    try:
        change = hgvsp[2:] if hgvsp.startswith("p.") else hgvsp
        m = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+|Ter|=|fs|del|ins)", change)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
        m = re.match(r"([A-Za-z]+)([0-9]+)([A-Za-z]+)", change)
        if m:
            return m.group(1), int(m.group(2)), m.group(3)
    except Exception:
        pass
    return None, None, None


@st.cache_resource(show_spinner=False)
def load_pickle_model():
    with open(SCALER_PATH, "rb") as f:
        payload = pickle.load(f)
    # Accept legacy 4-/5-tuples (model, selector, scaler, ...).
    model, selector, scaler = payload[:3]
    return model, selector, scaler


@st.cache_resource(show_spinner=False)
def load_esm():
    from transformers import AutoTokenizer, EsmModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
    esm_model.eval()
    return tokenizer, esm_model, device


@st.cache_data(ttl=86400, show_spinner=False)
def get_uniprot_id(gene: str):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&fields=accession&format=json"
    try:
        data = requests.get(url, timeout=15).json()
        results = data.get("results", [])
        if results:
            return results[0]["primaryAccession"]
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_protein_sequence(uniprot_id: str):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return "".join(resp.text.splitlines()[1:])
    except Exception:
        pass
    return None


def _parse_evo2_msg(msg):
    """Parse '|esm=..|evo2=..|llr=..' segments from a prediction msg.

    Returns dict with float values; empty when Evo2 was offline.
    """
    info = {}
    for seg in str(msg or "").split("|"):
        if "=" in seg:
            k, _, v = seg.partition("=")
            try:
                info[k.strip()] = float(v)
            except ValueError:
                pass
    return info


def run_prediction(gene: str, hgvsp_str: str):
    wt3, pos, mt3 = parse_hgvsp(hgvsp_str)
    if wt3 is None:
        return None, 0.0, "Could not parse HGVSp string."
    wt_aa = AA_MAP.get(wt3)
    mt_aa = AA_MAP.get(mt3)
    if not wt_aa or not mt_aa:
        return None, 0.0, f"Unknown residue: {wt3} or {mt3}."
    uniprot_id = get_uniprot_id(gene)
    if not uniprot_id:
        return None, 0.0, f"UniProt ID not found for gene {gene}."
    seq = get_protein_sequence(uniprot_id)
    if not seq or pos > len(seq):
        return None, 0.0, f"Position {pos} out of range (sequence length {len(seq)})."
    model, selector, scaler = load_pickle_model()
    tokenizer, esm_model, device = load_esm()
    start = max(0, pos - 1 - WINDOW_SIZE)
    end = min(len(seq), pos + WINDOW_SIZE)
    rel_pos = (pos - 1) - start
    wt_window = seq[start:end]
    mt_list = list(wt_window)
    mt_list[rel_pos] = mt_aa
    mt_window = "".join(mt_list)
    with torch.no_grad():
        tok_wt = tokenizer(wt_window, return_tensors="pt", padding=True, truncation=True, max_length=514).to(device)
        tok_mt = tokenizer(mt_window, return_tensors="pt", padding=True, truncation=True, max_length=514).to(device)
        out_wt = esm_model(**tok_wt).last_hidden_state[0]
        out_mt = esm_model(**tok_mt).last_hidden_state[0]
        emb_wt = out_wt[rel_pos + 1].cpu().numpy()
        emb_mt = out_mt[rel_pos + 1].cpu().numpy()
        feature = np.concatenate([emb_wt, emb_mt]).reshape(1, -1)
        feature_sel = selector.transform(feature)
        feature_sc = scaler.transform(feature_sel)
        esm_prob = float(model.predict_proba(feature_sc)[0][1])
    # Evo2-7B second opinion (DNA view) blended with ESM-2 (protein view).
    # Any Evo2 failure falls back to ESM-only -- never breaks prediction.
    final_prob, evo_seg = esm_prob, ""
    try:
        from src.evo2 import evo2_opinion, blend_probs
        evo2_prob, llr, _detail = evo2_opinion(gene, wt_aa, pos, mt_aa)
        if evo2_prob is not None:
            final_prob = blend_probs(esm_prob, evo2_prob)
            evo_seg = (f"|esm={esm_prob:.4f}|evo2={evo2_prob:.4f}"
                       f"|llr={llr:.2f}")
    except Exception:
        pass
    label = int(final_prob >= 0.5)
    return label, final_prob, "OK" + evo_seg


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_alphafold_coords(uniprot_id: str):
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    resp = requests.get(api_url, timeout=20, verify=False)
    if resp.status_code != 200:
        return None, None, None, None, f"AlphaFold API returned status {resp.status_code}"
    info = resp.json()[0]
    pdb_url = info.get("pdbUrl") or info.get("cifUrl")
    if not pdb_url:
        return None, None, None, None, "No PDB URL in AlphaFold response"
    pdb_resp = requests.get(pdb_url, timeout=60, verify=False)
    if pdb_resp.status_code != 200:
        return None, None, None, None, f"PDB download failed (status {pdb_resp.status_code})"
    coords, b_factors, atom_names, resseqs = [], [], [], []
    for line in pdb_resp.text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfac = float(line[60:66].strip() or 0)
            resseq = int(line[22:26].strip())
        except (ValueError, IndexError):
            continue
        coords.append([x, y, z])
        b_factors.append(bfac)
        atom_names.append(atom_name)
        resseqs.append(resseq)
    if not coords:
        return None, None, None, None, "No valid atoms parsed from PDB"
    return np.array(coords), np.array(b_factors), atom_names, resseqs, None


def build_3d_plot(coords, b_factors, atom_names, resseqs, mutation_pos, hgvsp_str):
    mut_resseq = int(re.search(r"(\d+)", hgvsp_str).group(1)) if re.search(r"(\d+)", hgvsp_str) else mutation_pos
    mut_coords = coords[(np.array(resseqs) == mut_resseq) & (np.array(atom_names) == "CA")]
    if mut_coords.size == 0:
        mid = len(resseqs) // 2
        mut_coords = coords[mid:mid + 1]
    mut_xyz = mut_coords[0]
    distances = np.linalg.norm(coords - mut_xyz, axis=1)
    ca_mask = np.array(atom_names) == "CA"
    ca_coords = coords[ca_mask]
    ca_resseqs = np.array(resseqs)[ca_mask]
    ca_dists = distances[ca_mask]

    trace_backbone = go.Scatter3d(
        x=ca_coords[:, 0], y=ca_coords[:, 1], z=ca_coords[:, 2],
        mode="lines+markers",
        line=dict(color="#2b66ff", width=6),
        marker=dict(size=3, color="#2b66ff", opacity=0.9),
        name="Backbone (CA)", hoverinfo="skip",
    )
    n_mask = np.array(atom_names) == "N"
    c_mask = np.array(atom_names) == "C"
    n_coords = coords[n_mask]
    c_coords = coords[c_mask]
    trace_n = go.Scatter3d(
        x=n_coords[:, 0], y=n_coords[:, 1], z=n_coords[:, 2],
        mode="markers", marker=dict(size=2, color="#7b2ff7", opacity=0.5),
        name="N atoms", hoverinfo="skip", showlegend=True,
    ) if len(n_coords) > 0 else None
    trace_c = go.Scatter3d(
        x=c_coords[:, 0], y=c_coords[:, 1], z=c_coords[:, 2],
        mode="markers", marker=dict(size=2, color="#00bb7f", opacity=0.5),
        name="C atoms", hoverinfo="skip", showlegend=True,
    ) if len(c_coords) > 0 else None

    scatter = go.Scatter3d(
        x=ca_coords[:, 0], y=ca_coords[:, 1], z=ca_coords[:, 2],
        mode="markers",
        marker=dict(
            size=5, color=ca_dists,
            colorscale=[[0, "#2b66ff"], [0.25, "#7b2ff7"], [0.5, "#ff007f"], [0.75, "#fb2c36"], [1, "#ffd166"]],
            colorbar=dict(title=dict(text="Distance (A)", font=dict(color="#888", size=11)), thickness=12, len=0.6, tickfont=dict(color="#888", size=10)),
            opacity=0.95, line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        ),
        text=[f"Residue {r} | {d:.1f}A from mutation" for r, d in zip(ca_resseqs, ca_dists)],
        hoverinfo="text", name="CA Atoms",
    )
    star_trace = go.Scatter3d(
        x=[mut_xyz[0]], y=[mut_xyz[1]], z=[mut_xyz[2]],
        mode="markers+text",
        marker=dict(symbol="diamond", size=20, color="#ff007f", line=dict(color="white", width=2)),
        text=[f"  {hgvsp_str}"], textposition="middle right",
        textfont=dict(size=14, color="#ff007f", family="IBM Plex Mono"),
        name="MUTATION", hovertext=f"<b>{hgvsp_str}</b><br>Position: {mut_resseq}", hoverinfo="text",
    )
    traces = [trace_backbone, scatter, star_trace]
    if trace_n:
        traces.append(trace_n)
    if trace_c:
        traces.append(trace_c)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            bgcolor="rgba(0,0,0,0.95)", aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.6), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.8)", font=dict(color="#ddd", size=11), bordercolor="#333", borderwidth=1),
        height=550, showlegend=True,
    )
    return go.Figure(data=traces, layout=layout)


def build_dna_helix():
    t = np.linspace(0, 4 * np.pi, 800)
    radius, pitch = 10, 3.5
    x1 = radius * np.cos(t)
    y1 = radius * np.sin(t)
    z1 = pitch * t
    x2 = radius * np.cos(t + np.pi)
    y2 = radius * np.sin(t + np.pi)
    z2 = pitch * t

    strand1 = go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", line=dict(color="#2b66ff", width=5), name="5' -> 3'", hoverinfo="skip")
    strand2 = go.Scatter3d(x=x2, y=y2, z=z2, mode="lines", line=dict(color="#ff007f", width=5), name="3' -> 5'", hoverinfo="skip")

    nucleotide_colors = {"A": "#fb2c36", "T": "#00bb7f", "G": "#ffd166", "C": "#2b66ff"}
    bp_x1, bp_y1, bp_z1, bp_x2, bp_y2, bp_z2 = [], [], [], [], [], []
    seq1, seq2 = "ATGCATGCATGCATGCATGCATGCATGC", "TACGTACGTACGTACGTACGTACGTACG"

    for i in range(0, len(t) - 20, 20):
        bp_x1.append(x1[i])
        bp_y1.append(y1[i])
        bp_z1.append(z1[i])
        bp_x2.append(x2[i])
        bp_y2.append(y2[i])
        bp_z2.append(z2[i])

    # Build rungs as individual line segments (Plotly connects with None separators)
    rung_x, rung_y, rung_z = [], [], []
    for i in range(len(bp_x1)):
        rung_x += [bp_x1[i], bp_x2[i], None]
        rung_y += [bp_y1[i], bp_y2[i], None]
        rung_z += [bp_z1[i], bp_z2[i], None]
    rungs = go.Scatter3d(
        x=rung_x, y=rung_y, z=rung_z,
        mode="lines", line=dict(color="#6a6a9e", width=3), name="Base pairs", hoverinfo="skip",
    )
    nuc_x = bp_x1 + bp_x2
    nuc_y = bp_y1 + bp_y2
    nuc_z = bp_z1 + bp_z2
    nuc_colors = [nucleotide_colors.get(s, "#555") for s in seq1[:len(bp_x1)]] + [nucleotide_colors.get(s, "#555") for s in seq2[:len(bp_x2)]]
    nuc_markers = go.Scatter3d(
        x=nuc_x, y=nuc_y, z=nuc_z, mode="markers",
        marker=dict(size=4, color=nuc_colors, opacity=0.9, line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
        name="Nucleotides", hoverinfo="skip",
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            bgcolor="rgba(0,0,0,0.95)", aspectmode="data",
            camera=dict(eye=dict(x=2.0, y=0.5, z=0.3), up=dict(x=0, y=0, z=1)),
            dragmode="orbit",
        ),
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.8)", font=dict(color="#ddd", size=10), bordercolor="#333", borderwidth=1),
        height=550, showlegend=True,
    )
    fig = go.Figure(data=[strand1, strand2, rungs, nuc_markers], layout=layout)
    # Auto-rotate: smooth orbit animation
    n_frames, angle_step = 60, 6
    frames = []
    eye_base = dict(x=2.0, y=0.5, z=0.3)
    for k in range(n_frames):
        ang = np.radians(k * angle_step)
        ex = eye_base["x"] * np.cos(ang) - eye_base["y"] * np.sin(ang)
        ey = eye_base["x"] * np.sin(ang) + eye_base["y"] * np.cos(ang)
        frames.append(go.Frame(
            layout=dict(scene=dict(camera=dict(eye=dict(x=ex, y=ey, z=eye_base["z"]), up=dict(x=0, y=0, z=1)))),
            name=f"f{k}",
        ))
    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(43,102,255,0.15)", bordercolor="#2b66ff", borderwidth=1,
            font=dict(color="#fff", size=10, family="IBM Plex Mono"),
            buttons=[dict(
                label="▶ ROTATE",
                method="animate",
                args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, mode="immediate",
                                 transition=dict(duration=0))],
            )],
        )],
    )
    return fig


def build_esi_bar(prob):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[prob], y=["Pathogenic"], orientation="h", marker=dict(color="#fb2c36"), showlegend=False))
    fig.add_trace(go.Bar(x=[1 - prob], y=["Benign"], orientation="h", marker=dict(color="#00bb7f"), showlegend=False))
    fig.update_layout(barmode="stack", height=40, margin=dict(l=0, r=0, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False))
    return fig


def build_confusion_matrix():
    cm = ESM_METRICS.get("confusion", {})
    TN = int(cm.get("tn", 228))
    FP = int(cm.get("fp", 47))
    FN = int(cm.get("fn", 58))
    TP = int(cm.get("tp", 226))
    matrix = np.array([[TN, FP], [FN, TP]])
    text = [[f"TN<br>{TN}", f"FP<br>{FP}"], [f"FN<br>{FN}", f"TP<br>{TP}"]]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Benign", "Pathogenic"], y=["Benign", "Pathogenic"],
        text=text, texttemplate="%{text}", textfont=dict(size=16, color="white"),
        colorscale=[[0, "#001a1a"], [1, "#1a000a"]], showscale=False,
    ))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), xaxis=dict(side="top"))
    return fig


def build_roc_curve():
    roc = ESM_METRICS.get("roc", {})
    fpr = np.asarray(roc.get("fpr", [0.0, 1.0]), dtype=float)
    tpr = np.asarray(roc.get("tpr", [0.0, 1.0]), dtype=float)
    auc = float(ESM_METRICS.get("auc", 0.8951))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"XGBoost (AUC = {auc:.3f})", line=dict(color="#2b66ff", width=3), fill="tozeroy", fillcolor="rgba(43,102,255,0.08)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#333", width=1, dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0.5)"))
    return fig


def build_prediction_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        title=dict(text="PATHOGENICITY", font=dict(size=11, color="#888", family="IBM Plex Mono")),
        number=dict(suffix="%", font=dict(size=32, color="#fb2c36" if prob > 0.5 else "#00bb7f", family="IBM Plex Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#333"),
            bar=dict(color="#fb2c36" if prob > 0.5 else "#00bb7f"),
            bgcolor="#0a0a0f", borderwidth=0,
            steps=[dict(range=[0, 50], color="#001210"), dict(range=[50, 100], color="#120010")],
            threshold=dict(line=dict(color="white", width=2), thickness=0.8, value=50),
        ),
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_feature_importance():
    top = ESM_METRICS.get("importance_top", [])
    if top:
        chosen = list(reversed(top[:15]))
        features = [f"ESM dim {d['dim']}" for d in chosen]
        importance = [d["importance"] for d in chosen]
    else:
        features = ["feature importance unavailable"]
        importance = [0.0]
    fig = go.Figure(go.Bar(
        x=importance, y=features, orientation="h",
        marker=dict(color="#2b66ff", line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1%}" for v in importance], textposition="inside", textfont=dict(color="white", size=11),
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"), xaxis=dict(visible=False), yaxis=dict(autorange="reversed"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-tag">// DNA MUTATION EFFECT PREDICTOR</div>
        <div class="hero-title">Predict. Classify.<br>Understand.</div>
        <div class="hero-subtitle">
            Classifying missense mutations as pathogenic or benign using
            ESM-2 protein language model embeddings and gradient-boosted classification.
        </div>
        <div class="hero-dna">A T G C U A T G C &mdash; A T G C U A T G C &mdash; A T G C U A T G C</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    stats = [
        (f"{float(ESM_METRICS.get('accuracy', 0.8122)):.1%}", "Accuracy", "🎯"),
        ("2,792", "Variants", "📊"),
        ("150M", "Parameters", "🧠"),
        ("1,280-dim", "Embedding", "📐"),
    ]
    for col, (val, label, icon) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card fade-in">
                <span class="stat-icon">{icon}</span>
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">// PIPELINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Gene Lookup", "Map gene symbol to UniProt accession via REST API"),
        ("02", "Sequence Window", "Extract ±128 residue window around mutation site"),
        ("03", "ESM-2 Embedding", "Feed wild-type and mutant windows through Facebook ESM-2 (150M params)"),
        ("04", "Classification", "XGBoost classifier on 400 selected features from 1280-dim embedding"),
        ("05", "Prediction", "Binary label (Pathogenic/Benign) + probability score"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step fade-in">
            <div class="step-num">{num}</div>
            <div class="step-content">
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">// EXPLORE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Try It Now</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--text-dim);font-size:0.85rem;margin-bottom:1rem;">Select a mutation to jump to the Predict page:</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (gene, hgvsp, desc, label) in enumerate(EXAMPLES):
        with cols[i % 4]:
            gene_color = GENE_COLORS.get(gene, "#2b66ff")
            tag_color = "#fb2c36" if label == "pathogenic" else "#00bb7f"
            tag_txt = "PATHOGENIC" if label == "pathogenic" else "BENIGN"
            st.markdown(f"""
            <div style="border-left:3px solid {gene_color};padding-left:0.6rem;margin-bottom:0.3rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;font-weight:700;color:{gene_color};">{gene}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:var(--text-dim);">{hgvsp}</div>
                <div style="font-size:0.68rem;color:var(--text-muted);margin:0.15rem 0 0.25rem;">{desc}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;color:{tag_color};">&#9679; {tag_txt}</div>
            </div>
            """, unsafe_allow_html=True)
            st.button("Analyze →", key=f"home_ex_{i}", width="stretch",
                      on_click=_go_predict, args=(gene, hgvsp))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">// STRUCTURE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">DNA Double Helix</div>', unsafe_allow_html=True)
    st.plotly_chart(build_dna_helix(), width="stretch", key="dna_helix_home", on_select="ignore")


def page_predict():
    st.markdown('<div class="section-header">// ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predict Mutation Effect</div>', unsafe_allow_html=True)

    st.markdown('<div style="color:var(--text-dim);font-size:0.8rem;margin-bottom:0.8rem;">Quick examples:</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (gene, hgvsp, desc, _) in enumerate(EXAMPLES[:4]):
        with cols[i]:
            st.button(f"{gene} {hgvsp}", key=f"pred_ex_{i}", width="stretch",
                      on_click=_fill_predict, args=(gene, hgvsp))

    st.session_state.setdefault("gene_input", "")
    st.session_state.setdefault("hgvsp_input", "")
    with st.form(key="predict_form", clear_on_submit=False):
        col_gene, col_hgvsp = st.columns([1, 1])
        with col_gene:
            gene = st.text_input("GENE SYMBOL", placeholder="e.g. TP53", key="gene_input")
        with col_hgvsp:
            hgvsp = st.text_input("HGVSp NOTATION", placeholder="e.g. p.Cys61Gly", key="hgvsp_input")
        submitted = st.form_submit_button("CLASSIFY MUTATION", type="primary", width="stretch")

    if submitted:
        if not gene.strip() or not hgvsp.strip():
            st.error("Enter both gene symbol and HGVSp string.")
            return

        with st.spinner("Running ESM-2 + Evo2-7B inference..."):
            t0 = time.time()
            label, prob, msg = run_prediction(gene.strip(), hgvsp.strip())
            elapsed = time.time() - t0

        if label is None:
            st.error(f"Prediction failed: {msg}")
            return

        status = "Pathogenic" if label == 1 else "Benign"
        css_class = "pathogenic" if label == 1 else "benign"
        _push_history(gene.strip(), hgvsp.strip(), status, prob)
        st.toast(f"{gene} {hgvsp}: {status} ({prob:.1%})", icon="🧬")

        evo = _parse_evo2_msg(msg)
        if "evo2" in evo:
            breakdown = (f"ESM-2 {evo.get('esm', prob):.1%} &bull; Evo2-7B "
                         f"{evo['evo2']:.1%} (LLR {evo.get('llr', 0.0):+.2f}) &bull; blended")
        else:
            breakdown = "ESM-2 only (Evo2 offline)"

        st.markdown("---")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="result-label">{status.upper()}</div>
                <div class="result-gene">{gene} &bull; {hgvsp}</div>
                <div class="result-prob">{prob:.1%}</div>
                <div style="color:var(--text-muted);font-family:'IBM Plex Mono',monospace;font-size:0.7rem;letter-spacing:0.1em;">
                    pathogenic probability &bull; {elapsed:.1f}s inference
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:var(--text-dim);margin-top:0.35rem;">
                    {breakdown}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_esi_bar(prob), width="stretch", key="prob_bar", on_select="ignore")

        with c2:
            st.plotly_chart(build_prediction_gauge(prob), width="stretch", key="gauge", on_select="ignore")
            uniprot_id = get_uniprot_id(gene.strip())
            if uniprot_id:
                wt3, pos, mt3 = parse_hgvsp(hgvsp)
                st.markdown(f"""
                <div class="blueprint-card" style="margin-top:1rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--blue);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">Protein Info</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;line-height:1.8;">
                        UniProt: <span style="color:#2b66ff;">{uniprot_id}</span><br>
                        WT: <span style="color:#00bb7f;">{wt3} ({AA_MAP.get(wt3, '?')})</span><br>
                        Position: <span style="color:#ffd166;">{pos}</span><br>
                        MT: <span style="color:#fb2c36;">{mt3} ({AA_MAP.get(mt3, '?')})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if uniprot_id:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">// 3D STRUCTURE</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Protein Structure</div>', unsafe_allow_html=True)
            coords, b_factors, atom_names, resseqs, error = fetch_alphafold_coords(uniprot_id)
            if coords is not None:
                wt3, pos, _ = parse_hgvsp(hgvsp)
                fig3d = build_3d_plot(coords, b_factors, atom_names, resseqs, pos, hgvsp)
                st.plotly_chart(fig3d, width="stretch", key="structure_3d", on_select="ignore")
                st.caption(f"AlphaFold structure for **{uniprot_id}** — {len(coords)} backbone atoms, coloured by distance from mutation site")
            else:
                st.info(f"Structure unavailable ({error}). Prediction above is still valid.")

    hist = st.session_state.get("history", [])
    if hist:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">// HISTORY</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.1rem !important;">Recent Predictions</div>', unsafe_allow_html=True)
        hcols = st.columns(min(len(hist), 5))
        for hc, h in zip(hcols, hist[:5]):
            with hc:
                hcol = "#fb2c36" if h["status"] == "Pathogenic" else "#00bb7f"
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:0.7rem;text-align:center;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;font-weight:700;">{h["gene"]}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:var(--text-dim);">{h["hgvsp"]}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;font-weight:700;color:{hcol};margin-top:0.2rem;">{h["status"]} {h["prob"]:.0%}</div>
                </div>
                """, unsafe_allow_html=True)


def page_batch():
    st.markdown('<div class="section-header">// BATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--text-dim);font-size:0.85rem;">Upload a CSV with <code style="color:#2b66ff;">gene</code> and <code style="color:#2b66ff;">hgvsp</code> columns. Max {MAX_UPLOAD_MB}MB / {MAX_BATCH_ROWS} rows.</div>'.format(MAX_UPLOAD_MB=MAX_UPLOAD_MB, MAX_BATCH_ROWS=MAX_BATCH_ROWS), unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv_upload")

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large. Max {MAX_UPLOAD_MB}MB.")
            return

        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        if not {"gene", "hgvsp"}.issubset(df.columns):
            st.error(f"CSV must have columns: gene, hgvsp. Found: {list(df.columns)}")
            return

        if len(df) > MAX_BATCH_ROWS:
            st.warning(f"Too many rows ({len(df)}). Truncating to {MAX_BATCH_ROWS}.")
            df = df.head(MAX_BATCH_ROWS)

        st.dataframe(df.head(20), width="stretch")
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:var(--text-dim);">{len(df)} variants loaded</div>', unsafe_allow_html=True)

        if st.button("RUN BATCH PREDICTIONS", type="primary", width="stretch"):
            progress = st.progress(0, text="Starting...")
            results = []
            total = len(df)
            for i, row in df.iterrows():
                gene_val = str(row["gene"]).strip()
                hgvsp_val = str(row["hgvsp"]).strip()
                label, prob, msg = run_prediction(gene_val, hgvsp_val)
                evo = _parse_evo2_msg(msg)
                results.append({
                    "gene": gene_val, "hgvsp": hgvsp_val,
                    "prediction": "Pathogenic" if label == 1 else ("Benign" if label == 0 else "Error"),
                    "probability": round(prob, 4) if label is not None else None,
                    "esm_prob": round(evo["esm"], 4) if "esm" in evo else None,
                    "evo2_prob": round(evo["evo2"], 4) if "evo2" in evo else None,
                    "evo2_llr": round(evo["llr"], 2) if "llr" in evo else None,
                })
                progress.progress((i + 1) / total, text=f"{i + 1}/{total}: {gene_val} {hgvsp_val}")

            res_df = pd.DataFrame(results)
            st.markdown("---")
            st.dataframe(res_df, width="stretch")

            pathogenic_count = (res_df["prediction"] == "Pathogenic").sum()
            benign_count = (res_df["prediction"] == "Benign").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(res_df))
            c2.metric("Pathogenic", pathogenic_count, delta=f"{pathogenic_count/len(res_df)*100:.1f}%")
            c3.metric("Benign", benign_count, delta=f"{benign_count/len(res_df)*100:.1f}%")

            csv_buf = io.BytesIO()
            res_df.to_csv(csv_buf, index=False)
            st.download_button("Download Results (CSV)", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")
            st.toast(f"Batch complete: {pathogenic_count} pathogenic / {benign_count} benign", icon="✅")
    else:
        st.markdown("""
        <div class="blueprint-card" style="text-align:center;padding:3rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📁</div>
            <div style="color:var(--text-dim);font-size:0.85rem;">Drop a CSV file here or click to browse</div>
            <div style="color:var(--text-muted);font-size:0.7rem;margin-top:0.3rem;">Columns: gene, hgvsp</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        sample_csv = "gene,hgvsp\nTP53,p.Cys61Gly\nKRAS,p.Gly12Val\nBRCA1,p.Gly1738Arg\nPTEN,p.Arg130Gln\n"
        st.download_button("⬇ Download sample CSV template", data=sample_csv,
                           file_name="sample_variants.csv", mime="text/csv",
                           key="sample_csv_dl")


def page_performance():
    st.markdown('<div class="section-header">// METRICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    acc = float(ESM_METRICS.get("accuracy", 0.8122))
    prec = float(ESM_METRICS.get("precision", 0.8278))
    rec = float(ESM_METRICS.get("recall", 0.7958))
    f1 = float(ESM_METRICS.get("f1", 0.8115))
    spec = float(ESM_METRICS.get("specificity", 0.8291))
    mcc = float(ESM_METRICS.get("mcc", 0.6250))
    auc = float(ESM_METRICS.get("auc", 0.8951))
    n_test = int(ESM_METRICS.get("n_test", 559))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.1%}")
    c2.metric("Precision", f"{prec:.1%}")
    c3.metric("Recall", f"{rec:.1%}")
    c4.metric("F1 Score", f"{f1:.1%}")

    st.caption(
        "ESM-2-only head, measured on the frozen 559-variant hold-out. "
        "The CLI blends this view with the population model (96.9% accuracy) "
        "when gnomAD/SIFT/PolyPhen data is available."
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-header">// CONFUSION</div>', unsafe_allow_html=True)
        st.plotly_chart(build_confusion_matrix(), width="stretch", on_select="ignore")
    with col_right:
        st.markdown('<div class="section-header">// ROC CURVE</div>', unsafe_allow_html=True)
        st.plotly_chart(build_roc_curve(), width="stretch", on_select="ignore")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Specificity", f"{spec:.1%}")
    c2.metric("MCC", f"{mcc:.3f}")
    c3.metric("Test Set", f"{n_test:,}")
    c4.metric("AUC-ROC", f"{auc:.3f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">// TOP SELECTED FEATURES</div>', unsafe_allow_html=True)
    st.plotly_chart(build_feature_importance(), width="stretch", on_select="ignore")

    st.markdown("""
    <div class="blueprint-card" style="margin-top:1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--blue);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;">Training Details</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;">
            <div>Model: <span style="color:#2b66ff;">XGBoost</span></div>
            <div>Estimators: <span style="color:#7b2ff7;">500</span></div>
            <div>Max Depth: <span style="color:#ff007f;">6</span></div>
            <div>Learning Rate: <span style="color:#00bb7f;">0.05</span></div>
            <div>Feature Selection: <span style="color:#ffd166;">SelectKBest k=400</span></div>
            <div>Scaler: <span style="color:#2b66ff;">StandardScaler</span></div>
            <div>Train: <span style="color:#7b2ff7;">2,233</span></div>
            <div>Test: <span style="color:#ff007f;">559</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_about():
    st.markdown('<div class="section-header">// ABOUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About This Tool</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="blueprint-card" style="margin-bottom:1.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--blue);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;">How It Works</div>
        <div style="color:var(--text-dim);line-height:1.8;font-size:0.9rem;">
            The DNA Mutation Effect Predictor uses a <b style="color:#fff;">protein language model</b> (ESM-2 by Meta AI) to embed
            wild-type and mutant protein sequences, then classifies the mutation as <b style="color:#fb2c36;">pathogenic</b> or <b style="color:#00bb7f;">benign</b>
            using a gradient-boosted decision tree (XGBoost).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">// STACK</div>', unsafe_allow_html=True)
    tech_data = [
        ["Frontend", "Streamlit + Plotly", "Interactive web dashboard"],
        ["3D Visualization", "Plotly 3D Scatter", "AlphaFold protein structures"],
        ["Protein LM", "ESM-2 (150M)", "Meta AI protein embeddings"],
        ["Classifier", "XGBoost", "Gradient-boosted trees"],
        ["Sequence Data", "UniProt API", "Protein sequences & IDs"],
        ["Structure Data", "AlphaFold DB", "Predicted 3D structures"],
        ["Framework", "Python + PyTorch", "Deep learning runtime"],
    ]
    df_tech = pd.DataFrame(tech_data, columns=["Layer", "Technology", "Description"])
    st.dataframe(df_tech, width="stretch", hide_index=True)

    st.markdown('<div class="section-header">// DATA SOURCES</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:var(--text-dim);font-size:0.85rem;line-height:1.8;">
        <b style="color:#fff;">ClinVar</b> — Annotated pathogenic/benign missense variants (66,210 variants)<br>
        <b style="color:#fff;">UniProt</b> — Canonical human protein sequences and gene-to-accession mapping<br>
        <b style="color:#fff;">AlphaFold DB</b> — Predicted 3D protein structures from EBI<br>
        <b style="color:#fff;">ESM-2</b> — Evolutionary-scale protein language model (Meta AI)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(251,44,54,0.08);border:1px solid rgba(251,44,54,0.2);border-radius:8px;padding:1.2rem;margin-top:1.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--red);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">Disclaimer</div>
        <div style="color:var(--text-dim);font-size:0.85rem;">
            This tool is for <b style="color:#fff;">research and educational purposes only</b>.
            Predictions should not be used as the sole basis for clinical decisions.
            Always consult a qualified geneticist or medical professional.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "Home": page_home,
    "Predict": page_predict,
    "Batch": page_batch,
    "Performance": page_performance,
    "About": page_about,
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.8rem;">
        <div style="font-size:1.8rem;margin-bottom:0.3rem;">🧬</div>
        <div style="font-family:'Anton',sans-serif;font-size:1.1rem;letter-spacing:0.05em;
            background:linear-gradient(135deg,#2b66ff,#ff007f);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            DNA PREDICTOR
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;color:var(--text-muted);font-size:0.6rem;letter-spacing:0.15em;margin-top:0.2rem;">ESM-2 + XGBOOST</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:0.5rem 0;"></div>', unsafe_allow_html=True)

    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = "Home"
    selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed", key="nav_radio")

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:0.5rem 0;"></div>', unsafe_allow_html=True)

    n_hist = len(st.session_state.get("history", []))
    hist_line = f"<div>HISTORY: {n_hist}/10</div>" if n_hist else ""
    st.markdown(f"""
    <div style="text-align:center;padding:0.5rem;">
        <div style="font-family:'IBM Plex Mono',monospace;color:var(--text-muted);font-size:0.6rem;letter-spacing:0.1em;">
            v1.0 &bull; ESM-2 150M + XGBoost &bull; research use only
            {hist_line}
        </div>
    </div>
    """, unsafe_allow_html=True)

PAGES[selected]()

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-section">
    <div class="footer-badges">
        <span class="tech-badge">ESM-2</span>
        <span class="tech-badge">XGBoost</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">PyTorch</span>
        <span class="tech-badge">UniProt</span>
        <span class="tech-badge">AlphaFold</span>
        <span class="tech-badge">ClinVar</span>
    </div>
    <div class="footer-credits">
        DNA Mutation Effect Predictor &bull; Built with ESM-2 + XGBoost + Streamlit &bull; Data from ClinVar, UniProt, AlphaFold
    </div>
</div>
""", unsafe_allow_html=True)
