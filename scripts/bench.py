"""Measure inference efficiency of the prediction pipeline.

Reports cold-start load time, warm per-prediction latency (population
view vs sequence fallback), and peak memory. Run with no other heavy
python processes for representative numbers.

Usage:
    venv\\Scripts\\python scripts/bench.py
"""
import os
import sys
import time

os.environ.setdefault("EVO2_DISABLE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

t_import0 = time.perf_counter()


try:
    import psutil
    PROC = psutil.Process(os.getpid())
except Exception:  # psutil optional -- memory numbers are a bonus
    PROC = None


def rss_mb():
    if PROC is None:
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(c), c.cb)
            return c.WorkingSetSize / 1024 / 1024
        except Exception:
            return 0.0
    return PROC.memory_info().rss / 1024 / 1024


def stage(msg, t0):
    dt = time.perf_counter() - t0
    print(f"  {msg:42s} {dt:7.2f}s   rss={rss_mb():7.1f}MB")
    return dt


print("== cold start ==")
t0 = time.perf_counter()
from src.predict import (predict_with_steps, tabular_opinion,  # noqa: E402
                         OnnxEsm, ESM_MODEL_NAME, ONNX_PATH)
stage("import src.predict", t0)

t0 = time.perf_counter()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
stage("tokenizer", t0)

t0 = time.perf_counter()
from src.predict import load_model  # noqa: E402
model, selector, scaler, sc_extra, base = load_model()
stage("load scaler.pkl (ESM head)", t0)

t0 = time.perf_counter()
esm = OnnxEsm(ONNX_PATH)
stage("ONNX ESM-2 session", t0)

t0 = time.perf_counter()
p, d = tabular_opinion("TP53", "R", 175, "H")
stage("tabular model + cache lookup", t0)

engine = {"model": model, "selector": selector, "scaler": scaler,
          "tokenizer": tok, "esm_model": esm, "device": None,
          "base_models": base, "sc_extra": sc_extra}
print(f"  tabular TP53 R175H p={p:.4f}")


def run(gene, hg):
    gen = predict_with_steps(gene, hg, **engine)
    n = 0
    while True:
        try:
            next(gen)
            n += 1
        except StopIteration as e:
            return e.value, n, n + 1


print("\n== warm latency, mapped variant (population + ESM) ==")
for gene, hg in [("TP53", "p.Arg175His"), ("BRCA1", "p.Cys61Gly")]:
    t0 = time.perf_counter()
    lab, prob, msg, n = run(gene, hg)
    dt = time.perf_counter() - t0
    print(f"  {gene} {hg:16s} p={prob:.4f} {'PATHO' if lab else 'benign':6s} "
          f"{dt:6.2f}s ({n} steps)")

print("\n== warm latency, unmapped variant (ESM fallback) ==")
for gene, hg in [("MTHFR", "p.Ala222Cys"), ("SOD1", "p.Gly93Ser")]:
    t0 = time.perf_counter()
    lab, prob, msg, n = run(gene, hg)
    dt = time.perf_counter() - t0
    print(f"  {gene} {hg:16s} p={prob:.4f} {'PATHO' if lab else 'benign':6s} "
          f"{dt:6.2f}s ({n} steps)")

print("\n== breakdown: sequence embed vs tabular ==")
t0 = time.perf_counter()
for _ in range(20):
    tabular_opinion("TP53", "R", 175, "H")
print(f"  tabular_opinion x20 (cached)      "
      f"{(time.perf_counter() - t0) / 20 * 1000:7.1f}ms each")

t0 = time.perf_counter()
for _ in range(5):
    w = "M" * 257
    i = tok(w, return_tensors="np", padding=True, truncation=True,
            max_length=514)
    esm(input_ids=i["input_ids"], attention_mask=i["attention_mask"])
print(f"  ESM ONNX 257aa forward x5         "
      f"{(time.perf_counter() - t0) / 5 * 1000:7.1f}ms each")

print(f"\npeak rss = {rss_mb():.1f} MB")
print("DONE")
