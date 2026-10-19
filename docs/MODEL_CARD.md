# Model Card — DNA Mutation Effect Predictor v1.0.0

## Model details

| | |
|---|---|
| **Model name** | DNA Mutation Effect Predictor |
| **Version** | 1.0.0 |
| **Task** | Binary classification of human missense variants: pathogenic vs. benign |
| **Input** | Gene symbol + HGVSp protein substitution (e.g. `TP53`, `p.Arg175His`) |
| **Output** | Probability in `[0, 1]`; threshold 0.5 for the reported label |
| **Protein encoder** | `facebook/esm2_t30_150M_UR50D` (ESM-2 150M), frozen |
| **Protein head** | XGBoost, 500 trees, max depth 6, learning rate 0.05 |
| **Population head** | scikit-learn `GradientBoostingClassifier` on 6 tabular features |
| **Serving blend** | 0.80 · population + 0.20 · ESM-2 when population coverage exists |
| **License** | MIT |
| **Author** | NOOBHEKER |

## Intended use

**Primary use:** research, teaching, and exploratory triage of missense variants
where a quick, transparent baseline is useful.

**Intended users:** bioinformatics practitioners, students, and researchers who
understand the limitations of computational variant-effect predictors.

**Out of scope:**

- Clinical diagnosis, treatment selection, or reproductive counselling.
- Nonsense, frameshift, splice-site, structural, or indel variants.
- Somatic variant interpretation without additional tumour context.
- Any decision that requires a validated, regulated medical device.

## Training data

- **Source:** ClinVar missense variants, filtered to binary pathogenic/benign
  labels.
- **Indexed sample:** 66,210 variants (`data/variants.csv`), balanced
  33,105 / 33,105.
- **Embedded cohort:** 2,792 variants in `data/processed_data.pt`.
- **Split:** stratified 80/20, frozen at 2,233 train / 559 test
  (`data/test_indices.pt`). The split is never re-rolled between runs.
- **Sequence source:** UniProt canonical human protein sequences.
- **Population annotations:** MyVariant.info / gnomAD and Ensembl VEP local
  caches. Coverage is limited to the 66k-variant ClinVar sample.

The 2,792-row ESM cohort is smaller than the 66k index because embedding every
variant window is expensive and because only canonical missense substitutions
with resolvable protein coordinates were retained.

## Evaluation

### Protein head (ESM-2)

Frozen hold-out, `n = 559`:

| Metric | Value |
|---|---:|
| Accuracy | 0.8122 |
| Balanced accuracy | 0.8124 |
| ROC-AUC | 0.8951 |
| Precision | 0.8278 |
| Recall | 0.7958 |
| F1 | 0.8115 |
| Specificity | 0.8291 |
| MCC | 0.6250 |

Confusion matrix (rows = actual, columns = predicted):

|  | Benign | Pathogenic |
|---|---:|---:|
| **Benign** | 228 (TN) | 47 (FP) |
| **Pathogenic** | 58 (FN) | 226 (TP) |

### Population head

Hold-out, `n = 545` (different cohort because population features are not
available for every embedded variant):

| Metric | Value |
|---|---:|
| Accuracy | 0.9688 |
| Balanced accuracy | 0.9680 |
| ROC-AUC | 0.9928 |

Conservative subset — only variants observed in gnomAD:

| Metric | Value |
|---|---:|
| Accuracy | 0.9282 |
| ROC-AUC | 0.9786 |

### What did not help

Measured on the same frozen split:

- Stacking (XGB + LGBM + CatBoost → logistic regression): 0.8068 accuracy.
- Soft voting: 0.8068 accuracy.
- More features (k = 800): 0.8104 accuracy.
- Deeper/wider trees and a 12-config sweep: 0.798–0.810 CV accuracy.
- Signed difference / product feature representations: 0.7782–0.8157.

The shipped model is the simpler single XGBoost head.

## Limitations and risks

1. **Population-signal leakage / ascertainment bias.** gnomAD allele frequency
   is observed for a much larger fraction of benign than pathogenic ClinVar
   records. The model can learn dataset composition rather than pure biology.
   The gnomAD-observed subset metrics are the conservative estimate.
2. **Coverage.** Variants outside the 66k ClinVar sample have no population
   features. They fall back to ESM-2 only, which is materially weaker.
3. **Label noise.** ClinVar assertions change over time and vary in review
   confidence. No clinical-significance review status was used as a weight.
4. **Population bias.** The training set is dominated by well-studied genes and
   ancestries represented in ClinVar/gnomAD.
5. **Blend not end-to-end validated.** The 0.80/0.20 weights are a serving
   choice. Component metrics are measured independently.
6. **No calibration analysis.** Probabilities are ranking scores, not
   necessarily calibrated risks.
7. **Optional DNA views are experimental.** Evo2 and DNABERT-2 are auxiliary,
   API/local dependencies, and can fail soft.

## Ethical considerations

- Do not present predictions to patients or clinicians as a diagnosis.
- Do not use the model to justify denying care, reproductive choices, or
  insurance coverage.
- Report the version and the exact input when discussing a prediction.
- Consider health equity: performance on underrepresented populations and genes
  is not established.

## Artifacts

| Artifact | Description |
|---|---|
| `data/scaler.pkl` | 3-tuple `(XGBClassifier, SelectKBest(k=400), StandardScaler)` |
| `data/tabular.pkl` | Population model dict + feature names |
| `data/processed_data.pt` | 2,792 × 1,280 ESM-2 features + labels |
| `data/test_indices.pt` | Frozen 559-row test indices |
| `data/metrics.json` | Machine-readable metrics for docs/dashboard |
| `data/variants.csv` | Variant index used by the CLI |
| `data/*_cache.json` | Local population annotations |

## Citation

See [CITATION.cff](../CITATION.cff). If you use the model, cite this repository
and the underlying resources (ClinVar, UniProt, ESM-2, gnomAD).
