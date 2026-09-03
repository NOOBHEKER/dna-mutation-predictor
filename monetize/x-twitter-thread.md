# X/Twitter Thread — Copy & Post

Post each tweet as a separate tweet. Use the numbering to keep order.

---

## Tweet 1

I built a CLI tool that predicts if a DNA mutation causes disease.

No cloud API. No paid service. Just Python on your laptop.

Uses Facebook's ESM-2 protein language model + XGBoost. 82.5% accuracy.

Open source. Try it 👇

---

## Tweet 2

How it works:

1. Input: protein + mutation (e.g., BRCA1 A1708E)
2. Fetches sequence from UniProt
3. ESM-2 produces 640-dim embedding
4. XGBoost classifies: PATHOGENIC or BENIGN
5. Returns confidence score + protein graph

---

## Tweet 3

The stack:

• ESM-2 (150M params) — reads protein sequences
• XGBoost — lightweight classifier on top
• scikit-learn — feature selection + scaling
• 2,792 ClinVar samples for training

No GPU needed. Runs on CPU.

---

## Tweet 4

Key insight: ESM-2 embeddings are overkill for classification.

I selected 40 out of 640 features. That's it. The reduced model outperformed the full embedding.

Less is more.

---

## Tweet 5

The CLI is a full interactive bot:

• Tab completion for gene names
• Protein feature visualization
• ONNX fast-path for speed
• Demo mode for testing
• Works on Windows, macOS, Linux

---

## Tweet 6

What I'd do differently:

• ClinVar data cleaning took longer than model training
• CPU inference is ~30s per prediction — fine for research, slow for production
• ESM-2 ONNX export was harder than expected

---

## Tweet 7

It's open source (MIT).

Clone it, break it, improve it.

GitHub: https://github.com/NOOBHEKER/dna-mutation-predictor

If it saved you time: ⭐ star the repo
If it inspired you: ☕ buy me a coffee

---

## Posting Tips

- Post Tweet 1 first, then reply to it with Tweets 2-7 (creates a thread)
- Add a screenshot of the CLI output to Tweet 1 for visual engagement
- Best posting times: 9-11 AM EST (when bioinformatics/ML people are active)
- Tag @github and @huggingface in Tweet 1 for visibility
- Use hashtags sparingly: #bioinformatics #opensourcesoftware #machinelearning
