import pandas as pd
import requests
import gzip
import shutil
import os
from tqdm import tqdm

# ClinVar FTP URL for variant summary
CLINVAR_URL = "ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
RAW_DATA_PATH = "data/variant_summary.txt.gz"
FILTERED_DATA_PATH = "data/variants.csv"

def download_data():
    print(f"Downloading ClinVar data from {CLINVAR_URL}...")
    if os.path.exists(RAW_DATA_PATH):
        print("Raw data already exists. Skipping download.")
        return

    # Note: requests doesn't support ftp:// directly.
    # We use urllib or a shell command. For simplicity in Python,
    # we can use urllib.request.urlretrieve.
    import urllib.request

    try:
        urllib.request.urlretrieve(CLINVAR_URL, RAW_DATA_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

def filter_variants():
    print("Filtering variants...")

    # We use a streaming approach to handle the large file
    pathogenic_count = 0
    benign_count = 0

    # Column names for variant_summary.txt are not in the file itself usually,
    # but we can infer them or use a known list.
    # Typical columns:
    # 0: Variant ID, 1: Chromosome, 2: Start, 3: Stop, 4: Reference, 5: Alternate,
    # 6: GeneSymbol, 7: HGVSc, 8: HGVSp, 9: ClinicalSignificance, ...

    # We only need: GeneSymbol, HGVSp, ClinicalSignificance
    # HGVSp is usually index 8, ClinicalSignificance is index 9, GeneSymbol is index 6

    filtered_data = []

    # Open the gzipped file
    with gzip.open(RAW_DATA_PATH, 'rt') as f:
        # Skip header if present (usually the first line starts with #)
        for line in f:
            if line.startswith('#'):
                continue

            cols = line.strip().split('\t')
            if len(cols) < 10:
                continue

            gene_symbol = cols[4]
            name = cols[2]
            clin_sig = cols[6]

            # Extract HGVSp from Name column (e.g., "... (p.Gly1046Arg)")
            import re
            hgvsp_match = re.search(r'\(p\.[^)]+\)', name)
            if not hgvsp_match:
                continue
            hgvsp = hgvsp_match.group(0)[1:-1] # Remove parentheses

            # Filter for Missense: HGVSp should contain a change like p.Arg50Gln
            # We exclude deletions, insertions, frameshifts (fs), or terminations (Ter)
            if 'fs' in hgvsp or 'del' in hgvsp or 'ins' in hgvsp or 'Ter' in hgvsp:
                continue

            # Filter for Pathogenic vs Benign
            # ClinVar labels can be complex (e.g., "Pathogenic/Likely pathogenic")
            label = None
            clin_sig_lower = clin_sig.lower()
            if 'pathogenic' in clin_sig_lower and 'benign' not in clin_sig_lower:
                label = 1
                pathogenic_count += 1
            elif 'benign' in clin_sig_lower and 'pathogenic' not in clin_sig_lower:
                label = 0
                benign_count += 1

            if label is not None:
                filtered_data.append([gene_symbol, hgvsp, label])

                # Limit dataset size for initial implementation as discussed
                if len(filtered_data) >= 100000:
                    break

    df = pd.DataFrame(filtered_data, columns=['gene_symbol', 'hgvsp', 'label'])

    # Balance the dataset if necessary
    counts = df['label'].value_counts()
    print(f"Found counts:\n{counts}")

    if 0 not in counts or 1 not in counts:
        print("Error: One or both classes (Benign/Pathogenic) are missing from the filtered data.")
        print("Check the ClinVar file format and filtering logic.")
        df.to_csv(FILTERED_DATA_PATH, index=False)
        return

    min_count = min(counts[0], counts[1])

    df_benign = df[df['label'] == 0].sample(min_count, random_state=42)
    df_pathogenic = df[df['label'] == 1].sample(min_count, random_state=42)

    df_balanced = pd.concat([df_benign, df_pathogenic]).sample(frac=1, random_state=42)

    df_balanced.to_csv(FILTERED_DATA_PATH, index=False)
    print(f"Filtered and balanced data saved to {FILTERED_DATA_PATH}")
    print(f"Final dataset size: {len(df_balanced)} ( {min_count} Benign, {min_count} Pathogenic )")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_data()
    filter_variants()
