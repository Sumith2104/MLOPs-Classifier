"""
preprocess.py
--------------
1. Reads data/processed/labeled.csv
2. Cleans text (lowercasing, whitespace, special char removal)
3. Encodes labels as integers
4. Performs a stratified 80 / 10 / 10 split on combined intent+priority strata
5. Saves train.csv, val.csv, test.csv to data/processed/

Run:
    python training/scripts/preprocess.py
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
LABELED_CSV = BASE_DIR / "data" / "processed" / "labeled.csv"
OUT_DIR     = BASE_DIR / "data" / "processed"

# ── Label Maps ───────────────────────────────────────────────────────────────
INTENT_MAP   = {"complaint": 0, "inquiry": 1, "feedback": 2}
PRIORITY_MAP = {"high": 0, "medium": 1, "low": 2}


# ── Text Cleaning ─────────────────────────────────────────────────────────────
_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE   = re.compile(r"<[^>]+>")
_MULTI_RE  = re.compile(r"\s+")
_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9\s'.,!?-]")


def clean_text(text: str) -> str:
    text = str(text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    text = _MULTI_RE.sub(" ", text)
    return text.strip().lower()


def main():
    print(f"Reading: {LABELED_CSV}")
    df = pd.read_csv(LABELED_CSV)

    # ── Clean text ────────────────────────────────────────────────────────────
    print("Cleaning text …")
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text", "intent_label", "priority_label"], inplace=True)

    # ── Encode labels ─────────────────────────────────────────────────────────
    df["intent_label"]   = df["intent_label"].str.strip().str.lower().map(INTENT_MAP)
    df["priority_label"] = df["priority_label"].str.strip().str.lower().map(PRIORITY_MAP)
    df.dropna(subset=["intent_label", "priority_label"], inplace=True)
    df["intent_label"]   = df["intent_label"].astype(int)
    df["priority_label"] = df["priority_label"].astype(int)

    # ── Stratified split using combined stratum ───────────────────────────────
    df["_stratum"] = df["intent_label"].astype(str) + "_" + df["priority_label"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["_stratum"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["_stratum"], random_state=42
    )

    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["_stratum"], inplace=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv",   index=False)
    test_df.to_csv(OUT_DIR / "test.csv",  index=False)

    print(f"\nSplit sizes -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print("\nIntent label distribution (train):")
    print(train_df["intent_label"].value_counts().sort_index())
    print("\nPriority label distribution (train):")
    print(train_df["priority_label"].value_counts().sort_index())

    print("\n[SUCCESS] Preprocessing complete. Files saved to data/processed/")


if __name__ == "__main__":
    main()
