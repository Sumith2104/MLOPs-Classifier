"""
assign_priority_labels.py
--------------------------
Reads the raw customer_support_100k.csv dataset (which has only intent/category
labels) and assigns a 'priority_label' column using rule-based keyword heuristics.

Priority Rules:
  HIGH   – Contains urgency/severity keywords, or is a complaint with
            critical action words (refund, stolen, broken, etc.)
  LOW    – General inquiry (no urgency) OR positive feedback
  MEDIUM – Everything else

Output: data/processed/labeled.csv
"""

import os
import re
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]          # MLops/
RAW_CSV    = BASE_DIR / "data" / "raw" / "customer_support_100k.csv"
OUT_DIR    = BASE_DIR / "data" / "processed"
OUT_CSV    = OUT_DIR / "labeled.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Keyword Lists ─────────────────────────────────────────────────────────────
HIGH_KEYWORDS = [
    r"\burgent\b", r"\bimmediately\b", r"\basap\b", r"\bemergency\b",
    r"\bcritical\b", r"\boutage\b", r"\bfraud\b", r"\bscam\b",
    r"\bhacked\b", r"\bsuspended\b", r"\bblocked\b", r"\blegal\b",
    r"\blawsuit\b", r"\bcannot access\b", r"\bdata loss\b", r"\bdata breach\b",
    r"\bstolen\b", r"\bnot working\b", r"\bbroken\b", r"\bcharged\b",
    r"\bunauthorized\b", r"\brefund\b", r"\bdeadline\b", r"\bescalat\b",
]

POSITIVE_KEYWORDS = [
    r"\bgreat\b", r"\bexcellent\b", r"\blove\b", r"\bamazing\b",
    r"\bwonderful\b", r"\bhappy\b", r"\bsatisfied\b", r"\bthank(s| you)\b",
    r"\bappreciat\b", r"\bfantastic\b", r"\bperfect\b", r"\bawesome\b",
    r"\bimpressive\b", r"\bbrilliant\b", r"\boutstanding\b",
]

HIGH_RE     = re.compile("|".join(HIGH_KEYWORDS),     re.IGNORECASE)
POSITIVE_RE = re.compile("|".join(POSITIVE_KEYWORDS), re.IGNORECASE)


def assign_priority(row: pd.Series) -> str:
    """Return 'high', 'medium', or 'low' based on text + intent category."""
    text     = str(row["text"]).lower()
    category = str(row["category"]).lower()

    # HIGH – any urgency keyword present
    if HIGH_RE.search(text):
        return "high"

    # LOW – positive feedback OR generic inquiry with no urgency
    if category == "feedback" and POSITIVE_RE.search(text):
        return "low"
    if category == "inquiry" and not HIGH_RE.search(text):
        return "low"

    # MEDIUM – default
    return "medium"


def main():
    print(f"Loading dataset from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, encoding="utf-8", encoding_errors="replace")

    # Standardise column names FIRST, then apply
    df.rename(columns={"message": "text"}, inplace=True)
    df["category"] = df["category"].str.strip().str.lower()
    df["text"]     = df["text"].astype(str).str.strip()

    # Assign priority using cleaned columns
    df["priority_label"] = df.apply(assign_priority, axis=1)

    # Rename category → intent_label after apply (assign_priority uses 'category')
    df.rename(columns={"category": "intent_label"}, inplace=True)

    # Keep only relevant columns
    df_out = df[["text", "intent_label", "priority_label"]].copy()
    df_out.dropna(inplace=True)

    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved labeled dataset: {OUT_CSV}  ({len(df_out):,} rows)")

    print("\n-- Intent distribution --")
    print(df_out["intent_label"].value_counts())
    print("\n-- Priority distribution --")
    print(df_out["priority_label"].value_counts())


if __name__ == "__main__":
    main()
