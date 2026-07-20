# Task 2 - Step 1: Validate the fixed test set

import pandas as pd

df = pd.read_csv("Task2/test_set.csv")

# The 7 official strategy labels the models actually predict (per Task2_Description.docx)

OFFICIAL_LABELS = [
    "Synonymy", "Modulation", "Compression", "Explanation",
    "Syntactic Change", "Illocutionary Change", "Omission",
]
# Present in the CSV but NOT part of the model's predicted label set
EXTRA_COLUMNS = ["Transposition", "audit_Transcription"]

print(f"Total rows: {len(df)}")
print(f"Unique pair_id: {df['pair_id'].nunique()}")
print(f"Duplicated pair_id: {df['pair_id'].duplicated().sum()}")
print(f"All rows labelled (is_labelled == True): {(df['is_labelled'] == True).all()}")
print()

print("Missing values per column:")
print(df.isna().sum())
print()

print("Language distribution:")
print(df["language"].value_counts())
print()

print("Label prevalence (7 official labels the models predict):")
print(df[OFFICIAL_LABELS].sum().sort_values(ascending=False))
print()

print("Extra columns NOT in the official taxonomy:")
print(df[EXTRA_COLUMNS].sum())
print()

# Flag: rows whose ONLY gold label is one of the extra (non-predictable) columns

no_official_label = df[df[OFFICIAL_LABELS].sum(axis=1) == 0]
print(f"Rows with ZERO official-taxonomy labels: {len(no_official_label)}")
if len(no_official_label) > 0:
    print("These rows' actual gold label(s):")
    print(no_official_label[["pair_id", "labels"]].to_string(index=False))
    print()
    print("WARNING: these pairs cannot be evaluated against any of the 7 predicted")
    print("strategies, since their only gold label falls outside the official taxonomy.")
    print("This needs a decision: exclude these pairs from evaluation, or handle separately.")

# Check for missing alignment/span data needed for evaluation later

alignment_related_cols = [c for c in df.columns if any(k in c.lower() for k in ["align", "edit", "span"])]
print()
print(f"Alignment/edit/span columns found in test_set.csv: {alignment_related_cols if alignment_related_cols else 'NONE'}")
if not alignment_related_cols:
    print("WARNING: no word_alignments/edit_type/edited_tokens_or_spans columns exist.")
    print("This data will need to be joined in from Task 1's source_tokens.csv / word-level")
    print("alignment files using pair_id or source_text matching, before Precision@K etc. can be computed.")