import pandas as pd

df = pd.read_csv("data/raw/test_set_with_spans.csv")

NEW_LABELS = ["Synonymy", "Modulation", "Compression", "Explanation", "Syntactic Change", "Omission"]

n_no_label = (df[NEW_LABELS].sum(axis=1) == 0).sum()
print(f"Total pairs: {len(df)}")
print(f"Pairs with zero labels under the NEW 6-label taxonomy: {n_no_label}")

if n_no_label > 0:
    orphans = df[df[NEW_LABELS].sum(axis=1) == 0]
    print()
    print("These pairs' actual gold label(s):")
    print(orphans[["pair_id", "labels"]].to_string(index=False))