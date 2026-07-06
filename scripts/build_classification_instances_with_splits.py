"""
build_classification_instances.py

Generates data/processed/classification_instances_with_splits.csv from
data/processed/pairs_with_splits.csv.

Each sentence pair becomes TWO classifier instances:
  - one "original" instance (the complex sentence), label = 1 (complex)
  - one "simplified" instance (the simple sentence), label = 0 (easy)

This is the format a complexity classifier is actually trained on: single
sentences with a binary label, not paired complex/simple rows.

Run this from the repo root, e.g.:
    python3 scripts/build_classification_instances.py

Output:
    data/processed/classification_instances_with_splits.csv
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

def main():

    # 1. Load the pair-level table we already built

    pairs = pd.read_csv(PROCESSED_DIR / "pairs_with_splits.csv", encoding="utf-8-sig")
    assert len(pairs) == 1930, f"Expected 1930 pairs, got {len(pairs)}"

    # 2. Build the "original" (complex) instance for every pair

    original = pairs[["pair_id", "language", "collection", "document_id", "split", "source_text", "n_source_tokens"]].copy()
    original = original.rename(columns={"source_text": "text", "n_source_tokens": "n_tokens"})
    original["instance_id"] = original["pair_id"] + "_original"
    original["side"] = "original"
    original["label"] = 1  # complex

    # 3. Build the "simplified" (easy) instance for every pair

    simplified = pairs[["pair_id", "language", "collection", "document_id", "split", "target_text", "n_target_tokens"]].copy()
    simplified = simplified.rename(columns={"target_text": "text", "n_target_tokens": "n_tokens"})
    simplified["instance_id"] = simplified["pair_id"] + "_simplified"
    simplified["side"] = "simplified"
    simplified["label"] = 0  # easy

    # 4. Stack them into one long table, ordered so each pair's two instances sit together

    instances = pd.concat([original, simplified], ignore_index=True)
    instances = instances.sort_values(["pair_id", "side"]).reset_index(drop=True)

    # 5. Column order

    instances = instances[
        [
            "instance_id",
            "pair_id",
            "side",
            "label",
            "text",
            "n_tokens",
            "language",
            "collection",
            "document_id",
            "split",
        ]
    ]

    # 6. Sanity checks before saving

    assert len(instances) == 3860, f"Expected 3860 instances, got {len(instances)}"
    assert instances["instance_id"].is_unique, "instance_id must be unique"

    per_pair_counts = instances.groupby("pair_id").size()
    assert (per_pair_counts == 2).all(), "Every pair_id must have exactly one original and one simplified instance"

    per_pair_splits = instances.groupby("pair_id")["split"].nunique()
    assert (per_pair_splits == 1).all(), "Every pair_id must appear in exactly one split"

    label_side_check = instances[
        ((instances["side"] == "original") & (instances["label"] != 1))
        | ((instances["side"] == "simplified") & (instances["label"] != 0))
    ]
    assert len(label_side_check) == 0, "label must agree with side: original=1 (complex), simplified=0 (easy)"

    print(f"Final classification_instances_with_splits.csv: {len(instances)} rows, {len(instances.columns)} columns")
    print("Columns:", list(instances.columns))
    print("Label counts:")
    print(instances["label"].value_counts())
    print("Split counts:")
    print(instances["split"].value_counts())

    # 7. Save

    out_path = PROCESSED_DIR / "classification_instances_with_splits.csv"
    instances.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()