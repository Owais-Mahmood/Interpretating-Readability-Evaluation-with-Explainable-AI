"""
build_split_summary.py

Generates data/processed/split_summary.csv — a compact count-check table
used to cross-verify that pairs_with_splits.csv, source_tokens.csv and
classification_instances_with_splits.csv all agree on how many records
fall into each split (train/validation/test). This is what Cell 13 of the
notebook (split leakage checks) cross-references against.

Run this from the repo root, e.g.:
    python3 scripts/build_split_summary.py

Output:
    data/processed/split_summary.csv
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main():

    # 1. Load the three files already built

    pairs = pd.read_csv(PROCESSED_DIR / "pairs_with_splits.csv", encoding="utf-8-sig")
    tokens = pd.read_csv(PROCESSED_DIR / "source_tokens.csv", encoding="utf-8-sig")
    instances = pd.read_csv(PROCESSED_DIR / "classification_instances_with_splits.csv", encoding="utf-8-sig")

    assert len(pairs) == 1930, f"Expected 1930 pairs, got {len(pairs)}"
    assert len(instances) == 3860, f"Expected 3860 instances, got {len(instances)}"

    # 2. source_tokens.csv doesn't carry split directly, so bring it in from pairs

    tokens = tokens.merge(pairs[["pair_id", "split"]], on="pair_id", how="left")

    # 3. Count each table by split

    n_pairs = pairs.groupby("split").size().rename("n_pairs")
    n_languages = pairs.groupby("split")["language"].nunique().rename("n_languages")
    n_documents = pairs.groupby("split")["document_id"].nunique().rename("n_documents")
    n_source_tokens = tokens.groupby("split").size().rename("n_source_token_rows")
    n_instances = instances.groupby("split").size().rename("n_classifier_instances")

    summary = pd.concat(
        [n_pairs, n_languages, n_documents, n_source_tokens, n_instances], axis=1
    ).reset_index()

    # 4. Add an overall "all splits" row for convenience

    total_row = pd.DataFrame(
        [
            {
                "split": "all",
                "n_pairs": pairs.shape[0],
                "n_languages": pairs["language"].nunique(),
                "n_documents": pairs["document_id"].nunique(),
                "n_source_token_rows": tokens.shape[0],
                "n_classifier_instances": instances.shape[0],
            }
        ]
    )
    summary = pd.concat([summary, total_row], ignore_index=True)

    # 5. Sanity checks before saving

    per_split = summary[summary["split"] != "all"]
    assert per_split["n_pairs"].sum() == 1930, "Per-split pair counts must sum to 1930"
    assert per_split["n_classifier_instances"].sum() == 3860, "Per-split instance counts must sum to 3860"
    assert (per_split["n_classifier_instances"] == per_split["n_pairs"] * 2).all(), (
        "Each split must have exactly twice as many classifier instances as pairs"
    )

    print(f"Final split_summary.csv: {len(summary)} rows, {len(summary.columns)} columns")
    print(summary.to_string(index=False))

    # 6. Save

    out_path = PROCESSED_DIR / "split_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()