"""
Joins Task 1's deletion/addition span data onto Task 2's test_set.csv,
via text-content matching (the two files use different pair_id schemes).

Run from the repo root:
    python3 scripts/build_test_set_with_spans.py
"""

import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

test_set = pd.read_csv(REPO_ROOT / "Task2" / "test_set.csv")
pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "pairs_with_splits.csv")
spans = pd.read_csv(REPO_ROOT / "data" / "derived" / "derived_deletion_addition_spans.csv")

# Step 1: bridge test_set.csv's pair_id to Task 1's pair_id, via text content
crosswalk = test_set[["pair_id", "language", "source_text", "simplified_text"]].merge(
    pairs[["pair_id", "language", "source_text", "target_text"]],
    left_on=["language", "source_text", "simplified_text"],
    right_on=["language", "source_text", "target_text"],
    how="left",
    suffixes=("_task2", "_task1"),
)

n_unmatched = crosswalk["pair_id_task1"].isna().sum()
assert n_unmatched == 0, f"{n_unmatched} test_set rows could not be matched to Task 1 data"

id_map = dict(zip(crosswalk["pair_id_task2"], crosswalk["pair_id_task1"]))
test_set["task1_pair_id"] = test_set["pair_id"].map(id_map)

# Step 2: attach deletion (complex-side) and insertion (simple-side) spans
deletion_spans_by_pair = {}
insertion_spans_by_pair = {}

for task1_pid, group in spans.groupby("pair_id"):
    complex_spans = group[group["side"] == "complex"]
    simple_spans = group[group["side"] == "simple"]

    deletion_spans_by_pair[task1_pid] = [
        {
            "span_text": row["span_text"],
            "start_index": row["start_index"],
            "end_index": row["end_index"],
            "span_length": row["span_length"],
        }
        for _, row in complex_spans.iterrows()
    ]
    insertion_spans_by_pair[task1_pid] = [
        {
            "span_text": row["span_text"],
            "start_index": row["start_index"],
            "end_index": row["end_index"],
            "span_length": row["span_length"],
        }
        for _, row in simple_spans.iterrows()
    ]

test_set["deletion_spans"] = test_set["task1_pair_id"].map(
    lambda pid: json.dumps(deletion_spans_by_pair.get(pid, []))
)
test_set["insertion_spans"] = test_set["task1_pair_id"].map(
    lambda pid: json.dumps(insertion_spans_by_pair.get(pid, []))
)
test_set["n_deletion_spans"] = test_set["deletion_spans"].apply(lambda s: len(json.loads(s)))
test_set["n_insertion_spans"] = test_set["insertion_spans"].apply(lambda s: len(json.loads(s)))

# Step 3: sanity checks
n_no_spans_at_all = ((test_set["n_deletion_spans"] == 0) & (test_set["n_insertion_spans"] == 0)).sum()
print(f"Total test pairs: {len(test_set)}")
print(f"Pairs with at least one deletion span: {(test_set['n_deletion_spans'] > 0).sum()}")
print(f"Pairs with at least one insertion span: {(test_set['n_insertion_spans'] > 0).sum()}")
print(f"Pairs with NO spans at all (fully KEEP/SUBSTITUTE only): {n_no_spans_at_all}")

# Step 4: save
output_path = REPO_ROOT / "Task2" / "test_set_with_spans.csv"
test_set.to_csv(output_path, index=False)
print(f"Saved to {output_path}")