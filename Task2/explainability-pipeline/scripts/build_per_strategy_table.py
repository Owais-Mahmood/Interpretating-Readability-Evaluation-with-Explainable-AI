"""
Builds the full per-strategy results table for the report: Precision@K,
Recall@K, F1, and AUPRC for every strategy, model, and applicable
method, together with the real ground-truth sample count (n) for each
strategy -- how many test pairs actually carry that gold label, not
how many times a model happened to predict it.

Run from the repo root:
    python3 scripts/build_per_strategy_table.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter, LABELS


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")
    print(f"Loaded {len(examples)} examples from the final test set.")

    # Real ground-truth prevalence: how many examples actually carry
    # each gold label (a pair can have more than one gold strategy).
    # Gold labels are stored as example.labels, a dict of label -> 0/1.
    strategy_counts = {label: 0 for label in LABELS}
    for example in examples:
        for label in LABELS:
            if example.labels.get(label) == 1:
                strategy_counts[label] += 1

    print()
    print("Ground-truth sample counts (n) per strategy:")
    for label, count in strategy_counts.items():
        print(f"  {label}: {count}")

    per_strategy = pd.read_csv("outputs/tables/comparison_per_strategy.csv")

    rows = []
    for _, row in per_strategy.iterrows():
        strategy = row["slice_value"]
        if strategy not in strategy_counts:
            continue  # skip anything not in the current 6-label taxonomy
        rows.append({
            "strategy": strategy,
            "n": strategy_counts[strategy],
            "model": row["model"],
            "method": row["method"],
            "metric": row["metric"],
            "mean": round(row["mean"], 4),
        })

    full_table = pd.DataFrame(rows)
    output_path = Path("outputs/tables/per_strategy_full_table.csv")
    full_table.to_csv(output_path, index=False)
    print()
    print(f"Saved full per-strategy table to {output_path}")

    # Also produce a wide, report-ready pivot: one row per
    # strategy/model/method, one column per metric
    wide = full_table.pivot_table(
        index=["strategy", "n", "model", "method"], columns="metric", values="mean"
    ).reset_index()
    wide = wide[["strategy", "n", "model", "method", "precision_at_k", "recall_at_k", "f1_at_k", "auprc"]]
    wide_path = Path("outputs/tables/per_strategy_wide_table.csv")
    wide.to_csv(wide_path, index=False)
    print(f"Saved wide (report-ready) table to {wide_path}")
    print()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()