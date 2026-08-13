"""
Builds the three separate comparison analyses Nouran asked for:
1. Method comparison: best explainer WITHIN each model
2. Model comparison: best model under a SHARED method (fair comparison
   only across models that actually have that same method)
3. Strategy comparison: best model+method combination per strategy

Run from the repo root:
    python3 scripts/build_three_comparisons.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd


def method_comparison(overall: pd.DataFrame):
    """Within each model, which method scores highest?"""
    print("=" * 70)
    print("1. METHOD COMPARISON (best explainer WITHIN each model)")
    print("=" * 70)
    precision = overall[overall["metric"] == "precision_at_k"]
    for model in sorted(precision["model"].unique()):
        model_data = precision[precision["model"] == model].sort_values("mean", ascending=False)
        best = model_data.iloc[0]
        print(f"\n{model.upper()}:")
        for _, row in model_data.iterrows():
            marker = " <- BEST" if row["method"] == best["method"] else ""
            print(f"  {row['method']:30s} Precision@K = {row['mean']:.4f}{marker}")
        if len(model_data) > 1:
            margin = best["mean"] - model_data.iloc[1]["mean"]
            print(f"  Margin over next-best: {margin:.4f}")


def model_comparison(overall: pd.DataFrame):
    """Under each shared method, which model scores highest? Only
    compares models that actually share that exact method -- avoids
    the trap of concluding a model is 'better' when it was really just
    evaluated with a stronger method than its competitors."""
    print()
    print("=" * 70)
    print("2. MODEL COMPARISON (best model under a SHARED, comparable method)")
    print("=" * 70)
    precision = overall[overall["metric"] == "precision_at_k"]
    for method in sorted(precision["method"].unique()):
        method_data = precision[precision["method"] == method].sort_values("mean", ascending=False)
        if len(method_data) < 2:
            print(f"\n{method}: only 1 model has this method -- no comparison possible")
            continue
        best = method_data.iloc[0]
        print(f"\nUnder {method} (models that actually share this method: {list(method_data['model'])}):")
        for _, row in method_data.iterrows():
            marker = " <- BEST" if row["model"] == best["model"] else ""
            print(f"  {row['model']:10s} Precision@K = {row['mean']:.4f}{marker}")
        margin = best["mean"] - method_data.iloc[1]["mean"]
        print(f"  Margin over next-best: {margin:.4f}")


def strategy_comparison(per_strategy: pd.DataFrame):
    """For each strategy, which model+method combination scores highest?"""
    print()
    print("=" * 70)
    print("3. STRATEGY COMPARISON (best model+method combination per strategy)")
    print("=" * 70)
    precision = per_strategy[per_strategy["metric"] == "precision_at_k"]
    for strategy in sorted(precision["slice_value"].unique()):
        strategy_data = precision[precision["slice_value"] == strategy].sort_values("mean", ascending=False)
        best = strategy_data.iloc[0]
        print(f"\n{strategy}:")
        print(f"  BEST: {best['model']}/{best['method']} = {best['mean']:.4f}")
        if len(strategy_data) > 1:
            second = strategy_data.iloc[1]
            margin = best["mean"] - second["mean"]
            print(f"  Next-best: {second['model']}/{second['method']} = {second['mean']:.4f} (margin: {margin:.4f})")


def main():
    overall = pd.read_csv("outputs/tables/comparison_overall.csv")
    per_strategy = pd.read_csv("outputs/tables/comparison_per_strategy.csv")

    method_comparison(overall)
    model_comparison(overall)
    strategy_comparison(per_strategy)


if __name__ == "__main__":
    main()