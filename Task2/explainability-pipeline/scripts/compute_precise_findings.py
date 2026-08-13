"""
Computes the precise quantified deltas needed for Section 8's findings
paragraphs, matching Nouran's requested template style (e.g. "On XLM-R,
AttnLRP improves Precision@K by X and AUPRC by Y over the next-best
method"). Run from the repo root:

    python3 scripts/compute_precise_findings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd


def main():
    overall = pd.read_csv("outputs/tables/comparison_overall.csv")

    print("=== AttnLRP margin over next-best method, per model ===")
    for model in ["xlmr", "e5"]:
        for metric in ["precision_at_k", "auprc"]:
            model_data = overall[(overall["model"] == model) & (overall["metric"] == metric)].sort_values("mean", ascending=False)
            best = model_data.iloc[0]
            assert best["method"] == "attnlrp", f"Expected attnlrp to be best for {model}/{metric}"
            next_best = model_data.iloc[1]
            margin = best["mean"] - next_best["mean"]
            print(f"{model}: AttnLRP {metric} = {best['mean']:.4f}, next-best ({next_best['method']}) = {next_best['mean']:.4f}, margin = {margin:.4f}")

    print()
    print("=== mBERT spread across its 3 methods (no AttnLRP available) ===")
    for metric in ["precision_at_k", "auprc"]:
        mbert_data = overall[(overall["model"] == "mbert") & (overall["metric"] == metric)].sort_values("mean", ascending=False)
        best = mbert_data.iloc[0]
        next_best = mbert_data.iloc[1]
        margin = best["mean"] - next_best["mean"]
        print(f"mbert {metric}: best ({best['method']}) = {best['mean']:.4f}, next-best ({next_best['method']}) = {next_best['mean']:.4f}, margin = {margin:.4f}")

    print()
    print("=== Under Integrated Gradients specifically: which model wins on which metric? ===")
    for metric in ["precision_at_k", "auprc"]:
        ig_data = overall[(overall["method"] == "integrated_gradients") & (overall["metric"] == metric)].sort_values("mean", ascending=False)
        print(f"\n{metric} under Integrated Gradients:")
        for _, row in ig_data.iterrows():
            print(f"  {row['model']}: {row['mean']:.4f}")

    print()
    print("=== AttnLRP: XLM-R vs E5 specifically ===")
    for metric in ["precision_at_k", "auprc"]:
        xlmr_val = overall[(overall["model"] == "xlmr") & (overall["method"] == "attnlrp") & (overall["metric"] == metric)]["mean"].iloc[0]
        e5_val = overall[(overall["model"] == "e5") & (overall["method"] == "attnlrp") & (overall["metric"] == metric)]["mean"].iloc[0]
        margin = xlmr_val - e5_val
        print(f"{metric}: XLM-R = {xlmr_val:.4f}, E5 = {e5_val:.4f}, XLM-R higher by {margin:.4f}")


if __name__ == "__main__":
    main()