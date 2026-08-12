"""
Consolidates all metric results (plausibility, deletion/insertion,
attribution stability, processing time) into master comparison tables,
reported both overall and broken down per simplification strategy.

Run from the repo root, after the individual evaluation runs have
produced their CSV outputs in outputs/metrics/ and outputs/explanations/:

    python3 scripts/build_comparison_tables.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    print(f"NOT FOUND (skipped): {path}")
    return None


def main():
    outputs_dir = Path("outputs")
    metrics_dir = outputs_dir / "metrics"

    # Gather every metrics CSV we've produced so far
    sources = {
        "plausibility_mbert": metrics_dir / "plausibility_full_results.csv",
        "plausibility_encoders": metrics_dir / "plausibility_encoders_full_results.csv",
        "attnlrp_xlmr": metrics_dir / "attnlrp_xlmr_plausibility_results.csv",
        "attnlrp_e5": metrics_dir / "attnlrp_e5_plausibility_results.csv",
        "qwen": metrics_dir / "qwen_plausibility_results.csv",
    }

    frames = []
    for name, path in sources.items():
        df = load_if_exists(path)
        if df is not None:
            df["source"] = name
            frames.append(df)

    if not frames:
        print("No metric result files found -- nothing to aggregate.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Ensure a consistent "model" column exists (mbert results don't have
    # one, since that run only covered a single model)
    if "model" not in combined.columns:
        combined["model"] = "mbert"
    else:
        combined["model"] = combined["model"].fillna("mbert")

    # "Illocutionary Change" is part of mBERT's legacy 7-label taxonomy
    # (its trained classification head has 7 output neurons baked into
    # the model weights), not the current 6-label taxonomy used by every
    # other model. Per Nouran's instruction, this is separated out from
    # the primary analysis rather than mixed into the main comparison --
    # it's not deleted, just kept visible on its own as a clearly-labeled,
    # non-comparable legacy result.
    if "slice_value" in combined.columns:
        legacy_mask = combined["slice_value"] == "Illocutionary Change"
        legacy_results = combined[legacy_mask].copy()
        combined = combined[~legacy_mask].copy()

        if len(legacy_results) > 0:
            legacy_path = outputs_dir / "tables" / "legacy_illocutionary_change_mbert_only.csv"
            legacy_results.to_csv(legacy_path, index=False)
            print(f"Separated {len(legacy_results)} legacy 'Illocutionary Change' rows "
                  f"(mBERT 7-label taxonomy only, not comparable to the current 6-label "
                  f"taxonomy) -- saved separately to {legacy_path}")

    print()
    print(f"Total combined metric records (primary, 6-label taxonomy only): {len(combined)}")
    print()

    # Overall comparison table: mean metric value by model x method
    overall = combined.groupby(["model", "method", "metric"])["value"].agg(["mean", "std", "count"])
    overall_path = outputs_dir / "tables" / "comparison_overall.csv"
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(overall_path)
    print(f"Saved overall comparison table to {overall_path}")
    print()
    print("=== Overall comparison (mean value) ===")
    print(combined.groupby(["model", "method", "metric"])["value"].mean().unstack().to_string())

    # --- Per-strategy breakdown: mean metric value by model x method x strategy ---
    if "slice_value" in combined.columns:
        per_strategy = combined.groupby(["model", "method", "slice_value", "metric"])["value"].agg(
            ["mean", "std", "count"]
        )
        per_strategy_path = outputs_dir / "tables" / "comparison_per_strategy.csv"
        per_strategy.to_csv(per_strategy_path)
        print()
        print(f"Saved per-strategy breakdown to {per_strategy_path}")
        print()
        print("=== Per-strategy breakdown (mean value, precision_at_k only, as an example) ===")
        precision_only = combined[combined["metric"] == "precision_at_k"]
        if len(precision_only) > 0:
            print(
                precision_only.groupby(["model", "method", "slice_value"])["value"]
                .mean()
                .unstack()
                .to_string()
            )
        else:
            print("(no precision_at_k rows found in the combined data)")


if __name__ == "__main__":
    main()