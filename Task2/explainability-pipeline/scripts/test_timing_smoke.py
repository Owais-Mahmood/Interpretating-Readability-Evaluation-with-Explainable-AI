"""
Smoke test for the timing utility, using mBERT + Integrated Gradients
on a few real examples. Run from the repo root:

    python3 scripts/test_timing_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer
from xai_pipeline.utils_timing import explain_with_timing


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:5]
    print(f"Loaded {len(examples)} examples.")

    print("Loading mBERT...")
    model = MBERTModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)

    print()
    print("Timing Integrated Gradients...")
    ig_explainer = IntegratedGradientsExplainer(n_steps=50)
    ig_explanations, ig_timing = explain_with_timing(ig_explainer, examples, model, predictions)
    print(ig_timing[["example_id", "metric", "value"]].to_string(index=False))

    print()
    print("Timing Raw Attention...")
    attn_explainer = RawAttentionExplainer(layer=-1)
    attn_explanations, attn_timing = explain_with_timing(attn_explainer, examples, model, predictions)
    print(attn_timing[["example_id", "metric", "value"]].to_string(index=False))

    print()
    print("=== Mean seconds_total by method ===")
    import pandas as pd
    combined = pd.concat([ig_timing, attn_timing], ignore_index=True)
    print(combined[combined["metric"] == "seconds_total"].groupby("method")["value"].mean())


if __name__ == "__main__":
    main()