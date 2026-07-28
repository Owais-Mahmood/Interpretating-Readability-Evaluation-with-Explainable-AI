"""
Full-scale run: all 260 test pairs, all 3 working explainers
(Integrated Gradients, GradientSHAP, Raw Attention), evaluated against
real gold spans. Run from the repo root:

    python3 scripts/run_full_evaluation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.gradient_shap_impl import GradientShapExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer
from xai_pipeline.evaluators.plausibility_impl import PlausibilityEvaluator


def main():
    # 1. Load ALL examples (no [:3] slicing this time)
    dataset = SimplificationDatasetAdapter("data/raw/test_set_with_spans.csv")
    errors = dataset.validate()
    if errors:
        print("Dataset validation errors:", errors)
        return
    examples = dataset.load("test")
    print(f"Loaded {len(examples)} examples for the full run.")

    # 2. Load mBERT
    print("Loading mBERT...")
    model = MBERTModelAdapter()
    model.load()
    print("Model loaded successfully.")

    # 3. Run predictions for everything up front
    print("Running predictions on all examples...")
    predictions = model.predict(examples)
    print("Predictions done.")

    # 4. Run each explainer across ALL examples, one at a time, with progress
    explainers = {
        "integrated_gradients": IntegratedGradientsExplainer(n_steps=50),
        "gradient_shap": GradientShapExplainer(n_samples=25, n_baselines=5),
        "raw_attention": RawAttentionExplainer(layer=-1),
    }

    all_explanations = []
    for method_name, explainer in explainers.items():
        print()
        print(f"Running {method_name} on all {len(examples)} examples...")
        start = time.time()
        method_explanations = []
        for i, (example, prediction) in enumerate(zip(examples, predictions)):
            method_explanations.extend(explainer.explain([example], model, [prediction]))
            if (i + 1) % 20 == 0 or (i + 1) == len(examples):
                elapsed = time.time() - start
                print(f"  {method_name}: {i + 1}/{len(examples)} examples done ({elapsed:.0f}s elapsed)")
        all_explanations.extend(method_explanations)
        print(f"{method_name} finished in {time.time() - start:.0f}s, produced {len(method_explanations)} explanations.")

    # 5. Evaluate everything against the real gold spans
    print()
    print("Running plausibility evaluation on all explanations...")
    evaluator = PlausibilityEvaluator(tokenizer=model.tokenizer)
    results = evaluator.evaluate(examples, predictions, all_explanations)
    print(f"Evaluation produced {len(results)} metric records.")

    # 6. Save full per-example results
    results_path = Path("outputs/metrics/plausibility_full_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"Saved full results to {results_path}")

    # 7. Save and print aggregated summary by method
    summary = results.groupby(["method", "metric"])["value"].agg(["mean", "std", "count"])
    summary_path = Path("outputs/tables/plausibility_summary_by_method.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")
    print()
    print("=== Summary: mean metric value by method ===")
    print(results.groupby(["method", "metric"])["value"].mean().unstack().to_string())


if __name__ == "__main__":
    main()