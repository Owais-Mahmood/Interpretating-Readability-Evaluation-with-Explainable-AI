"""
Full-scale run for the new encoder models (XLM-R and E5): all 260 test
pairs, all 3 working explainers, evaluated against real gold spans.
Produces the model comparison Nouran asked for. Run from the repo root:

    python3 scripts/run_full_evaluation_encoders.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.encoder_models import XLMRModelAdapter, E5ModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.gradient_shap_impl import GradientShapExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer
from xai_pipeline.evaluators.plausibility_impl import PlausibilityEvaluator


def run_for_model(model_name, model_adapter_cls, examples):
    print()
    print(f"########## {model_name} ##########")
    print(f"Loading {model_name}...")
    model = model_adapter_cls()
    model.load()
    print(f"{model_name} loaded successfully.")

    print("Running predictions on all examples...")
    predictions = model.predict(examples)
    print("Predictions done.")

    explainers = {
        "integrated_gradients": IntegratedGradientsExplainer(n_steps=50),
        "gradient_shap": GradientShapExplainer(n_samples=25, n_baselines=5),
        "raw_attention": RawAttentionExplainer(layer=-1),
    }

    all_explanations = []
    for method_name, explainer in explainers.items():
        print()
        print(f"[{model_name}] Running {method_name} on all {len(examples)} examples...")
        start = time.time()
        method_explanations = []
        for i, (example, prediction) in enumerate(zip(examples, predictions)):
            method_explanations.extend(explainer.explain([example], model, [prediction]))
            if (i + 1) % 40 == 0 or (i + 1) == len(examples):
                elapsed = time.time() - start
                print(f"  [{model_name}] {method_name}: {i + 1}/{len(examples)} done ({elapsed:.0f}s)")
        all_explanations.extend(method_explanations)
        print(f"[{model_name}] {method_name} finished in {time.time() - start:.0f}s, {len(method_explanations)} explanations.")

    print(f"[{model_name}] Running plausibility evaluation...")
    evaluator = PlausibilityEvaluator(tokenizer=model.tokenizer)
    results = evaluator.evaluate(examples, predictions, all_explanations)
    results["model"] = model_name
    print(f"[{model_name}] Evaluation produced {len(results)} metric records.")

    del model
    return results


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_with_spans.csv")
    errors = dataset.validate()
    blocking_errors = [e for e in errors if not e.startswith("WARNING")]
    if errors:
        print("Dataset validation messages:", errors)
    if blocking_errors:
        print("Stopping due to blocking errors above.")
        return

    examples = dataset.load("test")
    print(f"Loaded {len(examples)} examples for the full run.")

    all_results = []
    for model_name, model_cls in [("xlmr", XLMRModelAdapter), ("e5", E5ModelAdapter)]:
        results = run_for_model(model_name, model_cls, examples)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)

    results_path = Path("outputs/metrics/plausibility_encoders_full_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(results_path, index=False)
    print(f"Saved full results to {results_path}")

    summary = combined.groupby(["model", "method", "metric"])["value"].mean().unstack()
    summary_path = Path("outputs/tables/plausibility_summary_by_model_and_method.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")
    print()
    print("=== Summary: mean metric value by model and method ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()