"""
Smoke test for the attribution stability evaluator, using mBERT +
Integrated Gradients. Run from the repo root:

    python3 scripts/test_attribution_stability_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.evaluators.attribution_stability_impl import AttributionStabilityEvaluator


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:2]
    print(f"Loaded {len(examples)} examples.")

    print("Loading mBERT...")
    model = MBERTModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)
    for example, prediction in zip(examples, predictions):
        print(f"{example.example_id}: predicted {prediction.predicted_label}")

    print()
    print("Running Integrated Gradients (original, unperturbed)...")
    explainer = IntegratedGradientsExplainer(n_steps=50)
    explanations = explainer.explain(examples, model, predictions)
    print(f"Produced {len(explanations)} explanations.")

    print()
    print("Running attribution stability evaluation (this re-runs the explainer 3x per explanation)...")
    evaluator = AttributionStabilityEvaluator(model=model, explainer=explainer, n_perturbations=3, noise_std=0.01)
    results = evaluator.evaluate(examples, predictions, explanations)
    print(results[["example_id", "slice_value", "metric", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()