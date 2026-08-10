"""
Smoke test for the deletion/insertion evaluator, using mBERT + Integrated
Gradients on a couple of real examples. Run from the repo root:

    python3 scripts/test_deletion_insertion_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.evaluators.deletion_insertion_impl import DeletionInsertionEvaluator


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
    print("Running Integrated Gradients...")
    explainer = IntegratedGradientsExplainer(n_steps=50)
    explanations = explainer.explain(examples, model, predictions)
    print(f"Produced {len(explanations)} explanations.")

    print()
    print("Running deletion/insertion evaluation...")
    evaluator = DeletionInsertionEvaluator(model=model, n_steps=10)
    results = evaluator.evaluate(examples, predictions, explanations)
    print(results[["example_id", "slice_value", "metric", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()