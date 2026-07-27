"""
Quick end-to-end smoke test: load a few real examples, load mBERT,
run predictions, then run Integrated Gradients on the first example's
predicted label(s). Run this from the repo root:

    python3 scripts/test_ig_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer


def main():
    # 1. Load a handful of real examples (adjust path to wherever test_set.csv actually is)
    dataset = SimplificationDatasetAdapter("data/raw/test_set.csv")
    errors = dataset.validate()
    if errors:
        print("Dataset validation errors:", errors)
        return
    examples = dataset.load("test")[:3]  # just the first 3, for a quick smoke test
    print(f"Loaded {len(examples)} examples for the smoke test.")

    # 2. Load mBERT
    print("Loading mBERT (this will download the model on first run)...")
    model = MBERTModelAdapter()
    model.load()
    print("Model loaded successfully.")

    # 3. Run predictions
    predictions = model.predict(examples)
    for example, prediction in zip(examples, predictions):
        print(f"{example.example_id}: predicted {prediction.predicted_label}")

    # 4. Run Integrated Gradients on the first example only (keep it small/fast)
    print()
    print("Running Integrated Gradients on the first example...")
    explainer = IntegratedGradientsExplainer(n_steps=50)
    explanations = explainer.explain(examples[:1], model, predictions[:1])

    for explanation in explanations:
        print(f"\nExplanation for {explanation.example_id}, target={explanation.target}")
        print(f"Convergence delta: {explanation.metadata['convergence_delta']:.4f}")
        for token, score in sorted(
            zip(explanation.units, explanation.scores),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )[:5]:
            print(f"  {token}: {score:.4f}")


if __name__ == "__main__":
    main()