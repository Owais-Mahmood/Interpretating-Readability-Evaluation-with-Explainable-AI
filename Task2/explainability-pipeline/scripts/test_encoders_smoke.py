"""
Smoke test for the new XLM-R and E5 encoder models: load a few real
examples, load each model, run predictions. Run from the repo root:

    python3 scripts/test_encoders_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.encoder_models import XLMRModelAdapter, E5ModelAdapter


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_with_spans.csv")
    errors = dataset.validate()
    blocking_errors = [e for e in errors if not e.startswith("WARNING")]
    if errors:
        print("Dataset validation messages:", errors)
    if blocking_errors:
        print("Stopping due to blocking errors above.")
        return

    examples = dataset.load("test")[:3]
    print(f"Loaded {len(examples)} examples for the smoke test.")

    for name, adapter_cls in [("XLM-R", XLMRModelAdapter), ("E5", E5ModelAdapter)]:
        print()
        print(f"=== Loading {name} ===")
        model = adapter_cls()
        model.load()
        print(f"{name} loaded successfully.")
        print(f"Thresholds used: {model.thresholds}")

        predictions = model.predict(examples)
        for example, prediction in zip(examples, predictions):
            print(f"{example.example_id}: predicted {prediction.predicted_label}")

        del model


if __name__ == "__main__":
    main()