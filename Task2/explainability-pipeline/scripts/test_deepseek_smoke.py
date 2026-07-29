"""
Smoke test for the DeepSeek QLoRA adapter: load a few real examples,
load DeepSeek, run predictions only (no explainers yet, given the size
and risk of this model). Run from the repo root:

    python3 scripts/test_deepseek_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.deepseek_e2r import DeepSeekModelAdapter


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_with_spans.csv")
    errors = dataset.validate()
    if errors:
        print("Dataset validation errors:", errors)
        return
    examples = dataset.load("test")[:2]  # just 2, this model is much heavier
    print(f"Loaded {len(examples)} examples for the smoke test.")

    print("Loading DeepSeek (this downloads several GB on first run)...")
    model = DeepSeekModelAdapter()
    model.load()
    print("Model loaded successfully.")

    predictions = model.predict(examples)
    for example, prediction in zip(examples, predictions):
        print(f"{example.example_id}: predicted {prediction.predicted_label}")


if __name__ == "__main__":
    main()