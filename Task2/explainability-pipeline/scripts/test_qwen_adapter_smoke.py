"""
Smoke test for the Qwen model adapter: load a few real examples, load
Qwen, run predictions via our framework's contract. Run from the repo root:

    python3 scripts/test_qwen_adapter_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.qwen_e2r import QwenModelAdapter


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    errors = dataset.validate()
    blocking_errors = [e for e in errors if not e.startswith("WARNING")]
    if errors:
        print("Dataset validation messages:", errors)
    if blocking_errors:
        print("Stopping due to blocking errors above.")
        return

    examples = dataset.load("test")[:2]  # just 2, Qwen is heavy
    print(f"Loaded {len(examples)} examples for the smoke test.")

    print("Loading Qwen (downloads several GB on first run)...")
    model = QwenModelAdapter()
    model.load()
    print("Model loaded successfully.")

    predictions = model.predict(examples)
    for example, prediction in zip(examples, predictions):
        print(f"{example.example_id}: predicted {prediction.predicted_label}")
        print(f"  probabilities: {dict(zip(['Synonymy','Modulation','Compression','Explanation','Syntactic Change','Omission'], prediction.scores.round(4)))}")


if __name__ == "__main__":
    main()