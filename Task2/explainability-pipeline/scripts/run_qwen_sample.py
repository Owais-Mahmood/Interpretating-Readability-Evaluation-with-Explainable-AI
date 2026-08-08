"""
Sample run: Qwen model + Integrated Gradients on a representative sample
of the test set (not the full 281, given the real per-label cost of
~37s observed in the smoke test). Run from the repo root:

    python3 scripts/run_qwen_sample.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.qwen_e2r import QwenModelAdapter
from xai_pipeline.explainers.qwen_integrated_gradients import QwenIntegratedGradientsExplainer

SAMPLE_SIZE = 30


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:SAMPLE_SIZE]
    print(f"Loaded {len(examples)} examples (sample of {SAMPLE_SIZE}).")

    print("Loading Qwen...")
    model = QwenModelAdapter()
    model.load()
    print("Loaded.")

    print("Running predictions...")
    predictions = model.predict(examples)
    for example, prediction in zip(examples, predictions):
        print(f"{example.example_id}: predicted {prediction.predicted_label}")

    print()
    print(f"Running Integrated Gradients on {len(examples)} examples (this will take a while)...")
    explainer = QwenIntegratedGradientsExplainer(n_steps=20)

    all_explanations = []
    start = time.time()
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        example_explanations = explainer.explain([example], model, [prediction])
        all_explanations.extend(example_explanations)
        elapsed = time.time() - start
        avg_per_example = elapsed / (i + 1)
        remaining = avg_per_example * (len(examples) - i - 1)
        print(f"  {i + 1}/{len(examples)} examples done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    print()
    print(f"Finished in {time.time() - start:.0f}s, {len(all_explanations)} total explanations.")

    # Save raw explanations (tokens + scores), since full plausibility
    # evaluation against gold spans needs Qwen-specific logic to locate
    # the sentence pair within the full prompt -- not yet built.
    rows = []
    for exp in all_explanations:
        rows.append({
            "example_id": exp.example_id,
            "method": exp.method,
            "target": exp.target,
            "tokens": " | ".join(exp.units),
            "scores": " | ".join(f"{s:.4f}" for s in exp.scores),
            "prompt_length": exp.metadata["prompt_length"],
            "n_steps": exp.metadata["n_steps"],
        })
    results_df = pd.DataFrame(rows)

    output_path = Path("outputs/explanations/qwen_ig_sample_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()