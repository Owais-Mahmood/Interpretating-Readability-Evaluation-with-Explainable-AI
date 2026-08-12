"""
Full run: Qwen model + Integrated Gradients on the COMPLETE final test
set (not a sample), as Nouran specifically requested for a fair,
matched comparison against the encoder models. Run from the repo root:
    python3 scripts/run_qwen_full.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd
from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.qwen_e2r import QwenModelAdapter
from xai_pipeline.explainers.qwen_integrated_gradients import QwenIntegratedGradientsExplainer


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")
    print(f"Loaded {len(examples)} examples (FULL final test set).")

    print("Loading Qwen...")
    model = QwenModelAdapter()
    model.load()
    print("Loaded.")

    print("Running predictions...")
    predictions = model.predict(examples)
    for example, prediction in zip(examples, predictions):
        print(f"{example.example_id}: predicted {prediction.predicted_label}")

    print()
    print(f"Running Integrated Gradients on {len(examples)} examples (this will take several hours)...")
    explainer = QwenIntegratedGradientsExplainer(n_steps=20)
    all_explanations = []
    start = time.time()

    # Save progress incrementally, so a crash partway through doesn't
    # lose everything computed so far.
    output_path = Path("outputs/explanations/qwen_ig_full_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        example_explanations = explainer.explain([example], model, [prediction])
        all_explanations.extend(example_explanations)
        elapsed = time.time() - start
        avg_per_example = elapsed / (i + 1)
        remaining = avg_per_example * (len(examples) - i - 1)
        print(f"  {i + 1}/{len(examples)} examples done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
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
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"  (checkpoint saved: {len(rows)} explanations so far)")

    print()
    print(f"Finished in {time.time() - start:.0f}s, {len(all_explanations)} total explanations.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()