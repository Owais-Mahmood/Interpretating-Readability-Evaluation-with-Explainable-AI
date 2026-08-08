"""
Minimal test of Qwen's Integrated Gradients: 1 example, checking
correctness and timing before attempting anything larger. Run from the
repo root:

    python3 scripts/test_qwen_ig_smoke.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.qwen_e2r import QwenModelAdapter
from xai_pipeline.explainers.qwen_integrated_gradients import QwenIntegratedGradientsExplainer


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:1]
    print(f"Loaded {len(examples)} example.")

    print("Loading Qwen...")
    model = QwenModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)
    print(f"Predicted: {predictions[0].predicted_label}")

    print()
    print("Trying Integrated Gradients on Qwen (n_steps=20, timing this)...")
    start = time.time()
    explainer = QwenIntegratedGradientsExplainer(n_steps=20)
    explanations = explainer.explain(examples, model, predictions)
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.1f}s for {len(explanations)} label(s) explained.")
    if len(explanations) > 0:
        print(f"Time per label explanation: {elapsed / len(explanations):.1f}s")

    for exp in explanations[:1]:
        print()
        print(f"Explanation for target={exp.target}, prompt length={exp.metadata['prompt_length']} tokens")
        sorted_pairs = sorted(zip(exp.units, exp.scores), key=lambda p: abs(p[1]), reverse=True)
        for token, score in sorted_pairs[:10]:
            print(f"  {token}: {score:.4f}")


if __name__ == "__main__":
    main()