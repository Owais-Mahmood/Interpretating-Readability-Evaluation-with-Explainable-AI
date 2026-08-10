"""
Visualization 5 (actual deletion/insertion curves, not just AUC) and
8 (error analysis: successful vs unsuccessful explanation examples).
Run from the repo root:

    python3 scripts/make_curve_and_error_analysis_viz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer

OUTPUT_DIR = Path("outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_deletion_insertion_curve(model, example, prediction, explanation, target_label):
    """Recomputes the actual curve (not just the AUC) for one explanation,
    reusing the same masking logic as the DeletionInsertionEvaluator."""
    from xai_pipeline.datasets.simplification import LABELS

    tokenizer = model.tokenizer
    device = next(model.model.parameters()).device
    pad_id = tokenizer.pad_token_id
    target_index = LABELS.index(target_label)

    encoded = tokenizer(
        example.inputs["source_text"], text_pair=example.inputs["simplified_text"],
        truncation=True, max_length=256, return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    n_tokens = input_ids.shape[1]

    importance = np.abs(explanation.scores)
    order = np.argsort(-importance)
    n_steps = 10
    step_points = np.linspace(0, n_tokens, n_steps + 1, dtype=int)

    def get_probability(ids):
        with torch.inference_mode():
            logits = model.model(input_ids=ids, attention_mask=attention_mask).logits[0]
            return float(torch.sigmoid(logits.float())[target_index])

    deletion_probs = []
    for k in step_points:
        masked = input_ids.clone()
        if k > 0:
            masked[0, order[:k]] = pad_id
        deletion_probs.append(get_probability(masked))

    insertion_probs = []
    baseline = torch.full_like(input_ids, pad_id)
    for k in step_points:
        revealed = baseline.clone()
        if k > 0:
            revealed[0, order[:k]] = input_ids[0, order[:k]]
        insertion_probs.append(get_probability(revealed))

    fig, ax = plt.subplots(figsize=(7, 5))
    fraction_points = step_points / n_tokens
    ax.plot(fraction_points, deletion_probs, marker="o", label="Deletion (remove top-K important tokens)")
    ax.plot(fraction_points, insertion_probs, marker="s", label="Insertion (reveal top-K important tokens)")
    ax.set_xlabel("Fraction of tokens removed / revealed")
    ax.set_ylabel(f"P({target_label})")
    ax.set_title(f"Deletion/Insertion curves -- Integrated Gradients, target: {target_label}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "deletion_insertion_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def make_error_analysis():
    """Finds a high-plausibility (successful) and low-plausibility
    (unsuccessful) example from the real full evaluation results, and
    shows their token heatmaps side by side for comparison."""
    results = pd.read_csv("outputs/metrics/plausibility_full_results.csv")
    precision_rows = results[
        (results["metric"] == "precision_at_k") & (results["method"] == "integrated_gradients")
    ]

    best_row = precision_rows.loc[precision_rows["value"].idxmax()]
    worst_row = precision_rows.loc[precision_rows["value"].idxmin()]

    print(f"Best example: {best_row['example_id']}, target={best_row['slice_value']}, precision={best_row['value']:.3f}")
    print(f"Worst example: {worst_row['example_id']}, target={worst_row['slice_value']}, precision={worst_row['value']:.3f}")

    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    all_examples = dataset.load("test")
    examples_by_id = {ex.example_id: ex for ex in all_examples}

    best_example = examples_by_id.get(best_row["example_id"])
    worst_example = examples_by_id.get(worst_row["example_id"])

    if best_example is None or worst_example is None:
        print("Could not find matching examples in the dataset -- skipping error analysis plot.")
        return

    print("Loading mBERT...")
    model = MBERTModelAdapter()
    model.load()
    print("Loaded.")

    explainer = IntegratedGradientsExplainer(n_steps=50)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4))

    for ax, example, row, label in [
        (axes[0], best_example, best_row, "SUCCESSFUL"),
        (axes[1], worst_example, worst_row, "UNSUCCESSFUL"),
    ]:
        prediction = model.predict([example])[0]
        target = row["slice_value"]
        if target not in prediction.predicted_label:
            # explainer only explains predicted labels -- force this one
            prediction.predicted_label = [target]
        exps = explainer.explain([example], model, [prediction])
        matching = [e for e in exps if e.target == target]
        if not matching:
            continue
        exp = matching[0]

        norm_scores = exp.scores / (np.abs(exp.scores).max() + 1e-10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"{label} -- {example.example_id}, target={target}, precision@k={row['value']:.3f}", fontsize=10, loc="left")

        x = 0.01
        y = 0.5
        for token, score in zip(exp.units, norm_scores):
            color = plt.cm.RdBu_r((score + 1) / 2)
            text_obj = ax.text(
                x, y, token, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none"),
                transform=ax.transAxes, va="center",
            )
            fig.canvas.draw()
            bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_axes = bbox.transformed(ax.transAxes.inverted())
            x += bbox_axes.width + 0.01
            if x > 0.95:
                x = 0.01
                y -= 0.4

    plt.tight_layout()
    path = OUTPUT_DIR / "error_analysis_success_vs_failure.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:1]

    print("Loading mBERT for the curve plot...")
    model = MBERTModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)
    target_label = predictions[0].predicted_label[0]

    explainer = IntegratedGradientsExplainer(n_steps=50)
    explanations = explainer.explain(examples, model, predictions)
    matching = [e for e in explanations if e.target == target_label]
    if matching:
        make_deletion_insertion_curve(model, examples[0], predictions[0], matching[0], target_label)

    make_error_analysis()


if __name__ == "__main__":
    main()