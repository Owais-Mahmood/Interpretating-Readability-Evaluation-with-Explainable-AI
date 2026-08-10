"""
Visualization 1 (token-level heatmaps) and 2 (side-by-side method
comparison), using real mBERT explanations. Run from the repo root:

    python3 scripts/make_token_heatmap_viz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.gradient_shap_impl import GradientShapExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer

OUTPUT_DIR = Path("outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_token_heatmap(ax, tokens, scores, title):
    """Renders one row of colored-background tokens on the given axis."""
    norm_scores = scores / (np.abs(scores).max() + 1e-10)  # normalise to [-1, 1]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")

    x = 0.01
    y = 0.5
    renderer_fig = ax.figure
    for token, score in zip(tokens, norm_scores):
        color = plt.cm.RdBu_r((score + 1) / 2)  # red=positive, blue=negative
        text_obj = ax.text(
            x, y, token, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none"),
            transform=ax.transAxes, va="center",
        )
        renderer_fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=renderer_fig.canvas.get_renderer())
        bbox_axes = bbox.transformed(ax.transAxes.inverted())
        x += (bbox_axes.width) + 0.01
        if x > 0.95:
            x = 0.01
            y -= 0.3


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:1]  # one example is enough for a clear visual
    print(f"Loaded {len(examples)} example.")

    print("Loading mBERT...")
    model = MBERTModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)
    print(f"Predicted: {predictions[0].predicted_label}")

    explainers = {
        "Integrated Gradients": IntegratedGradientsExplainer(n_steps=50),
        "GradientSHAP": GradientShapExplainer(n_samples=25, n_baselines=5),
        "Raw Attention": RawAttentionExplainer(layer=-1),
    }

    # Use the first predicted label as the target to visualize
    target_label = predictions[0].predicted_label[0]
    print(f"Visualizing explanations for target label: {target_label}")

    all_explanations = {}
    for name, explainer in explainers.items():
        exps = explainer.explain(examples, model, predictions)
        matching = [e for e in exps if e.target == target_label]
        if matching:
            all_explanations[name] = matching[0]

    # Visualization 1: single heatmap (Integrated Gradients only)
    if "Integrated Gradients" in all_explanations:
        exp = all_explanations["Integrated Gradients"]
        fig, ax = plt.subplots(figsize=(12, 2))
        render_token_heatmap(ax, exp.units, exp.scores, f"Integrated Gradients -- target: {target_label}")
        plt.tight_layout()
        path = OUTPUT_DIR / "token_heatmap_single_method.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # Visualization 2: side-by-side, all 3 methods stacked
    fig, axes = plt.subplots(len(all_explanations), 1, figsize=(12, 2 * len(all_explanations)))
    if len(all_explanations) == 1:
        axes = [axes]
    for ax, (name, exp) in zip(axes, all_explanations.items()):
        render_token_heatmap(ax, exp.units, exp.scores, f"{name} -- target: {target_label}")
    plt.tight_layout()
    path = OUTPUT_DIR / "token_heatmap_side_by_side.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()