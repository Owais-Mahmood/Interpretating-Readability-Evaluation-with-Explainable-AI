"""
Visualization 4 (radar chart) and 7 (box plots of attribution score
distribution). Run from the repo root:

    python3 scripts/make_radar_and_boxplot_viz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.mbert_e2r import MBERTModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.gradient_shap_impl import GradientShapExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer

OUTPUT_DIR = Path("outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_radar_chart():
    overall = pd.read_csv("outputs/tables/comparison_overall.csv")
    per_strategy = pd.read_csv("outputs/tables/comparison_per_strategy.csv")
    mbert_overall = overall[overall["model"] == "mbert"]
    mbert_strategy = per_strategy[(per_strategy["model"] == "mbert") & (per_strategy["metric"] == "precision_at_k")]

    # Use 4 genuinely distinct dimensions (2 metrics need at least 3 to form
    # a real polygon shape, not a degenerate flat line)
    methods = mbert_overall["method"].unique()

    dimension_labels = ["Precision@K (overall)", "AUPRC (overall)", "Precision@K (Compression)", "Precision@K (Omission)"]

    values_by_method = {}
    for method in methods:
        overall_data = mbert_overall[mbert_overall["method"] == method]
        strategy_data = mbert_strategy[mbert_strategy["method"] == method]

        precision_overall = overall_data[overall_data["metric"] == "precision_at_k"]["mean"]
        auprc_overall = overall_data[overall_data["metric"] == "auprc"]["mean"]
        precision_compression = strategy_data[strategy_data["slice_value"] == "Compression"]["mean"]
        precision_omission = strategy_data[strategy_data["slice_value"] == "Omission"]["mean"]

        values = [
            precision_overall.iloc[0] if len(precision_overall) > 0 else 0,
            auprc_overall.iloc[0] if len(auprc_overall) > 0 else 0,
            precision_compression.iloc[0] if len(precision_compression) > 0 else 0,
            precision_omission.iloc[0] if len(precision_omission) > 0 else 0,
        ]
        values_by_method[method] = values

    n_dims = len(dimension_labels)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for method, values in values_by_method.items():
        values_closed = values + values[:1]
        ax.plot(angles, values_closed, label=method, linewidth=2)
        ax.fill(angles, values_closed, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimension_labels, fontsize=9)
    ax.set_title("Explainability method comparison (mBERT) -- radar", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    path = OUTPUT_DIR / "radar_chart_mbert_methods.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def make_boxplot():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")[:3]
    print(f"Loaded {len(examples)} examples for box plot data.")

    print("Loading mBERT...")
    model = MBERTModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)

    explainers = {
        "Integrated Gradients": IntegratedGradientsExplainer(n_steps=50),
        "GradientSHAP": GradientShapExplainer(n_samples=25, n_baselines=5),
        "Raw Attention": RawAttentionExplainer(layer=-1),
    }

    data_for_boxplot = []
    labels_for_boxplot = []
    for name, explainer in explainers.items():
        exps = explainer.explain(examples, model, predictions)
        all_scores = np.concatenate([np.abs(e.scores) for e in exps])
        data_for_boxplot.append(all_scores)
        labels_for_boxplot.append(name)
        print(f"{name}: {len(all_scores)} token scores collected")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_for_boxplot, tick_labels=labels_for_boxplot)
    ax.set_ylabel("|Token importance score| (normalised)")
    ax.set_title("Distribution of attribution scores by method (mBERT)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = OUTPUT_DIR / "boxplot_attribution_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    make_radar_chart()
    make_boxplot()


if __name__ == "__main__":
    main()