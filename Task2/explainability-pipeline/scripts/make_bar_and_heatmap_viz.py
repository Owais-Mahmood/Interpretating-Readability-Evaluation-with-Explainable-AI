"""
Visualization 3 (bar charts) and 6 (strategy heatmaps), built from the
real comparison tables already computed. Run from the repo root:

    python3 scripts/make_bar_and_heatmap_viz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("outputs/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_bar_chart():
    overall = pd.read_csv("outputs/tables/comparison_overall.csv")
    # Focus on precision_at_k, the primary faithfulness metric
    subset = overall[overall["metric"] == "precision_at_k"]

    pivot = subset.pivot(index="method", columns="model", values="mean")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Precision@K (mean)")
    ax.set_xlabel("Explainer method")
    ax.set_title("Precision@K by method and model")
    ax.legend(title="Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = OUTPUT_DIR / "bar_chart_precision_by_method_model.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def make_strategy_heatmap():
    per_strategy = pd.read_csv("outputs/tables/comparison_per_strategy.csv")
    subset = per_strategy[per_strategy["metric"] == "precision_at_k"]

    for model in subset["model"].unique():
        model_data = subset[subset["model"] == model]
        pivot = model_data.pivot(index="method", columns="slice_value", values="mean")

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Precision@K by strategy -- {model}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Precision@K")
        plt.tight_layout()

        path = OUTPUT_DIR / f"strategy_heatmap_{model}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")


def main():
    make_bar_chart()
    make_strategy_heatmap()


if __name__ == "__main__":
    main()