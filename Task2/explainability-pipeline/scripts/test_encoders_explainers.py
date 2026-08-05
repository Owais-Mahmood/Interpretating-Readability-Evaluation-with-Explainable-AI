"""
Test whether the existing explainers (Integrated Gradients, GradientSHAP,
Raw Attention) work on the new XLM-R and E5 models. Run from the repo root:

    python3 scripts/test_encoders_explainers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter
from xai_pipeline.models.encoder_models import E5ModelAdapter as XLMRModelAdapter
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.gradient_shap_impl import GradientShapExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_with_spans.csv")
    examples = dataset.load("test")[:1]  # just 1, to test the mechanics first
    print(f"Loaded {len(examples)} example(s).")

    print("Loading XLM-R...")
    model = XLMRModelAdapter()
    model.load()
    print("Loaded.")

    predictions = model.predict(examples)
    print(f"Predicted: {predictions[0].predicted_label}")

    print()
    print("Trying Integrated Gradients...")
    try:
        ig_explainer = IntegratedGradientsExplainer(n_steps=50)
        ig_explanations = ig_explainer.explain(examples, model, predictions)
        print(f"SUCCESS: produced {len(ig_explanations)} explanations")
        for exp in ig_explanations[:1]:
            print(f"  target={exp.target}, convergence_delta={exp.metadata['convergence_delta']:.4f}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    print()
    print("Trying GradientSHAP...")
    try:
        gshap_explainer = GradientShapExplainer(n_samples=25, n_baselines=5)
        gshap_explanations = gshap_explainer.explain(examples, model, predictions)
        print(f"SUCCESS: produced {len(gshap_explanations)} explanations")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    print()
    print("Trying Raw Attention...")
    try:
        attn_explainer = RawAttentionExplainer(layer=-1)
        attn_explanations = attn_explainer.explain(examples, model, predictions)
        print(f"SUCCESS: produced {len(attn_explanations)} explanations")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()