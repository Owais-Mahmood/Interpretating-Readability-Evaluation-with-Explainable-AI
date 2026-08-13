"""
Generates the full qualitative data needed for item 7: for every
strategy, one successful and one unsuccessful real example, with the
sentence pair, gold spans, model prediction/confidence, and every
method's highlighted tokens (top-K by importance, matching the same K
used in evaluation). Uses XLM-R throughout, since it has all 4 methods
implemented, giving a genuine like-for-like comparison across methods.

Run from the repo root:
    python3 scripts/generate_qualitative_examples.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter, LABELS
from xai_pipeline.contracts import Prediction
from xai_pipeline.explainers.integrated_gradients_impl import IntegratedGradientsExplainer
from xai_pipeline.explainers.gradient_shap_impl import GradientShapExplainer
from xai_pipeline.explainers.raw_attention_impl import RawAttentionExplainer

REPO_ID = "hannah-khallaf/e2r-strategy-xlmr-large-focal"
FALLBACK_THRESHOLDS = {label: 0.46 for label in LABELS}


class SimpleModelAdapter:
    """Minimal adapter matching what the explainers expect."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def get_top_k_tokens(tokens, scores, k):
    importance = np.abs(scores)
    top_k_idx = np.argsort(-importance)[:k]
    return [tokens[i] for i in sorted(top_k_idx)]


def main():
    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")
    print(f"Loaded {len(examples)} examples.")

    print("Loading XLM-R...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, trust_remote_code=True)
    model.eval()
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        for name, module in model.named_modules():
            if name.endswith("word_embeddings"):
                model.get_input_embeddings = lambda m=module: m
                break
    print("Loaded.")

    adapter = SimpleModelAdapter(model, tokenizer)
    explainers = {
        "GradientSHAP": GradientShapExplainer(n_samples=25, n_baselines=5),
        "Integrated Gradients": IntegratedGradientsExplainer(n_steps=50),
        "Raw Attention": RawAttentionExplainer(layer=-1),
    }

    # Load AttnLRP's already-saved raw explanations (tokens + scores),
    # rather than re-running it (which needs the separate isolated
    # environment with an older transformers version).
    import json
    attnlrp_raw = pd.read_csv("outputs/explanations/attnlrp_xlmr_raw_explanations.csv")

    # Load real per-example plausibility scores (AttnLRP) to rank examples
    attnlrp_results = pd.read_csv("outputs/metrics/attnlrp_xlmr_plausibility_results.csv")
    precision_rows = attnlrp_results[attnlrp_results["metric"] == "precision_at_k"]

    examples_by_id = {ex.example_id: ex for ex in examples}
    device = next(model.parameters()).device

    output_rows = []

    for strategy in LABELS:
        strategy_rows = precision_rows[precision_rows["slice_value"] == strategy]
        if len(strategy_rows) == 0:
            print(f"No AttnLRP results for {strategy}, skipping.")
            continue

        best_row = strategy_rows.loc[strategy_rows["value"].idxmax()]
        worst_row = strategy_rows.loc[strategy_rows["value"].idxmin()]

        for label, row in [("SUCCESSFUL", best_row), ("UNSUCCESSFUL", worst_row)]:
            example = examples_by_id.get(row["example_id"])
            if example is None:
                continue

            encoded = tokenizer(
                example.inputs["source_text"], text_pair=example.inputs["simplified_text"],
                truncation=True, max_length=256, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits[0]
                probs = torch.sigmoid(logits.float()).cpu().numpy()
            target_index = LABELS.index(strategy)
            confidence = float(probs[target_index])

            row_data = {
                "strategy": strategy,
                "label": label,
                "example_id": example.example_id,
                "source_text": example.inputs["source_text"],
                "simplified_text": example.inputs["simplified_text"],
                "deletion_spans": example.references.get("deletion_spans", []),
                "insertion_spans": example.references.get("insertion_spans", []),
                "confidence": confidence,
                "attnlrp_precision_at_k": row["value"],
            }

            for method_name, explainer in explainers.items():
                prediction = Prediction(
                    example_id=example.example_id, predicted_label=[strategy], target_label=None, scores=probs, metadata={}
                )
                exps = explainer.explain([example], adapter, [prediction])
                matching = [e for e in exps if e.target == strategy]
                if matching:
                    exp = matching[0]
                    n_gold = len(row_data["deletion_spans"]) + len(row_data["insertion_spans"])
                    k = max(n_gold, 3)
                    top_tokens = get_top_k_tokens(exp.units, exp.scores, k)
                    row_data[f"{method_name}_tokens"] = " | ".join(top_tokens)
                else:
                    row_data[f"{method_name}_tokens"] = "(not available)"

            # AttnLRP: look up from the already-saved full explanation file
            attnlrp_match = attnlrp_raw[
                (attnlrp_raw["example_id"] == example.example_id) & (attnlrp_raw["target"] == strategy)
            ]
            if len(attnlrp_match) > 0:
                attnlrp_tokens = json.loads(attnlrp_match.iloc[0]["tokens"])
                attnlrp_scores = np.array(json.loads(attnlrp_match.iloc[0]["scores"]))
                n_gold = len(row_data["deletion_spans"]) + len(row_data["insertion_spans"])
                k = max(n_gold, 3)
                row_data["AttnLRP_tokens"] = " | ".join(get_top_k_tokens(attnlrp_tokens, attnlrp_scores, k))
            else:
                row_data["AttnLRP_tokens"] = "(not found in saved AttnLRP results)"

            output_rows.append(row_data)
            print(f"{strategy} / {label}: {example.example_id}, confidence={confidence:.3f}")

    output_df = pd.DataFrame(output_rows)
    output_path = Path("outputs/tables/qualitative_examples.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(output_df)} rows to {output_path}")


if __name__ == "__main__":
    main()