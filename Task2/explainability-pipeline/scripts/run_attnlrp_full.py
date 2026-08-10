"""
Full AttnLRP run for XLM-R and E5: saves BOTH the raw numerical
attribution scores / token-level explanations (as Nouran explicitly
asked) AND runs them through the plausibility evaluator to get
comparable metrics for the comparison tables.

Run from Task2/explainability-pipeline, with attnlrp_venv activated:
    python3 scripts/run_attnlrp_full.py xlmr
    python3 scripts/run_attnlrp_full.py e5
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xlmr_attnlrp_patch import apply_xlmr_attnlrp_patch

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter, LABELS
from xai_pipeline.contracts import Explanation, Prediction

MODEL_REPOSITORIES = {
    "xlmr": "hannah-khallaf/e2r-strategy-xlmr-large-focal",
    "e5": "hannah-khallaf/e2r-strategy-multilingual-e5-large-bce",
}
FALLBACK_THRESHOLDS = {
    "xlmr": {label: 0.46 for label in LABELS},
    "e5": {label: 0.23 for label in LABELS},
}


def main():
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "xlmr"
    assert model_choice in MODEL_REPOSITORIES, "First argument must be 'xlmr' or 'e5'"
    repo_id = MODEL_REPOSITORIES[model_choice]

    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")
    print(f"Loaded {len(examples)} examples.")

    print(f"Loading {model_choice} ({repo_id})...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, trust_remote_code=True)
    model.eval()
    print("Loaded.")

    apply_xlmr_attnlrp_patch(verbose=True)

    try:
        model.get_input_embeddings()
        embedding_layer = model.get_input_embeddings()
    except NotImplementedError:
        embedding_layer = None
        for name, module in model.named_modules():
            if name.endswith("word_embeddings"):
                embedding_layer = module
                model.get_input_embeddings = lambda m=module: m
                break

    thresholds = np.array([FALLBACK_THRESHOLDS[model_choice][label] for label in LABELS])
    device = next(model.parameters()).device

    # Step 1: predict all examples first
    predictions = []
    for example in examples:
        prefix = "query: " if model_choice == "e5" else ""
        encoded = tokenizer(
            prefix + example.inputs["source_text"], text_pair=prefix + example.inputs["simplified_text"],
            truncation=True, max_length=256, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
            probs = torch.sigmoid(logits.float()).cpu().numpy()
        predicted_flags = (probs >= thresholds).astype(int)
        predicted_labels = [LABELS[j] for j in range(len(LABELS)) if predicted_flags[j] == 1]
        predictions.append(Prediction(
            example_id=example.example_id, predicted_label=predicted_labels,
            target_label=None, scores=probs, metadata={},
        ))
    print("Predictions done.")

    # Step 2: run AttnLRP on all (example, predicted label) pairs
    explanations = []
    start = time.time()
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        prefix = "query: " if model_choice == "e5" else ""
        for label in prediction.predicted_label:
            target_index = LABELS.index(label)
            encoded = tokenizer(
                prefix + example.inputs["source_text"], text_pair=prefix + example.inputs["simplified_text"],
                truncation=True, max_length=256, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            embeds = embedding_layer(input_ids).clone().detach().requires_grad_()
            outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
            target_logit = outputs.logits[0][target_index]
            target_logit.backward()
            scores = (embeds * embeds.grad).sum(-1).squeeze(0).detach().cpu().numpy()

            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
            explanations.append(Explanation(
                example_id=example.example_id, method="attnlrp", target=label,
                units=tokens, scores=scores, unit_type="subword_token", signed=True, metadata={},
            ))
        if (i + 1) % 40 == 0 or (i + 1) == len(examples):
            elapsed = time.time() - start
            print(f"  {i + 1}/{len(examples)} done ({elapsed:.0f}s elapsed)")

    print(f"AttnLRP finished in {time.time() - start:.0f}s, {len(explanations)} explanations.")

    # Step 3: save raw explanations (tokens + scores), as Nouran asked
    rows = []
    for exp in explanations:
        rows.append({
            "example_id": exp.example_id, "method": exp.method, "target": exp.target,
            "tokens": json.dumps(exp.units), "scores": json.dumps(exp.scores.tolist()),
        })
    raw_df = pd.DataFrame(rows)
    raw_path = Path(f"outputs/explanations/attnlrp_{model_choice}_raw_explanations.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw explanations to {raw_path}")

    # Step 4: run plausibility evaluation for comparable metrics
    from xai_pipeline.evaluators.plausibility_impl import PlausibilityEvaluator
    prefix = "query: " if model_choice == "e5" else ""
    evaluator = PlausibilityEvaluator(tokenizer=tokenizer, text_prefix=prefix)
    results = evaluator.evaluate(examples, predictions, explanations)
    results["model"] = model_choice
    metrics_path = Path(f"outputs/metrics/attnlrp_{model_choice}_plausibility_results.csv")
    results.to_csv(metrics_path, index=False)
    print(f"Saved plausibility metrics to {metrics_path}")
    print()
    print(results.groupby("metric")["value"].mean())


if __name__ == "__main__":
    main()