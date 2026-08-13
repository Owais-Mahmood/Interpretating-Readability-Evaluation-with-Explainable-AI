"""
Qwen-specific plausibility evaluation: locates the source/simplified
sentence pair within Qwen's full prompt (which also includes the system
message, taxonomy card, and instructions), extracts just those tokens'
scores, then applies the same gold-span matching logic as the standard
PlausibilityEvaluator -- scoped to just the sentence-pair portion, not
the whole prompt.

Works entirely from the already-saved qwen_ig_sample_results.csv (tokens
+ scores), no need to reload the model. Run from the repo root:

    python3 scripts/evaluate_qwen_plausibility.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer

from xai_pipeline.datasets.simplification import SimplificationDatasetAdapter

REPO_ID = "hannah-khallaf/e2r-strategy-qwen2.5-7b-pairwise-qlora"


def find_token_range_for_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int] | None:
    """Finds which TOKEN indices correspond to a given character range."""
    token_indices = [
        i for i, (tok_start, tok_end) in enumerate(offsets)
        if tok_start < char_end and tok_end > char_start and tok_start != tok_end
    ]
    if not token_indices:
        return None
    return min(token_indices), max(token_indices) + 1


def main():
    results_df = pd.read_csv("outputs/explanations/qwen_ig_full_results.csv")
    print(f"Loaded {len(results_df)} raw Qwen explanations.")

    dataset = SimplificationDatasetAdapter("data/raw/test_set_full_with_spans.csv")
    examples = dataset.load("test")
    examples_by_id = {ex.example_id: ex for ex in examples}

    print("Loading Qwen tokenizer and reference implementation (lightweight, no model weights needed)...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=False)

    import importlib
    from huggingface_hub import hf_hub_download
    root = Path("./qwen_e2r_reference").resolve()
    for filename in [
        "reference_implementation/__init__.py", "reference_implementation/binary_relevance.py",
        "reference_implementation/taxonomy.py", "e2r_taxonomy.yaml",
    ]:
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    from reference_implementation import binary_relevance, taxonomy as taxonomy_module
    from xai_pipeline.models.qwen_e2r import _construct_taxonomy
    taxonomy = _construct_taxonomy(taxonomy_module, root / "e2r_taxonomy.yaml")
    print("Loaded.")

    records = []
    n_skipped_no_match = 0
    n_skipped_length_mismatch = 0
    n_skipped_no_gold = 0

    for _, row in results_df.iterrows():
        example = examples_by_id.get(row["example_id"])
        if example is None:
            continue

        tokens = row["tokens"].split(" | ")
        scores = np.array([float(s) for s in row["scores"].split(" | ")])
        if len(tokens) != len(scores):
            continue

        row_dict = {"source_text": example.inputs["source_text"], "simplified_text": example.inputs["simplified_text"]}
        system_prompt, user_prompt = binary_relevance.build_binary_prompt(
            row_dict, row["target"], taxonomy,
            include_definition=True, demonstrations=(),
            request_confidence=False, response_format="boolean", input_mode="pair",
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        encoded = tokenizer(prompt, return_offsets_mapping=True, truncation=True)
        offsets = encoded["offset_mapping"]

        if len(offsets) != len(scores):
            n_skipped_length_mismatch += 1
            continue

        source_char_start = prompt.find(example.inputs["source_text"])
        simplified_char_start = prompt.find(example.inputs["simplified_text"])
        if source_char_start == -1 or simplified_char_start == -1:
            n_skipped_no_match += 1
            continue

        source_range = find_token_range_for_span(
            offsets, source_char_start, source_char_start + len(example.inputs["source_text"])
        )
        simplified_range = find_token_range_for_span(
            offsets, simplified_char_start, simplified_char_start + len(example.inputs["simplified_text"])
        )
        if source_range is None or simplified_range is None:
            n_skipped_no_match += 1
            continue

        deletion_spans = example.references.get("deletion_spans", [])
        insertion_spans = example.references.get("insertion_spans", [])
        gold_mask = np.zeros(len(tokens), dtype=int)

        src_start, src_end = source_range
        for span in deletion_spans:
            span_text = span["span_text"]
            local_start = example.inputs["source_text"].find(span_text)
            if local_start == -1:
                continue
            char_start_in_prompt = source_char_start + local_start
            char_end_in_prompt = char_start_in_prompt + len(span_text)
            for i in range(src_start, src_end):
                tok_start, tok_end = offsets[i]
                if tok_start < char_end_in_prompt and tok_end > char_start_in_prompt:
                    gold_mask[i] = 1

        simp_start, simp_end = simplified_range
        for span in insertion_spans:
            span_text = span["span_text"]
            local_start = example.inputs["simplified_text"].find(span_text)
            if local_start == -1:
                continue
            char_start_in_prompt = simplified_char_start + local_start
            char_end_in_prompt = char_start_in_prompt + len(span_text)
            for i in range(simp_start, simp_end):
                tok_start, tok_end = offsets[i]
                if tok_start < char_end_in_prompt and tok_end > char_start_in_prompt:
                    gold_mask[i] = 1

        n_gold = int(gold_mask.sum())
        if n_gold == 0:
            n_skipped_no_gold += 1
            continue

        importance = np.abs(scores)
        k = n_gold
        top_k_indices = np.argsort(-importance)[:k]
        predicted_mask = np.zeros_like(gold_mask)
        predicted_mask[top_k_indices] = 1

        true_positives = int((predicted_mask & gold_mask).sum())
        precision_at_k = true_positives / k if k > 0 else 0.0
        recall_at_k = true_positives / n_gold if n_gold > 0 else 0.0
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0

        try:
            auprc = average_precision_score(gold_mask, importance)
        except ValueError:
            auprc = float("nan")

        for metric_name, value in [
            ("precision_at_k", precision_at_k), ("recall_at_k", recall_at_k),
            ("f1_at_k", f1_at_k), ("auprc", auprc),
        ]:
            records.append({
                "run_id": "task2_qwen_plausibility", "example_id": row["example_id"],
                "method": "integrated_gradients_qwen", "metric": metric_name, "value": value,
                "slice_name": "target", "slice_value": row["target"], "model": "qwen",
            })

    print(f"Skipped (length mismatch, re-tokenization differs from saved): {n_skipped_length_mismatch}")
    print(f"Skipped (couldn't locate sentence pair in prompt): {n_skipped_no_match}")
    print(f"Skipped (no gold tokens in sentence-pair range): {n_skipped_no_gold}")
    print(f"Produced {len(records)} metric records.")

    results = pd.DataFrame(records)
    output_path = Path("outputs/metrics/qwen_plausibility_full_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    if len(results) > 0:
        print()
        print(results.groupby("metric")["value"].mean())


if __name__ == "__main__":
    main()