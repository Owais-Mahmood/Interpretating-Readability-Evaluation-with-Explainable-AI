from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from xai_pipeline.contracts import Example, Explanation, MetricRecord, Prediction
from xai_pipeline.registry import EVALUATORS


def _locate_span_char_range(text: str, span_text: str) -> tuple[int, int] | None:
    """Find the character start/end of a gold span's text within the full
    sentence. If `text` already includes a prefix (e.g. E5's 'query: '),
    pass the PREFIXED text here directly -- .find() already returns the
    correct absolute position within it, no separate offset needed."""
    start = text.find(span_text)
    if start == -1:
        return None
    return start, start + len(span_text)


def _gold_token_mask(offset_mapping, sequence_ids, side_index: int, char_ranges: list[tuple[int, int]]) -> np.ndarray:
    n_tokens = len(offset_mapping)
    mask = np.zeros(n_tokens, dtype=int)
    for i, (char_start, char_end) in enumerate(offset_mapping):
        if sequence_ids[i] != side_index:
            continue
        if char_start == char_end:
            continue
        for gold_start, gold_end in char_ranges:
            if char_start < gold_end and char_end > gold_start:
                mask[i] = 1
                break
    return mask


@EVALUATORS.register("plausibility")
class PlausibilityEvaluator:
    """Compares each explanation's token importance scores against the real
    human-edited spans (from Task 1), using Precision@K, Recall@K, F1
    overlap, and AUPRC.

    NOTE / ASSUMPTION: K is set adaptively per example, equal to the number
    of real gold tokens for that example's side -- a common convention,
    worth confirming with Nouran rather than an explicit task-sheet rule.

    NOTE: supports an optional text_prefix (e.g. "query: " for E5), which
    MUST match whatever prefix was used when the explanation itself was
    generated -- otherwise the re-tokenization here won't line up with the
    explanation's actual token count, and rows get silently dropped by the
    length-mismatch safety check below. Found and fixed this exact bug:
    E5's explanations without prefix-awareness here caused ~97% of rows
    to be silently dropped (only 20/787 explanations survived).
    """

    name = "plausibility"

    def __init__(self, tokenizer, text_prefix: str = "") -> None:
        self.tokenizer = tokenizer
        self.text_prefix = text_prefix

    def evaluate(
        self,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
        explanations: Sequence[Explanation],
    ) -> pd.DataFrame:
        examples_by_id = {ex.example_id: ex for ex in examples}
        records: list[MetricRecord] = []

        n_length_mismatches = 0

        for explanation in explanations:
            example = examples_by_id[explanation.example_id]
            deletion_spans = example.references.get("deletion_spans", [])
            insertion_spans = example.references.get("insertion_spans", [])

            source_text = self.text_prefix + example.inputs["source_text"]
            simplified_text = self.text_prefix + example.inputs["simplified_text"]

            encoded = self.tokenizer(
                source_text, text_pair=simplified_text,
                truncation=True, max_length=256, return_offsets_mapping=True,
            )
            offset_mapping = encoded["offset_mapping"]
            sequence_ids = encoded.sequence_ids()

            deletion_char_ranges = [
                r for r in (
                    _locate_span_char_range(source_text, s["span_text"])
                    for s in deletion_spans
                ) if r is not None
            ]
            insertion_char_ranges = [
                r for r in (
                    _locate_span_char_range(simplified_text, s["span_text"])
                    for s in insertion_spans
                ) if r is not None
            ]

            deletion_mask = _gold_token_mask(offset_mapping, sequence_ids, 0, deletion_char_ranges)
            insertion_mask = _gold_token_mask(offset_mapping, sequence_ids, 1, insertion_char_ranges)
            gold_mask = np.maximum(deletion_mask, insertion_mask)

            n_gold = int(gold_mask.sum())
            if n_gold == 0:
                continue

            scores = np.asarray(explanation.scores)
            if len(scores) != len(gold_mask):
                n_length_mismatches += 1
                continue

            importance = np.abs(scores)
            k = n_gold
            top_k_indices = np.argsort(-importance)[:k]

            predicted_mask = np.zeros_like(gold_mask)
            predicted_mask[top_k_indices] = 1

            true_positives = int((predicted_mask & gold_mask).sum())
            precision_at_k = true_positives / k if k > 0 else 0.0
            recall_at_k = true_positives / n_gold if n_gold > 0 else 0.0
            f1_at_k = (
                2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
                if (precision_at_k + recall_at_k) > 0 else 0.0
            )

            try:
                auprc = average_precision_score(gold_mask, importance)
            except ValueError:
                auprc = float("nan")

            for metric_name, value in [
                ("precision_at_k", precision_at_k),
                ("recall_at_k", recall_at_k),
                ("f1_at_k", f1_at_k),
                ("auprc", auprc),
            ]:
                records.append(
                    MetricRecord(
                        run_id="task2_plausibility",
                        example_id=explanation.example_id,
                        method=explanation.method,
                        metric=metric_name,
                        value=value,
                        slice_name="target",
                        slice_value=str(explanation.target),
                        metadata={"n_gold_tokens": n_gold, "k": k},
                    )
                )

        if n_length_mismatches > 0:
            print(f"WARNING: {n_length_mismatches} explanations skipped due to token length mismatch "
                  f"(check text_prefix is set correctly for this model)")

        return pd.DataFrame([asdict(r) for r in records])

    def validate(self) -> list[str]:
        return []