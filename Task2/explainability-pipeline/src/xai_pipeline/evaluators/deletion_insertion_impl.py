from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from xai_pipeline.contracts import Example, Explanation, MetricRecord, Prediction
from xai_pipeline.registry import EVALUATORS


def _trapezoidal_auc(values: list[float]) -> float:
    """Area under the curve, normalised to [0, 1] by dividing by the
    number of steps (so it's comparable across examples with different
    sequence lengths / number of steps)."""
    if len(values) < 2:
        return float("nan")
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in newer numpy
    return float(trapz_fn(values) / (len(values) - 1))


@EVALUATORS.register("deletion_insertion")
class DeletionInsertionEvaluator:
    """Deletion/insertion curves: a faithfulness check for classification
    models (mBERT, XLM-R, E5 -- NOT Qwen, which needs a different approach
    given its prompted/generative architecture).

    Deletion: progressively mask out the top-K most important tokens
    (per the explanation), track how fast the target probability drops.
    A good explanation causes a FAST drop (low deletion AUC).

    Insertion: start from an all-masked baseline, progressively reveal
    the top-K most important tokens, track how fast the target
    probability rises. A good explanation causes a FAST rise (high
    insertion AUC).

    NOTE / ASSUMPTION: uses 10 evenly-spaced steps along the curve
    (not testing every single token position individually), to keep
    compute cost tractable -- worth confirming with Nouran whether a
    finer-grained curve is needed for the final report.
    """

    name = "deletion_insertion"

    def __init__(self, model, n_steps: int = 10) -> None:
        self.model = model
        self.n_steps = n_steps

    def evaluate(
        self,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
        explanations: Sequence[Explanation],
    ) -> pd.DataFrame:
        from xai_pipeline.datasets.simplification import LABELS

        examples_by_id = {ex.example_id: ex for ex in examples}
        tokenizer = self.model.tokenizer
        device = next(self.model.model.parameters()).device
        pad_id = tokenizer.pad_token_id

        records: list[MetricRecord] = []

        for explanation in explanations:
            example = examples_by_id[explanation.example_id]
            target_index = LABELS.index(explanation.target)

            encoded = tokenizer(
                example.inputs["source_text"],
                text_pair=example.inputs["simplified_text"],
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            n_tokens = input_ids.shape[1]

            if n_tokens != len(explanation.scores):
                continue  # length mismatch (truncation edge case) -- skip safely

            importance = np.abs(explanation.scores)
            order = np.argsort(-importance)  # most important first

            step_points = np.linspace(0, n_tokens, self.n_steps + 1, dtype=int)

            def get_probability(ids: torch.Tensor) -> float:
                with torch.inference_mode():
                    logits = self.model.model(input_ids=ids, attention_mask=attention_mask).logits[0]
                    prob = torch.sigmoid(logits.float())[target_index]
                return float(prob)

            deletion_probs = []
            for k in step_points:
                masked_ids = input_ids.clone()
                if k > 0:
                    top_k_positions = order[:k]
                    masked_ids[0, top_k_positions] = pad_id
                deletion_probs.append(get_probability(masked_ids))

            insertion_probs = []
            baseline_ids_full = torch.full_like(input_ids, pad_id)
            for k in step_points:
                revealed_ids = baseline_ids_full.clone()
                if k > 0:
                    top_k_positions = order[:k]
                    revealed_ids[0, top_k_positions] = input_ids[0, top_k_positions]
                insertion_probs.append(get_probability(revealed_ids))

            deletion_auc = _trapezoidal_auc(deletion_probs)
            insertion_auc = _trapezoidal_auc(insertion_probs)

            for metric_name, value in [
                ("deletion_auc", deletion_auc),
                ("insertion_auc", insertion_auc),
            ]:
                records.append(
                    MetricRecord(
                        run_id="task2_deletion_insertion",
                        example_id=explanation.example_id,
                        method=explanation.method,
                        metric=metric_name,
                        value=value,
                        slice_name="target",
                        slice_value=str(explanation.target),
                        metadata={"n_steps": self.n_steps, "n_tokens": n_tokens},
                    )
                )

        return pd.DataFrame([asdict(r) for r in records])

    def validate(self) -> list[str]:
        return []