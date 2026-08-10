from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from xai_pipeline.contracts import Example, Explanation, MetricRecord, Prediction
from xai_pipeline.registry import EVALUATORS


@EVALUATORS.register("attribution_stability")
class AttributionStabilityEvaluator:
    """Attribution stability: how consistent an explanation stays under
    small input perturbations. Works generically with ANY explainer,
    by temporarily adding small noise to the model's embedding layer
    weights, re-running the SAME explainer, and measuring the rank
    correlation (Spearman) between the original and perturbed
    per-token importance scores.

    A stable/trustworthy explainer should give HIGH correlation (close
    to 1.0) under tiny noise -- if small, meaningless input changes
    drastically reorder which tokens look important, that's a red flag
    about the explanation's reliability, not the underlying model.

    NOTE / ASSUMPTION: noise_std is expressed as a fraction of the
    embedding weights' own standard deviation (default 1%), chosen to
    be small enough to test LOCAL stability without destroying the
    explanation outright -- worth confirming with Nouran whether a
    different perturbation magnitude is expected for the final report.
    """

    name = "attribution_stability"

    def __init__(self, model, explainer, n_perturbations: int = 3, noise_std: float = 0.01) -> None:
        self.model = model
        self.explainer = explainer
        self.n_perturbations = n_perturbations
        self.noise_std = noise_std

    def evaluate(
        self,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
        explanations: Sequence[Explanation],
    ) -> pd.DataFrame:
        embedding_layer = self.model.model.get_input_embeddings()
        original_weights = embedding_layer.weight.data.clone()
        weight_std = original_weights.std().item()

        examples_by_id = {ex.example_id: ex for ex in examples}
        predictions_by_id = {p.example_id: p for p in predictions}

        records: list[MetricRecord] = []

        try:
            for original_exp in explanations:
                if original_exp.method != self.explainer.name:
                    continue  # only re-run explanations matching this evaluator's explainer

                example = examples_by_id[original_exp.example_id]
                prediction = predictions_by_id[original_exp.example_id]

                correlations = []
                for _ in range(self.n_perturbations):
                    noise = torch.randn_like(embedding_layer.weight) * self.noise_std * weight_std
                    embedding_layer.weight.data = original_weights + noise

                    perturbed_explanations = self.explainer.explain([example], self.model, [prediction])
                    matching = [e for e in perturbed_explanations if e.target == original_exp.target]
                    if not matching:
                        continue
                    perturbed_exp = matching[0]

                    if len(perturbed_exp.scores) != len(original_exp.scores):
                        continue  # length mismatch -- skip safely

                    corr, _ = spearmanr(original_exp.scores, perturbed_exp.scores)
                    if not np.isnan(corr):
                        correlations.append(corr)

                embedding_layer.weight.data = original_weights.clone()  # restore before next example

                if not correlations:
                    continue

                records.append(
                    MetricRecord(
                        run_id="task2_attribution_stability",
                        example_id=original_exp.example_id,
                        method=original_exp.method,
                        metric="stability_spearman",
                        value=float(np.mean(correlations)),
                        slice_name="target",
                        slice_value=str(original_exp.target),
                        metadata={
                            "n_perturbations": len(correlations),
                            "noise_std_fraction": self.noise_std,
                        },
                    )
                )
        finally:
            # Always restore original weights, even if something raised
            embedding_layer.weight.data = original_weights

        return pd.DataFrame([asdict(r) for r in records])

    def validate(self) -> list[str]:
        return []