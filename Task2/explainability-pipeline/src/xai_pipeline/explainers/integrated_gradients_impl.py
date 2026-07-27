from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from xai_pipeline.contracts import Example, Explanation, Prediction
from xai_pipeline.registry import EXPLAINERS


@EXPLAINERS.register("integrated_gradients")
class IntegratedGradientsExplainer:
    """Captum Integrated Gradients, wrapped to match the framework's
    Explainer contract. Reuses the same LayerIntegratedGradients approach
    already verified against the real mBERT model."""

    name = "integrated_gradients"

    def __init__(self, n_steps: int = 50) -> None:
        self.n_steps = n_steps

    def _model_forward(self, model_adapter, input_ids, attention_mask, token_type_ids):
        return model_adapter.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

    def explain(
        self,
        examples: Sequence[Example],
        model,
        predictions: Sequence[Prediction],
    ) -> list[Explanation]:
        explanations: list[Explanation] = []
        device = next(model.model.parameters()).device
        tokenizer = model.tokenizer

        from xai_pipeline.models.mbert_e2r import LABELS

        def forward_func(input_ids, attention_mask, token_type_ids):
            return self._model_forward(model, input_ids, attention_mask, token_type_ids)

        lig = LayerIntegratedGradients(forward_func, model.model.get_input_embeddings())

        for example, prediction in zip(examples, predictions):
            for target_label in prediction.predicted_label:
                target_index = LABELS.index(target_label)

                encoded = tokenizer(
                    example.inputs["source_text"],
                    text_pair=example.inputs["simplified_text"],
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                token_type_ids = encoded.get(
                    "token_type_ids", torch.zeros_like(input_ids)
                )

                baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

                attributions, delta = lig.attribute(
                    inputs=input_ids,
                    baselines=baseline_ids,
                    additional_forward_args=(attention_mask, token_type_ids),
                    target=target_index,
                    n_steps=self.n_steps,
                    return_convergence_delta=True,
                )

                token_scores = attributions.sum(dim=-1).squeeze(0)
                norm = torch.norm(token_scores)
                if norm > 0:
                    token_scores = token_scores / norm

                tokens = tokenizer.convert_ids_to_tokens(
                    input_ids.squeeze(0).tolist()
                )

                explanations.append(
                    Explanation(
                        example_id=example.example_id,
                        method=self.name,
                        target=target_label,
                        units=tokens,
                        scores=token_scores.detach().cpu().numpy(),
                        unit_type="subword_token",
                        signed=True,
                        metadata={
                            "convergence_delta": float(delta.item()),
                            "n_steps": self.n_steps,
                        },
                    )
                )

        return explanations

    def validate(self) -> list[str]:
        return []