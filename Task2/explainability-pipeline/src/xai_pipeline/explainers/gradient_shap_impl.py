from __future__ import annotations

from collections.abc import Sequence

import torch
from captum.attr import GradientShap

from xai_pipeline.contracts import Example, Explanation, Prediction
from xai_pipeline.registry import EXPLAINERS


@EXPLAINERS.register("gradient_shap")
class GradientShapExplainer:
    """Captum GradientSHAP, wrapped to match the framework's Explainer
    contract. Reuses the same approach already verified against the real
    mBERT model: averages over several randomly-perturbed baselines around
    an all-PAD embedding, giving a SHAP-style approximation per token."""

    name = "gradient_shap"

    def __init__(self, n_samples: int = 25, n_baselines: int = 5) -> None:
        self.n_samples = n_samples
        self.n_baselines = n_baselines

    def explain(
        self,
        examples: Sequence[Example],
        model,
        predictions: Sequence[Prediction],
    ) -> list[Explanation]:
        from xai_pipeline.models.mbert_e2r import LABELS

        explanations: list[Explanation] = []
        device = next(model.model.parameters()).device
        tokenizer = model.tokenizer
        embedding_layer = model.model.get_input_embeddings()

        def forward_from_embeds(embeds, attention_mask, token_type_ids):
            return model.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

        gshap = GradientShap(forward_from_embeds)

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

                input_embeds = embedding_layer(input_ids)
                baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
                baseline_embeds = embedding_layer(baseline_ids)
                baseline_dist = baseline_embeds.repeat(self.n_baselines, 1, 1)

                attributions = gshap.attribute(
                    inputs=input_embeds,
                    baselines=baseline_dist,
                    target=target_index,
                    n_samples=self.n_samples,
                    additional_forward_args=(attention_mask, token_type_ids),
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
                            "n_samples": self.n_samples,
                            "n_baselines": self.n_baselines,
                        },
                    )
                )

        return explanations

    def validate(self) -> list[str]:
        return []