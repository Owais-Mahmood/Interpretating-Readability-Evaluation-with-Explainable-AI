from __future__ import annotations

from collections.abc import Sequence

import torch

from xai_pipeline.contracts import Example, Explanation, Prediction
from xai_pipeline.registry import EXPLAINERS


@EXPLAINERS.register("raw_attention")
class RawAttentionExplainer:
    """Uses the model's own attention scores directly as an importance
    measure. For each token, sums the attention it RECEIVES from the
    [CLS] token (the representation actually used for classification)
    across all heads in a chosen layer, averaged if layer=None across
    all layers."""

    name = "raw_attention"

    def __init__(self, layer: int | None = -1) -> None:
        # layer=-1 means "last layer" (closest to the classification head,
        # generally the most task-relevant); layer=None averages all layers
        self.layer = layer

    def explain(
        self,
        examples: Sequence[Example],
        model,
        predictions: Sequence[Prediction],
    ) -> list[Explanation]:
        explanations: list[Explanation] = []
        device = next(model.model.parameters()).device
        tokenizer = model.tokenizer

        # The custom E2RClassifier wrapper builds its BERT backbone
        # separately, so attn_implementation="eager" passed at load() time
        # doesn't propagate down to it. Force it directly here instead.
        model.model.config._attn_implementation = "eager"
        if hasattr(model.model, "backbone"):
            model.model.backbone.config._attn_implementation = "eager"

        for example, prediction in zip(examples, predictions):
            encoded = tokenizer(
                example.inputs["source_text"],
                text_pair=example.inputs["simplified_text"],
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = model.model(**encoded, output_attentions=True)

            # outputs.attentions: tuple of (num_layers) tensors, each
            # shape (batch=1, num_heads, seq_len, seq_len)
            attentions = outputs.attentions
            if self.layer is None:
                # average across all layers
                stacked = torch.stack(attentions, dim=0)  # (n_layers, 1, heads, seq, seq)
                layer_attention = stacked.mean(dim=0)
            else:
                layer_attention = attentions[self.layer]

            # Average across heads, then take the attention FROM the [CLS]
            # token (index 0) TO every other token -- i.e. how much the
            # representation used for classification attended to each token
            head_avg = layer_attention.mean(dim=1)  # (1, seq, seq)
            cls_attention = head_avg[0, 0, :]  # (seq,)

            tokens = tokenizer.convert_ids_to_tokens(
                encoded["input_ids"].squeeze(0).tolist()
            )
            scores = cls_attention.detach().cpu().numpy()

            for target_label in prediction.predicted_label:
                explanations.append(
                    Explanation(
                        example_id=example.example_id,
                        method=self.name,
                        target=target_label,
                        units=tokens,
                        scores=scores,
                        unit_type="subword_token",
                        # Attention weights are always non-negative (softmax
                        # output), so this is not a signed measure the way
                        # gradient-based methods are
                        signed=False,
                        metadata={"layer": self.layer},
                    )
                )

        return explanations

    def validate(self) -> list[str]:
        return []