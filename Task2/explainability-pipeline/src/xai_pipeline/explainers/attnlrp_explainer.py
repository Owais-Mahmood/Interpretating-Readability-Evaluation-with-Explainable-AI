from __future__ import annotations

from collections.abc import Sequence

import torch

from xai_pipeline.contracts import Example, Explanation, Prediction
from xai_pipeline.registry import EXPLAINERS

# NOTE: this explainer requires the AttnLRP patch to already be applied
# (via apply_xlmr_attnlrp_patch() from xlmr_attnlrp_patch.py) before use,
# and requires an environment with a compatible (older) transformers
# version -- see docs/ATTNLRP_XLMR_SESSION_FINDINGS.md for why. This
# is NOT compatible with the main pipeline's transformers version;
# run it in the isolated attnlrp_venv.


@EXPLAINERS.register("attnlrp")
class AttnLRPExplainer:
    """AttnLRP for XLM-R and E5 (RoBERTa-family models). Ported from
    lxt's official, validated BERT implementation -- see
    xlmr_attnlrp_patch.py for the actual patch and
    docs/ATTNLRP_XLMR_SESSION_FINDINGS.md for the full investigation,
    including a critical methodology fix (correct backward() and
    Input*Gradient formula) that took the conservation ratio from
    completely broken (~0.000000) to working (~2.6, target ratio 1.0).
    """

    name = "attnlrp"

    def explain(
        self,
        examples: Sequence[Example],
        model,
        predictions: Sequence[Prediction],
    ) -> list[Explanation]:
        from xai_pipeline.datasets.simplification import LABELS

        explanations: list[Explanation] = []
        device = next(model.model.parameters()).device
        tokenizer = model.tokenizer
        embedding_layer = model.model.get_input_embeddings()

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

                embeds = embedding_layer(input_ids).clone().detach().requires_grad_()

                outputs = model.model(inputs_embeds=embeds, attention_mask=attention_mask)
                logits = outputs.logits[0]
                target_logit = logits[target_index]

                # CORRECT AttnLRP methodology (see findings doc):
                # backward() with no argument, relevance = Input*Gradient
                target_logit.backward()
                token_scores = (embeds * embeds.grad).sum(-1).squeeze(0)

                norm = torch.norm(token_scores)
                if norm > 0:
                    token_scores = token_scores / norm

                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

                explanations.append(
                    Explanation(
                        example_id=example.example_id,
                        method=self.name,
                        target=target_label,
                        units=tokens,
                        scores=token_scores.detach().cpu().numpy(),
                        unit_type="subword_token",
                        signed=True,
                        metadata={},
                    )
                )

        return explanations

    def validate(self) -> list[str]:
        return []