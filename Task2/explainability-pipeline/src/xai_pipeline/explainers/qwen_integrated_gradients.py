from __future__ import annotations

from collections.abc import Sequence

import torch
from captum.attr import LayerIntegratedGradients

from xai_pipeline.contracts import Example, Explanation, Prediction
from xai_pipeline.registry import EXPLAINERS


def _true_false_token_ids(tokenizer):
    true_ids = {
        int(tokenizer.encode(value, add_special_tokens=False)[0])
        for value in ("true", " true", "True", " True")
        if tokenizer.encode(value, add_special_tokens=False)
    }
    false_ids = {
        int(tokenizer.encode(value, add_special_tokens=False)[0])
        for value in ("false", " false", "False", " False")
        if tokenizer.encode(value, add_special_tokens=False)
    }
    return true_ids, false_ids


@EXPLAINERS.register("integrated_gradients_qwen")
class QwenIntegratedGradientsExplainer:
    """Integrated Gradients adapted for Qwen's prompted binary-relevance
    architecture. Unlike the classification-head models, there is no fixed
    target index -- the quantity being explained IS the scalar P(true)
    computed from the last token position's logits, restricted to the
    true/false vocabulary tokens (same computation as
    binary_relevance.boolean_token_probability, made differentiable here).

    ASSUMPTION: explains the FULL prompt (system message + taxonomy card +
    instructions + sentence pair), since that's literally everything fed
    to the model. This means most of the explained tokens are boilerplate
    taxonomy-card text, not the sentence pair itself -- when evaluating
    against gold spans later, only the sentence-pair portion of the prompt
    should be extracted for comparison, similar to how encoder-model
    explanations are filtered via tokenizer offset mapping.
    """

    name = "integrated_gradients_qwen"

    def __init__(self, n_steps: int = 20) -> None:
        # NOTE: default n_steps is lower than the other explainers (50),
        # since Qwen's prompts are much longer (full taxonomy card +
        # instructions), making each step more expensive on a 7B model.
        self.n_steps = n_steps

    def explain(
        self,
        examples: Sequence[Example],
        model,
        predictions: Sequence[Prediction],
    ) -> list[Explanation]:
        from xai_pipeline.models.qwen_e2r import LABELS

        explanations: list[Explanation] = []
        device = next(model.model.parameters()).device
        tokenizer = model.tokenizer
        true_ids, false_ids = _true_false_token_ids(tokenizer)
        selected_ids = torch.tensor(sorted(true_ids | false_ids), device=device)
        true_mask = torch.tensor([tid in true_ids for tid in sorted(true_ids | false_ids)], device=device)

        embedding_layer = model.model.get_input_embeddings()

        def forward_func(input_ids, attention_mask):
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # last token position
            selected_logits = logits[:, selected_ids]
            probs = torch.softmax(selected_logits, dim=-1)
            true_prob = probs[:, true_mask].sum(dim=-1)
            return true_prob  # scalar per batch element -- no target= needed

        lig = LayerIntegratedGradients(forward_func, embedding_layer)

        for example, prediction in zip(examples, predictions):
            row = {
                "source_text": example.inputs["source_text"],
                "simplified_text": example.inputs["simplified_text"],
            }
            for label in prediction.predicted_label:
                system_prompt, user_prompt = model.binary_relevance.build_binary_prompt(
                    row, label, model.taxonomy,
                    include_definition=True, demonstrations=(),
                    request_confidence=False, response_format="boolean", input_mode="pair",
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or tokenizer.eos_token_id)

                attributions = lig.attribute(
                    inputs=input_ids,
                    baselines=baseline_ids,
                    additional_forward_args=(attention_mask,),
                    n_steps=self.n_steps,
                    internal_batch_size=1,
                )

                token_scores = attributions.sum(dim=-1).squeeze(0)
                norm = torch.norm(token_scores)
                if norm > 0:
                    token_scores = token_scores / norm

                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

                explanations.append(
                    Explanation(
                        example_id=example.example_id,
                        method=self.name,
                        target=label,
                        units=tokens,
                        scores=token_scores.detach().cpu().numpy(),
                        unit_type="subword_token",
                        signed=True,
                        metadata={"n_steps": self.n_steps, "prompt_length": len(tokens)},
                    )
                )

        return explanations

    def validate(self) -> list[str]:
        return []