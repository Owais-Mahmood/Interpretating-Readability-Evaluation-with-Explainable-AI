"""
Tests AttnLRP conservation with the CORRECT methodology, per lxt's own
official documentation quickstart (BERT Classifier example):
  1. backward() called with NO argument (default gradient=1.0)
  2. relevance = (input_embeds * input_embeds.grad).sum(-1) -- true
     Input*Gradient, not just the raw gradient alone

Run from Task2/explainability-pipeline, with attnlrp_venv activated:
    python3 test_xlmr_attnlrp_conservation_v2.py
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel

from xlmr_attnlrp_patch import apply_xlmr_attnlrp_patch

apply_xlmr_attnlrp_patch(verbose=True)

print()
print("=== Dummy small model, multiple seeds ===")
for seed in [0, 1, 2, 3, 4]:
    torch.manual_seed(seed)
    config = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    backbone = XLMRobertaModel(config)
    classifier = nn.Linear(32, 6)

    input_ids = torch.randint(0, 100, (1, 8))
    attention_mask = torch.ones(1, 8)

    embeds = backbone.get_input_embeddings()(input_ids).clone().detach().requires_grad_()
    outputs = backbone(inputs_embeds=embeds, attention_mask=attention_mask)
    pooled = outputs.last_hidden_state[:, 0, :]
    logits = classifier(pooled)
    target_logit = logits[0, 3]

    # CORRECT methodology: backward() with no argument
    target_logit.backward()

    # CORRECT methodology: multiply by the embeddings themselves (Input*Gradient)
    relevance = (embeds * embeds.grad).sum(-1).squeeze(0)
    ratio = (relevance.sum() / target_logit).item()
    print(f"seed={seed}: target={target_logit.item():.4f}, sum(relevance)={relevance.sum().item():.6f}, ratio={ratio:.6f}")