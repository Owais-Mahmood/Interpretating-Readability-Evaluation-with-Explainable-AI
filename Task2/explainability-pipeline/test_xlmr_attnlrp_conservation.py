"""
Tests the XLM-RoBERTa AttnLRP patch against LRP's core conservation
property: the sum of all token relevance scores should approximately
equal the actual output value being explained. This is the same rigor
check that caught the original mBERT AttnLRP bug.

Run from the repo root (with attnlrp_venv activated):
    python3 test_xlmr_attnlrp_conservation.py
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel

from xlmr_attnlrp_patch import apply_xlmr_attnlrp_patch

torch.manual_seed(0)

# Small dummy XLM-RoBERTa config, no download needed
config = XLMRobertaConfig(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=32,
)
backbone = XLMRobertaModel(config)
classifier = nn.Linear(32, 6)  # 6 labels, matching our real taxonomy

apply_xlmr_attnlrp_patch(verbose=True)

torch.manual_seed(0)
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

# Trace gradient magnitude through each encoder layer by hooking outputs
layer_outputs = []
hooks = []
for i, layer in enumerate(backbone.encoder.layer):
    def make_hook(idx):
        def hook(module, input, output):
            output[0].retain_grad()
            layer_outputs.append((idx, output[0]))
        return hook
    hooks.append(layer.register_forward_hook(make_hook(i)))

outputs = backbone(inputs_embeds=embeds, attention_mask=attention_mask)
pooled = outputs.last_hidden_state[:, 0, :]
logits = classifier(pooled)
target_logit = logits[0, 3]
target_logit.backward(target_logit)

for h in hooks:
    h.remove()

print()
print("Gradient magnitude (sum of abs) at each layer's output:")
for idx, layer_out in layer_outputs:
    if layer_out.grad is not None:
        print(f"  Layer {idx} output: grad sum(abs) = {layer_out.grad.abs().sum().item():.6e}")
    else:
        print(f"  Layer {idx} output: grad is None")

print(f"embeds.grad sum(abs) = {embeds.grad.abs().sum().item():.6e}")
print(f"target_logit = {target_logit.item():.6f}")