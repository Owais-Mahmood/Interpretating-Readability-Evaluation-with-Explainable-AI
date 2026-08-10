"""
Traces gradient magnitude through each of the real XLM-R model's 24
layers, to find exactly where relevance collapses to zero (unlike the
tiny 2-layer dummy test, which showed substantial gradient throughout).

Run from Task2/explainability-pipeline, with attnlrp_venv activated:
    python3 trace_real_xlmr_layers.py
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xlmr_attnlrp_patch import apply_xlmr_attnlrp_patch

REPO_ID = "hannah-khallaf/e2r-strategy-xlmr-large-focal"

print("Loading real XLM-R model...")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, trust_remote_code=True)
model.eval()
print("Loaded.")

apply_xlmr_attnlrp_patch(verbose=True)

try:
    model.get_input_embeddings()
    embedding_layer = model.get_input_embeddings()
except NotImplementedError:
    embedding_layer = None
    for name, module in model.named_modules():
        if name.endswith("word_embeddings"):
            embedding_layer = module
            model.get_input_embeddings = lambda m=module: m
            break

source_text = "The committee postponed the implementation of the measure."
simplified_text = "The committee decided to use the measure later."

encoded = tokenizer(source_text, simplified_text, return_tensors="pt", truncation=True, max_length=256)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

embeds = embedding_layer(input_ids).clone().detach().requires_grad_()

# Find the actual encoder (list of transformer layers) inside the model,
# same way we found the embedding layer -- search by name since the
# custom wrapper class doesn't expose it under a fixed attribute path.

encoder_layers = None
for name, module in model.named_modules():
    if name.endswith("encoder.layer"):
        encoder_layers = module
        break

if encoder_layers is None:
    print("Could not find encoder.layer list -- checking model structure:")
    for name, module in model.named_modules():
        print(" ", name, type(module).__name__)
else:
    print(f"Found {len(encoder_layers)} encoder layers.")

    hooks = []
    layer_outputs = []
    for i, layer in enumerate(encoder_layers):
        def make_hook(idx):
            def hook(module, input, output):
                output[0].retain_grad()
                layer_outputs.append((idx, output[0]))
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    logits = outputs.logits[0]
    target_logit = logits[0]
    target_logit.backward(target_logit)

    for h in hooks:
        h.remove()

    print()
    print("Gradient magnitude (sum of abs) at each layer's output:")
    for idx, layer_out in layer_outputs:
        if layer_out.grad is not None:
            print(f"  Layer {idx}: grad sum(abs) = {layer_out.grad.abs().sum().item():.6e}")
        else:
            print(f"  Layer {idx}: grad is None")

    print(f"embeds.grad sum(abs) = {embeds.grad.abs().sum().item():.6e}")
    print(f"target_logit = {target_logit.item():.6f}")