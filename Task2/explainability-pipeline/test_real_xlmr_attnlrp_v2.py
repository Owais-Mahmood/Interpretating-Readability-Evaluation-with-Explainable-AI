"""
Tests AttnLRP conservation on the REAL, trained XLM-R model, using the
CORRECTED methodology (backward() with no argument, true Input*Gradient).

Run from Task2/explainability-pipeline, with attnlrp_venv activated:
    python3 test_real_xlmr_attnlrp_v2.py
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

outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
logits = outputs.logits[0]
target_index = 0  # Synonymy
target_logit = logits[target_index]

# CORRECT methodology
target_logit.backward()
relevance = (embeds * embeds.grad).sum(-1).squeeze(0)

tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

print()
print("Per-token relevance:")
for tok, r in zip(tokens, relevance.tolist()):
    print(f"  {tok}: {r:.4f}")

print()
print(f"Sum of relevance: {relevance.sum().item():.6f}")
print(f"Target logit value: {target_logit.item():.6f}")
ratio = (relevance.sum() / target_logit).item()
print(f"Conservation ratio (sum/target): {ratio:.6f}")