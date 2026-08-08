"""
Tests AttnLRP conservation on the REAL, trained XLM-R model (not a
random dummy), to check whether the near-zero-sum discrepancy found
with tiny untrained models also appears with a real, trained model.

Run from Task2/explainability-pipeline, with attnlrp_venv activated:
    python3 test_real_xlmr_attnlrp.py
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xlmr_attnlrp_patch import apply_xlmr_attnlrp_patch

REPO_ID = "hannah-khallaf/e2r-strategy-xlmr-large-focal"

print("Loading real XLM-R model (should use HF cache, fast)...")
tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, trust_remote_code=True)
model.eval()
print("Loaded.")

apply_xlmr_attnlrp_patch(verbose=True)

# Same fix discovered earlier tonight: the custom E2R wrapper class
# doesn't implement get_input_embeddings() correctly -- find the real
# word embedding layer by name and patch it in directly.
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
    if embedding_layer is None:
        raise RuntimeError("Could not find word embeddings layer.")

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

target_logit.backward(target_logit)

relevance = embeds.grad.sum(-1).squeeze(0)
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
print()
if abs(ratio - 1.0) < 0.2:
    print("PASS: conservation property holds reasonably well (ratio close to 1.0)")
else:
    print("FAIL: conservation property does NOT hold (ratio far from 1.0)")