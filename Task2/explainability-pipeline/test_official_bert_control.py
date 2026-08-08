"""
Control test: same conservation-property methodology, but using lxt's
OWN official, fully-validated BERT patch (full module replacement),
not our hand-ported XLM-R version. If this also fails with the same
pattern, the bug is in the test methodology. If it passes, the bug is
specific to our XLM-R port.
"""

import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

# Bypass the broken lxt.efficient.models package __init__.py (qwen3 issue),
# same workaround as before, but this time we DO want the full bert.py
# reimplementation, loaded directly.
import lxt
_lxt_root = Path(lxt.__file__).parent


def _load_module_direct(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_rules = _load_module_direct("lxt.efficient.rules", _lxt_root / "efficient" / "rules.py")
_patches = _load_module_direct("lxt.efficient.patches", _lxt_root / "efficient" / "patches.py")
_bert = _load_module_direct("lxt_bert_official", _lxt_root / "efficient" / "models" / "bert.py")

# _bert is already a fully self-contained file with LRP rules baked
# directly into BertModel/BertEncoder/BertLayer/etc -- no need to
# replace_module anything, just use its classes directly.
torch.nn.LayerNorm.forward = _patches.layer_norm_forward
torch.nn.Dropout.forward = _patches.dropout_forward

print("Applied lxt's official BERT patch (self-contained module) + generic LayerNorm/Dropout patches.")
print()

for seed in [0, 1, 2, 3, 4]:
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    backbone = _bert.BertModel(config)
    classifier = nn.Linear(32, 6)

    input_ids = torch.randint(0, 100, (1, 8))
    attention_mask = torch.ones(1, 8)

    embeds = backbone.get_input_embeddings()(input_ids).clone().detach().requires_grad_()
    outputs = backbone(inputs_embeds=embeds, attention_mask=attention_mask)
    pooled = outputs.last_hidden_state[:, 0, :]
    logits = classifier(pooled)
    target_logit = logits[0, 3]
    target_logit.backward(target_logit)

    relevance = embeds.grad.sum(-1).squeeze(0)
    ratio = (relevance.sum() / target_logit).item()
    print(f"seed={seed}: target={target_logit.item():.4f}, sum(relevance)={relevance.sum().item():.6e}, ratio={ratio:.6e}")