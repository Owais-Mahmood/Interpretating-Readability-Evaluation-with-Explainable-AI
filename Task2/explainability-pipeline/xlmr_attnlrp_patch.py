"""
AttnLRP patch for XLM-RoBERTa (covers both XLM-R and E5, since both are
RoBERTa-family architectures). Built by porting lxt's already-correct,
paper-validated BERT attention math (from lxt.efficient.models.bert),
since RoBERTa's self-attention forward pass is functionally identical to
BERT's -- confirmed by direct source comparison, not assumed.

Applies four patches:
1. torch.nn.LayerNorm.forward -> identity rule (reused verbatim from lxt)
2. torch.nn.Dropout.forward -> no-op (reused verbatim from lxt)
3. XLMRobertaSelfAttention.forward -> same as HuggingFace's original,
   with divide_gradient() inserted at the two bilinear matmul points
   (Q·K^T and attention_probs·V) -- this is the actual AttnLRP
   contribution, Equation 7 of the paper
4. XLMRobertaIntermediate.forward -> identity rule wrapping the GELU
   activation (Equation 9 of the paper)

Run from the repo root (with the attnlrp_venv activated):
    python3 test_xlmr_attnlrp_conservation.py
"""

import math
import sys
import importlib.util
from pathlib import Path

import torch
from transformers.models.xlm_roberta import modeling_xlm_roberta as xlmr_module


def _load_module_direct(name, file_path):
    """Load a specific lxt submodule directly from its file, bypassing
    lxt.efficient's __init__.py (which eagerly imports every supported
    architecture, including ones this transformers version doesn't have,
    e.g. qwen3 -- even though we only need rules.py and patches.py)."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import lxt
_lxt_root = Path(lxt.__file__).parent
_rules = _load_module_direct("lxt.efficient.rules", _lxt_root / "efficient" / "rules.py")
_patches = _load_module_direct("lxt.efficient.patches", _lxt_root / "efficient" / "patches.py")

divide_gradient = _rules.divide_gradient
identity_rule_implicit = _rules.identity_rule_implicit
layer_norm_forward = _patches.layer_norm_forward
dropout_forward = _patches.dropout_forward


_CALL_COUNT = {"count": 0}


def patched_self_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    """Identical to XLMRobertaSelfAttention.forward, except for the two
    divide_gradient() calls marked below (the AttnLRP contribution)."""
    _CALL_COUNT["count"] += 1
    mixed_query_layer = self.query(hidden_states)

    is_cross_attention = encoder_hidden_states is not None
    if is_cross_attention and past_key_value is not None:
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    use_cache = past_key_value is not None
    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = divide_gradient(attention_scores, 2)  # <-- AttnLRP

    if self.position_embedding_type in ("relative_key", "relative_key_query"):
        query_length, key_length = query_layer.shape[2], key_layer.shape[2]
        if use_cache:
            position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        else:
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = divide_gradient(context_layer, 2)  # <-- AttnLRP

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


def patched_intermediate_forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = identity_rule_implicit(self.intermediate_act_fn, hidden_states)
    return hidden_states


def apply_xlmr_attnlrp_patch(verbose=True):
    torch.nn.LayerNorm.forward = layer_norm_forward
    torch.nn.Dropout.forward = dropout_forward
    xlmr_module.XLMRobertaSelfAttention.forward = patched_self_attention_forward
    xlmr_module.XLMRobertaIntermediate.forward = patched_intermediate_forward

    # IMPORTANT: XLMRobertaSdpaSelfAttention overrides forward independently
    # (doesn't call super().forward()), so patching only the base class is
    # silently bypassed when the SDPA variant is instantiated (the modern
    # default). Patch it too, with the same function -- it doesn't depend
    # on any SDPA-specific internals.
    if hasattr(xlmr_module, "XLMRobertaSdpaSelfAttention"):
        xlmr_module.XLMRobertaSdpaSelfAttention.forward = patched_self_attention_forward

    if verbose:
        print("Patched: LayerNorm, Dropout, XLMRobertaSelfAttention, "
              "XLMRobertaSdpaSelfAttention, XLMRobertaIntermediate")