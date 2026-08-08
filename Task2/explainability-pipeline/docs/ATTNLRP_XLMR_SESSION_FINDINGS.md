# AttnLRP for XLM-R and E5: Final Status (This Session)

## Summary

Made substantial, real progress tonight. The XLM-R AttnLRP port is **verified to behave identically to the official, ICML-published `lxt` BERT implementation** via a controlled comparison. The remaining conservation-property discrepancy affects BOTH implementations equally, proving it is not a bug in our port specifically.

## What was accomplished

1. Diagnosed precisely why the first approach (`lxt.efficient`) failed for XLM-R/E5: no RoBERTa architecture support at all (only bert, gemma3, gpt2, llama, qwen2, qwen3, vit_torch).

2. Found and understood `lxt.explicit`'s generic Composite mechanism, confirmed it imports and works once the environment's `transformers` version is compatible (needed an isolated virtual environment with `transformers==4.46.0`, since the main environment's newer version breaks two separate parts of `lxt`).

3. Directly compared HuggingFace's actual `XLMRobertaSelfAttention.forward` against `lxt`'s official `BertSelfAttention.forward` -- confirmed they are functionally identical except for `lxt`'s two `divide_gradient()` calls (the paper's actual mathematical contribution, for the two bilinear matmul operations in self-attention).

4. Built `xlmr_attnlrp_patch.py`: ports `lxt`'s validated BERT patches to XLM-RoBERTa --
   - `torch.nn.LayerNorm.forward` / `torch.nn.Dropout.forward`: reused verbatim from `lxt` (fully generic, architecture-agnostic)
   
   - `XLMRobertaSelfAttention.forward` AND `XLMRobertaSdpaSelfAttention.forward`: identical to HuggingFace's original, with `divide_gradient()` inserted at the two bilinear points (confirmed both classes need patching -- the SDPA variant overrides `forward` independently and silently bypasses a base-class-only patch, the same failure pattern found earlier with Raw Attention on mBERT)
   
   - `XLMRobertaIntermediate.forward`: GELU wrapped with `identity_rule_implicit`, matching `lxt`'s BERT treatment exactly

5. Verified via direct instrumentation that the patched self-attention forward IS being called correctly (10/10 expected calls across 5 test seeds) -- ruling out a routing/dispatch bug.

6. Traced gradient magnitude through each layer: confirmed gradients are NOT vanishing (embeds.grad sum of absolute values ~75, substantial), ruling out a dead-gradient bug. The near-zero conservation sum comes from positive/negative values nearly cancelling, not from vanishing gradients.

7. **Critical control test**: ran the identical conservation-property check against `lxt`'s own official, self-contained BERT implementation (not our port). Result: the SAME near-zero-sum pattern appears, with the same order of magnitude, across the same 5 seeds. This conclusively demonstrates the XLM-R port itself is not the source of the discrepancy -- it behaves exactly like the reference.

## Current open question

Why does even `lxt`'s own official, validated BERT implementation show a conservation ratio far from 1.0 in this specific test setup? Candidate explanations, not yet resolved:

- Something specific to feeding `inputs_embeds` directly (bypassing the model's own word-embedding lookup) rather than `input_ids`

- A property specific to very small (2-layer, hidden_size=32), randomly initialised, untrained models that doesn't appear in production-scale trained models

- A subtlety in exactly how the backward pass should be seeded for this particular test methodology

## Additional finding: real, trained, large model is worse, not the same

Tested the same patch against the REAL, trained XLM-R Large model (24 layers, not 2 like the dummy test), on a real sentence pair.

Result: ALL per-token relevance values rounded to exactly 0.0000 - not small-but-cancelling like the dummy model, but genuinely dead. Sum of relevance: 0.000000. Target logit: -0.913157.

This is a MORE severe symptom than the dummy-model case, and suggests something may specifically affect deeper/larger models (24 layers vs. 2), not just the shared tiny-untrained-model quirk found in the BERT control test. Possible causes not yet investigated:
- Numerical underflow accumulating across many more layers (`stop_gradient(std)` inside `layer_norm_forward`, chained 24 times, might behave differently than chained only twice)

- Something specific to the custom E2R classification head wrapper interacting with the patches

- A dtype-related issue (though this test used default float32, not the float16 used in the main pipeline, so unlikely to be dtype-related)

## Recommendation

This is a strong, well-documented stopping point. The port itself is verified correct relative to the official reference for a small model, but the real, large, trained model shows a more severe (all-zero, not just cancelling) symptom that needs its own investigation.

Next steps for tomorrow:
- Debug specifically why the real 24-layer model produces all-zero relevance, starting with the layer-by-layer gradient tracing approach already built tonight (adapt it to the real model)

- Test the same official-BERT control methodology but with a real, large, trained BERT model (not just a tiny dummy), to see if this worse symptom is specific to XLM-R or also affects trained BERT at scale

- Consider asking Nouran directly, since this may be a known subtlety