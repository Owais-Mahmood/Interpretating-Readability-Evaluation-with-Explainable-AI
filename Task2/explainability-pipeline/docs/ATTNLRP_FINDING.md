# AttnLRP: Compatibility Finding (Not Implemented)

## Summary

AttnLRP was NOT successfully implemented for this project. This is documented here as a genuine technical finding, not left silent, since real investigation was done before concluding it's currently blocked.

## What was tried

The official reference implementation (`lxt`, the "LRP-eXplains-Transformers" library cited in the task references [R14]/[R15]) was installed and tested against our mBERT setup.

`lxt`'s BERT support (`lxt.efficient.models.bert`) works by monkey-patching. BERT's internal layers (LayerNorm, Dropout, and the module's forward methods) so that a standard PyTorch backward pass computes LRP relevance scores instead of ordinary gradients.

## The blocker

1. `lxt`'s BERT module fails to import against the current `transformers` version (5.14.1) -- it expects a function (`find_pruneable_heads_and_indices`) that has since been removed from `transformers.pytorch_utils`. This alone was fixable with a small shim (adding the function back in, since it's unrelated to the LRP computation itself -- it's only used for an unrelated head-pruning feature we don't use).

2. However, after applying that shim and successfully patching a test BERT model, a correctness check revealed the deeper problem: LRP has a fundamental mathematical property called "conservation" -- the sum of all per-token relevance scores should approximately equal the actual output value being explained. In our test, this ratio was off by roughly SIX ORDERS OF MAGNITUDE (should be ~1.0, was ~0.000001), meaning the computed relevance scores are not mathematically valid AttnLRP output, even though the code runs without crashing.

## Conclusion

This points to a deeper incompatibility between `lxt`'s patching logic and the internals of this newer `transformers` version (likely related to how attention/softmax is implemented internally now, e.g. `sdpa` vs `eager` paths, beyond just the one missing function). This is a real, unresolved upstream compatibility gap, not a bug in our own explainer code.

## Options going forward (for discussion with Nouran)

- Downgrade `transformers` specifically for AttnLRP runs (risks breaking the CUDA/GPU setup already working for the other 3 methods; would need a separate environment)

- Attempt a from-scratch, more minimal AttnLRP implementation rather than relying on the official library (significant additional time investment)

- Substitute a different method in place of AttnLRP for this project, given the compatibility gap

- Wait for `lxt` to release an update compatible with newer `transformers`