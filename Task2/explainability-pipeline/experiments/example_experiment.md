# Experiment: baseline explainability comparison

## Objective
Compare one attention, one gradient, and one perturbation method on the same
model predictions.

## Methods
- Raw attention
- Gradient × input
- Leave-one-out

## Shared alignment
Word-level, special tokens removed, subwords summed.

## Evaluation
Comprehensiveness, sufficiency, deletion AUC, runtime, and human-edit token F1
when available.

## Statistics
Paired bootstrap confidence intervals, paired tests, effect sizes, and corrected
comparisons on per-example metrics.

## Slices
Language, label, correctness, length, domain, and strategy.
