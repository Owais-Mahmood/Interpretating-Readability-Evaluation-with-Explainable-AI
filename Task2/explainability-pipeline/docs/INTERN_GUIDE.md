# Intern implementation guide

## Goal

Build one reusable explainability pipeline rather than disconnected notebooks.

## Phase 0: define the experiment

Record the task, label space, analysis unit, data splits, explained model output,
reference evidence, and planned statistical comparison. Distinguish
faithfulness from plausibility before implementing a method.

## Phase 1: data contract

Deliver a dataset adapter, stable identifiers, validation report, leakage checks,
distributions, and explicit treatment of excluded or audit-only examples.

## Phase 2: model contract

Deliver a model adapter, reproducible predictions, explicit scalar target score,
token offsets, and slice-level model performance.

## Phase 3: baseline explainers

Start with at least two method families:

- raw attention: low-cost diagnostic baseline;
- gradient × input: local gradient baseline;
- integrated gradients: path-based gradient method;
- leave-one-out or occlusion: perturbation baseline.

Save the target, units, raw and aligned scores, sign, runtime, configuration,
status, and failure reason for every explanation.

## Phase 4: alignment

For NLP:

```text
model tokens -> subwords -> words -> human spans or edit spans
```

For sentence pairs:

```text
source tokens -> source words
 target tokens -> target words
 source-target word alignment
 edit or strategy spans
```

Never compare human word spans directly with raw subword scores.

## Phase 5: evaluation order

1. Model correctness and confidence.
2. Sanity checks.
3. Faithfulness.
4. Plausibility.
5. Stability.
6. Efficiency.
7. Class specificity and method agreement.
8. Language, label, domain, and strategy slices.

A plausible explanation can still be unfaithful.

## Phase 6: statistics

Use paired examples, bootstrap confidence intervals, effect sizes, paired tests,
and multiple-comparison correction. Aggregate by example before testing.

## Phase 7: error analysis

Inspect high and low faithfulness, high-plausibility/low-faithfulness cases,
method disagreement, correct and incorrect predictions, languages, labels,
lengths, and strategy combinations.

## Definition of done for one method

- implementation adapter;
- YAML configuration;
- unit and smoke tests;
- raw output schema;
- alignment support;
- at least one faithfulness metric;
- runtime and failure handling;
- method card;
- one worked example.
