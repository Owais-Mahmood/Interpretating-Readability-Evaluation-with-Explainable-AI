# Adding a component

## Dataset
Implement the adapter, return `Example` objects, register it, add YAML, and add
validation tests.

## Model
Implement the model contract, define one target score, preserve offsets, declare
capabilities, and verify prediction equivalence.

## Explainer
Implement the contract, declare requirements, save raw scores, add alignment,
faithfulness evaluation, tests, and a method card.

## Evaluator
Return one row per example, method, and metric. Do not hide missing cases or
aggregate inside the core evaluator.
