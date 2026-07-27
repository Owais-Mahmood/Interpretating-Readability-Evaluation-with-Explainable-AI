# Model contract

A model adapter provides `load`, `predict`, `score`, and `tokenise`.

The explained scalar must be explicit: a class logit, probability, margin,
multilabel logit, sequence score, or token-level score. Never compare methods
that explain different target functions.
