# Data contract

Each example contains a stable `example_id`, one or more `inputs`, optional
`labels`, useful `metadata`, and optional `references` such as human spans,
edits, concepts, or alignments.

Preserve raw input, normalised input, and character offsets separately. Record
excluded examples and reasons. Fingerprint the exact dataset used in each run.
