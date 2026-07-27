# Data layout

Do not commit restricted or personally identifiable data.

- `raw/`: immutable source files.
- `interim/`: partially processed data.
- `processed/`: validated model-ready data.
- `reference/`: human rationales, edits, concepts, or alignments.
- `audit/`: excluded examples and quality-control records.

Every processed dataset should have a data card and fingerprint.
