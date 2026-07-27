# Pipeline specification

| Stage | Input | Output |
|---|---|---|
| Validate | Configuration and environment | Validation report |
| Ingest | Dataset files | Normalised examples and fingerprint |
| Model | Checkpoint and configuration | Evaluation-mode adapter |
| Predict | Examples and model | Predictions and target scores |
| Select | Predictions and metadata | Frozen explanation cohort |
| Explain | Cohort, target and model | Raw explanations |
| Align | Raw explanations and offsets | Shared-unit explanations |
| Evaluate | Explanations and references | Per-example metrics |
| Statistics | Paired metric records | Intervals, tests, effects and groups |
| Report | All run artefacts | Tables, figures and error report |

The frozen selection stage ensures every method receives the same examples.
