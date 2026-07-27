# Output schemas

## Predictions

`run_id, example_id, split, language, gold_label, predicted_label, target_label, confidence, score_vector, correct, model_id`

## Explanations

`run_id, example_id, method, target_label, unit_index, unit_text, character_start, character_end, raw_score, aligned_score, rank, signed, runtime_seconds, status, error_message`

## Metrics

`run_id, example_id, method, metric, value, slice_name, slice_value, valid, missing_reason`

Use Parquet for large tables and JSON for nested metadata.
