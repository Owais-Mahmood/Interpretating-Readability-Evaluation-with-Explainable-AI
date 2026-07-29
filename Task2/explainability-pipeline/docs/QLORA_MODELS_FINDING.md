# DeepSeek & Mistral: Missing Classification Head (Both Blocked)

## Summary

Both QLoRA classifiers could NOT be used, for a genuine, verified, and SYSTEMATIC reason affecting both repos: the published adapters do not contain the trained classification head weights.

- `hannah-khallaf/e2r-deepseek-r1-qwen-7b-qlora-merged7`
- `hannah-khallaf/e2r-mistral-7b-qlora-merged7`

## What was found

Both repos show the identical pattern:

| | DeepSeek | Mistral |
|---|---|---|
| `adapter_config.json` task_type | `CAUSAL_LM` (should be `SEQ_CLS`) | `CAUSAL_LM` (should be `SEQ_CLS`) |
| `modules_to_save` | `null` | `null` |
| Total safetensors keys | 392 | 448 |
| Keys matching "score"/"classif" | 0 | 0 |

After fixing the `task_type` metadata bug for DeepSeek (a quick config edit, same pattern as the mBERT repo's `problem_type` bug), loading still failed with `KeyError: 'base_model.model.score.weight'`. Direct inspection of both adapters' `adapter_model.safetensors` files confirms this is not a loading-code problem: neither file contains any classification-head weights at all -- only LoRA adapter weights for the attention/MLP layers.

## Why this matters

Without the real trained classification head, `score.weight` is randomly initialised. Any predictions produced would be meaningless noise, not a reflection of anything the model actually learned. This is exactly the scenario both notebooks (`use_public_deepseek_merged7.ipynb` and `use_public_mistral_merged7.ipynb`) explicitly anticipate and guard against in their introductions: *"The notebook stops rather than returning unreliable predictions when the trained classification head cannot be loaded."*

## This is not a code bug on our side

Verified directly at the file level (safetensors key inspection) for BOTH models, not just inferred from an error message.

## Recommendation

This needs Nouran's input before either model can be used:

- Likely cause: when both adapters were saved, `modules_to_save` should have included `["score"]` (the newly added classification head), but didn't for either export -- a systematic mistake across both QLoRA exports, not a one-off.

- Both repos would need to be re-exported with the classification head weights included, or the trained head weights provided separately.

- Since this affects both QLoRA models identically, it's likely the exact same training/export script was used for both, and the fix would need to happen once for the shared pipeline, not twice separately.