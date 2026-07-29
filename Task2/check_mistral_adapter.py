import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "hannah-khallaf/e2r-mistral-7b-qlora-merged7"

root = Path(snapshot_download(repo_id=REPO_ID, repo_type="model", token=False))
print("Downloaded to:", root)
print()

adapter_config_path = next(root.rglob("adapter_config.json"))
with adapter_config_path.open() as f:
    config = json.load(f)
print("task_type:", config.get("task_type"))
print("modules_to_save:", config.get("modules_to_save"))
print()

from safetensors import safe_open

adapter_weights_path = next(root.rglob("adapter_model.safetensors"))
with safe_open(adapter_weights_path, framework="pt") as f:
    keys = list(f.keys())

print(f"Total keys: {len(keys)}")
score_keys = [k for k in keys if "score" in k.lower() or "classif" in k.lower()]
print("Keys containing 'score' or 'classif':", score_keys if score_keys else "NONE FOUND")