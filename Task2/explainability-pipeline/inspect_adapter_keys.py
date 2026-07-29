from safetensors import safe_open

path = "/users/jpdj5670/.cache/huggingface/hub/models--hannah-khallaf--e2r-deepseek-r1-qwen-7b-qlora-merged7/snapshots/c15589da91b5a277ea9093498e06dde5056a5366/adapter_model.safetensors"

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())

print(f"Total keys: {len(keys)}")
print()
print("Keys containing 'score' or 'classif' (looking for the classification head):")
for k in keys:
    if "score" in k.lower() or "classif" in k.lower():
        print(" ", k)

print()
print("First 10 keys overall (to see the naming pattern):")
for k in keys[:10]:
    print(" ", k)

print()
print("Last 10 keys overall:")
for k in keys[-10:]:
    print(" ", k)