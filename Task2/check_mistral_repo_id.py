import json

with open("use_public_mistral_merged7.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"][:3]):
    src = "".join(cell["source"])
    print(f"--- Cell {i} ---")
    print(src[:500])
    print()