import json

with open("use_public_deepseek_merged7.ipynb") as f:
    nb = json.load(f)

print("Total cells:", len(nb["cells"]))
for i, cell in enumerate(nb["cells"][:8]):
    src = "".join(cell["source"])
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print(src[:1200])
    print()