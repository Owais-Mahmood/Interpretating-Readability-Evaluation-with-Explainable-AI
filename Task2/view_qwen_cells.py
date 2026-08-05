import json

with open("Test_E2R_Strategy_Models.ipynb") as f:
    nb = json.load(f)

print("Total cells:", len(nb["cells"]))
print()
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "qwen" in src.lower() or "pairwise" in src.lower() or "taxonomy" in src.lower():
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        print(src[:2000])
        print()