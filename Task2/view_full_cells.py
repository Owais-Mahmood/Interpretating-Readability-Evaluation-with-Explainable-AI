import json

with open("Test_E2R_Strategy_Models.ipynb") as f:
    nb = json.load(f)

# Print cells 7 and 9 in full (the shared utilities and encoder loading/prediction functions)
for i in [7, 9]:
    src = "".join(nb["cells"][i]["source"])
    print(f"=== Cell {i} (FULL) ===")
    print(src)
    print()