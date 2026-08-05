import json

with open("Test_E2R_Strategy_Models.ipynb") as f:
    nb = json.load(f)

src = "".join(nb["cells"][11]["source"])
print("=== Cell 11 (FULL) ===")
print(src)
print()

src = "".join(nb["cells"][12]["source"])
print("=== Cell 12 (FULL) ===")
print(src)