import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xai_pipeline.models.encoder_models import XLMRModelAdapter

model = XLMRModelAdapter()
model.load()

print("Top-level attributes of model.model:")
for name, child in model.model.named_children():
    print(f"  {name}: {type(child).__name__}")

print()
print("Looking for anything named 'embed' at any depth:")
for name, module in model.model.named_modules():
    if "embed" in name.lower():
        print(f"  {name}: {type(module).__name__}")