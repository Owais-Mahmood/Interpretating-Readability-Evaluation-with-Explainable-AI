import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xai_pipeline.models.encoder_models import XLMRModelAdapter

model = XLMRModelAdapter()
model.load()

embeddings = model.model.get_input_embeddings()
print("get_input_embeddings() works:", embeddings)
print("Embedding shape:", embeddings.weight.shape)