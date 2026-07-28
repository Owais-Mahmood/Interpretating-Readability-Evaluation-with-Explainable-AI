from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import list_repo_files, snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai_pipeline.contracts import Example, Prediction
from xai_pipeline.registry import MODELS

LABELS = [
    "Compression",
    "Explanation",
    "Illocutionary Change",
    "Modulation",
    "Omission",
    "Synonymy",
    "Syntactic Change",
]

MAX_LENGTH = 256


@MODELS.register("mbert_e2r")
class MBERTModelAdapter:
    """Wraps the public hannah-khallaf/e2r-mbert-merged7 classifier so it
    fits the framework's ModelAdapter contract. Reuses the same loading and
    prediction logic already verified against the real model and test set."""

    def __init__(self, repo_id: str = "hannah-khallaf/e2r-mbert-merged7") -> None:
        self.repo_id = repo_id
        self.model = None
        self.tokenizer = None
        self.thresholds: np.ndarray | None = None

    def load(self) -> None:
        root = Path(
            snapshot_download(repo_id=self.repo_id, repo_type="model", token=False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(root)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            root, trust_remote_code=True, attn_implementation="eager"
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

        # Load per-label decision thresholds (dev_thresholds.json in the repo)
        import json
        thresholds_path = root / "dev_thresholds.json"
        with thresholds_path.open("r", encoding="utf-8") as handle:
            raw_thresholds = json.load(handle)
        self.thresholds = np.array([raw_thresholds[label] for label in LABELS])

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "source_text": [ex.inputs["source_text"] for ex in examples],
                "simplified_text": [ex.inputs["simplified_text"] for ex in examples],
            }
        )
        device = self._device()

        encoded = self.tokenizer(
            frame["source_text"].tolist(),
            text_pair=frame["simplified_text"].tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()

        predicted_flags = (probabilities >= self.thresholds.reshape(1, -1)).astype(int)

        predictions = []
        for i, example in enumerate(examples):
            predicted_labels = [
                LABELS[j] for j in range(len(LABELS)) if predicted_flags[i, j] == 1
            ]
            predictions.append(
                Prediction(
                    example_id=example.example_id,
                    predicted_label=predicted_labels,
                    target_label=None,
                    scores=probabilities[i],
                    metadata={"thresholds": self.thresholds.tolist()},
                )
            )
        return predictions

    def score(self, examples: Sequence[Example], targets: Sequence[Any]) -> np.ndarray:
        """One scalar probability per example, for the given target label
        (used by explainers that need a single number to attribute, e.g.
        Integrated Gradients / GradientSHAP)."""
        predictions = self.predict(examples)
        scores = []
        for prediction, target in zip(predictions, targets):
            target_index = LABELS.index(target)
            scores.append(prediction.scores[target_index])
        return np.array(scores)

    def tokenise(self, examples: Sequence[Example]) -> Any:
        return self.tokenizer(
            [ex.inputs["source_text"] for ex in examples],
            text_pair=[ex.inputs["simplified_text"] for ex in examples],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def validate(self) -> list[str]:
        errors = []
        if self.model is None:
            errors.append("Model has not been loaded yet -- call load() first.")
        return errors