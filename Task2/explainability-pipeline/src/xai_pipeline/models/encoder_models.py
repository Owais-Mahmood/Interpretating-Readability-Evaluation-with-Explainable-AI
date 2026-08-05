from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from xai_pipeline.contracts import Example, Prediction
from xai_pipeline.registry import MODELS

LABELS = [
    "Synonymy",
    "Modulation",
    "Compression",
    "Explanation",
    "Syntactic Change",
    "Omission",
]

MODEL_REPOSITORIES = {
    "xlmr": "hannah-khallaf/e2r-strategy-xlmr-large-focal",
    "e5": "hannah-khallaf/e2r-strategy-multilingual-e5-large-bce",
}

FALLBACK_THRESHOLDS = {
    "xlmr": {label: 0.46 for label in LABELS},
    "e5": {label: 0.23 for label in LABELS},
}

MAX_LENGTH = 512


class EncoderModelAdapter:
    """Shared loading/prediction logic for the XLM-R and Multilingual-E5
    classifiers, following Nouran's Test_E2R_Strategy_Models.ipynb.
    Both are standard AutoModelForSequenceClassification releases (no
    separate PEFT adapter step needed, unlike the old QLoRA models).
    """

    def __init__(self, model_choice: str) -> None:
        if model_choice not in MODEL_REPOSITORIES:
            raise ValueError("model_choice must be 'xlmr' or 'e5'")
        self.model_choice = model_choice
        self.repo_id = MODEL_REPOSITORIES[model_choice]
        self.model = None
        self.tokenizer = None
        self.thresholds: np.ndarray | None = None

    def load(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=True)

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device.type == "cuda":
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_id, **load_kwargs)
        self.model.to(device)
        self.model.eval()
        self._patch_get_input_embeddings()

        # Thresholds: prefer the model's own saved config, fall back to the
        # fixed values from Nouran's notebook if not present.
        config_data = getattr(self.model.config, "e2r_classifier", None)
        if isinstance(config_data, dict) and isinstance(config_data.get("thresholds"), dict):
            saved = config_data["thresholds"]
            self.thresholds = np.array([float(saved[label]) for label in LABELS])
        else:
            self.thresholds = np.array(
                [FALLBACK_THRESHOLDS[self.model_choice][label] for label in LABELS]
            )

    def _patch_get_input_embeddings(self) -> None:
        """Some custom E2R wrapper classes (e.g. the XLM-R one) don't
        implement get_input_embeddings(), which Integrated Gradients and
        GradientSHAP both need. Try the real method first; if it's broken,
        find the word/token embedding layer by name and patch it in
        directly on this model instance."""
        try:
            self.model.get_input_embeddings()
            return  # already works fine, nothing to do
        except NotImplementedError:
            pass

        for name, module in self.model.named_modules():
            if name.endswith("word_embeddings"):
                self.model.get_input_embeddings = lambda m=module: m
                return

        raise RuntimeError(
            f"Could not find a word embeddings layer to patch get_input_embeddings "
            f"with, for model_choice={self.model_choice!r}."
        )

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        device = self._device()
        all_probabilities = []

        for example in examples:
            source = example.inputs["source_text"]
            simplified = example.inputs["simplified_text"]
            if self.model_choice == "e5":
                first_sequence = "query: " + source
                second_sequence = "query: " + simplified
            else:
                first_sequence = source
                second_sequence = simplified

            encoded = self.tokenizer(
                first_sequence,
                second_sequence,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**encoded).logits[0]
                probabilities = torch.sigmoid(logits).float().cpu().numpy()
            all_probabilities.append(probabilities)

        probabilities = np.array(all_probabilities)
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
        predictions = self.predict(examples)
        scores = []
        for prediction, target in zip(predictions, targets):
            target_index = LABELS.index(target)
            scores.append(prediction.scores[target_index])
        return np.array(scores)

    def tokenise(self, examples: Sequence[Example]) -> Any:
        first = []
        second = []
        for example in examples:
            source = example.inputs["source_text"]
            simplified = example.inputs["simplified_text"]
            if self.model_choice == "e5":
                first.append("query: " + source)
                second.append("query: " + simplified)
            else:
                first.append(source)
                second.append(simplified)
        return self.tokenizer(first, second, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True)

    def validate(self) -> list[str]:
        errors = []
        if self.model is None:
            errors.append("Model has not been loaded yet -- call load() first.")
        return errors


@MODELS.register("xlmr_e2r")
class XLMRModelAdapter(EncoderModelAdapter):
    def __init__(self) -> None:
        super().__init__("xlmr")


@MODELS.register("e5_e2r")
class E5ModelAdapter(EncoderModelAdapter):
    def __init__(self) -> None:
        super().__init__("e5")