from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

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

MAX_LENGTH = 2048


def _find_adapter_dir(root: Path) -> Path:
    matches = sorted(root.rglob("adapter_config.json"))
    if not matches:
        raise FileNotFoundError(f"No adapter_config.json found under {root}")
    return matches[0].parent


def _make_quantization_config() -> BitsAndBytesConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 7B QLoRA checkpoint.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


@MODELS.register("deepseek_e2r")
class DeepSeekModelAdapter:
    """Wraps the public hannah-khallaf/e2r-deepseek-r1-qwen-7b-qlora-merged7
    classifier: a 7B causal LM base model with a QLoRA adapter and a
    multi-label classification head, following the same loading pattern
    as Nouran's use_public_deepseek_merged7.ipynb notebook.
    """

    def __init__(
        self,
        repo_id: str = "hannah-khallaf/e2r-deepseek-r1-qwen-7b-qlora-merged7",
        base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ) -> None:
        self.repo_id = repo_id
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.thresholds: np.ndarray | None = None

    def load(self) -> None:
        from peft import PeftModel

        root = Path(snapshot_download(repo_id=self.repo_id, repo_type="model", token=False))
        adapter_dir = _find_adapter_dir(root)

        with (adapter_dir / "adapter_config.json").open("r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_name = adapter_config.get("base_model_name_or_path", self.base_model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_name
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad or EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # QLoRA/causal-LM models conventionally pad on the left for
        # classification, matching the pattern from Nouran's notebook.
        self.tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(
            base_name,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={label: i for i, label in enumerate(LABELS)},
            problem_type="multi_label_classification",
        )

        base = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            config=config,
            quantization_config=_make_quantization_config(),
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

        # Per-label decision thresholds, same file convention as mBERT
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

        # QLoRA models are much heavier per example than mBERT -- process
        # one at a time rather than batching, to keep GPU memory safe.
        all_probabilities = []
        for _, row in frame.iterrows():
            encoded = self.tokenizer(
                row["source_text"],
                text_pair=row["simplified_text"],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            all_probabilities.append(probabilities[0])

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