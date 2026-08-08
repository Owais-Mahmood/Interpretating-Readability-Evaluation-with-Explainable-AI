from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BitsAndBytesConfig

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

FALLBACK_THRESHOLDS = {
    "Synonymy": 0.63,
    "Modulation": 0.54,
    "Compression": 0.09,
    "Explanation": 0.40,
    "Syntactic Change": 0.20,
    "Omission": 0.32,
}

REPO_ID = "hannah-khallaf/e2r-strategy-qwen2.5-7b-pairwise-qlora"


def _download_reference_package(repo_id: str, local_root: str = "./qwen_e2r_reference") -> Path:
    root = Path(local_root).resolve()
    filenames = [
        "reference_implementation/__init__.py",
        "reference_implementation/binary_relevance.py",
        "reference_implementation/taxonomy.py",
        "e2r_taxonomy.yaml",
    ]
    for filename in filenames:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    return root


def _construct_taxonomy(taxonomy_module: Any, taxonomy_path: Path) -> Any:
    """Same multi-attempt construction logic as Nouran's notebook, since
    the exact taxonomy constructor signature isn't fixed."""
    import inspect

    taxonomy_class = taxonomy_module.E2RTaxonomy
    taxonomy_data = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))

    attempts = []
    for name in ("load_taxonomy", "load_e2r_taxonomy", "read_taxonomy"):
        loader = getattr(taxonomy_module, name, None)
        if callable(loader):
            attempts.extend([lambda loader=loader: loader(taxonomy_path), lambda loader=loader: loader(str(taxonomy_path))])
    for name in dir(taxonomy_class):
        lowered = name.lower()
        if not any(token in lowered for token in ("yaml", "file", "path", "load")):
            continue
        loader = getattr(taxonomy_class, name, None)
        if callable(loader):
            attempts.extend([lambda loader=loader: loader(taxonomy_path), lambda loader=loader: loader(str(taxonomy_path))])
    attempts.extend([
        lambda: taxonomy_class(taxonomy_path),
        lambda: taxonomy_class(str(taxonomy_path)),
        lambda: taxonomy_class(taxonomy_data),
    ])

    errors = []
    for attempt in attempts:
        try:
            taxonomy = attempt()
            taxonomy.render_macro_card(LABELS[0], include_descendants=True, include_examples=True)
            return taxonomy
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")

    signature = inspect.signature(taxonomy_class)
    raise RuntimeError(
        f"Could not construct E2RTaxonomy. Constructor signature: {signature}. "
        f"Last errors: " + " | ".join(errors[-5:])
    )


def _render_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@MODELS.register("qwen_e2r")
class QwenModelAdapter:
    """Wraps the Qwen2.5-7B pairwise QLoRA classifier. Unlike mBERT/XLM-R/E5,
    this is NOT a classification head -- for each of the 6 labels, it builds
    a taxonomy-card prompt and computes P(true) vs P(false) from the next
    token's probability distribution, following Nouran's own reference
    implementation (binary_relevance.py, downloaded directly from the model
    repo) as closely as possible.
    """

    def __init__(self, repo_id: str = REPO_ID) -> None:
        self.repo_id = repo_id
        self.model = None
        self.tokenizer = None
        self.taxonomy = None
        self.binary_relevance = None
        self.thresholds = np.array([FALLBACK_THRESHOLDS[label] for label in LABELS])

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen requires a CUDA GPU.")

        from peft import AutoPeftModelForCausalLM

        reference_root = _download_reference_package(self.repo_id)
        from reference_implementation import binary_relevance, taxonomy as taxonomy_module
        self.binary_relevance = binary_relevance

        taxonomy_path = reference_root / "e2r_taxonomy.yaml"
        self.taxonomy = _construct_taxonomy(taxonomy_module, taxonomy_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=False)

        quantisation = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.repo_id,
            quantization_config=quantisation,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        self.model.eval()

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        predictions = []
        for example in examples:
            row = {
                "source_text": example.inputs["source_text"],
                "simplified_text": example.inputs["simplified_text"],
            }
            probabilities = np.zeros(len(LABELS))
            for i, label in enumerate(LABELS):
                system_prompt, user_prompt = self.binary_relevance.build_binary_prompt(
                    row, label, self.taxonomy,
                    include_definition=True, demonstrations=(),
                    request_confidence=False, response_format="boolean", input_mode="pair",
                )
                prompt = _render_prompt(self.tokenizer, system_prompt, user_prompt)
                probabilities[i] = self.binary_relevance.boolean_token_probability(
                    self.model, self.tokenizer, prompt
                )

            predicted_flags = (probabilities >= self.thresholds).astype(int)
            predicted_labels = [LABELS[j] for j in range(len(LABELS)) if predicted_flags[j] == 1]
            predictions.append(
                Prediction(
                    example_id=example.example_id,
                    predicted_label=predicted_labels,
                    target_label=None,
                    scores=probabilities,
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
        # Not meaningful in the same batched sense as the other models,
        # since each example x label pair gets its own distinct prompt.
        raise NotImplementedError(
            "Qwen builds one prompt per (example, label) pair -- use predict() directly."
        )

    def validate(self) -> list[str]:
        errors = []
        if self.model is None:
            errors.append("Model has not been loaded yet -- call load() first.")
        return errors