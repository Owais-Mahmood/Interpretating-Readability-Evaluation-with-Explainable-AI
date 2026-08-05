"""
Diagnostic script: runs Nouran's own Qwen loading + prediction code as
closely as possible to the original notebook, to identify exactly where
it fails. Run from Task2/ (same folder as Test_E2R_Strategy_Models.ipynb):

    python3 diagnose_qwen.py
"""

from __future__ import annotations

import gc
import importlib
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BitsAndBytesConfig

LABELS = [
    "Synonymy",
    "Modulation",
    "Compression",
    "Explanation",
    "Syntactic Change",
    "Omission",
]

LABEL_CODES = {
    "Synonymy": "SYN",
    "Modulation": "MOD",
    "Compression": "COMP",
    "Explanation": "EXPL",
    "Syntactic Change": "SYNT",
    "Omission": "OMIT",
}

MODEL_REPOSITORIES = {
    "qwen": "hannah-khallaf/e2r-strategy-qwen2.5-7b-pairwise-qlora",
}

FALLBACK_THRESHOLDS = {
    "qwen": {
        "Synonymy": 0.63,
        "Modulation": 0.54,
        "Compression": 0.09,
        "Explanation": 0.40,
        "Syntactic Change": 0.20,
        "Omission": 0.32,
    },
}


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def download_qwen_reference_package(repo_id: str, local_root: str = "./qwen_e2r_reference") -> Path:
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


def construct_taxonomy(taxonomy_module: Any, taxonomy_path: Path) -> Any:
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


def load_qwen_model_and_helpers():
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen requires a CUDA GPU.")

    from peft import AutoPeftModelForCausalLM

    repo_id = MODEL_REPOSITORIES["qwen"]

    print("STEP 1: downloading reference implementation package...")
    reference_root = download_qwen_reference_package(repo_id)
    print("STEP 1 OK:", reference_root)

    print("STEP 2: importing reference_implementation modules...")
    from reference_implementation import binary_relevance, taxonomy as taxonomy_module
    print("STEP 2 OK")

    print("STEP 3: constructing taxonomy from YAML...")
    taxonomy_path = reference_root / "e2r_taxonomy.yaml"
    taxonomy = construct_taxonomy(taxonomy_module, taxonomy_path)
    print("STEP 3 OK")

    print("STEP 4: loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False)
    print("STEP 4 OK")

    print("STEP 5: loading Qwen model (4-bit QLoRA, this downloads several GB)...")
    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=quantisation,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.eval()
    print("STEP 5 OK")

    return model, tokenizer, taxonomy, binary_relevance


def render_qwen_prompt(tokenizer, system_prompt, user_prompt) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    standard_sentence = "The committee postponed the implementation of the measure."
    easy_to_read_sentence = "The committee decided to use the measure later."

    try:
        model, tokenizer, taxonomy, binary_relevance = load_qwen_model_and_helpers()
    except Exception:
        print()
        print("=" * 60)
        print("FAILED during model/helper loading. Full traceback:")
        print("=" * 60)
        traceback.print_exc()
        return

    print()
    print("Model and helpers loaded successfully. Trying one label prediction...")

    row = {"source_text": standard_sentence, "simplified_text": easy_to_read_sentence}

    all_probabilities = {}
    for label in LABELS:
        try:
            print(f"Testing label '{label}'...")
            system_prompt, user_prompt = binary_relevance.build_binary_prompt(
                row, label, taxonomy,
                include_definition=True, demonstrations=(),
                request_confidence=False, response_format="boolean", input_mode="pair",
            )
            prompt = render_qwen_prompt(tokenizer, system_prompt, user_prompt)
            probability = binary_relevance.boolean_token_probability(model, tokenizer, prompt)
            all_probabilities[label] = float(probability)
            print(f"  {label}: {probability:.6f}")
        except Exception:
            print(f"  FAILED on label '{label}':")
            traceback.print_exc()

    print()
    print("=== All label probabilities ===")
    for label, prob in all_probabilities.items():
        threshold = FALLBACK_THRESHOLDS["qwen"][label]
        predicted = "PREDICTED" if prob >= threshold else ""
        print(f"  {label}: {prob:.4f} (threshold {threshold}) {predicted}")


if __name__ == "__main__":
    main()