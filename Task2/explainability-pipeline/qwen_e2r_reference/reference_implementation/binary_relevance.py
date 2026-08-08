from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .taxonomy import E2RTaxonomy


DEFAULT_CONFUSABLE_LABELS: Mapping[str, tuple[str, ...]] = {
    "Compression": ("Omission", "Syntactic Change", "Explanation"),
    "Omission": ("Compression", "Syntactic Change"),
    "Synonymy": ("Modulation",),
    "Modulation": ("Synonymy", "Syntactic Change"),
    "Syntactic Change": ("Compression", "Omission", "Explanation", "Modulation"),
    "Explanation": ("Compression", "Syntactic Change", "Omission"),
}


@dataclass(frozen=True)
class BinaryDecision:
    present: bool
    confidence: float
    evidence: str
    valid: bool
    raw_value: str = ""


def parse_binary_response(text: object) -> BinaryDecision:
    raw = str(text or "").strip()
    payload: Any = None
    candidates = [raw]
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            payload = value
            break
    if payload is None:
        lowered = raw.casefold()
        if re.search(r"\b(true|yes|present)\b", lowered):
            return BinaryDecision(True, 0.5, raw, False, raw)
        if re.search(r"\b(false|no|absent)\b", lowered):
            return BinaryDecision(False, 0.5, raw, False, raw)
        return BinaryDecision(False, 0.0, raw, False, raw)
    value = payload.get("present")
    valid = isinstance(value, bool)
    if isinstance(value, str):
        norm = value.strip().casefold()
        if norm in {"true", "yes", "present", "1"}:
            value = True
        elif norm in {"false", "no", "absent", "0"}:
            value = False
        else:
            value = False
        valid = False
    present = bool(value)
    confidence_raw = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
        valid = False
    confidence = float(np.clip(confidence, 0.0, 1.0))
    evidence = str(payload.get("evidence", "")).strip()
    probability = confidence if present else 1.0 - confidence
    return BinaryDecision(present, probability, evidence, valid, str(value))


def build_binary_prompt(
    row: Mapping[str, Any],
    label: str,
    taxonomy: E2RTaxonomy,
    *,
    include_definition: bool = True,
    demonstrations: Sequence[str] = (),
    request_confidence: bool = True,
    response_format: str = "json",
    input_mode: str = "pair",
) -> tuple[str, str]:
    system = (
        "You are a careful binary classifier for Easy-to-Read transformation strategies. "
        "Decide only whether the named strategy is present. Other strategies may also be present."
    )
    definition = ""
    if include_definition:
        definition = "\n\nOFFICIAL TAXONOMY CARD:\n" + taxonomy.render_macro_card(
            label, include_descendants=True, include_examples=True
        )
    demo_text = ""
    if demonstrations:
        demo_text = "\n\nTRAINING EXAMPLES:\n" + "\n\n".join(demonstrations)
    confidence_field = ',"confidence":0.0' if request_confidence else ""
    if response_format == "boolean":
        output_instruction = (
            "Answer with exactly one lowercase token: true or false. "
            "Use false when the evidence is insufficient."
        )
    elif response_format == "json":
        output_instruction = (
            "Return exactly one JSON object: "
            f'{{"present":true{confidence_field},"evidence":"brief text-grounded reason"}}. '
            "Use false when the evidence is insufficient."
        )
    else:
        raise ValueError("response_format must be json or boolean")
    mode = str(input_mode).lower()
    if mode == "source_only":
        task_input = f"STANDARD SENTENCE:\n{row['source_text']}"
    elif mode == "pair":
        task_input = (
            f"STANDARD SENTENCE:\n{row['source_text']}\n\n"
            f"EASY-TO-READ SENTENCE:\n{row['simplified_text']}"
        )
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    user = (
        f"Candidate strategy: {label}{definition}{demo_text}\n\n"
        f"{task_input}\n\n" + output_instruction
    )
    return system, user


def format_binary_demonstration(
    row: Mapping[str, Any], label: str, input_mode: str = "pair"
) -> str:
    gold = bool(int(row[label]))
    if str(input_mode).lower() == "source_only":
        text = f"STANDARD SENTENCE: {row['source_text']}"
    else:
        text = (
            f"STANDARD SENTENCE: {row['source_text']}\n"
            f"EASY-TO-READ SENTENCE: {row['simplified_text']}"
        )
    return f"{text}\nANSWER: {json.dumps({'present': gold}, separators=(',', ':'))}"


def select_binary_demonstrations(
    train_frame: pd.DataFrame,
    label: str,
    *,
    n_positive: int = 1,
    n_negative: int = 1,
    seed: int = 2026,
    excluded_group: str | None = None,
    input_mode: str = "pair",
) -> tuple[list[str], list[str]]:
    frame = train_frame.copy()
    if excluded_group is not None:
        frame = frame[frame["group_id"].astype(str) != str(excluded_group)]
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for value, count in ((1, n_positive), (0, n_negative)):
        candidates = frame[frame[label].astype(int) == value]
        if candidates.empty or count <= 0:
            continue
        indices = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        selected.append(candidates.iloc[np.sort(indices)])
    if not selected:
        return [], []
    examples = pd.concat(selected, ignore_index=True)
    return (
        [format_binary_demonstration(row, label, input_mode) for _, row in examples.iterrows()],
        examples["pair_id"].astype(str).tolist(),
    )


def build_pairwise_training_frame(
    frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    negative_strategy: str = "all",
    negative_ratio: float = 1.0,
    seed: int = 42,
    confusable: Mapping[str, Sequence[str]] = DEFAULT_CONFUSABLE_LABELS,
) -> pd.DataFrame:
    """Expand sentence pairs into one row per candidate label.

    Positive rows are always retained. Negative sampling is performed only on
    the supplied frame, which must be the locked training split.
    """

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        active = {label for label in labels if int(row[label]) == 1}
        for label in labels:
            rows.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "group_id": str(row["group_id"]),
                    "language": str(row["language"]),
                    "source_text": str(row["source_text"]),
                    "simplified_text": str(row["simplified_text"]),
                    "candidate_label": label,
                    "target": int(label in active),
                    "active_labels": json.dumps(sorted(active), ensure_ascii=False),
                }
            )
    table = pd.DataFrame(rows)
    strategy = str(negative_strategy).lower()
    if strategy == "all":
        return table.reset_index(drop=True)
    positives = table[table["target"] == 1]
    negatives = table[table["target"] == 0]
    rng = random.Random(seed)
    selected_negative_indices: set[int] = set()
    if strategy == "balanced":
        for label in labels:
            pos_count = int((positives["candidate_label"] == label).sum())
            candidates = negatives[negatives["candidate_label"] == label].index.tolist()
            rng.shuffle(candidates)
            keep = min(len(candidates), max(1, int(math_ceil(pos_count * negative_ratio))))
            selected_negative_indices.update(candidates[:keep])
    elif strategy in {"hard", "confusable"}:
        for index, row in negatives.iterrows():
            active = set(json.loads(row["active_labels"]))
            confusing = set(confusable.get(str(row["candidate_label"]), ()))
            if active & confusing:
                selected_negative_indices.add(int(index))
        # Retain a small random background set so every label has ordinary negatives.
        for label in labels:
            candidates = [
                int(index)
                for index in negatives[negatives["candidate_label"] == label].index
                if int(index) not in selected_negative_indices
            ]
            rng.shuffle(candidates)
            background = max(1, int(len(candidates) * min(max(negative_ratio, 0.0), 1.0) * 0.1))
            selected_negative_indices.update(candidates[:background])
    else:
        raise ValueError("negative_strategy must be all, balanced, hard, or confusable")
    sampled = pd.concat([positives, table.loc[sorted(selected_negative_indices)]], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def math_ceil(value: float) -> int:
    return int(np.ceil(float(value)))


def aggregate_pairwise_rows(
    pairwise: pd.DataFrame,
    pair_order: Sequence[str],
    labels: Sequence[str],
    *,
    probability_column: str = "probability",
) -> np.ndarray:
    required = {"pair_id", "candidate_label", probability_column}
    missing = required - set(pairwise.columns)
    if missing:
        raise ValueError(f"Pairwise table is missing columns: {sorted(missing)}")
    duplicated = pairwise.duplicated(["pair_id", "candidate_label"])
    if duplicated.any():
        raise ValueError("Pairwise predictions contain duplicate pair/label decisions.")
    lookup = {
        (str(row["pair_id"]), str(row["candidate_label"])): float(row[probability_column])
        for _, row in pairwise.iterrows()
    }
    matrix = np.zeros((len(pair_order), len(labels)), dtype=float)
    for i, pair_id in enumerate(pair_order):
        for j, label in enumerate(labels):
            key = (str(pair_id), str(label))
            if key not in lookup:
                raise ValueError(f"Missing pairwise decision for pair={pair_id}, label={label}")
            matrix[i, j] = np.clip(lookup[key], 0.0, 1.0)
    return matrix


def validate_pairwise_demonstrations(
    query: Mapping[str, Any],
    demonstration_ids: Iterable[str],
    canonical: pd.DataFrame,
    train_ids: set[str],
) -> None:
    indexed = canonical.assign(pair_id=canonical["pair_id"].astype(str)).set_index("pair_id")
    query_group = str(query["group_id"])
    for pair_id in demonstration_ids:
        pair_id = str(pair_id)
        if pair_id not in train_ids:
            raise ValueError(f"Pairwise demonstration is not in training split: {pair_id}")
        if pair_id not in indexed.index:
            raise KeyError(f"Unknown demonstration pair_id: {pair_id}")
        if str(indexed.loc[pair_id, "group_id"]) == query_group:
            raise ValueError(f"Pairwise demonstration leaks query group: {pair_id}")


def boolean_token_probability(model: Any, tokenizer: Any, prompt: str) -> float:
    """Return P(true) from the next-token distribution.

    The function supports tokenizers where true/false have several spellings by
    summing the probability of unique first tokens for leading-space and plain variants.
    """

    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    true_ids = {
        int(tokenizer.encode(value, add_special_tokens=False)[0])
        for value in ("true", " true", "True", " True")
        if tokenizer.encode(value, add_special_tokens=False)
    }
    false_ids = {
        int(tokenizer.encode(value, add_special_tokens=False)[0])
        for value in ("false", " false", "False", " False")
        if tokenizer.encode(value, add_special_tokens=False)
    }
    if not true_ids or not false_ids:
        raise ValueError("Tokenizer does not provide true/false tokens.")
    with torch.no_grad():
        logits = model(**encoded).logits[0, -1]
    selected = torch.tensor(sorted(true_ids | false_ids), device=logits.device)
    probs = torch.softmax(logits[selected], dim=0)
    token_to_prob = {int(token): float(prob) for token, prob in zip(selected, probs)}
    true_prob = sum(token_to_prob[token] for token in true_ids)
    false_prob = sum(token_to_prob[token] for token in false_ids)
    return float(true_prob / max(true_prob + false_prob, 1e-12))
