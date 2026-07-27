from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Example:
    example_id: str
    inputs: Mapping[str, Any]
    labels: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    references: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    example_id: str
    predicted_label: Any
    target_label: Any
    scores: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Explanation:
    example_id: str
    method: str
    target: Any
    units: Sequence[str]
    scores: np.ndarray
    unit_type: str
    signed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricRecord:
    run_id: str
    example_id: str
    method: str
    metric: str
    value: float
    slice_name: str = "all"
    slice_value: str = "all"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    def load(self, split: str) -> list[Example]: ...
    def validate(self) -> list[str]: ...
    def fingerprint(self) -> str: ...


class ModelAdapter(Protocol):
    def load(self) -> None: ...
    def predict(self, examples: Sequence[Example]) -> list[Prediction]: ...
    def score(self, examples: Sequence[Example], targets: Sequence[Any]) -> np.ndarray: ...
    def tokenise(self, examples: Sequence[Example]) -> Any: ...


class Explainer(Protocol):
    name: str

    def explain(
        self,
        examples: Sequence[Example],
        model: ModelAdapter,
        predictions: Sequence[Prediction],
    ) -> list[Explanation]: ...


class AlignmentAdapter(Protocol):
    def align(self, explanation: Explanation, example: Example) -> Explanation: ...


class Evaluator(Protocol):
    name: str

    def evaluate(
        self,
        examples: Sequence[Example],
        predictions: Sequence[Prediction],
        explanations: Sequence[Explanation],
    ) -> pd.DataFrame: ...
