from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            if key in self._items:
                raise KeyError(f"{key!r} is already registered in {self.name}")
            self._items[key] = factory
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"{key!r} is not registered in {self.name}. Available: {available}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)


DATASETS = Registry("datasets")
MODELS = Registry("models")
EXPLAINERS = Registry("explainers")
ALIGNERS = Registry("aligners")
EVALUATORS = Registry("evaluators")
REPORTERS = Registry("reporters")
