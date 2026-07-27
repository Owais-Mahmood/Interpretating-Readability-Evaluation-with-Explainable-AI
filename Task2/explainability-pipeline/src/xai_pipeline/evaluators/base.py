from __future__ import annotations

from abc import ABC, abstractmethod


class EvaluatorBase(ABC):
    @abstractmethod
    def validate(self) -> list[str]:
        raise NotImplementedError
