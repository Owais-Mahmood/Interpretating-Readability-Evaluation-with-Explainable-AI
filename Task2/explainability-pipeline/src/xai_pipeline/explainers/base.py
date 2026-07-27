from __future__ import annotations

from abc import ABC, abstractmethod


class ExplainerBase(ABC):
    name: str

    @abstractmethod
    def validate(self) -> list[str]:
        raise NotImplementedError
