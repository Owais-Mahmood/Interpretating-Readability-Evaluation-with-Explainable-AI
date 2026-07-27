from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetBase(ABC):
    @abstractmethod
    def validate(self) -> list[str]:
        """Return validation errors, or an empty list when valid."""
        raise NotImplementedError
