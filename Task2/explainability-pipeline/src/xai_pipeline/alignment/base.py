from __future__ import annotations

from abc import ABC, abstractmethod


class AlignmentBase(ABC):
    @abstractmethod
    def validate(self) -> list[str]:
        raise NotImplementedError
