from __future__ import annotations

from pathlib import Path

from xai_pipeline.contracts import Example


class ExampleDatasetAdapter:
    """Replace with a project-specific dataset adapter."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, split: str) -> list[Example]:
        raise NotImplementedError("Parse the project dataset into Example objects.")

    def validate(self) -> list[str]:
        return [] if self.path.exists() else [f"Dataset file does not exist: {self.path}"]

    def fingerprint(self) -> str:
        raise NotImplementedError("Hash stable identifiers and relevant dataset content.")
