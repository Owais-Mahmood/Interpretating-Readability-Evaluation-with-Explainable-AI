from __future__ import annotations


class AlignStage:
    """Placeholder for the align stage."""

    name = "align"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
