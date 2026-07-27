from __future__ import annotations


class EvaluateStage:
    """Placeholder for the evaluate stage."""

    name = "evaluate"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
