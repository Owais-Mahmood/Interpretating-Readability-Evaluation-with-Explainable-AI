from __future__ import annotations


class InferStage:
    """Placeholder for the infer stage."""

    name = "infer"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
