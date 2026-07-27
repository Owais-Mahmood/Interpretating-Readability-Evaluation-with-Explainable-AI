from __future__ import annotations


class ValidateStage:
    """Placeholder for the validate stage."""

    name = "validate"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
