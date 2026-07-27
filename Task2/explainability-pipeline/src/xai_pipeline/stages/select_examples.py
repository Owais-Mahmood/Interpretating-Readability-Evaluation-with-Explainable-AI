from __future__ import annotations


class SelectExamplesStage:
    """Placeholder for the select examples stage."""

    name = "select_examples"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
