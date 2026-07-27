from __future__ import annotations


class ExplainStage:
    """Placeholder for the explain stage."""

    name = "explain"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
