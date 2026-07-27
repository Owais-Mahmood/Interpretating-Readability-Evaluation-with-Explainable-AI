from __future__ import annotations


class ReportStage:
    """Placeholder for the report stage."""

    name = "report"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
