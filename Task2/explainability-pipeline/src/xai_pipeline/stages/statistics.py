from __future__ import annotations


class StatisticsStage:
    """Placeholder for the statistics stage."""

    name = "statistics"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
