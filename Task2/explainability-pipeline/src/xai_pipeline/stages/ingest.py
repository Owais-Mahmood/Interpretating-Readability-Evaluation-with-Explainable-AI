from __future__ import annotations


class IngestStage:
    """Placeholder for the ingest stage."""

    name = "ingest"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
