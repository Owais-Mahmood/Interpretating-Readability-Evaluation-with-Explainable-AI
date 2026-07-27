from __future__ import annotations


class LoadModelStage:
    """Placeholder for the load model stage."""

    name = "load_model"

    def run(self, context: dict) -> dict:
        raise NotImplementedError(
            "Implement this stage after its input and output contract is agreed."
        )
