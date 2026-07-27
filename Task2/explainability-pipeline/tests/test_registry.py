from xai_pipeline.registry import Registry


def test_registry_round_trip() -> None:
    registry = Registry("test")

    @registry.register("component")
    def component() -> str:
        return "ok"

    assert registry.get("component")() == "ok"
    assert registry.names() == ["component"]
