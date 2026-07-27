from pathlib import Path

from xai_pipeline.config import load_pipeline_config


def test_base_config_loads() -> None:
    config = load_pipeline_config(Path("configs/base.yaml"))
    assert config.project.modality == "text"
    assert config.project.seed == 13
    assert len(config.explainers) >= 1
