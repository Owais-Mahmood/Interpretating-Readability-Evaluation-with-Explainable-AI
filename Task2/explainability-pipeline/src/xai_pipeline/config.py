from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    modality: str
    task_type: str
    languages: list[str] = Field(default_factory=list)
    seed: int = 13


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_root: Path
    output_root: Path
    cache_root: Path


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig
    paths: PathsConfig
    dataset: dict[str, Any]
    model: dict[str, Any]
    selection: dict[str, Any]
    explainers: list[dict[str, Any]]
    alignment: dict[str, Any]
    evaluation: dict[str, Any]
    reporting: dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return data


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    return PipelineConfig.model_validate(load_yaml(path))
