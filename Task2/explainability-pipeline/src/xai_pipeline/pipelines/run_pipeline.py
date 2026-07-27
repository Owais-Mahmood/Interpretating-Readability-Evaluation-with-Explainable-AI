from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from xai_pipeline.config import load_pipeline_config
from xai_pipeline.logging_utils import configure_logging

STAGES = [
    "validate", "ingest", "model", "predict", "select", "explain",
    "align", "evaluate", "statistics", "report",
]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest(config_path: Path, config: Any, run_id: str, dry_run: bool) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "config_path": str(config_path),
        "project": config.project.model_dump(mode="json"),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "stages": STAGES,
        "status": "validated" if dry_run else "not_implemented",
    }


def run_pipeline(config_path: Path, dry_run: bool = False, stage: str = "all") -> Path:
    config = load_pipeline_config(config_path)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    output_root = Path(config.paths.output_root)
    configure_logging(output_root / "logs" / f"{run_id}.log")

    manifest = _manifest(config_path, config, run_id, dry_run)
    manifest["requested_stage"] = stage
    manifest_path = output_root / "manifests" / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if dry_run:
        print(f"Dry run complete. Manifest: {manifest_path}")
        return manifest_path

    raise NotImplementedError(
        "Implement the registered dataset, model, explainer, alignment, evaluator, "
        "and reporter components before a full run."
    )
