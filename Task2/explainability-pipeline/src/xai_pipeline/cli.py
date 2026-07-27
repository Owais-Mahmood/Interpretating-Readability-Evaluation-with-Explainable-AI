from __future__ import annotations

import argparse
from pathlib import Path

from xai_pipeline.config import load_pipeline_config
from xai_pipeline.pipelines.run_pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General explainability pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)

    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--config", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--stage", default="all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        config = load_pipeline_config(args.config)
        print(f"Valid configuration: {config.project.name}")
        return
    if args.command == "dry-run":
        run_pipeline(Path(args.config), dry_run=True)
        return
    if args.command == "run":
        run_pipeline(Path(args.config), dry_run=False, stage=args.stage)
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
