from __future__ import annotations

import argparse
from pathlib import Path

from xai_pipeline.pipelines.run_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(Path(args.config), dry_run=args.dry_run, stage=args.stage)


if __name__ == "__main__":
    main()
