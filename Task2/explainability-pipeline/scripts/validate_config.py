from __future__ import annotations

import argparse

from xai_pipeline.config import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_pipeline_config(args.config)
    print(f"Configuration is valid: {config.project.name}")
    print(f"Configured explainers: {[item['name'] for item in config.explainers]}")


if __name__ == "__main__":
    main()
