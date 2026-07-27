#!/usr/bin/env bash
#SBATCH --job-name=xai-validate
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
source .venv/bin/activate
python scripts/validate_config.py --config "${CONFIG:-configs/base.yaml}"
python scripts/run_pipeline.py --config "${CONFIG:-configs/base.yaml}" --dry-run
