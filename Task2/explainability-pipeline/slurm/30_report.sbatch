#!/usr/bin/env bash
#SBATCH --job-name=xai-report
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
source .venv/bin/activate
CONFIG="${CONFIG:-configs/base.yaml}"
# Replace after the reporter stage is implemented:
# xai-pipeline run --config "$CONFIG" --stage report
