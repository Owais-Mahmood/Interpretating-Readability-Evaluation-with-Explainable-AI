#!/usr/bin/env bash
#SBATCH --job-name=xai-evaluate
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail
source .venv/bin/activate
CONFIG="${CONFIG:-configs/base.yaml}"
# Replace after the evaluator stage is implemented:
# xai-pipeline run --config "$CONFIG" --stage evaluate
