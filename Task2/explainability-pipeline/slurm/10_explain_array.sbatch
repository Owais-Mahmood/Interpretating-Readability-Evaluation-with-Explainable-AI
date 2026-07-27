#!/usr/bin/env bash
#SBATCH --job-name=xai-explain
#SBATCH --array=0-3%1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -euo pipefail
source .venv/bin/activate

METHODS=(raw_attention gradient_x_input integrated_gradients leave_one_out)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="${CONFIG:-configs/base.yaml}"

echo "Method: $METHOD"
# Replace with the implemented stage command, for example:
# xai-pipeline run --config "$CONFIG" --stage "explain:$METHOD"
