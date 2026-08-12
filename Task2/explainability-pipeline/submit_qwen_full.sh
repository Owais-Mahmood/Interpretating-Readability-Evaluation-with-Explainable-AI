#!/bin/bash
#SBATCH --job-name=qwen_full_ig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=qwen_full_run_%j.log
#SBATCH --error=qwen_full_run_%j.log

module load python/3.13.0

cd /scratch/jpdj5670/XAI/Interpretating-Readability-Evaluation-with-Explainable-AI/Task2/explainability-pipeline

echo "Job started: $(date)"
echo "Running on node: $(hostname)"

python3 -u scripts/run_qwen_full.py 2>&1

echo "Job finished: $(date)"