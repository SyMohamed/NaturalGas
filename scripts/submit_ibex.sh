#!/bin/bash
#SBATCH --job-name=natgas-cnn
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

# ── Activate environment ────────────────────────────────────────
module load anaconda3
conda activate natgas          # change to your env name

# ── Run training ────────────────────────────────────────────────
python -m src.train --config configs/config.yaml
