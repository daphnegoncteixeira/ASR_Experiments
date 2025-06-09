#!/bin/bash
#SBATCH --job-name=whisper_kriol_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# === ENV SETUP ===
source ~/.bashrc
conda activate training

# === MOVE TO PROJECT DIRECTORY ===
cd /path/to/your/project/

# === CREATE LOG FOLDER IF MISSING ===
mkdir -p logs

# === RUN TRAINING SCRIPT ===
python whisper_trainer.py
