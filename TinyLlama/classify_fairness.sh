#!/bin/bash
#SBATCH --job-name=classify_TinyLlama
#SBATCH --output=logs/classify.out
#SBATCH --error=logs/classify.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# === Activate environment ===
source /home/njuttu_umass_edu/venvs/torch311_env/bin/activate

# === Set paths ===
export PYTHONPATH="/home/njuttu_umass_edu/venvs/torch311_env/lib/python3.11/site-packages"
export PATH="/home/njuttu_umass_edu/venvs/torch311_env/bin:$PATH"

# === Navigate to project directory ===
cd /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection-1/TinyLlama

echo "Running classification with TinyLlama..."
python classify_fairness.py

echo "Done with inference!"
