#!/bin/bash
#SBATCH --job-name=finetune_TinyLlama
#SBATCH --output=logs/FT.out
#SBATCH --error=logs/FT.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Activate environment
source /home/njuttu_umass_edu/venvs/torch311_env/bin/activate

# Force Python to use venv
export PYTHONPATH="/home/njuttu_umass_edu/venvs/torch311_env/lib/python3.11/site-packages"
export PATH="/home/njuttu_umass_edu/venvs/torch311_env/bin:$PATH"

# Just in case - force reinstall
# pip install --force-reinstall --upgrade "transformers==4.51.3"

cd /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/TinyLlama

# echo "Running data preparation..."
# python load_data.py

echo "Running fine-tuning..."
python run_finetune.py

echo "Evaluation check:"
python evaluation.py
