#!/bin/bash
#SBATCH --job-name=download_tinyllama
#SBATCH --output=logs/dl_%j.out
#SBATCH --error=logs/dl_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu

source /home/njuttu_umass_edu/venvs/torch_env/bin/activate
cd /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/TinyLlama

python baseline.py