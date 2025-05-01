#!/bin/bash
#SBATCH --job-name=finetune_TinyLlama
#SBATCH --output=logs/FT_%j.out
#SBATCH --error=logs/FT_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Activate environment
source /home/njuttu_umass_edu/venvs/torch_env/bin/activate
cd /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/TinyLlama

# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Create logs directory if not exists
mkdir -p logs data

# Step 1: Prepare dataset
echo "Running data preparation..."
python load_data.py

# Step 2: Run fine-tuning
echo "Running fine-tuning..."
python run_finetune.py