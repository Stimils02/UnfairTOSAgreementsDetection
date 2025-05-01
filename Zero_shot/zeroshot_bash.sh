#!/bin/bash
#SBATCH --job-name=zeroshot_lexglue
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

module load anaconda
conda activate your_env_name

python run_zeroshot.py