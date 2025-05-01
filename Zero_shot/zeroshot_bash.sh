#!/bin/bash
#SBATCH --job-name=zeroshot_lexglue
#SBATCH --output=logs/zeroshot_%j.out
#SBATCH --error=logs/zeroshot_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

source /home/njuttu_umass_edu/venvs/torch_env/bin/activate
cd /home/njuttu_umass_edu/685/ZeroShotAnomolyDetection/Zero_shot 

export OPENAI_API_KEY="RQ9xqvwX_H9elDWqNqkwRcfqN61fOpWfzFjuwDSJqc6D9eoiWqu8ZZETLMEw5y4jC05MdVhL3KT3BlbkFJVnLKRTjI6gVQWOLsKbr5-ECNPuxXCPLQfmCTLWSR56VGNh25eUYysl95F0TU7xKNJqCtUcSJ4A" 
python run_zeroshot.py