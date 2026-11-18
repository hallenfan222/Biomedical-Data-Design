#!/bin/bash
#SBATCH --job-name=bdd-xgb-tune
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/expanse/lustre/projects/jhu159/zfan11/xgb-tuning-20261117/logs/%j.log
set -eo pipefail
source /expanse/lustre/projects/jhu159/zfan11/bdd-runtime/activate.sh
cd /expanse/lustre/projects/jhu159/zfan11/xgb-tuning-20261117
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi
python -u runner.py "$@"
