#!/bin/bash
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=05:55:00
#SBATCH --output=logs/%j.log
set -eo pipefail
source ../bdd-runtime/activate.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi
python -u campaign.py
