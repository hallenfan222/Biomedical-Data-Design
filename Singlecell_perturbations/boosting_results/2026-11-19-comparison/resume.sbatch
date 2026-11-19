#!/bin/bash
#SBATCH --account=jhu159
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%j.log
set -eo pipefail
export BDD_JOB_DEADLINE=$(( $(date +%s) + 8940 ))
source ../bdd-runtime/activate.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi
python -u - <<'PY'
import json, subprocess, sys
from pathlib import Path
for lib in ['xgb','pyboost']:
    w=json.loads(Path('locked.json').read_text())[lib]
    subprocess.run([sys.executable,'-u','fit.py',lib,w['design'],w['stage'],'final',json.dumps(w['overrides'])],check=True)
PY
