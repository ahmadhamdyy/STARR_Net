#!/bin/bash
#SBATCH --job-name=starr-full
#SBATCH --output=logs/starr-full-%j.out
#SBATCH --error=logs/starr-full-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Override VENV_PATH when submitting if your environment is elsewhere:
# sbatch --export=ALL,VENV_PATH=/path/to/venv launch_starr.slurm
VENV_PATH="${VENV_PATH:-$HOME/.venvs/starr}"
source "${VENV_PATH}/bin/activate"

python -u starr_train.py \
  --variant starr_full \
  --data-root /scratch/group/p.cis230407.000/Major-Metro-Central-TX_1024x1024 \
  --time-csv-root time_data \
  --mscm-checkpoint ckpts/pre-trained_weights/pre-trained_weights_mscm.pth \
  --geoclip-weights geoclip/model/weights/location_encoder_weights.pth \
  --output-dir runs/starr_full_seed0 \
  --batch-size 4 \
  --workers 2 \
  --image-size 768 \
  --learning-rate 1e-4 \
  --dropout 0.7 \
  --seed 0
