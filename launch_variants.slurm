#!/bin/bash
#SBATCH --job-name=STARR_VARIANTS
#SBATCH --output=/scratch/user/u.ae271721/log_files/STARR/STARR_VARIANTS_%A_%a.log
#SBATCH --error=/scratch/user/u.ae271721/log_files/STARR/STARR_VARIANTS_%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=742G
#SBATCH --gres=gpu:a30:2
#SBATCH --time=48:00:00

cd /scratch/user/u.ae271721/BetaRisk

conda init bash
source ~/.bashrc
conda activate /scratch/user/u.ae271721/BetaRisk/env

set -euo pipefail

variants=(mscm_ms betarisk starr_loc starr_time starr_full)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
output_dir="runs/${variant}_seed0"
resume_args=()
if [[ -f "${output_dir}/last.pth" ]]; then
  echo "Resuming ${variant} from ${output_dir}/last.pth"
  resume_args=(--resume "${output_dir}/last.pth")
else
  echo "Starting ${variant} from scratch"
fi

python -u starr_train.py \
  --variant "${variant}" \
  --data-root /scratch/group/p.cis230407.000/Major-Metro-Central-TX_1024x1024 \
  --time-csv-root time_data \
  --mscm-checkpoint ckpts/pre-trained_weights/pre-trained_weights_mscm.pth \
  --geoclip-weights geoclip/model/weights/location_encoder_weights.pth \
  --output-dir "${output_dir}" \
  --epochs 200 \
  "${resume_args[@]}" \
  --batch-size 16 \
  --workers 30 \
  --image-size 768 \
  --learning-rate 1e-4 \
  --dropout 0.7 \
  --seed 0
