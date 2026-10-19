#!/bin/bash
#SBATCH --job-name=STARR_EVAL
#SBATCH --output=/scratch/user/u.ae271721/log_files/STARR/STARR_EVAL_%A_%a.log
#SBATCH --error=/scratch/user/u.ae271721/log_files/STARR/STARR_EVAL_%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:a30:2
#SBATCH --time=08:00:00

cd /scratch/user/u.ae271721/BetaRisk

conda init bash
source ~/.bashrc
conda activate /scratch/user/u.ae271721/BetaRisk/env

set -euo pipefail

variants=(mscm_ms betarisk starr_loc starr_time starr_full)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
output_dir="runs/${variant}_seed0"

if [[ -f "${output_dir}/best.pth" ]]; then
  checkpoint="${output_dir}/best.pth"
elif [[ -f "${output_dir}/last.pth" ]]; then
  checkpoint="${output_dir}/last.pth"
else
  echo "No checkpoint found for ${variant} in ${output_dir}"
  exit 1
fi

echo "Evaluating ${variant} from ${checkpoint}"

python -u starr_train.py \
  --variant "${variant}" \
  --eval-only \
  --eval-checkpoint "${checkpoint}" \
  --data-root /scratch/group/p.cis230407.000/Major-Metro-Central-TX_1024x1024 \
  --time-csv-root time_data \
  --mscm-checkpoint ckpts/pre-trained_weights/pre-trained_weights_mscm.pth \
  --geoclip-weights geoclip/model/weights/location_encoder_weights.pth \
  --output-dir "${output_dir}" \
  --batch-size 8 \
  --workers 16 \
  --image-size 768 \
  --dropout 0.7 \
  --seed 0
