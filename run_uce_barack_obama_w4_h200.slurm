#!/bin/bash
#SBATCH --job-name=uce_bo_w4
#SBATCH --output=logs/uce_barack_obama_w4_%j.log
#SBATCH --error=logs/uce_barack_obama_w4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=h200
#SBATCH --qos=h200_qos
#SBATCH --account=somabiswas
#SBATCH --nodelist=cn10
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/somabiswas/karthikeyav/new/PCR
mkdir -p logs ckpt

source /home/somabiswas/karthikeyav/miniconda3/etc/profile.d/conda.sh
conda activate pcr_env

echo "Job started at: $(date)"
echo "Node: $(hostname)"
nvidia-smi

python benchmark/eval-sd-test.py \
  --linear-a-bit 16 \
  --linear-w-bit 4 \
  --conv-a-bit 16 \
  --conv-w-bit 4 \
  --num-calibrate 200 \
  --cali-batch-size 4 \
  --num-generate 200 \
  --model-path ../models/uce_models/Barack_Obama \
  --path-to-cali-text prompts/coco_train2017.txt \
  --path-to-test-text prompts/coco_val2017.txt \
  --exp-name UCE_barack_obama_w4_weightonly_full \
  --method Separate \
  --relax-first-last-layer \
  --recon \
  --save-ckpt-dir ckpt

echo "Job finished at: $(date)"