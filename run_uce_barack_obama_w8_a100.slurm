#!/bin/bash
#SBATCH --job-name=uce_bo_g1_w8
#SBATCH --output=logs/uce_barack_obama_gamma25_w8_%j.log
#SBATCH --error=logs/uce_barack_obama_gamma25_w8_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:A100:1

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
  --linear-w-bit 8 \
  --conv-a-bit 16 \
  --conv-w-bit 8 \
  --num-calibrate 200 \
  --cali-batch-size 4 \
  --num-generate 200 \
  --model-path ../models/uce_models/Barack_Obama_reg_25 \
  --path-to-cali-text prompts/coco_train2017.txt \
  --path-to-test-text prompts/coco_val2017.txt \
  --exp-name UCE_barack_obama_reg_25_w8_weightonly_full \
  --method Separate \
  --relax-first-last-layer \
  --recon \
  --save-ckpt-dir ckpt

echo "Job finished at: $(date)"