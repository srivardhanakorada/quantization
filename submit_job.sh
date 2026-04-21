#!/bin/bash
#SBATCH --job-name=AdaVD_flex
#SBATCH --partition=allgpu           # Uses the catch-all partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # Lowering slightly to fit into 'mix' nodes easier
#SBATCH --gres=gpu:1                 # Request 1 GPU of any type
#SBATCH --mem=32G                    # Lowering to 32G to increase chances of scheduling
#SBATCH --time=04:00:00              # Shorter time (4hrs) helps "backfill" (jump the queue)
#SBATCH --output=adavd_flex_%j.out
#SBATCH --error=adavd_flex_%j.err

# --- 1. Environment Setup ---
# Initialize Conda for the script environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pcr_env

# --- 2. Run the Command ---
# Replace 'train.py' and arguments with your actual execution command
# python -W ignore adavd_quantize_erase.py \
#   --quant_ckpt  ckpt/sd_quantized.ckpt \
#   --schema_path generation/single_schema.json \
#   --out_dir  generation/outputs/8bit/quantize_then_erase/adavd/single \
#   --mode original,erase,retain \
#   --batch_size 1 \
#   --include_anchor \
#   --record_type values \
#   --decomp_timestep 0 \
#   --sigmoid_a 100 \
#   --sigmoid_b 0.93 \
#   --sigmoid_c 2

python -W ignore fp_adavd.py \
  --model_name runwayml/stable-diffusion-v1-5 \
  --schema_path generation/single_schema.json \
  --out_dir  generation/outputs/8bit/fp/adavd/single \
  --mode original,erase,retain \
  --batch_size 1 \
  --include_anchor \
  --record_type values \
  --decomp_timestep 0 \
  --sigmoid_a 100 \
  --sigmoid_b 0.93 \
  --sigmoid_c 2