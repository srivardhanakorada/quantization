#!/bin/bash
#SBATCH --job-name=AdaVD_flex
#SBATCH --partition=allgpu           # Uses the catch-all partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # Lowering slightly to fit into 'mix' nodes easier
#SBATCH --gres=gpu:1                 # Request 1 GPU of any type
#SBATCH --mem=32G                    # Lowering to 32G to increase chances of scheduling
#SBATCH --time=04:00:00              # Shorter time (4hrs) helps "backfill" (jump the queue)
#SBATCH --output=adavd_flex.out
#SBATCH --error=adavd_flex.err

# --- 1. Environment Setup ---
# Initialize Conda for the script environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pcr_env

# --- 2. Run the Command ---
# Replace 'train.py' and arguments with your actual execution command
# python -W ignore adavd_quantize_erase.py \
#   --quant_ckpt ckpt/sd_quantized.ckpt \
#   --schema_path generation/single_schema.json \
#   --save_root generation/outputs/8bit/quantize_then_erase/adavd/single \
#   --mode retain \
#   --record_type values \
#   --include_anchor

# python -W ignore fp_adavd.py \
#   --sd_ckpt ~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14 \
#   --schema_path generation/single_schema.json \
#   --save_root generation/outputs/8bit/fp/adavd/single \
#   --mode retain \
#   --record_type values \
#   --include_anchor

# python -W ignore generation/scripts/generate_from_diffusers_schema.py \
#   --model_path generation/models/8bit/fp/esd/donald_trump_full_model \
#   --schema_path generation/single_schema.json \
#   --out_dir generation/outputs/8bit/fp/esd/single

# python -W ignore generation/scripts/generate_from_quant_ckpt.py \
#     --model_path generation/models/8bit/erase_then_quantize/esd/esd.ckpt \
#     --schema_path generation/single_schema.json \
#     --out_dir generation/outputs/8bit/erase_then_quantize/esd/single

# python -W ignore esd_quantize_erase.py \
#     --quant_ckpt ckpt/sd_quantized.ckpt \
#     --erase_concept "donald trump" \
#     --erase_from "" \
#     --train_method esd-x \
#     --iterations 200 \
#     --lr 5e-5 \
#     --batchsize 1 \
#     --num_inference_steps 50 \
#     --guidance_scale 3 \
#     --negative_guidance 1 \
#     --save_path generation/models/8bit/quantize_then_erase/esd/esd.ckpt

python -W ignore generation/scripts/generate_from_quant_ckpt.py \
    --model_path generation/models/8bit/quantize_then_erase/esd/esd.ckpt \
    --schema_path generation/single_schema.json \
    --out_dir generation/outputs/8bit/quantize_then_erase/esd/single