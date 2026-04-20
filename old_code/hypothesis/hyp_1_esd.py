import os
import sys
import gc
import json
from typing import Dict, List

import torch
import pandas as pd
from diffusers import StableDiffusionPipeline

# Make PCR imports work when running from repo root
sys.path.append(os.getcwd())

from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub


# ============================================================
# Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
BASE_MODEL_LOCAL_FILES_ONLY = False
OUTPUT_DIR = "analysis_results_h1_esd"

FP_CHANGE_EPS = 1e-12
Q_BASE_EPS = 1e-12

# Set this to True if you want only attn2.to_k / attn2.to_v
# For ESD, I recommend False first.
EDITED_ONLY_ATTN2_KV = False

TARGET_SUFFIXES_FP = (
    "attn2.to_k.weight",
    "attn2.to_v.weight",
)

TARGET_SUFFIXES_QUANT = (
    "attn2.to_k.core.weight",
    "attn2.to_v.core.weight",
)

models_to_run = [
    {
        "name": "base_fp",
        "type": "fp",
        "path": "runwayml/stable-diffusion-v1-5",
        "local_files_only": BASE_MODEL_LOCAL_FILES_ONLY,
    },
    {
        "name": "esd_fp",
        "type": "fp",
        "path": "../models/esd-models/Barack_Obama/barack_obama_full_model",
        "local_files_only": True,
    },
    {
        "name": "esd_int8",
        "type": "quant",
        "path": "ckpt/quant-ESD_barack_obama_w8_weightonly-16-8-16-8.ckpt",
    },
]


# ============================================================
# Loading helpers
# ============================================================
def free_pipe(pipe):
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_fp_model(model_path: str, local_files_only: bool = True):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        safety_checker=None,
        torch_dtype=DTYPE,
        local_files_only=local_files_only,
    ).to(DEVICE)
    pipe.unet.eval()
    return pipe


def load_quantized_model(ckpt_path: str):
    pipe = torch.load(ckpt_path, map_location="cpu")
    pipe.to(DEVICE)
    pipe.unet.eval()

    layers_linear = find_layers(pipe.unet, (LinearQuantHub,))
    layers_conv = find_layers(pipe.unet, (Conv2dQuantHub,))
    all_quant_layers = {**layers_linear, **layers_conv}
    return pipe, all_quant_layers


# ============================================================
# Key filtering / canonicalization
# ============================================================
def is_weight_key(k: str) -> bool:
    return k.endswith(".weight")


def is_target_fp_key(k: str) -> bool:
    if EDITED_ONLY_ATTN2_KV:
        return any(k.endswith(suf) for suf in TARGET_SUFFIXES_FP)
    return is_weight_key(k)


def is_target_quant_key(k: str) -> bool:
    if EDITED_ONLY_ATTN2_KV:
        return any(k.endswith(suf) for suf in TARGET_SUFFIXES_QUANT)
    # For quantized checkpoints, effective tensors usually end with ".core.weight"
    return k.endswith(".core.weight")


def canonicalize_fp_key(k: str) -> str:
    return k


def canonicalize_quant_key(k: str) -> str:
    # Quantized names often look like:
    # down_blocks.0.attentions.0.transformer_blocks.0.core.attn2.to_k.core.weight
    # and should map to:
    # down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight
    k = k.replace(".core.attn2.", ".attn2.")
    k = k.replace(".core.attn1.", ".attn1.")
    k = k.replace(".core.ff.", ".ff.")
    k = k.replace(".core.proj_in.core.weight", ".proj_in.weight")
    k = k.replace(".core.proj_out.core.weight", ".proj_out.weight")
    k = k.replace(".core.conv1.core.weight", ".conv1.weight")
    k = k.replace(".core.conv2.core.weight", ".conv2.weight")
    k = k.replace(".core.conv_shortcut.core.weight", ".conv_shortcut.weight")
    k = k.replace(".core.conv_in.core.weight", ".conv_in.weight")
    k = k.replace(".core.conv_out.core.weight", ".conv_out.weight")
    k = k.replace(".to_q.core.weight", ".to_q.weight")
    k = k.replace(".to_k.core.weight", ".to_k.weight")
    k = k.replace(".to_v.core.weight", ".to_v.weight")
    k = k.replace(".to_out.0.core.weight", ".to_out.0.weight")
    return k


def get_filtered_fp_state_dict(pipe) -> Dict[str, torch.Tensor]:
    sd = pipe.unet.state_dict()
    out = {}
    for k, v in sd.items():
        if is_target_fp_key(k):
            out[canonicalize_fp_key(k)] = v.detach().float().cpu().clone()
    return out


def get_filtered_quant_state_dict(pipe) -> Dict[str, torch.Tensor]:
    sd = pipe.unet.state_dict()
    out = {}
    for k, v in sd.items():
        if is_target_quant_key(k):
            out[canonicalize_quant_key(k)] = v.detach().float().cpu().clone()
    return out


# ============================================================
# Key metadata
# ============================================================
def get_stage_from_key(k: str) -> str:
    if k.startswith("down_blocks."):
        return "down"
    if k.startswith("mid_block."):
        return "mid"
    if k.startswith("up_blocks."):
        return "up"
    return "other"


def get_family_from_key(k: str) -> str:
    if ".attn2.to_k.weight" in k:
        return "attn2_to_k"
    if ".attn2.to_v.weight" in k:
        return "attn2_to_v"
    if ".attn2.to_q.weight" in k:
        return "attn2_to_q"
    if ".attn2.to_out.0.weight" in k:
        return "attn2_to_out"
    if ".attn1.to_k.weight" in k:
        return "attn1_to_k"
    if ".attn1.to_v.weight" in k:
        return "attn1_to_v"
    if ".attn1.to_q.weight" in k:
        return "attn1_to_q"
    if ".attn1.to_out.0.weight" in k:
        return "attn1_to_out"
    if ".ff.net." in k:
        return "ff"
    if ".proj_in.weight" in k:
        return "proj_in"
    if ".proj_out.weight" in k:
        return "proj_out"
    if ".conv_in.weight" in k:
        return "conv_in"
    if ".conv_out.weight" in k:
        return "conv_out"
    if ".conv_shortcut.weight" in k:
        return "conv_shortcut"
    if ".downsamplers." in k:
        return "downsampler"
    if ".upsamplers." in k:
        return "upsampler"
    if ".conv1.weight" in k:
        return "resnet_conv1"
    if ".conv2.weight" in k:
        return "resnet_conv2"
    return "other"


# ============================================================
# Analysis helpers
# ============================================================
def safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.mean().item())


def safe_max(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.max().item())


def safe_sum_bool(x: torch.Tensor) -> int:
    if x.numel() == 0:
        return 0
    return int(x.sum().item())


def ratio(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def analyze_quantized_survival_for_key(
    key_name: str,
    w_base: torch.Tensor,
    w_esd_fp: torch.Tensor,
    w_quant: torch.Tensor,
) -> dict:
    fp_delta = w_esd_fp - w_base
    fp_abs_delta = fp_delta.abs()

    fp_changed_mask = fp_abs_delta > FP_CHANGE_EPS
    fp_changed_count = safe_sum_bool(fp_changed_mask)
    total_count = int(w_base.numel())

    q_delta_to_base = w_quant - w_base
    q_abs_delta_to_base = q_delta_to_base.abs()

    dist_to_base = (w_quant - w_base).abs()
    dist_to_fp = (w_quant - w_esd_fp).abs()

    fp_delta_changed = fp_delta[fp_changed_mask]
    fp_abs_delta_changed = fp_abs_delta[fp_changed_mask]
    q_delta_changed = q_delta_to_base[fp_changed_mask]
    q_abs_delta_changed = q_abs_delta_to_base[fp_changed_mask]
    dist_to_base_changed = dist_to_base[fp_changed_mask]
    dist_to_fp_changed = dist_to_fp[fp_changed_mask]

    q_survival_mask = q_abs_delta_changed > Q_BASE_EPS
    q_survival_count = safe_sum_bool(q_survival_mask)

    closer_to_fp_mask = dist_to_fp_changed < dist_to_base_changed
    closer_to_base_mask = dist_to_base_changed < dist_to_fp_changed
    tie_mask = dist_to_base_changed == dist_to_fp_changed

    closer_to_fp_count = safe_sum_bool(closer_to_fp_mask)
    closer_to_base_count = safe_sum_bool(closer_to_base_mask)
    tie_count = safe_sum_bool(tie_mask)

    same_sign_mask = torch.sign(q_delta_changed) == torch.sign(fp_delta_changed)
    same_sign_count = safe_sum_bool(same_sign_mask)

    shrinkage_vals = torch.zeros_like(q_abs_delta_changed)
    nonzero_fp_mask = fp_abs_delta_changed > FP_CHANGE_EPS
    shrinkage_vals[nonzero_fp_mask] = (
        q_abs_delta_changed[nonzero_fp_mask] / fp_abs_delta_changed[nonzero_fp_mask]
    )
    shrinkage_vals = torch.clamp(shrinkage_vals, min=0.0)

    l1_delta = float(fp_abs_delta.sum().item())
    l2_delta_sq = float((fp_delta ** 2).sum().item())

    return {
        "key": key_name,
        "stage": get_stage_from_key(key_name),
        "family": get_family_from_key(key_name),
        "numel": total_count,
        "fp_changed_count": fp_changed_count,
        "fp_changed_ratio": ratio(fp_changed_count, total_count),
        "fp_mean_abs_delta": safe_mean(fp_abs_delta),
        "fp_max_abs_delta": safe_max(fp_abs_delta),
        "fp_mean_abs_delta_changed_only": safe_mean(fp_abs_delta_changed),
        "q_mean_abs_delta_to_base_all": safe_mean(q_abs_delta_to_base),
        "q_mean_abs_delta_to_base_changed_only": safe_mean(q_abs_delta_changed),
        "q_survival_count": q_survival_count,
        "q_survival_ratio_over_fp_changed": ratio(q_survival_count, fp_changed_count),
        "closer_to_fp_count": closer_to_fp_count,
        "closer_to_base_count": closer_to_base_count,
        "tie_count": tie_count,
        "closer_to_fp_ratio_over_fp_changed": ratio(closer_to_fp_count, fp_changed_count),
        "closer_to_base_ratio_over_fp_changed": ratio(closer_to_base_count, fp_changed_count),
        "tie_ratio_over_fp_changed": ratio(tie_count, fp_changed_count),
        "same_sign_count": same_sign_count,
        "same_sign_ratio_over_fp_changed": ratio(same_sign_count, fp_changed_count),
        "mean_shrinkage_ratio_changed_only": safe_mean(shrinkage_vals),
        "median_shrinkage_ratio_changed_only": float(
            torch.median(shrinkage_vals).item()
        ) if shrinkage_vals.numel() > 0 else 0.0,
        "l1_delta": l1_delta,
        "l2_delta_sq": l2_delta_sq,
    }


def build_summary(per_key_df: pd.DataFrame) -> pd.DataFrame:
    total_fp_changed = int(per_key_df["fp_changed_count"].sum())
    total_numel = int(per_key_df["numel"].sum())
    total_q_survival = int(per_key_df["q_survival_count"].sum())
    total_closer_to_fp = int(per_key_df["closer_to_fp_count"].sum())
    total_closer_to_base = int(per_key_df["closer_to_base_count"].sum())
    total_tie = int(per_key_df["tie_count"].sum())
    total_same_sign = int(per_key_df["same_sign_count"].sum())

    return pd.DataFrame([{
        "keys_analyzed": int(len(per_key_df)),
        "total_numel": total_numel,
        "total_fp_changed": total_fp_changed,
        "total_fp_changed_ratio": ratio(total_fp_changed, total_numel),
        "weighted_q_survival_ratio_over_fp_changed": ratio(total_q_survival, total_fp_changed),
        "weighted_closer_to_fp_ratio_over_fp_changed": ratio(total_closer_to_fp, total_fp_changed),
        "weighted_closer_to_base_ratio_over_fp_changed": ratio(total_closer_to_base, total_fp_changed),
        "weighted_tie_ratio_over_fp_changed": ratio(total_tie, total_fp_changed),
        "weighted_same_sign_ratio_over_fp_changed": ratio(total_same_sign, total_fp_changed),
        "mean_of_key_fp_mean_abs_delta": float(per_key_df["fp_mean_abs_delta"].mean()),
        "mean_of_key_fp_mean_abs_delta_changed_only": float(per_key_df["fp_mean_abs_delta_changed_only"].mean()),
        "mean_of_key_q_mean_abs_delta_to_base_changed_only": float(per_key_df["q_mean_abs_delta_to_base_changed_only"].mean()),
        "mean_of_key_shrinkage": float(per_key_df["mean_shrinkage_ratio_changed_only"].mean()),
        "median_of_key_shrinkage": float(per_key_df["median_shrinkage_ratio_changed_only"].median()),
    }])


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col).agg({
        "key": "count",
        "numel": "sum",
        "fp_changed_count": "sum",
        "l1_delta": "sum",
        "l2_delta_sq": "sum",
        "closer_to_fp_count": "sum",
        "closer_to_base_count": "sum",
    }).reset_index()

    grouped = grouped.rename(columns={"key": "num_tensors"})
    grouped["fp_changed_ratio_weighted"] = grouped["fp_changed_count"] / grouped["numel"]

    total_l1 = grouped["l1_delta"].sum()
    total_l2_sq = grouped["l2_delta_sq"].sum()

    grouped["l1_share_global"] = grouped["l1_delta"] / total_l1 if total_l1 > 0 else 0.0
    grouped["l2_energy_share_global"] = grouped["l2_delta_sq"] / total_l2_sq if total_l2_sq > 0 else 0.0
    grouped["closer_to_fp_ratio"] = grouped["closer_to_fp_count"] / grouped["fp_changed_count"].clip(lower=1)
    grouped["closer_to_base_ratio"] = grouped["closer_to_base_count"] / grouped["fp_changed_count"].clip(lower=1)

    grouped = grouped.sort_values("l1_share_global", ascending=False).reset_index(drop=True)
    return grouped


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loaded = {}

    print(f"Running on device: {DEVICE}")
    print(f"EDITED_ONLY_ATTN2_KV = {EDITED_ONLY_ATTN2_KV}")
    print("Loading models...")

    for cfg in models_to_run:
        name = cfg["name"]
        mtype = cfg["type"]
        path = cfg["path"]

        print(f"\nLoading {name} from {path}")

        if mtype == "fp":
            pipe = load_fp_model(
                model_path=path,
                local_files_only=cfg.get("local_files_only", True),
            )
        else:
            pipe, qlayers = load_quantized_model(path)
            print(f"{name}: found {len(qlayers)} quantized layers")

        loaded[name] = pipe

    print("\nExtracting canonicalized keys...")
    base_sd = get_filtered_fp_state_dict(loaded["base_fp"])
    esd_fp_sd = get_filtered_fp_state_dict(loaded["esd_fp"])
    esd_int8_sd = get_filtered_quant_state_dict(loaded["esd_int8"])

    print(f"base keys:    {len(base_sd)}")
    print(f"esd_fp keys:  {len(esd_fp_sd)}")
    print(f"esd_int8 keys:{len(esd_int8_sd)}")

    common_keys = sorted(
        set(base_sd.keys()) &
        set(esd_fp_sd.keys()) &
        set(esd_int8_sd.keys())
    )

    print(f"\nCommon keys to analyze: {len(common_keys)}")
    for k in common_keys[:50]:
        print(" ", k)
    if len(common_keys) > 50:
        print("  ...")

    if len(common_keys) == 0:
        debug_path = os.path.join(OUTPUT_DIR, "debug_key_dump.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_keys": sorted(list(base_sd.keys())),
                    "esd_fp_keys": sorted(list(esd_fp_sd.keys())),
                    "esd_int8_keys": sorted(list(esd_int8_sd.keys())),
                },
                f,
                indent=2,
            )
        print(f"No common keys found. Saved debug dump to {debug_path}")
        return

    per_key_rows: List[dict] = []

    print("\nAnalyzing ESD INT8...")
    for key_name in common_keys:
        per_key_rows.append(
            analyze_quantized_survival_for_key(
                key_name=key_name,
                w_base=base_sd[key_name],
                w_esd_fp=esd_fp_sd[key_name],
                w_quant=esd_int8_sd[key_name],
            )
        )

    per_key_df = pd.DataFrame(per_key_rows)
    summary_df = build_summary(per_key_df)
    family_df = summarize_group(per_key_df, "family")
    stage_df = summarize_group(per_key_df, "stage")
    top_df = per_key_df.sort_values("l1_delta", ascending=False).reset_index(drop=True)

    per_key_csv = os.path.join(OUTPUT_DIR, "esd_h1_quant_survival_per_key.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "esd_h1_quant_survival_summary.csv")
    family_csv = os.path.join(OUTPUT_DIR, "esd_h1_family_summary.csv")
    stage_csv = os.path.join(OUTPUT_DIR, "esd_h1_stage_summary.csv")
    top_csv = os.path.join(OUTPUT_DIR, "esd_h1_top_keys.csv")

    per_key_df.to_csv(per_key_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    family_df.to_csv(family_csv, index=False)
    stage_df.to_csv(stage_csv, index=False)
    top_df.to_csv(top_csv, index=False)

    print("\nSaved:")
    print(" ", per_key_csv)
    print(" ", summary_csv)
    print(" ", family_csv)
    print(" ", stage_csv)
    print(" ", top_csv)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Family summary ===")
    print(family_df.to_string(index=False))

    print("\n=== Stage summary ===")
    print(stage_df.to_string(index=False))

    for pipe in loaded.values():
        free_pipe(pipe)


if __name__ == "__main__":
    main()