import os
import sys
import gc
import re
import json
import math
import argparse
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Make PCR imports work when running from repo root
sys.path.append(os.getcwd())

from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

FP_CHANGE_EPS = 1e-12


# =========================================================
# helpers
# =========================================================
def free_pipe(pipe):
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.mean().item())


def safe_max(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.max().item())


def safe_median(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.median(x).item())


def safe_sum_bool(x: torch.Tensor) -> int:
    if x.numel() == 0:
        return 0
    return int(x.sum().item())


def ratio(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def get_block_name(param_name: str):
    if param_name.startswith("down_blocks."):
        m = re.match(r"(down_blocks\.\d+)", param_name)
        return m.group(1) if m else "down_blocks.?"
    elif param_name.startswith("up_blocks."):
        m = re.match(r"(up_blocks\.\d+)", param_name)
        return m.group(1) if m else "up_blocks.?"
    elif param_name.startswith("mid_block."):
        return "mid_block"
    else:
        return "other"


def get_family(param_name: str):
    if param_name.endswith("to_k.weight"):
        return "to_k"
    elif param_name.endswith("to_v.weight"):
        return "to_v"
    return "other"


# =========================================================
# key filtering / canonicalization
# =========================================================
TARGET_SUFFIXES_FP = (
    "attn2.to_k.weight",
    "attn2.to_v.weight",
)

TARGET_SUFFIXES_QUANT = (
    "attn2.to_k.core.weight",
    "attn2.to_v.core.weight",
)


def is_target_fp_key(k: str) -> bool:
    return any(k.endswith(suf) for suf in TARGET_SUFFIXES_FP)


def is_target_quant_key(k: str) -> bool:
    return any(k.endswith(suf) for suf in TARGET_SUFFIXES_QUANT)


def canonicalize_fp_key(k: str) -> str:
    return k


def canonicalize_quant_key(k: str) -> str:
    # Example:
    # down_blocks.0.attentions.0.transformer_blocks.0.core.attn2.to_k.core.weight
    # -> down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight
    k = k.replace(".transformer_blocks.0.core.attn2.", ".transformer_blocks.0.attn2.")
    k = k.replace(".to_k.core.weight", ".to_k.weight")
    k = k.replace(".to_v.core.weight", ".to_v.weight")
    return k


def canonical_key_to_quanthub_key(canonical_key: str) -> str:
    # inverse mapping used to find actual quant hub module
    # canonical:
    # down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight
    # quant hub:
    # down_blocks.0.attentions.0.transformer_blocks.0.core.attn2.to_k
    base = canonical_key.replace(".weight", "")
    base = base.replace(".transformer_blocks.0.attn2.", ".transformer_blocks.0.core.attn2.")
    return base


# =========================================================
# model loading
# =========================================================
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


# =========================================================
# qparam extraction from quant hubs
# =========================================================
def get_qparams_for_key(
    canonical_key: str,
    quant_layers: Dict[str, torch.nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Returns:
        scale, zero_point, source
    scale can be:
        - [out_features] or [out_features, 1]
        - scalar tensor
    """
    qhub_key = canonical_key_to_quanthub_key(canonical_key)

    if qhub_key not in quant_layers:
        return None, None, "not_found"

    layer = quant_layers[qhub_key]

    if not hasattr(layer, "quantizer") or len(layer.quantizer) == 0:
        return None, None, "no_quantizer"

    q = layer.quantizer[0]

    if hasattr(q, "w_scale") and hasattr(q, "w_zero_point"):
        scale = q.w_scale.detach().float().cpu().clone()
        zero_point = q.w_zero_point.detach().float().cpu().clone()
        return scale, zero_point, "saved_w_scale"

    return None, None, "missing_w_scale"


def broadcast_qparams_like_weight(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
):
    """
    Match the common PCR cases:
    - per-channel affine: scale shape [out] or [out,1]
    - per-tensor affine: scalar
    """
    if scale is None or zero_point is None:
        return None, None

    if scale.numel() == 1:
        scale_b = scale.reshape(1, 1).expand_as(w)
        zp_b = zero_point.reshape(1, 1).expand_as(w)
        return scale_b, zp_b

    # common linear weight case: [out, in]
    if w.ndim == 2:
        if scale.ndim == 1 and scale.shape[0] == w.shape[0]:
            scale_b = scale.view(-1, 1).expand_as(w)
            zp_b = zero_point.view(-1, 1).expand_as(w)
            return scale_b, zp_b

        if scale.ndim == 2 and scale.shape[0] == w.shape[0] and scale.shape[1] == 1:
            scale_b = scale.expand_as(w)
            zp_b = zero_point.expand_as(w)
            return scale_b, zp_b

    return None, None


def estimate_qparams_from_fp_weight(w_fp: torch.Tensor):
    """
    Fallback approximation:
    per-channel affine uint8 along dim 0 for linear weights.
    """
    if w_fp.ndim != 2:
        x_min = w_fp.min()
        x_max = w_fp.max()
        scale = (x_max - x_min) / 255.0
        scale = torch.tensor(max(scale.item(), 1e-8))
        zp = torch.round(-x_min / scale).clamp(0, 255)
        return scale, zp, "estimated_per_tensor"

    x_min = w_fp.min(dim=1).values
    x_max = w_fp.max(dim=1).values
    scale = (x_max - x_min) / 255.0
    scale = torch.where(scale <= 0, torch.ones_like(scale) * 1e-8, scale)
    zp = torch.round(-x_min / scale).clamp(0, 255)
    return scale, zp, "estimated_per_channel"


def quantize_to_indices(
    w: torch.Tensor,
    scale_b: torch.Tensor,
    zp_b: torch.Tensor,
):
    q_idx = torch.round(w / scale_b + zp_b)
    q_idx = torch.clamp(q_idx, 0, 255)
    w_dequant = (q_idx - zp_b) * scale_b
    return q_idx, w_dequant


# =========================================================
# analysis
# =========================================================
def analyze_quantized_bucket_behavior_for_key(
    key_name: str,
    w_base: torch.Tensor,
    w_uce_fp: torch.Tensor,
    w_quant_actual: torch.Tensor,
    quant_layers: Dict[str, torch.nn.Module],
) -> dict:
    fp_delta = w_uce_fp - w_base
    fp_abs_delta = fp_delta.abs()

    fp_changed_mask = fp_abs_delta > FP_CHANGE_EPS
    fp_changed_count = safe_sum_bool(fp_changed_mask)
    total_count = int(w_base.numel())

    # ---- tensorwise FP change ----
    base_norm = torch.norm(w_base).item()
    diff_norm = torch.norm(fp_delta).item()
    pct_l2_change = 100.0 * diff_norm / (base_norm + 1e-12)
    pct_scalar_changed = 100.0 * ratio(fp_changed_count, total_count)

    # ---- get quantization bucket info ----
    scale, zero_point, qparam_source = get_qparams_for_key(key_name, quant_layers)
    if scale is None or zero_point is None:
        scale, zero_point, qparam_source = estimate_qparams_from_fp_weight(w_uce_fp)

    scale_b, zp_b = broadcast_qparams_like_weight(w_uce_fp, scale, zero_point)
    if scale_b is None or zp_b is None:
        # final fallback: scalar
        scale = torch.tensor(float(scale.flatten()[0].item()))
        zero_point = torch.tensor(float(zero_point.flatten()[0].item()))
        scale_b = scale.reshape(1, 1).expand_as(w_uce_fp)
        zp_b = zero_point.reshape(1, 1).expand_as(w_uce_fp)
        qparam_source += "_scalar_fallback"

    bucket_width = scale_b.abs()

    # ---- update size relative to bucket width ----
    update_over_bucket = fp_abs_delta / (bucket_width + 1e-12)
    update_over_bucket_changed = update_over_bucket[fp_changed_mask]

    frac_ratio_lt_05 = safe_mean((update_over_bucket_changed < 0.5).float()) * 100.0
    frac_ratio_05_1 = safe_mean(((update_over_bucket_changed >= 0.5) & (update_over_bucket_changed < 1.0)).float()) * 100.0
    frac_ratio_ge_1 = safe_mean((update_over_bucket_changed >= 1.0).float()) * 100.0

    # ---- same bucket test: do base and erased FP land in same bucket? ----
    qidx_base, w_base_qsim = quantize_to_indices(w_base, scale_b, zp_b)
    qidx_uce_fp, w_uce_fp_qsim = quantize_to_indices(w_uce_fp, scale_b, zp_b)

    same_bucket_mask_all = (qidx_base == qidx_uce_fp)
    same_bucket_mask_changed = same_bucket_mask_all[fp_changed_mask]
    cross_bucket_mask_changed = ~same_bucket_mask_changed

    same_bucket_count = safe_sum_bool(same_bucket_mask_changed)
    cross_bucket_count = safe_sum_bool(cross_bucket_mask_changed)

    # ---- actual quantized checkpoint behavior ----
    dist_to_base = (w_quant_actual - w_base).abs()
    dist_to_uce = (w_quant_actual - w_uce_fp).abs()

    dist_to_base_changed = dist_to_base[fp_changed_mask]
    dist_to_uce_changed = dist_to_uce[fp_changed_mask]

    closer_to_uce_mask = dist_to_uce_changed < dist_to_base_changed
    closer_to_base_mask = dist_to_base_changed < dist_to_uce_changed
    tie_mask = dist_to_base_changed == dist_to_uce_changed

    closer_to_uce_count = safe_sum_bool(closer_to_uce_mask)
    closer_to_base_count = safe_sum_bool(closer_to_base_mask)
    tie_count = safe_sum_bool(tie_mask)

    # ---- collapse to base-quantized bucket/value ----
    collapse_to_base_q_mask = torch.isclose(
        w_quant_actual[fp_changed_mask],
        w_base_qsim[fp_changed_mask],
        atol=1e-8,
        rtol=0.0,
    )
    collapse_to_base_q_count = safe_sum_bool(collapse_to_base_q_mask)

    # ---- does actual quantized erased still differ from base? ----
    q_delta_to_base = w_quant_actual - w_base
    q_abs_delta_to_base = q_delta_to_base.abs()
    q_abs_delta_changed = q_abs_delta_to_base[fp_changed_mask]

    q_survival_mask = q_abs_delta_changed > FP_CHANGE_EPS
    q_survival_count = safe_sum_bool(q_survival_mask)

    # ---- direction consistency ----
    q_delta_changed = q_delta_to_base[fp_changed_mask]
    fp_delta_changed = fp_delta[fp_changed_mask]
    same_sign_mask = torch.sign(q_delta_changed) == torch.sign(fp_delta_changed)
    same_sign_count = safe_sum_bool(same_sign_mask)

    # ---- shrinkage ----
    fp_abs_delta_changed = fp_abs_delta[fp_changed_mask]
    shrinkage_vals = torch.zeros_like(q_abs_delta_changed)
    nonzero_fp_mask = fp_abs_delta_changed > FP_CHANGE_EPS
    shrinkage_vals[nonzero_fp_mask] = (
        q_abs_delta_changed[nonzero_fp_mask] / fp_abs_delta_changed[nonzero_fp_mask]
    )
    shrinkage_vals = torch.clamp(shrinkage_vals, min=0.0)

    return {
        "key": key_name,
        "family": get_family(key_name),
        "block": get_block_name(key_name),
        "numel": total_count,

        # FP32 erased vs base
        "fp_changed_count": fp_changed_count,
        "fp_changed_ratio": ratio(fp_changed_count, total_count),
        "pct_l2_change": pct_l2_change,
        "pct_scalar_changed": pct_scalar_changed,
        "fp_mean_abs_delta": safe_mean(fp_abs_delta),
        "fp_max_abs_delta": safe_max(fp_abs_delta),
        "fp_mean_abs_delta_changed_only": safe_mean(fp_abs_delta_changed),

        # bucket info
        "qparam_source": qparam_source,
        "bucket_mean": safe_mean(bucket_width),
        "bucket_median": safe_median(bucket_width),
        "update_over_bucket_mean_changed_only": safe_mean(update_over_bucket_changed),
        "update_over_bucket_median_changed_only": safe_median(update_over_bucket_changed),
        "frac_update_over_bucket_lt_05_changed_only": frac_ratio_lt_05,
        "frac_update_over_bucket_05_1_changed_only": frac_ratio_05_1,
        "frac_update_over_bucket_ge_1_changed_only": frac_ratio_ge_1,

        # same bucket
        "same_bucket_count": same_bucket_count,
        "cross_bucket_count": cross_bucket_count,
        "same_bucket_ratio_over_fp_changed": ratio(same_bucket_count, fp_changed_count),
        "cross_bucket_ratio_over_fp_changed": ratio(cross_bucket_count, fp_changed_count),

        # actual quantized behavior
        "q_mean_abs_delta_to_base_all": safe_mean(q_abs_delta_to_base),
        "q_mean_abs_delta_to_base_changed_only": safe_mean(q_abs_delta_changed),
        "q_survival_count": q_survival_count,
        "q_survival_ratio_over_fp_changed": ratio(q_survival_count, fp_changed_count),

        "closer_to_uce_count": closer_to_uce_count,
        "closer_to_base_count": closer_to_base_count,
        "tie_count": tie_count,
        "closer_to_uce_ratio_over_fp_changed": ratio(closer_to_uce_count, fp_changed_count),
        "closer_to_base_ratio_over_fp_changed": ratio(closer_to_base_count, fp_changed_count),
        "tie_ratio_over_fp_changed": ratio(tie_count, fp_changed_count),

        "collapse_to_base_q_count": collapse_to_base_q_count,
        "collapse_to_base_q_ratio_over_fp_changed": ratio(collapse_to_base_q_count, fp_changed_count),

        "same_sign_count": same_sign_count,
        "same_sign_ratio_over_fp_changed": ratio(same_sign_count, fp_changed_count),

        "mean_shrinkage_ratio_changed_only": safe_mean(shrinkage_vals),
        "median_shrinkage_ratio_changed_only": safe_median(shrinkage_vals),

        # for hist plot
        "_update_over_bucket_changed": update_over_bucket_changed.detach().cpu().numpy(),
    }


def build_summary(per_key_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total_fp_changed = int(per_key_df["fp_changed_count"].sum())
    total_numel = int(per_key_df["numel"].sum())
    total_q_survival = int(per_key_df["q_survival_count"].sum())
    total_closer_to_uce = int(per_key_df["closer_to_uce_count"].sum())
    total_closer_to_base = int(per_key_df["closer_to_base_count"].sum())
    total_tie = int(per_key_df["tie_count"].sum())
    total_same_sign = int(per_key_df["same_sign_count"].sum())
    total_same_bucket = int(per_key_df["same_bucket_count"].sum())
    total_cross_bucket = int(per_key_df["cross_bucket_count"].sum())
    total_collapse_to_base_q = int(per_key_df["collapse_to_base_q_count"].sum())

    rows.append({
        "keys_analyzed": int(len(per_key_df)),
        "total_numel": total_numel,
        "total_fp_changed": total_fp_changed,
        "total_fp_changed_ratio": ratio(total_fp_changed, total_numel),

        "weighted_same_bucket_ratio_over_fp_changed": ratio(total_same_bucket, total_fp_changed),
        "weighted_cross_bucket_ratio_over_fp_changed": ratio(total_cross_bucket, total_fp_changed),
        "weighted_q_survival_ratio_over_fp_changed": ratio(total_q_survival, total_fp_changed),
        "weighted_closer_to_uce_ratio_over_fp_changed": ratio(total_closer_to_uce, total_fp_changed),
        "weighted_closer_to_base_ratio_over_fp_changed": ratio(total_closer_to_base, total_fp_changed),
        "weighted_tie_ratio_over_fp_changed": ratio(total_tie, total_fp_changed),
        "weighted_collapse_to_base_q_ratio_over_fp_changed": ratio(total_collapse_to_base_q, total_fp_changed),
        "weighted_same_sign_ratio_over_fp_changed": ratio(total_same_sign, total_fp_changed),

        "mean_of_key_pct_l2_change": float(per_key_df["pct_l2_change"].mean()),
        "mean_of_key_pct_scalar_changed": float(per_key_df["pct_scalar_changed"].mean()),
        "mean_of_key_bucket_mean": float(per_key_df["bucket_mean"].mean()),
        "mean_of_key_update_over_bucket_mean": float(per_key_df["update_over_bucket_mean_changed_only"].mean()),
        "median_of_key_update_over_bucket_median": float(per_key_df["update_over_bucket_median_changed_only"].median()),
        "mean_of_key_shrinkage": float(per_key_df["mean_shrinkage_ratio_changed_only"].mean()),
        "median_of_key_shrinkage": float(per_key_df["median_shrinkage_ratio_changed_only"].median()),
    })

    return pd.DataFrame(rows)


# =========================================================
# plotting
# =========================================================
def save_hist_update_over_bucket(per_key_rows: List[dict], save_path: str, title: str):
    vals = []
    for r in per_key_rows:
        arr = r["_update_over_bucket_changed"]
        if arr.size > 0:
            vals.append(arr)

    if len(vals) == 0:
        return

    vals = np.concatenate(vals, axis=0)

    plt.figure(figsize=(8, 6))
    plt.hist(vals, bins=100)
    plt.xlabel(r"$|W_{erased}-W_{base}| / \Delta$")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_bar(df: pd.DataFrame, x_col: str, y_col: str, save_path: str, title: str):
    if len(df) == 0:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col], df[y_col])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--erased_fp", type=str, default="../models/uce_models/Barack_Obama_reg_25")
    parser.add_argument("--erased_q", type=str, default="ckpt/quant-UCE_barack_obama_reg_25_w8_weightonly_full-16-8-16-8.ckpt")
    parser.add_argument("--outdir", type=str, default="hypothesis/uce_reg_25_qbucket")
    parser.add_argument("--base_local_files_only", action="store_true")
    parser.add_argument("--erased_fp_local_files_only", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    print(f"Running on device: {DEVICE}")

    # -------------------------
    # load models
    # -------------------------
    print("\nLoading base FP model...")
    base_pipe = load_fp_model(
        model_path=args.base,
        local_files_only=args.base_local_files_only,
    )

    print("\nLoading erased FP32 model...")
    erased_fp_pipe = load_fp_model(
        model_path=args.erased_fp,
        local_files_only=args.erased_fp_local_files_only,
    )

    print("\nLoading quantized erased model...")
    erased_q_pipe, erased_q_layers = load_quantized_model(args.erased_q)
    print(f"Found {len(erased_q_layers)} quantized hub layers")

    # -------------------------
    # extract matched tensors
    # -------------------------
    print("\nExtracting canonicalized keys...")
    base_sd = get_filtered_fp_state_dict(base_pipe)
    erased_fp_sd = get_filtered_fp_state_dict(erased_fp_pipe)
    erased_q_sd = get_filtered_quant_state_dict(erased_q_pipe)

    print(f"base keys:      {len(base_sd)}")
    print(f"erased_fp keys: {len(erased_fp_sd)}")
    print(f"erased_q keys:  {len(erased_q_sd)}")

    common_keys = sorted(
        set(base_sd.keys()) &
        set(erased_fp_sd.keys()) &
        set(erased_q_sd.keys())
    )

    print(f"\nCommon keys to analyze: {len(common_keys)}")
    for k in common_keys:
        print(" ", k)

    if len(common_keys) == 0:
        debug_path = os.path.join(args.outdir, "debug_key_dump.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_keys": sorted(list(base_sd.keys())),
                    "erased_fp_keys": sorted(list(erased_fp_sd.keys())),
                    "erased_q_keys": sorted(list(erased_q_sd.keys())),
                    "quant_layer_keys": sorted(list(erased_q_layers.keys())),
                },
                f,
                indent=2,
            )
        print(f"No common keys found. Saved debug dump to {debug_path}")
        free_pipe(base_pipe)
        free_pipe(erased_fp_pipe)
        free_pipe(erased_q_pipe)
        return

    # -------------------------
    # analysis
    # -------------------------
    per_key_rows: List[dict] = []

    print("\nRunning bucket analysis...")
    for key_name in common_keys:
        row = analyze_quantized_bucket_behavior_for_key(
            key_name=key_name,
            w_base=base_sd[key_name],
            w_uce_fp=erased_fp_sd[key_name],
            w_quant_actual=erased_q_sd[key_name],
            quant_layers=erased_q_layers,
        )
        per_key_rows.append(row)

    per_key_df = pd.DataFrame(per_key_rows)
    summary_df = build_summary(per_key_df)

    # family summary
    family_df = (
        per_key_df.groupby("family", as_index=False)
        .agg({
            "pct_l2_change": "mean",
            "pct_scalar_changed": "mean",
            "update_over_bucket_mean_changed_only": "mean",
            "same_bucket_ratio_over_fp_changed": "mean",
            "cross_bucket_ratio_over_fp_changed": "mean",
            "collapse_to_base_q_ratio_over_fp_changed": "mean",
            "closer_to_uce_ratio_over_fp_changed": "mean",
            "closer_to_base_ratio_over_fp_changed": "mean",
        })
    )

    block_df = (
        per_key_df.groupby("block", as_index=False)
        .agg({
            "same_bucket_ratio_over_fp_changed": "mean",
            "cross_bucket_ratio_over_fp_changed": "mean",
            "collapse_to_base_q_ratio_over_fp_changed": "mean",
            "closer_to_uce_ratio_over_fp_changed": "mean",
            "closer_to_base_ratio_over_fp_changed": "mean",
        })
    )

    # -------------------------
    # save tables
    # -------------------------
    per_key_csv = os.path.join(args.outdir, "bucket_analysis_per_key.csv")
    summary_csv = os.path.join(args.outdir, "bucket_analysis_summary.csv")
    family_csv = os.path.join(args.outdir, "bucket_analysis_family.csv")
    block_csv = os.path.join(args.outdir, "bucket_analysis_block.csv")

    per_key_df.drop(columns=["_update_over_bucket_changed"]).to_csv(per_key_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    family_df.to_csv(family_csv, index=False)
    block_df.to_csv(block_csv, index=False)

    # -------------------------
    # save plots
    # -------------------------
    save_hist_update_over_bucket(
        per_key_rows,
        os.path.join(args.outdir, "hist_update_over_bucket.png"),
        "Histogram of FP erasure update / quantization bucket width"
    )

    save_bar(
        family_df,
        "family",
        "same_bucket_ratio_over_fp_changed",
        os.path.join(args.outdir, "family_same_bucket_ratio.png"),
        "Mean same-bucket ratio by family"
    )

    save_bar(
        family_df,
        "family",
        "collapse_to_base_q_ratio_over_fp_changed",
        os.path.join(args.outdir, "family_collapse_to_base_ratio.png"),
        "Mean collapse-to-base ratio by family"
    )

    save_bar(
        block_df,
        "block",
        "same_bucket_ratio_over_fp_changed",
        os.path.join(args.outdir, "block_same_bucket_ratio.png"),
        "Mean same-bucket ratio by block"
    )

    save_bar(
        block_df,
        "block",
        "collapse_to_base_q_ratio_over_fp_changed",
        os.path.join(args.outdir, "block_collapse_to_base_ratio.png"),
        "Mean collapse-to-base ratio by block"
    )

    # -------------------------
    # print summary
    # -------------------------
    print("\nSaved:")
    print(" ", per_key_csv)
    print(" ", summary_csv)
    print(" ", family_csv)
    print(" ", block_csv)
    print(" ", os.path.join(args.outdir, "hist_update_over_bucket.png"))
    print(" ", os.path.join(args.outdir, "family_same_bucket_ratio.png"))
    print(" ", os.path.join(args.outdir, "family_collapse_to_base_ratio.png"))
    print(" ", os.path.join(args.outdir, "block_same_bucket_ratio.png"))
    print(" ", os.path.join(args.outdir, "block_collapse_to_base_ratio.png"))

    print("\nGlobal summary:")
    print(summary_df.to_string(index=False))

    print("\nPer-key summary:")
    print(
        per_key_df.drop(columns=["_update_over_bucket_changed"])[[
            "key",
            "pct_l2_change",
            "pct_scalar_changed",
            "bucket_mean",
            "update_over_bucket_mean_changed_only",
            "same_bucket_ratio_over_fp_changed",
            "collapse_to_base_q_ratio_over_fp_changed",
            "closer_to_uce_ratio_over_fp_changed",
            "closer_to_base_ratio_over_fp_changed",
            "qparam_source",
        ]].to_string(index=False)
    )

    # -------------------------
    # cleanup
    # -------------------------
    free_pipe(base_pipe)
    free_pipe(erased_fp_pipe)
    free_pipe(erased_q_pipe)


if __name__ == "__main__":
    main()