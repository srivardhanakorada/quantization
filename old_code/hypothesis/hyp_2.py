import os
import gc
import json
from typing import Dict, List, Tuple

import torch
import pandas as pd
from diffusers import StableDiffusionPipeline


# ============================================================
# Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
BASE_MODEL_LOCAL_FILES_ONLY = False
OUTPUT_DIR = "analysis_results_h2"

models_to_run = [
    {
        "name": "base_fp",
        "type": "fp",
        "path": "runwayml/stable-diffusion-v1-5",
        "local_files_only": BASE_MODEL_LOCAL_FILES_ONLY,
    },
    {
        "name": "uce_fp",
        "type": "fp",
        "path": "../models/uce_models/Barack_Obama",
        "local_files_only": True,
    },
]

FP_CHANGE_EPS = 1e-12


# ============================================================
# Utilities
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


def is_weight_key(k: str) -> bool:
    # Analyze only weight tensors, not biases/norm stats/etc.
    return k.endswith(".weight")


def get_unet_weight_state_dict(pipe) -> Dict[str, torch.Tensor]:
    sd = pipe.unet.state_dict()
    out = {}
    for k, v in sd.items():
        if is_weight_key(k):
            out[k] = v.detach().float().cpu().clone()
    return out


# ============================================================
# Key parsing
# ============================================================
def get_stage_from_key(k: str) -> str:
    if k.startswith("down_blocks."):
        return "down"
    if k.startswith("mid_block."):
        return "mid"
    if k.startswith("up_blocks."):
        return "up"
    return "other"


def get_block_id_from_key(k: str) -> str:
    parts = k.split(".")
    if len(parts) >= 2 and parts[0] in {"down_blocks", "up_blocks"}:
        return f"{parts[0]}.{parts[1]}"
    if parts[0] == "mid_block":
        return "mid_block"
    return "other"


def get_family_from_key(k: str) -> str:
    # Most important first
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
    if ".time_emb_proj.weight" in k:
        return "time_emb_proj"
    if ".time_embedding." in k:
        return "time_embedding"
    if ".conv1.weight" in k:
        return "resnet_conv1"
    if ".conv2.weight" in k:
        return "resnet_conv2"
    if ".to_out." in k:
        return "attn_other_to_out"
    if ".to_q.weight" in k or ".to_k.weight" in k or ".to_v.weight" in k:
        return "attn_other_qkv"
    return "other"


def get_attn_group_from_key(k: str) -> str:
    if ".attn2." in k:
        return "cross_attention"
    if ".attn1." in k:
        return "self_attention"
    return "non_attention"


# ============================================================
# Core analysis
# ============================================================
def analyze_base_vs_uce(
    base_sd: Dict[str, torch.Tensor],
    uce_sd: Dict[str, torch.Tensor],
) -> pd.DataFrame:
    common_keys = sorted(set(base_sd.keys()) & set(uce_sd.keys()))
    rows = []

    for k in common_keys:
        w_base = base_sd[k]
        w_uce = uce_sd[k]

        if w_base.shape != w_uce.shape:
            continue

        delta = w_uce - w_base
        abs_delta = delta.abs()

        numel = int(delta.numel())
        changed_mask = abs_delta > FP_CHANGE_EPS
        changed_count = int(changed_mask.sum().item())

        l1_delta = float(abs_delta.sum().item())
        l2_delta_sq = float((delta ** 2).sum().item())
        l2_delta = float(torch.sqrt((delta ** 2).sum()).item()) if numel > 0 else 0.0
        linf_delta = float(abs_delta.max().item()) if numel > 0 else 0.0
        mean_abs_delta = float(abs_delta.mean().item()) if numel > 0 else 0.0

        rows.append({
            "key": k,
            "stage": get_stage_from_key(k),
            "block_id": get_block_id_from_key(k),
            "family": get_family_from_key(k),
            "attn_group": get_attn_group_from_key(k),
            "shape": str(tuple(w_base.shape)),
            "numel": numel,
            "changed_count": changed_count,
            "changed_ratio": changed_count / numel if numel > 0 else 0.0,
            "l1_delta": l1_delta,
            "l2_delta_sq": l2_delta_sq,
            "l2_delta": l2_delta,
            "linf_delta": linf_delta,
            "mean_abs_delta": mean_abs_delta,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    total_l1 = df["l1_delta"].sum()
    total_l2_sq = df["l2_delta_sq"].sum()

    if total_l1 > 0:
        df["l1_share_global"] = df["l1_delta"] / total_l1
    else:
        df["l1_share_global"] = 0.0

    if total_l2_sq > 0:
        df["l2_energy_share_global"] = df["l2_delta_sq"] / total_l2_sq
    else:
        df["l2_energy_share_global"] = 0.0

    return df


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    grouped = df.groupby(group_col).agg({
        "key": "count",
        "numel": "sum",
        "changed_count": "sum",
        "l1_delta": "sum",
        "l2_delta_sq": "sum",
        "l2_delta": "mean",
        "linf_delta": "max",
        "mean_abs_delta": "mean",
    }).reset_index()

    grouped = grouped.rename(columns={"key": "num_tensors"})
    grouped["changed_ratio_weighted"] = grouped["changed_count"] / grouped["numel"]

    total_l1 = grouped["l1_delta"].sum()
    total_l2_sq = grouped["l2_delta_sq"].sum()

    grouped["l1_share_global"] = grouped["l1_delta"] / total_l1 if total_l1 > 0 else 0.0
    grouped["l2_energy_share_global"] = grouped["l2_delta_sq"] / total_l2_sq if total_l2_sq > 0 else 0.0

    grouped = grouped.sort_values("l1_share_global", ascending=False).reset_index(drop=True)
    return grouped


def build_headline_metrics(
    per_tensor_df: pd.DataFrame,
    family_df: pd.DataFrame,
    stage_df: pd.DataFrame,
) -> dict:
    def get_share(df, col_name, group_col, group_value):
        hit = df[df[group_col] == group_value]
        if len(hit) == 0:
            return 0.0
        return float(hit.iloc[0][col_name])

    attn2_k_l1 = get_share(family_df, "l1_share_global", "family", "attn2_to_k")
    attn2_v_l1 = get_share(family_df, "l1_share_global", "family", "attn2_to_v")
    attn2_k_l2 = get_share(family_df, "l2_energy_share_global", "family", "attn2_to_k")
    attn2_v_l2 = get_share(family_df, "l2_energy_share_global", "family", "attn2_to_v")

    cross_attn_df = summarize_group(per_tensor_df, "attn_group")
    cross_l1 = get_share(cross_attn_df, "l1_share_global", "attn_group", "cross_attention")
    self_l1 = get_share(cross_attn_df, "l1_share_global", "attn_group", "self_attention")
    non_attn_l1 = get_share(cross_attn_df, "l1_share_global", "attn_group", "non_attention")

    down_l1 = get_share(stage_df, "l1_share_global", "stage", "down")
    mid_l1 = get_share(stage_df, "l1_share_global", "stage", "mid")
    up_l1 = get_share(stage_df, "l1_share_global", "stage", "up")

    top5 = (
        per_tensor_df.sort_values("l1_share_global", ascending=False)
        .head(5)[["key", "family", "stage", "l1_share_global", "l2_energy_share_global"]]
        .to_dict(orient="records")
    )

    return {
        "share_l1_attn2_to_k": attn2_k_l1,
        "share_l1_attn2_to_v": attn2_v_l1,
        "share_l1_attn2_to_k_plus_v": attn2_k_l1 + attn2_v_l1,
        "share_l2_attn2_to_k": attn2_k_l2,
        "share_l2_attn2_to_v": attn2_v_l2,
        "share_l2_attn2_to_k_plus_v": attn2_k_l2 + attn2_v_l2,
        "share_l1_cross_attention_total": cross_l1,
        "share_l1_self_attention_total": self_l1,
        "share_l1_non_attention_total": non_attn_l1,
        "share_l1_down": down_l1,
        "share_l1_mid": mid_l1,
        "share_l1_up": up_l1,
        "top5_tensors_by_l1_share": top5,
    }


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loaded = {}

    print(f"Running on device: {DEVICE}")
    print("Loading models...")

    for cfg in models_to_run:
        print(f"\nLoading {cfg['name']} from {cfg['path']}")
        pipe = load_fp_model(
            model_path=cfg["path"],
            local_files_only=cfg.get("local_files_only", True),
        )
        loaded[cfg["name"]] = pipe

    print("\nExtracting UNet weight state_dicts...")
    base_sd = get_unet_weight_state_dict(loaded["base_fp"])
    uce_sd = get_unet_weight_state_dict(loaded["uce_fp"])

    print(f"base weight keys: {len(base_sd)}")
    print(f"uce  weight keys: {len(uce_sd)}")

    common_keys = sorted(set(base_sd.keys()) & set(uce_sd.keys()))
    print(f"common weight keys: {len(common_keys)}")

    if len(common_keys) == 0:
        raise RuntimeError("No common UNet weight keys found between base_fp and uce_fp.")

    print("\nRunning H2 localization analysis...")
    per_tensor_df = analyze_base_vs_uce(base_sd, uce_sd)

    if len(per_tensor_df) == 0:
        raise RuntimeError("Per-tensor delta table is empty.")

    family_df = summarize_group(per_tensor_df, "family")
    stage_df = summarize_group(per_tensor_df, "stage")
    attn_group_df = summarize_group(per_tensor_df, "attn_group")
    block_df = summarize_group(per_tensor_df, "block_id")

    top_tensors_df = per_tensor_df.sort_values("l1_share_global", ascending=False).reset_index(drop=True)

    headline = build_headline_metrics(per_tensor_df, family_df, stage_df)

    # Save outputs
    per_tensor_csv = os.path.join(OUTPUT_DIR, "h2_per_tensor_delta.csv")
    family_csv = os.path.join(OUTPUT_DIR, "h2_family_summary.csv")
    stage_csv = os.path.join(OUTPUT_DIR, "h2_stage_summary.csv")
    attn_group_csv = os.path.join(OUTPUT_DIR, "h2_attn_group_summary.csv")
    block_csv = os.path.join(OUTPUT_DIR, "h2_block_summary.csv")
    top_tensors_csv = os.path.join(OUTPUT_DIR, "h2_top_tensors.csv")
    headline_json = os.path.join(OUTPUT_DIR, "h2_headline_metrics.json")

    per_tensor_df.to_csv(per_tensor_csv, index=False)
    family_df.to_csv(family_csv, index=False)
    stage_df.to_csv(stage_csv, index=False)
    attn_group_df.to_csv(attn_group_csv, index=False)
    block_df.to_csv(block_csv, index=False)
    top_tensors_df.to_csv(top_tensors_csv, index=False)

    with open(headline_json, "w", encoding="utf-8") as f:
        json.dump(headline, f, indent=2)

    print("\nSaved:")
    print(" ", per_tensor_csv)
    print(" ", family_csv)
    print(" ", stage_csv)
    print(" ", attn_group_csv)
    print(" ", block_csv)
    print(" ", top_tensors_csv)
    print(" ", headline_json)

    print("\n=== Family summary (top by L1 share) ===")
    print(family_df.to_string(index=False))

    print("\n=== Stage summary ===")
    print(stage_df.to_string(index=False))

    print("\n=== Attention-group summary ===")
    print(attn_group_df.to_string(index=False))

    print("\n=== Headline metrics ===")
    print(json.dumps(headline, indent=2))

    for pipe in loaded.values():
        free_pipe(pipe)


if __name__ == "__main__":
    main()