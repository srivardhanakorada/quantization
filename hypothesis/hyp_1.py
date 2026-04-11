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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
BASE_MODEL_LOCAL_FILES_ONLY = False
OUTPUT_DIR = "analysis_results_h1"

FP_CHANGE_EPS = 1e-12
Q_BASE_EPS = 1e-12

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
        "name": "uce_fp",
        "type": "fp",
        "path": "../models/uce_models/Barack_Obama",
        "local_files_only": True,
    },
    {
        "name": "uce_int8",
        "type": "quant",
        "path": "ckpt/quant-UCE_barack_obama_w8_weightonly_full-16-8-16-8.ckpt",
    },
    {
        "name": "uce_int4",
        "type": "quant",
        "path": "ckpt/quant-UCE_barack_obama_w4_weightonly_full-16-4-16-4.ckpt",
    },
]


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


def analyze_quantized_survival_for_key(
    key_name: str,
    w_base: torch.Tensor,
    w_uce_fp: torch.Tensor,
    w_quant: torch.Tensor,
    quant_tag: str,
) -> dict:
    fp_delta = w_uce_fp - w_base
    fp_abs_delta = fp_delta.abs()

    fp_changed_mask = fp_abs_delta > FP_CHANGE_EPS
    fp_changed_count = safe_sum_bool(fp_changed_mask)
    total_count = int(w_base.numel())

    q_delta_to_base = w_quant - w_base
    q_abs_delta_to_base = q_delta_to_base.abs()

    dist_to_base = (w_quant - w_base).abs()
    dist_to_uce = (w_quant - w_uce_fp).abs()

    fp_delta_changed = fp_delta[fp_changed_mask]
    fp_abs_delta_changed = fp_abs_delta[fp_changed_mask]
    q_delta_changed = q_delta_to_base[fp_changed_mask]
    q_abs_delta_changed = q_abs_delta_to_base[fp_changed_mask]
    dist_to_base_changed = dist_to_base[fp_changed_mask]
    dist_to_uce_changed = dist_to_uce[fp_changed_mask]

    q_survival_mask = q_abs_delta_changed > Q_BASE_EPS
    q_survival_count = safe_sum_bool(q_survival_mask)

    closer_to_uce_mask = dist_to_uce_changed < dist_to_base_changed
    closer_to_base_mask = dist_to_base_changed < dist_to_uce_changed
    tie_mask = dist_to_base_changed == dist_to_uce_changed

    closer_to_uce_count = safe_sum_bool(closer_to_uce_mask)
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

    return {
        "quant_model": quant_tag,
        "key": key_name,
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
        "closer_to_uce_count": closer_to_uce_count,
        "closer_to_base_count": closer_to_base_count,
        "tie_count": tie_count,
        "closer_to_uce_ratio_over_fp_changed": ratio(closer_to_uce_count, fp_changed_count),
        "closer_to_base_ratio_over_fp_changed": ratio(closer_to_base_count, fp_changed_count),
        "tie_ratio_over_fp_changed": ratio(tie_count, fp_changed_count),
        "same_sign_count": same_sign_count,
        "same_sign_ratio_over_fp_changed": ratio(same_sign_count, fp_changed_count),
        "mean_shrinkage_ratio_changed_only": safe_mean(shrinkage_vals),
        "median_shrinkage_ratio_changed_only": float(
            torch.median(shrinkage_vals).item()
        ) if shrinkage_vals.numel() > 0 else 0.0,
    }


def build_summary(per_key_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quant_model in sorted(per_key_df["quant_model"].unique()):
        dfm = per_key_df[per_key_df["quant_model"] == quant_model].copy()

        total_fp_changed = int(dfm["fp_changed_count"].sum())
        total_numel = int(dfm["numel"].sum())
        total_q_survival = int(dfm["q_survival_count"].sum())
        total_closer_to_uce = int(dfm["closer_to_uce_count"].sum())
        total_closer_to_base = int(dfm["closer_to_base_count"].sum())
        total_tie = int(dfm["tie_count"].sum())
        total_same_sign = int(dfm["same_sign_count"].sum())

        rows.append({
            "quant_model": quant_model,
            "keys_analyzed": int(len(dfm)),
            "total_numel": total_numel,
            "total_fp_changed": total_fp_changed,
            "total_fp_changed_ratio": ratio(total_fp_changed, total_numel),
            "weighted_q_survival_ratio_over_fp_changed": ratio(total_q_survival, total_fp_changed),
            "weighted_closer_to_uce_ratio_over_fp_changed": ratio(total_closer_to_uce, total_fp_changed),
            "weighted_closer_to_base_ratio_over_fp_changed": ratio(total_closer_to_base, total_fp_changed),
            "weighted_tie_ratio_over_fp_changed": ratio(total_tie, total_fp_changed),
            "weighted_same_sign_ratio_over_fp_changed": ratio(total_same_sign, total_fp_changed),
            "mean_of_key_fp_mean_abs_delta": float(dfm["fp_mean_abs_delta"].mean()),
            "mean_of_key_fp_mean_abs_delta_changed_only": float(dfm["fp_mean_abs_delta_changed_only"].mean()),
            "mean_of_key_q_mean_abs_delta_to_base_changed_only": float(dfm["q_mean_abs_delta_to_base_changed_only"].mean()),
            "mean_of_key_shrinkage": float(dfm["mean_shrinkage_ratio_changed_only"].mean()),
            "median_of_key_shrinkage": float(dfm["median_shrinkage_ratio_changed_only"].median()),
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loaded = {}
    quant_maps = {}

    print(f"Running on device: {DEVICE}")
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
            loaded[name] = pipe
            quant_maps[name] = None
        else:
            pipe, qlayers = load_quantized_model(path)
            loaded[name] = pipe
            quant_maps[name] = qlayers
            print(f"{name}: found {len(qlayers)} quantized layers")

    print("\nExtracting canonicalized keys...")
    base_sd = get_filtered_fp_state_dict(loaded["base_fp"])
    uce_fp_sd = get_filtered_fp_state_dict(loaded["uce_fp"])
    uce_int8_sd = get_filtered_quant_state_dict(loaded["uce_int8"])
    uce_int4_sd = get_filtered_quant_state_dict(loaded["uce_int4"])

    print(f"base keys:    {len(base_sd)}")
    print(f"uce_fp keys:  {len(uce_fp_sd)}")
    print(f"uce_int8 keys:{len(uce_int8_sd)}")
    print(f"uce_int4 keys:{len(uce_int4_sd)}")

    common_keys = sorted(
        set(base_sd.keys())
        & set(uce_fp_sd.keys())
        & set(uce_int8_sd.keys())
        & set(uce_int4_sd.keys())
    )

    print(f"\nCommon keys to analyze: {len(common_keys)}")
    for k in common_keys:
        print(" ", k)

    if len(common_keys) == 0:
        debug_path = os.path.join(OUTPUT_DIR, "debug_key_dump.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_keys": sorted(list(base_sd.keys())),
                    "uce_fp_keys": sorted(list(uce_fp_sd.keys())),
                    "uce_int8_keys": sorted(list(uce_int8_sd.keys())),
                    "uce_int4_keys": sorted(list(uce_int4_sd.keys())),
                },
                f,
                indent=2,
            )
        print(f"No common keys found. Saved debug dump to {debug_path}")
        return

    per_key_rows: List[dict] = []

    print("\nAnalyzing INT8...")
    for key_name in common_keys:
        per_key_rows.append(
            analyze_quantized_survival_for_key(
                key_name=key_name,
                w_base=base_sd[key_name],
                w_uce_fp=uce_fp_sd[key_name],
                w_quant=uce_int8_sd[key_name],
                quant_tag="uce_int8",
            )
        )

    print("Analyzing INT4...")
    for key_name in common_keys:
        per_key_rows.append(
            analyze_quantized_survival_for_key(
                key_name=key_name,
                w_base=base_sd[key_name],
                w_uce_fp=uce_fp_sd[key_name],
                w_quant=uce_int4_sd[key_name],
                quant_tag="uce_int4",
            )
        )

    per_key_df = pd.DataFrame(per_key_rows)
    summary_df = build_summary(per_key_df)

    per_key_csv = os.path.join(OUTPUT_DIR, "h1_quant_survival_per_key.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "h1_quant_survival_summary.csv")

    per_key_df.to_csv(per_key_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved:")
    print(" ", per_key_csv)
    print(" ", summary_csv)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    for pipe in loaded.values():
        free_pipe(pipe)


if __name__ == "__main__":
    main()