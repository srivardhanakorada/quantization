#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from typing import Dict, Optional, Tuple, List

import torch
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quantization_tools.quantization.layers  # noqa: F401
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub
from diffusers import StableDiffusionPipeline, DiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
EPS = 1e-12

TARGET_SUFFIXES_FP = (
    "attn2.to_k.weight",
    "attn2.to_v.weight",
)

TARGET_SUFFIXES_QUANT = (
    "attn2.to_k.core.weight",
    "attn2.to_v.core.weight",
)


def is_torch_ckpt(path: str) -> bool:
    return path.endswith((".ckpt", ".pt", ".pth", ".bin"))


def load_fp_model(path: str):
    if os.path.exists(path) and is_torch_ckpt(path):
        pipe = torch.load(path, map_location="cpu").to(DEVICE)
    else:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                path,
                safety_checker=None,
                torch_dtype=DTYPE,
                local_files_only=True,
            ).to(DEVICE)
        except Exception:
            pipe = DiffusionPipeline.from_pretrained(
                path,
                safety_checker=None,
                torch_dtype=DTYPE,
                local_files_only=True,
            ).to(DEVICE)
    pipe.unet.eval()
    return pipe


def load_quant_model(path: str):
    pipe = torch.load(path, map_location="cpu").to(DEVICE)
    pipe.unet.eval()
    layers_linear = find_layers(pipe.unet, (LinearQuantHub,))
    layers_conv = find_layers(pipe.unet, (Conv2dQuantHub,))
    return pipe, {**layers_linear, **layers_conv}


def canonicalize_quant_key(k: str) -> str:
    k = k.replace(".transformer_blocks.0.core.attn2.", ".transformer_blocks.0.attn2.")
    k = k.replace(".to_k.core.weight", ".to_k.weight")
    k = k.replace(".to_v.core.weight", ".to_v.weight")
    return k


def canonical_key_to_quanthub_key(canonical_key: str) -> str:
    base = canonical_key.replace(".weight", "")
    base = base.replace(".transformer_blocks.0.attn2.", ".transformer_blocks.0.core.attn2.")
    return base


def get_fp_sd(pipe) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in pipe.unet.state_dict().items():
        if any(k.endswith(s) for s in TARGET_SUFFIXES_FP):
            out[k] = v.detach().float().cpu()
    return out


def get_quant_sd(pipe) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in pipe.unet.state_dict().items():
        if any(k.endswith(s) for s in TARGET_SUFFIXES_QUANT):
            out[canonicalize_quant_key(k)] = v.detach().float().cpu()
    return out


def get_qparams_for_key(
    canonical_key: str,
    quant_layers: Dict[str, torch.nn.Module],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
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
    if scale is None or zero_point is None:
        return None, None

    if scale.numel() == 1:
        scale_b = scale.reshape(1, 1).expand_as(w)
        zp_b = zero_point.reshape(1, 1).expand_as(w)
        return scale_b, zp_b

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


def quantize_to_indices(
    w: torch.Tensor,
    scale_b: torch.Tensor,
    zp_b: torch.Tensor,
):
    q_idx = torch.round(w / scale_b + zp_b)
    q_idx = torch.clamp(q_idx, 0, 255)
    return q_idx


def _estimate_plot_max(vals_cpu: torch.Tensor) -> float:
    n = vals_cpu.numel()
    sample_size = min(2_000_000, n)

    if n > sample_size:
        idx = torch.randperm(n)[:sample_size]
        vals_for_q = vals_cpu[idx]
    else:
        vals_for_q = vals_cpu

    vals_for_q = vals_for_q.sort().values
    q99_idx = int(0.99 * (vals_for_q.numel() - 1))
    q99 = float(vals_for_q[q99_idx].item())
    return max(2.0, q99)


def save_delta_histogram(
    ratio_vals: torch.Tensor,
    out_path: str,
    bins: int = 120,
    max_x: Optional[float] = None,
):
    vals_cpu = ratio_vals.detach().float().cpu().reshape(-1)

    frac_le_1 = float((vals_cpu <= 1.0).float().mean().item())
    frac_gt_1 = float((vals_cpu > 1.0).float().mean().item())

    if max_x is None:
        max_x = _estimate_plot_max(vals_cpu)

    vals_np = vals_cpu.numpy()
    vals_clipped = vals_np[vals_np <= max_x]

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(vals_clipped, bins=bins, edgecolor="black")
    plt.axvline(1.0, linestyle="--", linewidth=1.8)

    ymax = plt.gca().get_ylim()[1]

    plt.text(
        0.05 * max_x,
        0.90 * ymax,
        rf"$|\Delta W|/\delta \leq 1$: {100.0 * frac_le_1:.1f}%",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )
    plt.text(
        0.52 * max_x,
        0.90 * ymax,
        rf"$|\Delta W|/\delta > 1$: {100.0 * frac_gt_1:.1f}%",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    plt.xlabel(r"$|\Delta W| / \delta$")
    plt.ylabel("Count")
    plt.title("Distribution of normalized update magnitudes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "fraction_le_1": frac_le_1,
        "fraction_gt_1": frac_gt_1,
        "hist_max_x": max_x,
    }


def save_threshold_bar(ratio_vals: torch.Tensor, out_path: str):
    vals_cpu = ratio_vals.detach().float().cpu().reshape(-1)
    frac_le_1 = float((vals_cpu <= 1.0).float().mean().item())
    frac_gt_1 = float((vals_cpu > 1.0).float().mean().item())

    labels = [r"$|\Delta W|/\delta \leq 1$", r"$|\Delta W|/\delta > 1$"]
    values = [100.0 * frac_le_1, 100.0 * frac_gt_1]

    plt.figure(figsize=(6.0, 4.4))
    bars = plt.bar(labels, values, edgecolor="black")
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.ylabel("Percentage of changed weights")
    plt.ylim(0, 100)
    plt.title("Share of updates relative to one bucket width")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "fraction_le_1": frac_le_1,
        "fraction_gt_1": frac_gt_1,
    }


def save_bucket_bar(
    same_bucket_count: int,
    cross_bucket_count: int,
    out_path: str,
):
    labels = ["Same bucket", "Cross bucket"]
    values = [same_bucket_count, cross_bucket_count]

    plt.figure(figsize=(5.8, 4.5))
    bars = plt.bar(labels, values, edgecolor="black")
    total = max(1, same_bucket_count + cross_bucket_count)

    for bar, val in zip(bars, values):
        pct = 100.0 * val / total
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("Count")
    plt.title("Bucket transition statistics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fp", required=True, help="Base full-precision model/checkpoint")
    parser.add_argument("--erased_fp", required=True, help="Erased full-precision model/checkpoint")
    parser.add_argument("--base_q8", required=True, help="Base quantized checkpoint")
    parser.add_argument("--erased_q8", required=True, help="Erased quantized checkpoint")
    parser.add_argument("--out_dir", default="quant_erasure_figs", help="Directory to save outputs")
    parser.add_argument("--hist_bins", type=int, default=120, help="Number of bins for histogram")
    parser.add_argument("--hist_max_x", type=float, default=None, help="Optional x-axis max for histogram")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading FP base model...")
    base_fp_pipe = load_fp_model(args.base_fp)

    print("[INFO] Loading FP erased model...")
    erased_fp_pipe = load_fp_model(args.erased_fp)

    print("[INFO] Loading quantized base model...")
    base_q8_pipe, base_q8_layers = load_quant_model(args.base_q8)

    print("[INFO] Loading quantized erased model...")
    erased_q8_pipe, erased_q8_layers = load_quant_model(args.erased_q8)

    print("[INFO] Extracting target attention weights...")
    base_fp = get_fp_sd(base_fp_pipe)
    erased_fp = get_fp_sd(erased_fp_pipe)
    base_q8 = get_quant_sd(base_q8_pipe)
    erased_q8 = get_quant_sd(erased_q8_pipe)

    common = sorted(set(base_fp) & set(erased_fp) & set(base_q8) & set(erased_q8))
    if len(common) == 0:
        raise RuntimeError("No common attn2.to_k / attn2.to_v keys found")

    total_numel = 0
    total_fp_changed = 0
    total_cross_bucket = 0
    total_closer_to_erased = 0
    ratio_vals: List[torch.Tensor] = []

    for k in common:
        w_base = base_fp[k]
        w_erased = erased_fp[k]
        w_q_erased = erased_q8[k]

        fp_delta = w_erased - w_base
        fp_changed_mask = fp_delta.abs() > EPS

        total_numel += w_base.numel()
        fp_changed_count = int(fp_changed_mask.sum().item())
        total_fp_changed += fp_changed_count

        if fp_changed_count == 0:
            continue

        dist_to_base = (w_q_erased - w_base).abs()[fp_changed_mask]
        dist_to_erased = (w_q_erased - w_erased).abs()[fp_changed_mask]
        total_closer_to_erased += int((dist_to_erased < dist_to_base).sum().item())

        scale, zero_point, source = get_qparams_for_key(k, erased_q8_layers)
        if scale is None or zero_point is None:
            print(f"[WARN] Skipping qparam-dependent stats for {k} ({source})")
            continue

        scale_b, zp_b = broadcast_qparams_like_weight(w_erased, scale, zero_point)
        if scale_b is None or zp_b is None:
            print(f"[WARN] Could not broadcast qparams for {k}")
            continue

        ratio_vals.append(
            (fp_delta.abs()[fp_changed_mask] / scale_b[fp_changed_mask].clamp_min(1e-12)).reshape(-1)
        )

        qidx_base = quantize_to_indices(w_base, scale_b, zp_b)
        qidx_erased = quantize_to_indices(w_erased, scale_b, zp_b)
        same_bucket = (qidx_base[fp_changed_mask] == qidx_erased[fp_changed_mask])
        total_cross_bucket += int((~same_bucket).sum().item())

    if total_fp_changed == 0:
        raise RuntimeError("No FP-changed weights found in target UCE layers")

    if len(ratio_vals) == 0:
        raise RuntimeError("Could not recover any quantizer scales from erased q8 checkpoint")

    ratio_vals = torch.cat(ratio_vals)

    fp_changed_ratio = total_fp_changed / total_numel
    mean_abs_delta_over_delta = float(ratio_vals.mean().item())
    cross_bucket_ratio = total_cross_bucket / total_fp_changed
    closer_to_erased_ratio = total_closer_to_erased / total_fp_changed
    same_bucket_count = total_fp_changed - total_cross_bucket

    frac_le_1 = float((ratio_vals <= 1.0).float().mean().item())
    frac_gt_1 = float((ratio_vals > 1.0).float().mean().item())

    print("\nUSE THESE NUMBERS IN THE PAPER:\n")
    print(f"FP changed ratio         : {fp_changed_ratio:.6f}")
    print(f"Mean |ΔW| / δ            : {mean_abs_delta_over_delta:.6f}")
    print(f"Cross-bucket ratio       : {cross_bucket_ratio:.6f}")
    print(f"Closer-to-erased ratio   : {closer_to_erased_ratio:.6f}")
    print(f"Fraction |ΔW|/δ <= 1     : {frac_le_1:.6f}")
    print(f"Fraction |ΔW|/δ > 1      : {frac_gt_1:.6f}")

    metrics_json = {
        "fp_changed_ratio": fp_changed_ratio,
        "mean_abs_delta_over_delta": mean_abs_delta_over_delta,
        "cross_bucket_ratio": cross_bucket_ratio,
        "closer_to_erased_ratio": closer_to_erased_ratio,
        "fraction_le_1": frac_le_1,
        "fraction_gt_1": frac_gt_1,
        "same_bucket_count": int(same_bucket_count),
        "cross_bucket_count": int(total_cross_bucket),
        "total_fp_changed": int(total_fp_changed),
        "total_numel": int(total_numel),
        "num_ratio_values": int(ratio_vals.numel()),
    }

    metrics_path = os.path.join(args.out_dir, "uce_quant_exact_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)

    ratio_path = os.path.join(args.out_dir, "delta_over_delta_values.pt")
    torch.save(ratio_vals.cpu(), ratio_path)

    hist_path = os.path.join(args.out_dir, "delta_histogram.png")
    threshold_bar_path = os.path.join(args.out_dir, "delta_threshold_bar.png")
    bucket_bar_path = os.path.join(args.out_dir, "bucket_bar.png")

    print(f"[INFO] Saving histogram to: {hist_path}")
    hist_stats = save_delta_histogram(
        ratio_vals=ratio_vals,
        out_path=hist_path,
        bins=args.hist_bins,
        max_x=args.hist_max_x,
    )

    print(f"[INFO] Saving threshold summary bar plot to: {threshold_bar_path}")
    threshold_stats = save_threshold_bar(
        ratio_vals=ratio_vals,
        out_path=threshold_bar_path,
    )

    print(f"[INFO] Saving bucket bar plot to: {bucket_bar_path}")
    save_bucket_bar(
        same_bucket_count=same_bucket_count,
        cross_bucket_count=total_cross_bucket,
        out_path=bucket_bar_path,
    )

    metrics_json.update(hist_stats)
    metrics_json.update(threshold_stats)

    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"[INFO] Saved metrics JSON to: {metrics_path}")
    print(f"[INFO] Saved raw ratio values to: {ratio_path}")
    print("\nDONE.")


if __name__ == "__main__":
    main()