#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

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


def get_quant_sd(pipe) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in pipe.unet.state_dict().items():
        if any(k.endswith(s) for s in TARGET_SUFFIXES_QUANT):
            out[canonicalize_quant_key(k)] = v.detach().float().cpu()
    return out


def get_qparams_for_key(canonical_key: str, quant_layers: Dict[str, torch.nn.Module]):
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


def quantize_from_qparams(
    w: torch.Tensor,
    scale_b: torch.Tensor,
    zp_b: torch.Tensor,
):
    q_idx = torch.round(w / scale_b + zp_b)
    q_idx = torch.clamp(q_idx, 0, 255)
    wq = (q_idx - zp_b) * scale_b
    return wq


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = a.norm(p=2) * b.norm(p=2)
    if denom.item() < EPS:
        return 1.0
    return float(torch.dot(a, b).item() / denom.item())


def relative_frobenius_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).norm(p=2)
    den = b.norm(p=2).clamp_min(EPS)
    return float((num / den).item())


def shorten_layer_name(k: str) -> str:
    return (
        k.replace(".attentions.", ".attn.")
         .replace(".transformer_blocks.0.", "")
         .replace(".attn2.to_k.weight", ".to_k")
         .replace(".attn2.to_v.weight", ".to_v")
    )


def save_endpoint_similarity_plot(
    layer_names: List[str],
    cosine_vals: List[float],
    out_path: str,
):
    import numpy as np

    cosine_vals = np.array(cosine_vals, dtype=float)
    xs = np.arange(len(cosine_vals))   # layer indices: 0, 1, 2, ...

    y_min = max(0.0, float(cosine_vals.min()) - 0.002)
    y_max = min(1.0005, float(cosine_vals.max()) + 0.0003)

    plt.figure(figsize=(10.5, 4.8))
    plt.plot(xs, cosine_vals, marker='o', linewidth=2)

    plt.ylim(y_min, y_max)
    plt.xlabel("Layer index")
    plt.ylabel("Cosine similarity")
    plt.title(r"Endpoint alignment between $Q(W_e)$ and $W_q^{(e)}$")
    # plt.grid(True, axis='y', linestyle='--', alpha=0.4)

    mean_cos = float(np.mean(cosine_vals))
    min_cos = float(np.min(cosine_vals))
    plt.axhline(mean_cos, linestyle='--', linewidth=1.5, alpha=0.8, label=f"Mean = {mean_cos:.6f}")
    plt.axhline(min_cos, linestyle=':', linewidth=1.5, alpha=0.8, label=f"Min = {min_cos:.6f}")
    plt.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()

def save_endpoint_deviation_plot(
    layer_names: List[str],
    cosine_vals: List[float],
    out_path: str,
):
    import numpy as np

    cosine_vals = np.array(cosine_vals, dtype=float)
    deviation = 1.0 - cosine_vals
    xs = np.arange(len(cosine_vals))   # original layer order

    plt.figure(figsize=(10.5, 4.8))
    plt.plot(xs, deviation, marker='o', linewidth=2)

    plt.xlabel("Layer index")
    plt.ylabel(r"$1 -$ cosine similarity")
    plt.title(r"Deviation from perfect alignment")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)

    if np.all(deviation > 0):
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--erased_fp", required=True, help="Full-precision erased model/checkpoint")
    parser.add_argument("--erased_q8", required=True, help="Quantize-then-erase quantized checkpoint")
    parser.add_argument("--base_q8", required=True, help="Base quantized checkpoint used to provide quantizer params")
    parser.add_argument("--out_dir", default="quantize_first_analysis", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading full-precision erased model...")
    erased_fp_pipe = load_fp_model(args.erased_fp)

    print("[INFO] Loading quantize-then-erase model...")
    erased_q8_pipe, _ = load_quant_model(args.erased_q8)

    print("[INFO] Loading base quantized model for qparams...")
    _, base_q8_layers = load_quant_model(args.base_q8)

    print("[INFO] Extracting edited cross-attention weights...")
    erased_fp_quantized_endpoint_source = get_quant_sd(erased_q8_pipe)  # only for key set
    erased_q8 = get_quant_sd(erased_q8_pipe)

    # Get FP erased weights from the diffusers UNet directly, but only on canonical keys
    erased_fp_state = {}
    for k, v in erased_fp_pipe.unet.state_dict().items():
        if k.endswith("attn2.to_k.weight") or k.endswith("attn2.to_v.weight"):
            erased_fp_state[k] = v.detach().float().cpu()

    common = sorted(set(erased_fp_state) & set(erased_q8) & set(erased_fp_quantized_endpoint_source))
    if len(common) == 0:
        raise RuntimeError("No common edited attn2.to_k / attn2.to_v keys found")

    layer_names = []
    cosine_vals = []
    rel_gap_vals = []

    total_num_sq = 0.0
    total_den_sq = 0.0

    for k in common:
        w_fp_erased = erased_fp_state[k]
        w_q_then_erase = erased_q8[k]

        scale, zero_point, source = get_qparams_for_key(k, base_q8_layers)
        if scale is None or zero_point is None:
            print(f"[WARN] Skipping {k}: missing qparams from base quantized model ({source})")
            continue

        scale_b, zp_b = broadcast_qparams_like_weight(w_fp_erased, scale, zero_point)
        if scale_b is None or zp_b is None:
            print(f"[WARN] Skipping {k}: could not broadcast qparams")
            continue

        # This is Q(W_e): quantize the full-precision erased endpoint using the same quantizer grid.
        w_erase_then_quantize = quantize_from_qparams(w_fp_erased, scale_b, zp_b)

        cos_val = cosine_similarity_flat(w_q_then_erase, w_erase_then_quantize)
        gap_val = relative_frobenius_gap(w_q_then_erase, w_erase_then_quantize)

        layer_names.append(shorten_layer_name(k))
        cosine_vals.append(cos_val)
        rel_gap_vals.append(gap_val)

        num = (w_q_then_erase - w_erase_then_quantize).norm(p=2).item()
        den = w_erase_then_quantize.norm(p=2).item()
        total_num_sq += num * num
        total_den_sq += den * den

    if len(cosine_vals) == 0:
        raise RuntimeError("No valid layers available for endpoint comparison")

    mean_cos = float(sum(cosine_vals) / len(cosine_vals))
    min_cos = float(min(cosine_vals))
    mean_rel_gap = float(sum(rel_gap_vals) / len(rel_gap_vals))
    overall_rel_gap = float((total_num_sq ** 0.5) / max(total_den_sq ** 0.5, EPS))

    print("\nUSE THESE NUMBERS IN THE PAPER:\n")
    print(f"Mean layer-wise cosine similarity : {mean_cos:.6f}")
    print(f"Minimum layer-wise cosine sim.    : {min_cos:.6f}")
    print(f"Mean layer-wise relative gap      : {mean_rel_gap:.6f}")
    print(f"Overall relative Frobenius gap    : {overall_rel_gap:.6f}")

    metrics = {
        "num_layers": len(cosine_vals),
        "mean_layerwise_cosine_similarity": mean_cos,
        "min_layerwise_cosine_similarity": min_cos,
        "mean_layerwise_relative_gap": mean_rel_gap,
        "overall_relative_frobenius_gap": overall_rel_gap,
        "per_layer": [
            {
                "layer": ln,
                "cosine_similarity": float(cs),
                "relative_gap": float(rg),
            }
            for ln, cs, rg in zip(layer_names, cosine_vals, rel_gap_vals)
        ],
    }

    metrics_path = os.path.join(args.out_dir, "quantize_first_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    fig_path = os.path.join(args.out_dir, "endpoint_similarity.png")
    save_endpoint_similarity_plot(layer_names, cosine_vals, fig_path)

    dev_fig_path = os.path.join(args.out_dir, "endpoint_deviation.png")
    save_endpoint_deviation_plot(layer_names, cosine_vals, dev_fig_path)

    print(f"[INFO] Saved metrics JSON to: {metrics_path}")
    print("\nDONE.")


if __name__ == "__main__":
    main()