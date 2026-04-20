#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import huggingface_hub
import torch

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    import sys
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

import quantization_tools.quantization.layers  # noqa: F401
from diffusers import DiffusionPipeline


def is_torch_serialized_checkpoint(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".ckpt") or lowered.endswith(".pt") or lowered.endswith(".pth") or lowered.endswith(".bin")


def load_pipe(path_or_name: str, device: str = "cpu"):
    if os.path.exists(path_or_name) and is_torch_serialized_checkpoint(path_or_name):
        pipe = torch.load(path_or_name, map_location="cpu")
    else:
        pipe = DiffusionPipeline.from_pretrained(
            path_or_name,
            torch_dtype=torch.float32,
            safety_checker=None,
            local_files_only=True,
        )
    return pipe.to(device)


def get_unet_keys(pipe):
    sd = pipe.unet.state_dict()
    return sorted([k for k, v in sd.items() if torch.is_tensor(v)])


def filter_keys(keys, patterns):
    out = []
    for k in keys:
        if all(p in k for p in patterns):
            out.append(k)
    return out


def print_section(title, items, max_items=None):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if max_items is None:
        max_items = len(items)
    for x in items[:max_items]:
        print(x)
    print(f"\n[COUNT] {len(items)}")


def save_list(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(x + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fp", type=str, required=True)
    parser.add_argument("--erased_fp", type=str, required=True)
    parser.add_argument("--base_q8", type=str, required=True)
    parser.add_argument("--erased_q8", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Loading base_fp ...")
    base_fp = load_pipe(args.base_fp, device=args.device)
    print("[INFO] Loading erased_fp ...")
    erased_fp = load_pipe(args.erased_fp, device=args.device)
    print("[INFO] Loading base_q8 ...")
    base_q8 = load_pipe(args.base_q8, device=args.device)
    print("[INFO] Loading erased_q8 ...")
    erased_q8 = load_pipe(args.erased_q8, device=args.device)

    keysets = {
        "base_fp": get_unet_keys(base_fp),
        "erased_fp": get_unet_keys(erased_fp),
        "base_q8": get_unet_keys(base_q8),
        "erased_q8": get_unet_keys(erased_q8),
    }

    # Save full key dumps
    for name, keys in keysets.items():
        save_list(os.path.join(args.out_dir, f"{name}_all_unet_keys.txt"), keys)

    # Print counts
    print("\n" + "=" * 120)
    print("TOTAL UNET KEY COUNTS")
    print("=" * 120)
    for name, keys in keysets.items():
        print(f"{name:12s}: {len(keys)}")

    # Focused filters
    focused_filters = {
        "attn2": ["attn2"],
        "to_k": ["to_k"],
        "to_v": ["to_v"],
        "attn2_to_k": ["attn2", "to_k"],
        "attn2_to_v": ["attn2", "to_v"],
        "attn2_to_k_weight": ["attn2", "to_k", "weight"],
        "attn2_to_v_weight": ["attn2", "to_v", "weight"],
    }

    for model_name, keys in keysets.items():
        for filt_name, patterns in focused_filters.items():
            matched = filter_keys(keys, patterns)
            save_list(os.path.join(args.out_dir, f"{model_name}_{filt_name}.txt"), matched)

    # Print the most important filtered views
    for model_name, keys in keysets.items():
        print_section(f"{model_name} : keys containing attn2 + to_k", filter_keys(keys, ["attn2", "to_k"]))
        print_section(f"{model_name} : keys containing attn2 + to_v", filter_keys(keys, ["attn2", "to_v"]))

    # Intersections
    sets = {k: set(v) for k, v in keysets.items()}
    common_all = sorted(list(sets["base_fp"] & sets["erased_fp"] & sets["base_q8"] & sets["erased_q8"]))
    save_list(os.path.join(args.out_dir, "common_all_models.txt"), common_all)
    print_section("COMMON KEYS ACROSS ALL FOUR MODELS", common_all, max_items=200)

    common_attn2_to_k = [k for k in common_all if "attn2" in k and "to_k" in k]
    common_attn2_to_v = [k for k in common_all if "attn2" in k and "to_v" in k]
    save_list(os.path.join(args.out_dir, "common_attn2_to_k.txt"), common_attn2_to_k)
    save_list(os.path.join(args.out_dir, "common_attn2_to_v.txt"), common_attn2_to_v)

    print_section("COMMON attn2 + to_k KEYS ACROSS ALL FOUR", common_attn2_to_k)
    print_section("COMMON attn2 + to_v KEYS ACROSS ALL FOUR", common_attn2_to_v)

    # Also compare FP vs quantized separately
    fp_attn2_to_k = set(filter_keys(keysets["base_fp"], ["attn2", "to_k"]))
    q_attn2_to_k = set(filter_keys(keysets["base_q8"], ["attn2", "to_k"]))
    fp_attn2_to_v = set(filter_keys(keysets["base_fp"], ["attn2", "to_v"]))
    q_attn2_to_v = set(filter_keys(keysets["base_q8"], ["attn2", "to_v"]))

    print_section("FP attn2+to_k but NOT in q8", sorted(list(fp_attn2_to_k - q_attn2_to_k)))
    print_section("Q8 attn2+to_k but NOT in fp", sorted(list(q_attn2_to_k - fp_attn2_to_k)))
    print_section("FP attn2+to_v but NOT in q8", sorted(list(fp_attn2_to_v - q_attn2_to_v)))
    print_section("Q8 attn2+to_v but NOT in fp", sorted(list(q_attn2_to_v - fp_attn2_to_v)))

    print("\n[INFO] Wrote all key dumps to:", args.out_dir)


if __name__ == "__main__":
    main()