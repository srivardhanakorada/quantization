#!/usr/bin/env python3
import argparse
import os
import json
import sys

import torch
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create scaled UCE-style edited SD1.5 models in Diffusers format."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="models/stable-diffusion-v1-5",
        help="Base SD model (HF id, local diffusers dir, or local HF snapshot dir).",
    )
    parser.add_argument(
        "--erased_model",
        type=str,
        required=True,
        help="Path to erased model in diffusers format.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory where scaled diffusers pipelines will be saved.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default= [0.75,0.9],
        help="Scaling factors alpha.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Load/save dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use while constructing scaled models.",
    )
    parser.add_argument(
        "--edit_mode",
        type=str,
        default="uce",
        choices=["uce", "all_unet"],
        help=(
            "'uce' edits only attn2.to_k/to_v weights; "
            "'all_unet' interpolates the full UNet."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional model variant, e.g. fp16.",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def should_edit_param(name: str, edit_mode: str) -> bool:
    if edit_mode == "all_unet":
        return True

    return (
        "attn2" in name
        and (
            name.endswith("to_k.weight")
            or name.endswith("to_v.weight")
        )
    )


def load_pipe(model_path, dtype, cache_dir=None, variant=None):
    kwargs = dict(
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    )
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if variant is not None:
        kwargs["variant"] = variant

    return StableDiffusionPipeline.from_pretrained(model_path, **kwargs)


@torch.no_grad()
def build_scaled_unet_state(
    base_unet: torch.nn.Module,
    erased_unet: torch.nn.Module,
    alpha: float,
    edit_mode: str,
):
    base_sd = base_unet.state_dict()
    erased_sd = erased_unet.state_dict()

    scaled_sd = {}
    edited_names = []
    total_names = 0

    for name, base_tensor in base_sd.items():
        total_names += 1

        if name not in erased_sd:
            raise KeyError(f"{name} missing in erased UNet state_dict")

        erased_tensor = erased_sd[name]

        if base_tensor.shape != erased_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"{tuple(base_tensor.shape)} vs {tuple(erased_tensor.shape)}"
            )

        if should_edit_param(name, edit_mode):
            scaled_tensor = base_tensor + alpha * (erased_tensor - base_tensor)
            scaled_sd[name] = scaled_tensor
            edited_names.append(name)
        else:
            scaled_sd[name] = base_tensor.clone()

    return scaled_sd, edited_names, total_names


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    print(f"[INFO] Loading base model:   {args.base_model}")
    base_pipe = load_pipe(
        model_path=args.base_model,
        dtype=dtype,
        cache_dir=args.cache_dir,
        variant=args.variant,
    )
    base_pipe = base_pipe.to(device)

    print(f"[INFO] Loading erased model: {args.erased_model}")
    erased_pipe = load_pipe(
        model_path=args.erased_model,
        dtype=dtype,
        cache_dir=args.cache_dir,
        variant=args.variant,
    )
    erased_pipe = erased_pipe.to(device)

    metadata = {
        "base_model": args.base_model,
        "erased_model": args.erased_model,
        "dtype": args.dtype,
        "device": args.device,
        "edit_mode": args.edit_mode,
        "alphas": args.alphas,
        "cache_dir": args.cache_dir,
        "variant": args.variant,
        "saved_models": [],
    }

    for alpha in args.alphas:
        print(f"\n[INFO] Creating scaled model for alpha = {alpha}")

        scaled_pipe = load_pipe(
            model_path=args.base_model,
            dtype=dtype,
            cache_dir=args.cache_dir,
            variant=args.variant,
        ).to(device)

        scaled_unet_sd, edited_names, total_names = build_scaled_unet_state(
            base_unet=base_pipe.unet,
            erased_unet=erased_pipe.unet,
            alpha=alpha,
            edit_mode=args.edit_mode,
        )

        incompatible = scaled_pipe.unet.load_state_dict(scaled_unet_sd, strict=True)

        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])

        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(
                f"State load issue for alpha={alpha}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        alpha_tag = str(alpha).replace(".", "p")
        save_dir = os.path.join(args.output_root, f"alpha_{alpha_tag}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"[INFO] Saving diffusers pipeline to: {save_dir}")
        scaled_pipe.save_pretrained(save_dir)

        info = {
            "alpha": alpha,
            "save_dir": save_dir,
            "num_unet_tensors_total": total_names,
            "num_unet_tensors_edited": len(edited_names),
            "edited_param_names": edited_names,
        }
        metadata["saved_models"].append(info)

        with open(os.path.join(save_dir, "scaling_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        del scaled_pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata_path = os.path.join(args.output_root, "all_scaled_models.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()