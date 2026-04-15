import os
import sys
import json
import torch
from collections import Counter

sys.path.append(os.getcwd())

CKPTS = [
    "ckpt/quant-UCE_barack_obama_w8_weightonly_full-16-8-16-8.ckpt",
    # "ckpt/quant-UCE_barack_obama_w4_weightonly_full-16-4-16-4.ckpt",
]


def tensor_info(x):
    if not isinstance(x, torch.Tensor):
        return None
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(x.min().item()) if x.numel() > 0 else None,
        "max": float(x.max().item()) if x.numel() > 0 else None,
        "mean": float(x.float().mean().item()) if x.numel() > 0 else None,
    }


def print_obj_summary(obj, prefix="", max_items=30):
    if isinstance(obj, dict):
        print(f"{prefix}DICT with {len(obj)} keys")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}  ...")
                break
            print(f"{prefix}  [{i}] key={repr(k)} type={type(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__.upper()} len={len(obj)}")
        for i, v in enumerate(obj[:max_items]):
            print(f"{prefix}  [{i}] type={type(v)}")
        if len(obj) > max_items:
            print(f"{prefix}  ...")
    else:
        print(f"{prefix}OBJECT type={type(obj)}")


def inspect_pipe(pipe, tag):
    print(f"\n===== Inspecting pipe: {tag} =====")
    print("pipe type:", type(pipe))
    print("has unet:", hasattr(pipe, "unet"))

    if not hasattr(pipe, "unet"):
        return

    unet = pipe.unet
    print("unet type:", type(unet))

    sd = unet.state_dict()
    print("unet.state_dict() num keys:", len(sd))

    attn2_keys = [k for k in sd.keys() if "attn2" in k]
    print("attn2-related state_dict keys:", len(attn2_keys))
    for k in attn2_keys[:20]:
        print(" ", k)
    if len(attn2_keys) > 20:
        print("  ...")

    named_params = list(unet.named_parameters())
    print("unet.named_parameters() count:", len(named_params))
    for name, p in named_params[:20]:
        print(" ", name, tuple(p.shape), p.dtype)
    if len(named_params) > 20:
        print("  ...")

    named_buffers = list(unet.named_buffers())
    print("unet.named_buffers() count:", len(named_buffers))
    for name, b in named_buffers[:20]:
        shape = tuple(b.shape) if isinstance(b, torch.Tensor) else None
        print(" ", name, shape, getattr(b, "dtype", None))
    if len(named_buffers) > 20:
        print("  ...")

    print("\nSearching modules with 'attn2' in module name:")
    hits = 0
    for name, mod in unet.named_modules():
        if "attn2" in name:
            hits += 1
            print(f"  {name}: {type(mod)}")
            attrs = [a for a in ["weight", "bias", "quantizer", "qweight", "scales", "zeros"] if hasattr(mod, a)]
            if attrs:
                print("     attrs:", attrs)
    print("attn2 module hits:", hits)

    print("\nSearching for quantized modules:")
    quant_hits = 0
    for name, mod in unet.named_modules():
        cls = type(mod).__name__.lower()
        if "quant" in cls or hasattr(mod, "quantizer"):
            quant_hits += 1
            print(f"  {name}: {type(mod)}")
            attrs = [a for a in ["weight", "bias", "quantizer", "qweight", "scales", "zeros"] if hasattr(mod, a)]
            if attrs:
                print("     attrs:", attrs)
            if quant_hits >= 40:
                print("  ...")
                break
    print("quantized module hits:", quant_hits)


def inspect_checkpoint_file(path):
    print(f"\n\n==============================")
    print("FILE:", path)
    print("==============================")

    obj = torch.load(path, map_location="cpu")
    print("top-level loaded object type:", type(obj))

    if isinstance(obj, dict):
        print_obj_summary(obj, prefix="")
        # Look for common checkpoint containers
        for key in ["state_dict", "model", "pipe", "unet"]:
            if key in obj:
                print(f"\nTop-level key {key!r} found with type {type(obj[key])}")
                print_obj_summary(obj[key], prefix="  ")

    # If the checkpoint is directly a pipeline/module, inspect it
    if hasattr(obj, "unet"):
        inspect_pipe(obj, os.path.basename(path))
    else:
        # If dict contains a model-like object, inspect likely candidates
        if isinstance(obj, dict):
            for key in ["pipe", "model"]:
                if key in obj and hasattr(obj[key], "unet"):
                    inspect_pipe(obj[key], f"{os.path.basename(path)}::{key}")


def main():
    for path in CKPTS:
        inspect_checkpoint_file(path)


if __name__ == "__main__":
    main()