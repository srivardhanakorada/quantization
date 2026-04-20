#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import copy
import argparse
from typing import Dict, List, Tuple

import torch
torch.set_grad_enabled(False)

import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

import quantization_tools.quantization.layers  # noqa: F401
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32


def get_edit_layers_quantized(pipe):
    """
    Find quantized attn2.to_k / attn2.to_v layers in the UNet.
    """
    quant_layers = find_layers(pipe.unet, (LinearQuantHub,))
    uce_modules = []
    uce_module_names = []

    for name, module in pipe.unet.named_modules():
        if "attn2" in name and (name.endswith("to_v") or name.endswith("to_k")):
            if name in quant_layers:
                uce_modules.append(quant_layers[name])
                uce_module_names.append(name)

    if len(uce_modules) == 0:
        raise RuntimeError("No quantized attn2.to_k / attn2.to_v layers found")

    return uce_modules, uce_module_names


def get_layer_qparams(layer):
    if not hasattr(layer, "quantizer") or len(layer.quantizer) == 0:
        raise RuntimeError("Layer has no quantizer attached")

    q = layer.quantizer[0]
    if not hasattr(q, "w_scale") or not hasattr(q, "w_zero_point"):
        raise RuntimeError("Could not find w_scale / w_zero_point")

    scale = q.w_scale.detach().float().clone()
    zero_point = q.w_zero_point.detach().float().clone()
    return scale, zero_point


def broadcast_qparams_like_weight(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale.numel() == 1:
        scale_b = scale.reshape(1, 1).to(w.device).expand_as(w)
        zp_b = zero_point.reshape(1, 1).to(w.device).expand_as(w)
        return scale_b, zp_b

    if w.ndim == 2:
        if scale.ndim == 1 and scale.shape[0] == w.shape[0]:
            scale_b = scale.view(-1, 1).to(w.device).expand_as(w)
            zp_b = zero_point.view(-1, 1).to(w.device).expand_as(w)
            return scale_b, zp_b

        if scale.ndim == 2 and scale.shape[0] == w.shape[0] and scale.shape[1] == 1:
            scale_b = scale.to(w.device).expand_as(w)
            zp_b = zero_point.to(w.device).expand_as(w)
            return scale_b, zp_b

    raise RuntimeError(
        f"Unsupported qparam shape {tuple(scale.shape)} for weight shape {tuple(w.shape)}"
    )


@torch.no_grad()
def quantize_to_grid(
    w_fp: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    scale_b, zp_b = broadcast_qparams_like_weight(w_fp, scale, zero_point)
    q_idx = torch.round(w_fp / scale_b + zp_b)
    q_idx = torch.clamp(q_idx, 0, 255)
    w_q = (q_idx - zp_b) * scale_b
    return w_q


@torch.no_grad()
def get_last_token_embedding(pipe, text: str) -> torch.Tensor:
    """
    Match original UCE intent:
    - tokenize prompt
    - run text encoder directly
    - pick the last non-special token embedding

    Returns shape [1, hidden_dim]
    """
    toks = pipe.tokenizer(
        text,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = toks["input_ids"].to(DEVICE)
    attention_mask = toks["attention_mask"].to(DEVICE)

    text_out = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

    if hasattr(text_out, "last_hidden_state"):
        hidden = text_out.last_hidden_state
    else:
        hidden = text_out[0]

    last_token_idx = max(int(attention_mask.sum().item()) - 2, 0)
    last_token_emb = hidden[:, last_token_idx, :]   # [1, hidden_dim]
    return last_token_emb.detach()


@torch.no_grad()
def UCE_on_quantized_model(
    pipe,
    edit_concepts: List[str],
    guide_concepts: List[str],
    preserve_concepts: List[str],
    erase_scale: float,
    preserve_scale: float,
    lamb: float,
):
    start_time = time.time()

    # Find edit layers exactly as in original UCE, but on quantized modules
    uce_modules, uce_module_names = get_edit_layers_quantized(pipe)

    # Keep frozen copies of original quantized modules for guide-output computation
    original_modules = copy.deepcopy(uce_modules)

    print(f"[INFO] Found {len(uce_modules)} quantized UCE edit layers")

    # Collect concept embeddings exactly as original UCE
    uce_embeds = {}
    for e in edit_concepts + guide_concepts + preserve_concepts:
        if e in uce_embeds:
            continue
        uce_embeds[e] = get_last_token_embedding(pipe, e)

    # Collect guide outputs from the ORIGINAL frozen quantized modules
    # exactly as original UCE does
    uce_guide_outputs = {}
    for g in guide_concepts + preserve_concepts:
        if g in uce_guide_outputs:
            continue

        t_emb = uce_embeds[g].to(DEVICE, dtype=TORCH_DTYPE)
        uce_guide_outputs[g] = []

        for module in original_modules:
            module = module.to(DEVICE)
            module.eval()
            out = module(t_emb)
            uce_guide_outputs[g].append(out.detach())

    # Closed-form UCE update, with the quantization-specific change that:
    #   w_old = quantized-point weight
    #   final solved weight is projected back to same fixed grid
    for module_idx, module in enumerate(uce_modules):
        module = module.to(DEVICE)

        scale, zero_point = get_layer_qparams(module)

        # Use fixed quantized-point weight as the base matrix W_q
        w_old_fp = module.core.weight.detach().float().clone()
        w_old = quantize_to_grid(w_old_fp, scale, zero_point).to(DEVICE, dtype=TORCH_DTYPE)

        mat1 = lamb * w_old
        mat2 = lamb * torch.eye(w_old.shape[1], device=DEVICE, dtype=TORCH_DTYPE)

        # Erase concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            c_i = uce_embeds[erase_concept].to(DEVICE, dtype=TORCH_DTYPE).T
            v_i_star = uce_guide_outputs[guide_concept][module_idx].to(DEVICE, dtype=TORCH_DTYPE).T

            mat1 += erase_scale * (v_i_star @ c_i.T)
            mat2 += erase_scale * (c_i @ c_i.T)

        # Preserve concepts
        for preserve_concept in preserve_concepts:
            c_i = uce_embeds[preserve_concept].to(DEVICE, dtype=TORCH_DTYPE).T
            v_i_star = uce_guide_outputs[preserve_concept][module_idx].to(DEVICE, dtype=TORCH_DTYPE).T

            mat1 += preserve_scale * (v_i_star @ c_i.T)
            mat2 += preserve_scale * (c_i @ c_i.T)

        W_new_fp = mat1 @ torch.inverse(mat2.float()).to(TORCH_DTYPE)

        # Project back to SAME fixed quantization grid
        W_new_q = quantize_to_grid(W_new_fp, scale.to(DEVICE), zero_point.to(DEVICE))

        # Write back into live quantized model
        module.core.weight.data.copy_(W_new_q.to(module.core.weight.device, dtype=module.core.weight.dtype))

        if (module_idx + 1) % 10 == 0 or (module_idx + 1) == len(uce_modules):
            print(f"[INFO] Edited {module_idx + 1}/{len(uce_modules)} layers")

    end_time = time.time()
    print(f"\n[INFO] Quantized UCE finished in {end_time - start_time:.2f} seconds\n")

    return pipe


def main():
    parser = argparse.ArgumentParser(
        prog="QuantizedUCE",
        description="Apply original UCE-style editing on a quantized checkpoint"
    )

    parser.add_argument("--quant_ckpt", type=str, required=True, help="Base quantized checkpoint")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save the edited quantized checkpoint")

    parser.add_argument("--edit_concepts", type=str, required=True,
                        help="Concepts to erase separated by ;")
    parser.add_argument("--guide_concepts", type=str, default=None,
                        help="Guide concepts separated by ;")
    parser.add_argument("--preserve_concepts", type=str, default=None,
                        help="Preserve concepts separated by ;")

    parser.add_argument("--concept_type", choices=["art", "object"], type=str, default="object")
    parser.add_argument("--erase_scale", type=float, default=1.0)
    parser.add_argument("--preserve_scale", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=0.5)

    args = parser.parse_args()

    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(";")]

    guide_concepts = args.guide_concepts
    if guide_concepts is None:
        guide_concepts = ""
        if args.concept_type == "art":
            guide_concepts = "art"

    guide_concepts = [concept.strip() for concept in guide_concepts.split(";")]
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts * len(edit_concepts)

    if len(guide_concepts) != len(edit_concepts):
        raise ValueError("erase concepts and guide concepts must match in length")

    if args.preserve_concepts is None or args.preserve_concepts.strip() == "":
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(";")]

    print(f"\nErasing: {edit_concepts}")
    print(f"Guiding: {guide_concepts}")
    print(f"Preserving: {preserve_concepts}\n")

    print(f"[INFO] Loading quantized checkpoint: {args.quant_ckpt}")
    pipe = torch.load(args.quant_ckpt, map_location="cpu").to(DEVICE)
    pipe.unet.eval()
    pipe.text_encoder.eval()

    pipe = UCE_on_quantized_model(
        pipe=pipe,
        edit_concepts=edit_concepts,
        guide_concepts=guide_concepts,
        preserve_concepts=preserve_concepts,
        erase_scale=args.erase_scale,
        preserve_scale=args.preserve_scale,
        lamb=args.lamb,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(pipe, args.save_path)
    print(f"[INFO] Saved edited quantized checkpoint to: {args.save_path}")


if __name__ == "__main__":
    main()