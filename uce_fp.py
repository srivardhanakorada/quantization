#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

import huggingface_hub

# -------------------------------------------------------------------------
# Monkey patch for old diffusers + newer huggingface_hub
# -------------------------------------------------------------------------
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

# Offline mode for HPC
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file


PROMPT_TEMPLATES = [
    "a photo of {}",
    "a portrait of {}",
    "a close-up portrait of {}",
    "a professional photo of {}",
    "a studio portrait of {}",
    "a candid photograph of {}",
    "a headshot of {}",
    "a detailed portrait of {}",
    "a face photo of {}",
    "an image of {}",
]


def parse_concepts(x: str) -> List[str]:
    if x is None or x.strip() == "":
        return []
    return [c.strip() for c in x.split(";") if c.strip()]


def sanitize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def load_fp_pipe(model_id: str, device: str, dtype: torch.dtype):
    print(f"[INFO] Loading FP SD pipeline from: {model_id}")

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()

    return pipe


@torch.no_grad()
def apply_uce_safetensors(pipe, uce_path: str):
    """
    Load normal UCE safetensors and patch edited weights into pipe.unet.

    Expected keys:
        down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_k.weight
        down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_v.weight
        ...
    """
    print(f"[INFO] Loading UCE weights from: {uce_path}")
    uce_state = load_file(uce_path, device="cpu")

    module_map = dict(pipe.unet.named_modules())

    loaded = 0
    skipped = []

    for key, value in uce_state.items():
        if not key.endswith(".weight"):
            skipped.append(key)
            continue

        module_name = key[:-len(".weight")]

        if module_name not in module_map:
            skipped.append(key)
            continue

        module = module_map[module_name]

        if not hasattr(module, "weight"):
            skipped.append(key)
            continue

        if module.weight.shape != value.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: "
                f"model has {tuple(module.weight.shape)}, "
                f"safetensors has {tuple(value.shape)}"
            )

        module.weight.data.copy_(
            value.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        )

        loaded += 1

    print(f"[INFO] Loaded {loaded} UCE edited weights into UNet")

    if len(skipped) > 0:
        print(f"[WARN] Skipped {len(skipped)} keys")
        for k in skipped[:20]:
            print(f"  - {k}")

    if loaded == 0:
        raise RuntimeError("No UCE weights were loaded. Check safetensors key names.")


@torch.no_grad()
def generate_for_concept(
    pipe,
    concept: str,
    out_dir: Path,
    seeds: List[int],
    prompt_templates: List[str],
    device: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for template_idx, template in enumerate(prompt_templates):
        prompt = template.format(concept)

        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            filename = f"template{template_idx:02d}_seed{seed:04d}.png"
            save_path = out_dir / filename
            image.save(save_path)

            manifest_rows.append(
                {
                    "concept": concept,
                    "prompt": prompt,
                    "template_idx": template_idx,
                    "seed": seed,
                    "path": str(save_path),
                }
            )

    return manifest_rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from normal FP Stable Diffusion with UCE safetensors applied"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Local Diffusers model path, e.g. models/stable-diffusion-v1-5/",
    )

    parser.add_argument(
        "--uce_path",
        type=str,
        required=True,
        help="Path to normal UCE .safetensors file",
    )

    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="Output root directory",
    )

    parser.add_argument(
        "--target_concepts",
        type=str,
        required=True,
        help="Target concepts separated by ;",
    )

    parser.add_argument(
        "--preserve_concepts",
        type=str,
        default="",
        help="Preserve concepts separated by ;",
    )

    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=0)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16"],
        default="fp32",
        help="Use fp32 for exact paper runs unless memory is an issue",
    )

    args = parser.parse_args()

    device = args.device
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    target_concepts = parse_concepts(args.target_concepts)
    preserve_concepts = parse_concepts(args.preserve_concepts)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    print("\n[INFO] Generation config")
    print(f"Targets:   {target_concepts}")
    print(f"Preserve:  {preserve_concepts}")
    print(f"Seeds:     {seeds}")
    print(f"Save root: {args.save_root}\n")

    pipe = load_fp_pipe(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
    )

    apply_uce_safetensors(pipe, args.uce_path)

    save_root = Path(args.save_root)
    manifest_path = save_root / "generation_manifest.jsonl"
    save_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # Target concepts go under target/
    for concept in tqdm(target_concepts, desc="Generating target concepts"):
        concept_dir = save_root / "target" / sanitize_name(concept)

        rows = generate_for_concept(
            pipe=pipe,
            concept=concept,
            out_dir=concept_dir,
            seeds=seeds,
            prompt_templates=PROMPT_TEMPLATES,
            device=device,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )

        for r in rows:
            r["split"] = "target"
        all_rows.extend(rows)

    # Preserve concepts go under preserve/
    for concept in tqdm(preserve_concepts, desc="Generating preserve concepts"):
        concept_dir = save_root / "preserve" / sanitize_name(concept)

        rows = generate_for_concept(
            pipe=pipe,
            concept=concept,
            out_dir=concept_dir,
            seeds=seeds,
            prompt_templates=PROMPT_TEMPLATES,
            device=device,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )

        for r in rows:
            r["split"] = "preserve"
        all_rows.extend(rows)

    with open(manifest_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n[INFO] Done.")
    print(f"[INFO] Images saved under: {save_root}")
    print(f"[INFO] Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()