#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---- HF Hub compatibility patch: must run before quantization_tools/diffusers imports ----
try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download

    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = hf_hub_download
except Exception as e:
    print(f"[WARN] huggingface_hub cached_download monkeypatch failed: {e}")
# --------------------------------------------------------------------------------------

import torch
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything

from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub

all_quant_layers = {}


def step_start_callback(step: int, timestep: int):
    global all_quant_layers
    for _, layer in all_quant_layers.items():
        if not hasattr(layer, "quantizer"):
            continue
        for quantizer in layer.quantizer:
            if hasattr(quantizer, "set_curr_step"):
                quantizer.set_curr_step(step)


def safe_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "", text)
    return text[:180]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r") as f:
        schema = json.load(f)

    required = [
        "guidance_scale",
        "num_inference_steps",
        "negative_prompt",
        "seeds",
        "prompt_templates",
    ]
    for k in required:
        if k not in schema:
            raise ValueError(f"Missing required schema key: {k}")

    if "erase_concepts" not in schema:
        raise ValueError("Schema must contain 'erase_concepts'")

    if "preserve_concepts" not in schema:
        raise ValueError("Schema must contain 'preserve_concepts'")

    if "anchor_concepts" not in schema:
        schema["anchor_concepts"] = []

    return schema


def flatten_concepts(schema: Dict[str, Any], include_anchor: bool) -> Dict[str, List[str]]:
    """
    Returns:
        {
            "erase": [...],
            "preserve": [...],
            "anchor": [...]
        }
    Supports preserve_concepts as either:
      1) list[str]
      2) dict[str, list[str]]
    """
    erase = schema["erase_concepts"]

    preserve_raw = schema["preserve_concepts"]
    if isinstance(preserve_raw, dict):
        preserve = []
        for _, vals in preserve_raw.items():
            preserve.extend(vals)
        preserve = sorted(list(dict.fromkeys(preserve)))
    elif isinstance(preserve_raw, list):
        preserve = preserve_raw
    else:
        raise ValueError("preserve_concepts must be either a list or dict")

    anchor = schema.get("anchor_concepts", []) if include_anchor else []

    return {
        "erase": erase,
        "preserve": preserve,
        "anchor": anchor,
    }


def build_generation_plan(
    schema: Dict[str, Any],
    include_anchor: bool,
) -> List[Dict[str, Any]]:
    groups = flatten_concepts(schema, include_anchor=include_anchor)
    templates = schema["prompt_templates"]
    seeds = schema["seeds"]

    jobs = []
    for group_name, concepts in groups.items():
        for concept in concepts:
            for template_idx, template in enumerate(templates):
                prompt = template.format(concept)
                for seed in seeds:
                    jobs.append(
                        {
                            "group": group_name,
                            "concept": concept,
                            "template": template,
                            "template_idx": template_idx,
                            "seed": int(seed),
                            "prompt": prompt,
                        }
                    )
    return jobs


def save_metadata(out_dir: str, schema: Dict[str, Any], jobs: List[Dict[str, Any]], args):
    meta = {
        "schema_path": args.schema_path,
        "model_path": args.model_path,
        "schema_name": schema.get("schema_name", "unknown"),
        "guidance_scale": schema["guidance_scale"],
        "num_inference_steps": schema["num_inference_steps"],
        "negative_prompt": schema["negative_prompt"],
        "num_jobs": len(jobs),
        "batch_size": args.batch_size,
        "include_anchor": args.include_anchor,
        "device": args.device,
        "dtype": args.dtype,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "resolved_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    with open(os.path.join(out_dir, "generation_manifest.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


def save_image_and_sidecar(
    img: Image.Image,
    job: Dict[str, Any],
    out_dir: str,
    global_index: int,
):
    group_dir = os.path.join(out_dir, job["group"], safe_name(job["concept"]))
    ensure_dir(group_dir)

    base_name = (
        f"{global_index:05d}"
        f"_t{job['template_idx']:02d}"
        f"_s{job['seed']:02d}"
        f"_{safe_name(job['concept'])}"
    )

    img_path = os.path.join(group_dir, base_name + ".png")
    # meta_path = os.path.join(group_dir, base_name + ".json")

    img.save(img_path)

    # with open(meta_path, "w") as f:
    #     json.dump(job, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to quantized ckpt")
    parser.add_argument("--schema_path", required=True, type=str, help="Path to schema JSON")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--include_anchor", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    schema = load_schema(args.schema_path)
    jobs = build_generation_plan(schema, include_anchor=args.include_anchor)
    save_metadata(args.out_dir, schema, jobs, args)

    print(f"[INFO] Loading quantized checkpoint from: {args.model_path}")
    pipe = torch.load(args.model_path, map_location="cpu")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        pipe = pipe.to(args.device)
        device_for_generator = args.device
    else:
        pipe = pipe.to("cpu")
        device_for_generator = "cpu"

    global all_quant_layers
    layers_linear = find_layers(pipe.unet, (LinearQuantHub,))
    layers_conv = find_layers(pipe.unet, (Conv2dQuantHub,))
    all_quant_layers = {**layers_linear, **layers_conv}

    # Generation defaults aligned with your current setup
    width = 512
    height = 512

    pipe.set_progress_bar_config(disable=False)

    guidance_scale = float(schema["guidance_scale"])
    num_inference_steps = int(schema["num_inference_steps"])
    negative_prompt = schema["negative_prompt"]

    if isinstance(negative_prompt, str):
        neg_mode = "single"
    else:
        raise ValueError("negative_prompt in schema must be a string")

    print(f"[INFO] Total generations to run: {len(jobs)}")
    print(f"[INFO] Batch size: {args.batch_size}")

    global_idx = args.start_index

    with torch.no_grad():
        start = 0
        while start < len(jobs):
            end = min(start + args.batch_size, len(jobs))
            batch_jobs = jobs[start:end]

            prompts = [j["prompt"] for j in batch_jobs]
            seeds = [j["seed"] for j in batch_jobs]

            generators = []
            for seed in seeds:
                g = torch.Generator(device=device_for_generator)
                g.manual_seed(seed)
                generators.append(g)

            if neg_mode == "single":
                negative_prompts = [negative_prompt] * len(batch_jobs)
            else:
                negative_prompts = None

            print(f"[INFO] Generating batch {start}:{end}")

            result = pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generators,
                callback_on_start=step_start_callback,
            )

            images = result.images

            for img, job in zip(images, batch_jobs):
                save_image_and_sidecar(img, job, args.out_dir, global_idx)
                global_idx += 1

            start = end

    print("[DONE] Generation complete.")


if __name__ == "__main__":
    main()