#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Goes: generate/ -> GLoCE/ -> project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add GLoCE/ so that 'src' is importable
GLOCE_ROOT = Path(__file__).resolve().parent.parent
if str(GLOCE_ROOT) not in sys.path:
    sys.path.insert(0, str(GLOCE_ROOT))

QUANT_ROOT = PROJECT_ROOT / "quantization"
for p in [str(PROJECT_ROOT), str(GLOCE_ROOT), str(QUANT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from gloce.src.configs.config import parse_precision
from gloce.src.engine import train_util
from gloce.src.models.gloce import GLoCELayerOutProp, GLoCENetworkOutProp
import gloce.src.engine.gloce_util as gloce_util
from gloce.src.models.merge_gloce import load_state_dict

from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub

import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

device = torch.device("cuda:0")
torch.cuda.set_device(device)

all_quant_layers = {}


def step_start_callback(step: int, timestep: int):
    global all_quant_layers
    for _, layer in all_quant_layers.items():
        if not hasattr(layer, "quantizer"):
            continue
        for quantizer in layer.quantizer:
            if hasattr(quantizer, "set_curr_step"):
                quantizer.set_curr_step(step)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "", text)
    return text[:180]


def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r") as f:
        schema = json.load(f)

    required = [
        "guidance_scale",
        "num_inference_steps",
        "negative_prompt",
        "seeds",
        "prompt_templates",
        "erase_concepts",
        "preserve_concepts",
        "anchor_concepts",
    ]
    for k in required:
        if k not in schema:
            raise ValueError(f"Missing required schema key: {k}")

    return schema


def flatten_concepts(schema: Dict[str, Any], include_anchor: bool) -> Dict[str, List[str]]:
    groups = {
        "erase": schema["erase_concepts"],
        "preserve": schema["preserve_concepts"],
    }
    groups["anchor"] = schema.get("anchor_concepts", []) if include_anchor else []
    return groups


def build_generation_plan(schema: Dict[str, Any], include_anchor: bool) -> List[Dict[str, Any]]:
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
        "quantized_ckpt_path": args.quantized_ckpt_path,
        "model_path": args.model_path,
        "schema_name": schema.get("schema_name", "unknown"),
        "guidance_scale": schema["guidance_scale"],
        "num_inference_steps": schema["num_inference_steps"],
        "negative_prompt": schema["negative_prompt"],
        "num_jobs": len(jobs),
        "include_anchor": args.include_anchor,
        "device": str(device),
        "precision": args.precision,
        "width": args.width,
        "height": args.height,
        "gate_rank": args.gate_rank,
        "update_rank": args.update_rank,
        "degen_rank": args.degen_rank,
        "eta": args.eta,
        "st_timestep": args.st_timestep,
        "last_layer": args.last_layer,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(args.out_dir, "resolved_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    with open(os.path.join(args.out_dir, "generation_manifest.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


def save_image(img: Image.Image, job: Dict[str, Any], out_dir: str, global_index: int):
    group_dir = os.path.join(out_dir, job["group"], safe_name(job["concept"]))
    ensure_dir(group_dir)

    base_name = (
        f"{global_index:05d}"
        f"_t{job['template_idx']:02d}"
        f"_s{job['seed']:02d}"
        f"_{safe_name(job['concept'])}"
    )
    img.save(os.path.join(group_dir, base_name + ".png"))


def load_quantized_pipeline(quant_ckpt_path: str, weight_dtype):
    print(f"[INFO] Loading quantized checkpoint: {quant_ckpt_path}")
    pipe = torch.load(quant_ckpt_path, map_location="cpu")
    pipe = pipe.to(device)

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)
    unet = pipe.unet.to(device)

    text_encoder.eval()
    unet.eval()
    unet.requires_grad_(False)

    all_q = {
        **find_layers(unet, (LinearQuantHub,)),
        **find_layers(unet, (Conv2dQuantHub,)),
    }
    print(f"[INFO] Found {len(all_q)} quantized layers.")

    return tokenizer, text_encoder, unet, pipe, all_q


def discover_concept_ckpts(model_root: str, ckpt_name: str = "ckpt.safetensors") -> List[Path]:
    if os.path.isfile(model_root):
        return [Path(model_root)]

    concept_ckpts = []
    for ckpt in sorted(os.listdir(model_root)):
        cand = os.path.join(model_root, ckpt, ckpt_name)
        if os.path.isfile(cand):
            concept_ckpts.append(Path(cand))

    if len(concept_ckpts) == 0:
        raise RuntimeError(f"No concept checkpoints found under: {model_root}")

    return concept_ckpts


def infer_with_gloce(
    args,
    model_paths: List[Path],
    schema: Dict[str, Any],
    quant_ckpt_path: str,
    precision: str = "fp32",
):
    weight_dtype = parse_precision(precision)

    global all_quant_layers
    tokenizer, text_encoder, unet, pipe, all_quant_layers = load_quantized_pipeline(
        quant_ckpt_path=quant_ckpt_path,
        weight_dtype=weight_dtype,
    )

    quant_hub_map = {}
    for name, module in unet.named_modules():
        if name.endswith("attn2.to_out.0") and isinstance(module, LinearQuantHub):
            quant_hub_map[id(module.core)] = module

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.requires_grad_(False)
    unet.eval()

    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []

    for find_module_name in args.find_module_name:
        module_name, module_type = gloce_util.get_module_name_type(find_module_name)
        org_modules, module_name_list = gloce_util.get_modules_list(
            unet, text_encoder, find_module_name, module_name, module_type
        )

        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    cpes, metadatas = zip(*[load_state_dict(model_path, weight_dtype) for model_path in model_paths])

    assert all(metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas)

    network = GLoCENetworkOutProp(
        unet,
        text_encoder,
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=GLoCELayerOutProp,
        degen_rank=args.degen_rank,
        gate_rank=args.gate_rank,
        update_rank=args.update_rank,
        n_concepts=len(model_paths),
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names=args.find_module_name,
        last_layer=args.last_layer,
        st_step=args.st_timestep,
        quant_hub_map=quant_hub_map,
    ).to(device, dtype=weight_dtype)

    for n_concept in range(len(cpes)):
        print(f"[INFO] Loaded concept checkpoint {n_concept + 1}/{len(cpes)}")
        for k, m in network.named_modules():
            if m.__class__.__name__ == "GLoCELayerOutProp":
                m.eta = args.eta

                for k_child, m_child in m.named_children():
                    module_name = f"{k}.{k_child}"
                    ckpt_module_name = re.sub(r"_core_attn", "_attn", module_name)
                    ckpt_module_name = re.sub(r"_to_out_(\d+)_core\.", r"_to_out_\1.", ckpt_module_name)

                    if ("lora_update" in k_child) or ("lora_degen" in k_child):
                        m_child.weight.data[n_concept] = cpes[n_concept][ckpt_module_name + ".weight"]

                    elif "bias" in k_child:
                        m_child.weight.data[:, n_concept:n_concept + 1, :] = cpes[n_concept][
                            ckpt_module_name + ".weight"
                        ]

                    elif "selector" in k_child:
                        m_child.select_weight.weight.data[n_concept] = cpes[n_concept][
                            ckpt_module_name + ".select_weight.weight"
                        ].squeeze(0)
                        m_child.select_mean_diff.weight.data[n_concept] = cpes[n_concept][
                            ckpt_module_name + ".select_mean_diff.weight"
                        ].squeeze(0)
                        m_child.imp_center[n_concept] = cpes[n_concept][ckpt_module_name + ".imp_center"]
                        m_child.imp_slope[n_concept] = cpes[n_concept][ckpt_module_name + ".imp_slope"]

    network.to(device, dtype=weight_dtype)
    network.eval()

    jobs = build_generation_plan(schema, include_anchor=args.include_anchor)
    save_metadata(args.out_dir, schema, jobs, args)

    print(f"[INFO] Total generations to run: {len(jobs)}")

    global_idx = args.start_index
    pbar = tqdm(jobs, desc="Generating", total=len(jobs))

    with torch.no_grad():
        for job in pbar:
            group_dir = os.path.join(args.out_dir, job["group"], safe_name(job["concept"]))
            ensure_dir(group_dir)

            out_name = (
                f"{global_idx:05d}"
                f"_t{job['template_idx']:02d}"
                f"_s{job['seed']:02d}"
                f"_{safe_name(job['concept'])}.png"
            )
            out_path = os.path.join(group_dir, out_name)

            if os.path.exists(out_path):
                pbar.set_postfix_str(f"skip s={job['seed']}")
                global_idx += 1
                continue

            prompt = job["prompt"]
            prompt_embeds, _ = train_util.encode_prompts(
                tokenizer, text_encoder, [prompt], return_tokens=True
            )

            generator = torch.Generator(device=device)
            generator.manual_seed(int(job["seed"]))

            pbar.set_postfix_str(f"{job['group']} | {safe_name(job['concept'])} | s={job['seed']}")

            with network:
                images = pipe(
                    negative_prompt=schema["negative_prompt"],
                    width=args.width,
                    height=args.height,
                    num_inference_steps=int(schema["num_inference_steps"]),
                    guidance_scale=float(schema["guidance_scale"]),
                    callback_on_start=step_start_callback,
                    generator=generator,
                    num_images_per_prompt=1,
                    prompt_embeds=prompt_embeds,
                ).images

            save_image(images[0], job, args.out_dir, global_idx)
            global_idx += 1


def main(args):
    model_paths = discover_concept_ckpts(args.model_path)

    args.find_module_name = args.find_module_name.split(",")
    if isinstance(args.find_module_name, str):
        args.find_module_name = [args.find_module_name]

    schema = load_schema(args.schema_path)

    infer_with_gloce(
        args,
        model_paths=model_paths,
        schema=schema,
        quant_ckpt_path=args.quantized_ckpt_path,
        precision=args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str,
                        help="Directory containing <concept>/ckpt.safetensors or a single ckpt.safetensors")
    parser.add_argument("--quantized_ckpt_path", type=str, required=True,
                        help="Quantized SD checkpoint path")
    parser.add_argument("--schema_path", type=str, required=True,
                        help="Path to schema JSON")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")

    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gate_rank", type=int, default=1)
    parser.add_argument("--update_rank", type=int, default=16)
    parser.add_argument("--degen_rank", type=int, default=2)
    parser.add_argument("--st_timestep", type=int, default=10)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--find_module_name", type=str, default="unet_ca_out")
    parser.add_argument("--last_layer", type=str, default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--include_anchor", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)

    args = parser.parse_args()
    main(args)