#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import random
import argparse
from typing import Dict, List, Any

import numpy as np
import torch
from tqdm import tqdm

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# HF compatibility patch MUST happen before importing diffusers
# -------------------------------------------------------------------------
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GLOCE_ROOT = os.path.join(REPO_ROOT, "gloce")
if GLOCE_ROOT not in sys.path:
    sys.path.insert(0, GLOCE_ROOT)

from gloce.src.models.merge_gloce import load_state_dict  # type: ignore
from gloce.src.models.gloce import GLoCELayerOutProp, GLoCENetworkOutProp  # type: ignore
from gloce.src.engine import train_util  # type: ignore
import gloce.src.engine.gloce_util as gloce_util  # type: ignore
from gloce.src.models import model_util  # type: ignore
from gloce.src.configs.config import parse_precision  # type: ignore
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------
def set_global_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "", text)
    return text[:180]


# -------------------------------------------------------------------------
# Schema helpers
# -------------------------------------------------------------------------
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

    if not isinstance(schema["erase_concepts"], list):
        raise ValueError("erase_concepts must be a list")
    if not isinstance(schema["preserve_concepts"], list):
        raise ValueError("preserve_concepts must be a list")
    if not isinstance(schema["anchor_concepts"], list):
        raise ValueError("anchor_concepts must be a list")
    if not isinstance(schema["prompt_templates"], list):
        raise ValueError("prompt_templates must be a list")
    if not isinstance(schema["seeds"], list):
        raise ValueError("seeds must be a list")
    if not isinstance(schema["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")

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


def save_run_metadata(out_dir: str, schema: Dict[str, Any], jobs: List[Dict[str, Any]], args) -> None:
    meta = {
        "schema_path": args.schema_path,
        "schema_name": schema.get("schema_name", "unknown"),
        "base_model": args.base_model,
        "gloce_model_path": args.gloce_model_path,
        "guidance_scale": schema["guidance_scale"],
        "num_inference_steps": schema["num_inference_steps"],
        "negative_prompt": schema["negative_prompt"],
        "num_jobs": len(jobs),
        "include_anchor": args.include_anchor,
        "device": args.device,
        "width": args.width,
        "height": args.height,
        "precision": args.precision,
        "gate_rank": args.gate_rank,
        "update_rank": args.update_rank,
        "degen_rank": args.degen_rank,
        "eta": args.eta,
        "st_timestep": args.st_timestep,
        "find_module_name": args.find_module_name,
        "last_layer": args.last_layer,
        "mode": "FP base + runtime-attached GLoCE (manual concept-aware load)",
    }
    save_json(os.path.join(out_dir, "run_config.json"), meta)
    save_json(os.path.join(out_dir, "resolved_schema.json"), schema)

    with open(os.path.join(out_dir, "generation_manifest.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


# -------------------------------------------------------------------------
# GLoCE model discovery / loading
# -------------------------------------------------------------------------
def discover_concept_ckpts(model_root: str, ckpt_name: str = "ckpt.safetensors") -> List[str]:
    """
    Expects model_root to look like:
      <root>/<concept_1>/ckpt.safetensors
      <root>/<concept_2>/ckpt.safetensors
      ...

    If model_root itself is a ckpt.safetensors file, returns [model_root].
    """
    if os.path.isfile(model_root):
        return [model_root]

    concept_ckpts = []
    for entry in sorted(os.listdir(model_root)):
        full = os.path.join(model_root, entry)
        cand = os.path.join(full, ckpt_name)
        if os.path.isfile(cand):
            concept_ckpts.append(cand)

    if len(concept_ckpts) == 0:
        raise RuntimeError(f"No concept checkpoints found under: {model_root}")

    return concept_ckpts


def build_gloce_network(
    base_model: str,
    model_paths: List[str],
    precision: str,
    gate_rank: int,
    update_rank: int,
    degen_rank: int,
    eta: float,
    st_timestep: int,
    find_module_name: str,
    last_layer: str,
    device: str,
):
    """
    This mirrors the original generate_by_gloce.py loading semantics:
    - load base model through GLoCE utilities
    - collect original target modules through gloce_util
    - build GLoCENetworkOutProp with original GLoCELayerOutProp
    - manually load each concept checkpoint into the correct concept slot
    """
    weight_dtype = parse_precision(precision)

    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=False,
        weight_dtype=weight_dtype,
    )

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    # Register modules exactly the original way
    find_module_names = find_module_name.split(",")
    if isinstance(find_module_names, str):
        find_module_names = [find_module_names]

    org_modules_all = []
    module_name_list_all = []

    for fm in find_module_names:
        module_name, module_type = gloce_util.get_module_name_type(fm)
        org_modules, module_name_list = gloce_util.get_modules_list(
            unet, text_encoder, fm, module_name, module_type
        )
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    cpes, metadatas = zip(*[load_state_dict(Path(mp), weight_dtype) for mp in model_paths])

    assert all(md["rank"] == metadatas[0]["rank"] for md in metadatas), "Incompatible GLoCE ranks"

    network = GLoCENetworkOutProp(
        unet,
        text_encoder,
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=GLoCELayerOutProp,
        degen_rank=degen_rank,
        gate_rank=gate_rank,
        update_rank=update_rank,
        n_concepts=len(model_paths),
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names=find_module_names,
        last_layer=last_layer,
        st_step=st_timestep,
    ).to(device, dtype=weight_dtype)

    loaded_concepts = 0
    loaded_tensors = 0

    for n_concept in range(len(cpes)):
        loaded_concepts += 1
        for k, m in network.named_modules():
            if m.__class__.__name__ == "GLoCELayerOutProp":
                m.eta = eta

                for k_child, m_child in m.named_children():
                    module_name = f"{k}.{k_child}"

                    if ("lora_update" in k_child) or ("lora_degen" in k_child):
                        key = module_name + ".weight"
                        m_child.weight.data[n_concept] = cpes[n_concept][key]
                        loaded_tensors += 1

                    elif "bias" in k_child:
                        key = module_name + ".weight"
                        m_child.weight.data[:, n_concept:n_concept + 1, :] = cpes[n_concept][key]
                        loaded_tensors += 1

                    elif "selector" in k_child:
                        key_w = module_name + ".select_weight.weight"
                        key_m = module_name + ".select_mean_diff.weight"
                        key_c = module_name + ".imp_center"
                        key_s = module_name + ".imp_slope"

                        m_child.select_weight.weight.data[n_concept] = cpes[n_concept][key_w].squeeze(0)
                        m_child.select_mean_diff.weight.data[n_concept] = cpes[n_concept][key_m].squeeze(0)
                        m_child.imp_center[n_concept] = cpes[n_concept][key_c]
                        m_child.imp_slope[n_concept] = cpes[n_concept][key_s]
                        loaded_tensors += 4

    network.to(device, dtype=weight_dtype)
    network.eval()

    enabled = 0
    for module in network.modules():
        if hasattr(module, "use_prompt_tuning"):
            setattr(module, "use_prompt_tuning", True)
            enabled += 1

    print(f"[INFO] Loaded {loaded_concepts} concept checkpoint(s)")
    print(f"[INFO] Loaded {loaded_tensors} GLoCE tensor blocks manually")
    print(f"[SANITY] use_prompt_tuning enabled on {enabled} modules")

    return pipe, tokenizer, text_encoder, network, weight_dtype


# -------------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------------
def run_generation(
    pipe,
    network,
    jobs: List[Dict[str, Any]],
    out_dir: str,
    guidance_scale: float,
    num_inference_steps: int,
    negative_prompt: str,
    width: int,
    height: int,
    device: str,
    start_index: int = 0,
):
    ensure_dir(out_dir)

    pbar = tqdm(jobs, desc="Generating", total=len(jobs))
    global_idx = start_index

    for job in pbar:
        group_dir = os.path.join(out_dir, job["group"], safe_name(job["concept"]))
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

        generator = torch.Generator(device=device)
        generator.manual_seed(int(job["seed"]))

        pbar.set_postfix_str(
            f"{job['group']} | {safe_name(job['concept'])} | s={job['seed']}"
        )

        with torch.no_grad():
            with network:
                result = pipe(
                    prompt=job["prompt"],
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                )
                image = result.images[0]

        image.save(out_path)
        global_idx += 1


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fixed FP GLoCE inference with schema")

    parser.add_argument("--base_model", type=str, required=True, help="HF model id or local Diffusers model path")
    parser.add_argument(
        "--gloce_model_path",
        type=str,
        required=True,
        help="Either a single ckpt.safetensors file or a directory containing <concept>/ckpt.safetensors",
    )
    parser.add_argument("--schema_path", type=str, required=True, help="Path to schema JSON")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--include_anchor", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)

    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gate_rank", type=int, default=1)
    parser.add_argument("--update_rank", type=int, default=16)
    parser.add_argument("--degen_rank", type=int, default=2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--st_timestep", type=int, default=10)
    parser.add_argument("--find_module_name", type=str, default="unet_ca_out")
    parser.add_argument("--last_layer", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    set_global_determinism(0)
    ensure_dir(args.out_dir)

    schema = load_schema(args.schema_path)
    jobs = build_generation_plan(schema, include_anchor=args.include_anchor)
    save_run_metadata(args.out_dir, schema, jobs, args)

    model_paths = discover_concept_ckpts(args.gloce_model_path)

    print(f"[INFO] Loading FP base model: {args.base_model}")
    print(f"[INFO] Using GLoCE model path(s): {model_paths}")

    pipe, tokenizer, text_encoder, network, weight_dtype = build_gloce_network(
        base_model=args.base_model,
        model_paths=model_paths,
        precision=args.precision,
        gate_rank=args.gate_rank,
        update_rank=args.update_rank,
        degen_rank=args.degen_rank,
        eta=args.eta,
        st_timestep=args.st_timestep,
        find_module_name=args.find_module_name,
        last_layer=args.last_layer,
        device=args.device,
    )

    print(f"[INFO] Total generations to run: {len(jobs)}")
    print(f"[INFO] Schema name: {schema.get('schema_name', 'unknown')}")

    run_generation(
        pipe=pipe,
        network=network,
        jobs=jobs,
        out_dir=args.out_dir,
        guidance_scale=float(schema["guidance_scale"]),
        num_inference_steps=int(schema["num_inference_steps"]),
        negative_prompt=str(schema["negative_prompt"]),
        width=args.width,
        height=args.height,
        device=args.device,
        start_index=args.start_index,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()