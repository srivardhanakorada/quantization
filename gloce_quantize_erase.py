#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# HF compatibility patch BEFORE diffusers-related imports
# -------------------------------------------------------------------------
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

# -------------------------------------------------------------------------
# PCR imports required before torch.load(...)
# -------------------------------------------------------------------------
import quantization_tools.quantization.layers  # noqa: F401
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GLOCE_ROOT = os.path.join(REPO_ROOT, "gloce")
if GLOCE_ROOT not in sys.path:
    sys.path.insert(0, GLOCE_ROOT)

from gloce.src.models.merge_gloce import load_state_dict  # type: ignore
from gloce.src.models.gloce import GLoCENetworkOutProp, ParamModule, SimpleSelectorOutProp  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
GLOCE_FIND_NAME = "unet_ca_out"

all_quant_layers = {}


# -------------------------------------------------------------------------
# Quant callback logic
# -------------------------------------------------------------------------
def step_start_callback(step: int, timestep: int):
    global all_quant_layers
    for _, layer in all_quant_layers.items():
        if not hasattr(layer, "quantizer"):
            continue
        for quantizer in layer.quantizer:
            if hasattr(quantizer, "set_curr_step"):
                quantizer.set_curr_step(step)


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
        "quant_ckpt": args.quant_ckpt,
        "gloce_model_path": args.gloce_model_path,
        "guidance_scale": schema["guidance_scale"],
        "num_inference_steps": schema["num_inference_steps"],
        "negative_prompt": schema["negative_prompt"],
        "num_jobs": len(jobs),
        "include_anchor": args.include_anchor,
        "device": args.device,
        "width": args.width,
        "height": args.height,
        "gate_rank": args.gate_rank,
        "update_rank": args.update_rank,
        "degen_rank": args.degen_rank,
        "eta": args.eta,
        "st_timestep": args.st_timestep,
        "find_module_name": args.find_module_name,
        "last_layer": args.last_layer,
        "mode": "PCR quantized base + runtime-attached GLoCE (manual concept-aware load)",
    }
    save_json(os.path.join(out_dir, "run_config.json"), meta)
    save_json(os.path.join(out_dir, "resolved_schema.json"), schema)

    with open(os.path.join(out_dir, "generation_manifest.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


# -------------------------------------------------------------------------
# GLoCE model discovery
# -------------------------------------------------------------------------
def discover_concept_ckpts(model_root: str, ckpt_name: str = "ckpt.safetensors") -> List[str]:
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


# -------------------------------------------------------------------------
# PCR loading
# -------------------------------------------------------------------------
def load_pcr_quantized_pipe(quant_ckpt: str, device: str = DEVICE):
    print(f"[INFO] Loading PCR quantized checkpoint: {quant_ckpt}")
    pipe = torch.load(quant_ckpt, map_location="cpu")
    pipe = pipe.to(device)

    pipe.unet.eval()
    pipe.text_encoder.eval()

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    global all_quant_layers
    layers_linear = find_layers(pipe.unet, (LinearQuantHub,))
    layers_conv = find_layers(pipe.unet, (Conv2dQuantHub,))
    all_quant_layers = {**layers_linear, **layers_conv}

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    core_unet = getattr(pipe.unet, "model", pipe.unet)

    return pipe, tokenizer, text_encoder, core_unet


# -------------------------------------------------------------------------
# PCR-compatible module helpers
# -------------------------------------------------------------------------
def resolve_linear_like(module: nn.Module) -> nn.Module:
    seen = set()
    cur = module

    unwrap_attrs = [
        "org_module",
        "module",
        "layer",
        "linear",
        "fp_module",
        "base_layer",
        "inner",
        "core",
    ]

    while True:
        if id(cur) in seen:
            return cur
        seen.add(id(cur))

        if hasattr(cur, "out_features"):
            return cur

        if hasattr(cur, "weight") and isinstance(getattr(cur, "weight"), (torch.Tensor, nn.Parameter)):
            return cur

        nxt = None
        for attr in unwrap_attrs:
            if hasattr(cur, attr):
                cand = getattr(cur, attr)
                if isinstance(cand, nn.Module):
                    nxt = cand
                    break

        if nxt is None:
            return cur
        cur = nxt


def infer_out_dim(org_module: nn.Module) -> int:
    base = resolve_linear_like(org_module)

    if hasattr(base, "out_features"):
        return int(base.out_features)

    w = getattr(base, "weight", None)
    if isinstance(w, nn.Parameter):
        return int(w.data.shape[0])
    if isinstance(w, torch.Tensor):
        return int(w.shape[0])

    for p in base.parameters(recurse=False):
        if p is not None and p.ndim >= 1:
            return int(p.shape[0])

    raise RuntimeError(
        f"Could not infer out_dim for module type={type(org_module)} "
        f"(resolved base={type(base)})."
    )


def collect_target_modules(core_unet: nn.Module) -> Dict[str, nn.Module]:
    org_modules = {}
    for name, module in core_unet.named_modules():
        if name.endswith("attn2.to_out.0"):
            org_modules[name] = module

    if len(org_modules) == 0:
        raise RuntimeError("No modules ending with 'attn2.to_out.0' found in quantized UNet.")

    return org_modules


# -------------------------------------------------------------------------
# PCR-compatible runtime layer
# -------------------------------------------------------------------------
class PCRCompatibleGLoCELayerOutProp(nn.Module):
    """
    Same runtime attach style as before, but parameter loading is handled
    manually later in concept-aware fashion.
    """

    def __init__(
        self,
        find_name,
        gloce_name,
        gloce_org_name,
        org_module,
        multiplier=1.0,
        alpha=1.0,
        gate_rank=1,
        update_rank=16,
        degen_rank=2,
        n_concepts=1,
        last_layer_name="",
        use_update=True,
        use_degen=True,
        use_bias=True,
        use_gate=True,
        st_step=10,
        **kwargs,
    ):
        super().__init__()

        out_dim = infer_out_dim(org_module)

        self.find_name = find_name
        self.gloce_name = gloce_name
        self.gloce_org_name = gloce_org_name

        self.use_update = use_update
        self.use_degen = use_degen
        self.use_bias = use_bias
        self.use_gate = use_gate

        self.lora_update = ParamModule((n_concepts, out_dim, degen_rank))
        self.lora_degen = ParamModule((n_concepts, out_dim, degen_rank))
        self.bias = ParamModule((1, n_concepts, out_dim))
        self.debias = ParamModule((1, n_concepts, out_dim))

        nn.init.zeros_(self.lora_update.weight)
        nn.init.zeros_(self.lora_degen.weight)

        is_last_layer = (gloce_org_name == last_layer_name)
        self.selector = SimpleSelectorOutProp(
            gate_rank=gate_rank,
            d_model=out_dim,
            n_concepts=n_concepts,
            is_last_layer=is_last_layer,
        )

        self.multiplier = multiplier
        self.eta = 1.0
        self.st_step = st_step

        self.n_step = 51
        self.t_counter = 0
        self.use_prompt_tuning = False
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

    def forward(self, x):
        x = self.org_forward(x)
        self.t_counter += 1

        if self.use_prompt_tuning and (self.t_counter > self.st_step):
            idx, scale = self.selector(x)
            idx_bt = idx.squeeze(1)
            scale_bt1 = scale.permute(0, 2, 1)

            debias = self.debias.weight.squeeze(0)[idx_bt]
            x_debias = x - debias

            upd = self.lora_update.weight[idx_bt]
            deg = self.lora_degen.weight[idx_bt]

            mod = torch.einsum("btdh,btd->bth", upd, x_debias)
            mod = torch.einsum("btdh,bth->btd", deg, mod)

            bias = self.bias.weight.squeeze(0)[idx_bt]
            out = bias + self.eta * mod

            if not self.use_gate:
                scale_bt1 = torch.ones_like(scale_bt1)

            if self.t_counter == self.n_step:
                self.t_counter = 0

            return (1.0 - scale_bt1) * x + scale_bt1 * out

        return x


# -------------------------------------------------------------------------
# Key mapping for manual concept-aware load
# -------------------------------------------------------------------------
def normalize_module_key(k: str) -> str:
    out = k
    out = out.replace("_core_", "_")
    out = out.replace(".core.", ".")
    out = out.replace("__", "_")
    return out


def find_ckpt_key_for_sd(module_name: str, cpe: Dict[str, torch.Tensor], suffix: str) -> str:
    """
    Match PCR network module key to FP checkpoint key by removing 'core'
    from the network side when needed.
    """
    sd_key = module_name + suffix
    if sd_key in cpe:
        return sd_key

    norm_sd = normalize_module_key(sd_key)
    for ck in cpe.keys():
        if normalize_module_key(ck) == norm_sd:
            return ck

    raise KeyError(f"Could not map SD key '{sd_key}' to any checkpoint key")


# -------------------------------------------------------------------------
# Build network + manual load
# -------------------------------------------------------------------------
def build_quantized_gloce_network(
    core_unet: nn.Module,
    text_encoder: nn.Module,
    model_paths: List[str],
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
    Build PCR-compatible GLoCE runtime network and load checkpoint tensors
    manually per concept, mirroring generate_by_gloce.py semantics.
    """
    org_modules = collect_target_modules(core_unet)
    print(f"[INFO] Found {len(org_modules)} target modules for GLoCE attach")

    cpes, metadatas = zip(*[load_state_dict(Path(mp), DTYPE) for mp in model_paths])

    alpha = float(metadatas[0]["alpha"]) if "alpha" in metadatas[0] else 1.0

    network = GLoCENetworkOutProp(
        core_unet,
        text_encoder,
        multiplier=1.0,
        alpha=alpha,
        module=PCRCompatibleGLoCELayerOutProp,
        gate_rank=gate_rank,
        update_rank=update_rank,
        degen_rank=degen_rank,
        n_concepts=len(model_paths),
        org_modules_all=[org_modules],
        module_name_list_all=[list(org_modules.keys())],
        find_module_names=[GLOCE_FIND_NAME],
        last_layer=last_layer,
        st_step=st_timestep,
    ).to(device, dtype=DTYPE)

    loaded_concepts = 0
    loaded_tensors = 0
    examples = []

    for n_concept in range(len(cpes)):
        cpe = cpes[n_concept]
        loaded_concepts += 1

        for k, m in network.named_modules():
            if m.__class__.__name__ == "PCRCompatibleGLoCELayerOutProp":
                m.eta = eta

                for k_child, m_child in m.named_children():
                    module_name = f"{k}.{k_child}"

                    if ("lora_update" in k_child) or ("lora_degen" in k_child):
                        ck = find_ckpt_key_for_sd(module_name, cpe, ".weight")
                        m_child.weight.data[n_concept] = cpe[ck]
                        loaded_tensors += 1
                        if len(examples) < 6:
                            examples.append((module_name + ".weight", ck))

                    elif "bias" in k_child:
                        ck = find_ckpt_key_for_sd(module_name, cpe, ".weight")
                        m_child.weight.data[:, n_concept:n_concept + 1, :] = cpe[ck]
                        loaded_tensors += 1
                        if len(examples) < 6:
                            examples.append((module_name + ".weight", ck))

                    elif "selector" in k_child:
                        ck_w = find_ckpt_key_for_sd(module_name, cpe, ".select_weight.weight")
                        ck_m = find_ckpt_key_for_sd(module_name, cpe, ".select_mean_diff.weight")
                        ck_c = find_ckpt_key_for_sd(module_name, cpe, ".imp_center")
                        ck_s = find_ckpt_key_for_sd(module_name, cpe, ".imp_slope")

                        m_child.select_weight.weight.data[n_concept] = cpe[ck_w].squeeze(0)
                        m_child.select_mean_diff.weight.data[n_concept] = cpe[ck_m].squeeze(0)
                        m_child.imp_center[n_concept] = cpe[ck_c]
                        m_child.imp_slope[n_concept] = cpe[ck_s]
                        loaded_tensors += 4

                        if len(examples) < 6:
                            examples.append((module_name + ".select_weight.weight", ck_w))

    network.to(device, dtype=DTYPE)
    network.eval()

    enabled = 0
    for module in network.modules():
        if hasattr(module, "use_prompt_tuning"):
            setattr(module, "use_prompt_tuning", True)
            enabled += 1

    print(f"[INFO] Loaded {loaded_concepts} concept checkpoint(s)")
    print(f"[INFO] Loaded {loaded_tensors} GLoCE tensor blocks manually")
    print(f"[SANITY] use_prompt_tuning enabled on {enabled} modules")

    if len(examples) > 0:
        print("[DEBUG] Example mapped keys:")
        for sd_key, ck_key in examples:
            print(f"  SD   {sd_key}")
            print(f"  CKPT {ck_key}")

    return network


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
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_start=step_start_callback,
                )
                image = result.images[0]

        image.save(out_path)
        global_idx += 1


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fixed PCR GLoCE inference with schema")

    parser.add_argument("--quant_ckpt", type=str, required=True, help="PCR quantized pipeline checkpoint (.ckpt)")
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

    print(f"[INFO] Using GLoCE model path(s): {model_paths}")

    pipe, tokenizer, text_encoder, core_unet = load_pcr_quantized_pipe(
        args.quant_ckpt, device=args.device
    )

    network = build_quantized_gloce_network(
        core_unet=core_unet,
        text_encoder=text_encoder,
        model_paths=model_paths,
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