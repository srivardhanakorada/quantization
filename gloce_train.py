#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# HF compatibility patch BEFORE diffusers imports
# -------------------------------------------------------------------------
import os
import sys

import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

# -------------------------------------------------------------------------
# Standard imports
# -------------------------------------------------------------------------
from pathlib import Path
import random
import gc
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent
GLOCE_ROOT = REPO_ROOT / "gloce"

for p in [str(REPO_ROOT), str(GLOCE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -------------------------------------------------------------------------
# GLoCE imports
# -------------------------------------------------------------------------
from src.models.merge_gloce import *
from src.models.gloce import (
    GLoCELayerOutProp,
    GLoCENetworkOutProp,
)

import src.engine.train_util as train_util
import src.engine.gloce_util as gloce_util
from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.configs.config import RootConfig
from src.configs.prompt import PromptSettings

# -------------------------------------------------------------------------
# PCR quantization imports
# -------------------------------------------------------------------------
import quantization_tools.quantization.layers  # noqa: F401
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE_CUDA = torch.device("cuda")

all_quant_layers = {}

def map_org_modules_to_quant_wrappers(org_modules: dict, quant_hub_map: dict):
    mapped = {}
    for name, module in org_modules.items():
        wrapped = quant_hub_map.get(id(module), None)
        if wrapped is not None:
            mapped[name] = wrapped
            continue

        # fallback: common wrapped layouts
        found = None
        for attr in ["core", "module", "linear", "inner", "org_module"]:
            if hasattr(module, attr):
                sub = getattr(module, attr)
                found = quant_hub_map.get(id(sub), None)
                if found is not None:
                    break

        mapped[name] = found if found is not None else module

    return mapped

def normalize_module_name(name: str) -> str:
    name = name.replace(".core.", ".")
    if name.endswith(".core"):
        name = name[:-len(".core")]
    return name


def normalize_org_modules(org_modules: dict, module_name_list: list):
    new_org_modules = {}
    new_module_name_list = []

    for name, module in org_modules.items():
        norm_name = normalize_module_name(name)
        new_org_modules[norm_name] = module
        new_module_name_list.append(norm_name)

    return new_org_modules, new_module_name_list

def resolve_config_path(config_file: str, maybe_rel: str | None) -> str | None:
    if maybe_rel is None:
        return None

    p = Path(maybe_rel)

    # 1) absolute path: keep it
    if p.is_absolute():
        return str(p)

    # 2) if it already exists from current working directory, keep it
    if p.exists():
        return str(p.resolve())

    # 3) otherwise resolve relative to the config file location
    base = Path(config_file).resolve().parent
    rebased = (base / p).resolve()
    return str(rebased)


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


def step_start_callback(step: int, timestep: int):
    global all_quant_layers
    for _, layer in all_quant_layers.items():
        if not hasattr(layer, "quantizer"):
            continue
        for quantizer in layer.quantizer:
            if hasattr(quantizer, "set_curr_step"):
                quantizer.set_curr_step(step)


def load_quantized_pipeline(quant_ckpt_path: str, weight_dtype):
    print(f"[INFO] Loading quantized checkpoint: {quant_ckpt_path}")
    pipe = torch.load(quant_ckpt_path, map_location="cpu")
    pipe = pipe.to(DEVICE_CUDA)

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    unet = pipe.unet.to(DEVICE_CUDA, dtype=weight_dtype)

    text_encoder.eval()
    unet.eval()
    unet.requires_grad_(False)

    global all_quant_layers
    all_quant_layers = {
        **find_layers(unet, (LinearQuantHub,)),
        **find_layers(unet, (Conv2dQuantHub,)),
    }
    print(f"[INFO] Found {len(all_quant_layers)} quantized layers.")

    return tokenizer, text_encoder, unet, pipe


def patch_pipe_for_quant_callbacks(pipe):
    orig_call = pipe.__call__

    def wrapped_call(*args, **kwargs):
        if "callback_on_start" not in kwargs:
            kwargs["callback_on_start"] = step_start_callback
        return orig_call(*args, **kwargs)

    pipe.__call__ = wrapped_call
    return pipe


def train(
    config: RootConfig,
    prompts_target: list[PromptSettings],
    prompts_anchor: list[PromptSettings],
    prompts_update: list[PromptSettings],
    args,
):
    n_target_concepts = args.n_target_concepts
    tar_concept_idx = args.tar_concept_idx
    n_anchor_concepts = args.n_anchor_concepts
    st_timestep = args.st_timestep
    end_timestep = args.end_timestep
    n_avail_tokens = args.n_tokens
    gate_rank = args.gate_rank
    update_rank = args.update_rank
    degen_rank = args.degen_rank

    prompts_target = prompts_target[tar_concept_idx:tar_concept_idx + n_target_concepts]

    targets = [prompt.target for prompt in prompts_target]
    anchors = [prompt.target for prompt in prompts_anchor]
    surrogate = [prompts_target[0].neutral]
    updates = [prompt.target for prompt in prompts_update]

    save_path = f"{args.save_path}/{targets[0].replace(' ', '_')}"
    emb_cache_path = f"{args.emb_cache_path}/{targets[0].replace(' ', '_')}"
    register_buffer_path = f"{args.buffer_path}/{targets[0].replace(' ', '_')}"
    emb_cache_fn = args.emb_cache_fn

    if os.path.isfile(f"{save_path}/ckpt.safetensors"):
        print(f"[INFO] ckpt for {tar_concept_idx}-{targets[0]} exists, skipping.")
        return

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts_target]),
        "config": config.json(),
    }
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts_target]),
        "rank": str(config.network.rank),
        "alpha": str(config.network.alpha),
        "quantized_point": "true",
        "quantized_ckpt_path": args.quantized_ckpt_path,
    }

    if config.logging.verbose:
        print(metadata)

    weight_dtype = config_pkg.parse_precision(config.train.precision)
    save_weight_dtype = config_pkg.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, pipe = load_quantized_pipeline(
        args.quantized_ckpt_path,
        weight_dtype=weight_dtype,
    )
    pipe = patch_pipe_for_quant_callbacks(pipe)

    # map inner core linear -> wrapper hub
    quant_hub_map = {}
    for name, module in unet.named_modules():
        if name.endswith("attn2.to_out.0") and isinstance(module, LinearQuantHub):
            quant_hub_map[id(module.core)] = module

    module_types = []
    module_names = []
    org_modules_all = []
    module_name_list_all = []

    for find_module_name in args.find_module_name:
        module_name, module_type = gloce_util.get_module_name_type(find_module_name)
        org_modules, module_name_list = gloce_util.get_modules_list(
            unet, text_encoder, find_module_name, module_name, module_type
        )

        org_modules, module_name_list = normalize_org_modules(org_modules, module_name_list)
        org_modules = map_org_modules_to_quant_wrappers(org_modules, quant_hub_map)

        module_names.append(module_name)
        module_types.append(module_type)
        org_modules_all.append(org_modules)
        module_name_list_all.append(module_name_list)

    network = GLoCENetworkOutProp(
        unet,
        text_encoder,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=GLoCELayerOutProp,
        gate_rank=gate_rank,
        update_rank=update_rank,
        degen_rank=degen_rank,
        n_concepts=1,
        org_modules_all=org_modules_all,
        module_name_list_all=module_name_list_all,
        find_module_names=args.find_module_name,
        last_layer=args.last_layer,
        quant_hub_map=quant_hub_map,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    print()
    print("gate rank of network:", config.network.init_size)
    print()

    network.eval()

    with torch.no_grad():
        embedding_unconditional = train_util.encode_prompts(tokenizer, text_encoder, [""])

    emb_cache = gloce_util.prepare_text_embedding_token(
        args,
        config,
        prompts_target,
        prompts_anchor,
        prompts_update,
        tokenizer,
        text_encoder,
        train_util,
        DEVICE_CUDA,
        emb_cache_path,
        emb_cache_fn,
        n_avail_tokens=n_avail_tokens,
        n_anchor_concepts=n_anchor_concepts,
    )

    embeddings_surrogate_sel_base = emb_cache["embeddings_surrogate_sel_base"]
    embeddings_target_sel_base = emb_cache["embeddings_target_sel_base"]
    embeddings_anchor_sel_base = emb_cache["embeddings_anchor_sel_base"]
    embeddings_update_sel_base = emb_cache["embeddings_update_sel_base"]

    embeddings_surrogate_cache = emb_cache["embeddings_surrogate_cache"]
    embeddings_target_cache = emb_cache["embeddings_target_cache"]
    embeddings_anchor_cache = emb_cache["embeddings_anchor_cache"]
    embeddings_update_cache = emb_cache["embeddings_update_cache"]

    prmpt_scripts_sur = emb_cache["prmpt_scripts_sur"]
    prmpt_scripts_tar = emb_cache["prmpt_scripts_tar"]
    prmpt_scripts_anc = emb_cache["prmpt_scripts_anc"]
    prmpt_scripts_upd = emb_cache["prmpt_scripts_upd"]

    prompt_scripts_path = config.scripts_file
    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list = prompt_scripts_df["prompt"].to_list()
    len_prmpts_list = len(prompt_scripts_list) + 1

    use_prompt = ("unet_ca_v" in args.find_module_name) or ("unet_ca_outv" in args.find_module_name)
    if config.replace_word == "artist" and use_prompt:
        embeddings_surrogate_sel_base = embeddings_surrogate_cache
        embeddings_target_sel_base = embeddings_target_cache
        embeddings_anchor_sel_base = embeddings_anchor_cache
        embeddings_update_sel_base = embeddings_update_cache

        surrogate = prmpt_scripts_sur
        targets = prmpt_scripts_tar
        anchors = prmpt_scripts_anc
        updates = prmpt_scripts_upd

    target_selected = prmpt_scripts_tar[len_prmpts_list - 1::len_prmpts_list]
    print("target concept:", target_selected)

    anchor_selected = prmpt_scripts_anc[len_prmpts_list - 1::len_prmpts_list]
    print("anchor concept:", anchor_selected)

    surrogate_selected = prmpt_scripts_sur[len_prmpts_list - 1::len_prmpts_list]
    print("surrogate concept:", surrogate_selected)

    neutral_selected = prmpt_scripts_upd[len_prmpts_list - 1::len_prmpts_list]
    print("neutral concept:", neutral_selected)

    register_buffer_fn = "stacked_surrogate.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_surrogate = gloce_util.get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        surrogate,
        embeddings_surrogate_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn,
        register_func,
    )

    Vh_sur_dict = {}
    surrogate_mean_dict = {}
    for find_name in args.find_module_name:
        Vh_sur_dict[find_name] = {}
        surrogate_mean_dict[find_name] = {}

    for gloce_module in network.gloce_layers:
        n_forward = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]["n_forward"]
        n_sum_per_forward = buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]["n_sum_per_forward"]
        n_sum = n_forward * n_sum_per_forward

        stacked_buffer_surrogate = (
            buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]["data"] / n_sum
        )
        stacked_buffer_surrogate_mean = (
            buffer_sel_basis_surrogate[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum
        )
        stacked_buffer_surrogate_cov = (
            stacked_buffer_surrogate - stacked_buffer_surrogate_mean.T @ stacked_buffer_surrogate_mean
        )

        _, S_sur, Vh_sur = torch.linalg.svd(stacked_buffer_surrogate_cov, full_matrices=False)
        Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_sur
        surrogate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_surrogate_mean

        gloce_module.lora_degen.weight.data = Vh_sur[:degen_rank].T.contiguous()
        gloce_module.bias.weight.data = stacked_buffer_surrogate_mean.unsqueeze(0).clone().contiguous()

    register_buffer_fn = "stacked_target.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_target = gloce_util.get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        targets,
        embeddings_target_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn,
        register_func,
    )

    target_mean_dict = {}
    target_cov_dict = {}
    Vh_tar_dict = {}
    for find_name in args.find_module_name:
        target_mean_dict[find_name] = {}
        Vh_tar_dict[find_name] = {}
        target_cov_dict[find_name] = {}

    for gloce_module in network.gloce_layers:
        n_forward = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["n_forward"]
        n_sum_per_forward = buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["n_sum_per_forward"]
        n_sum = n_forward * n_sum_per_forward

        stacked_buffer_target_mean = (
            buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum
        )
        stacked_buffer_target = (
            buffer_sel_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["data"] / n_sum
        )
        stacked_buffer_target_cov = (
            stacked_buffer_target - stacked_buffer_target_mean.T @ stacked_buffer_target_mean
        )

        _, S_tar, Vh_tar = torch.linalg.svd(stacked_buffer_target_cov, full_matrices=False)
        Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_tar[:update_rank]
        target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_mean
        target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_target_cov

    for gloce_module in network.gloce_layers:
        Vh_upd = Vh_tar_dict[gloce_module.find_name][gloce_module.gloce_org_name][:update_rank]
        target_mean = target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name].squeeze(0)
        dim_emb = Vh_upd.size(1)

        Vh_sur = Vh_sur_dict[gloce_module.find_name][gloce_module.gloce_org_name][:degen_rank]
        gloce_module.lora_update.weight.data = (
            Vh_sur @ (torch.eye(dim_emb).to(DEVICE_CUDA) - Vh_upd.T @ Vh_upd)
        ).T.contiguous()
        gloce_module.debias.weight.data = target_mean.unsqueeze(0).unsqueeze(0).clone().contiguous()

    register_buffer_fn = "stacked_gate.pt"
    register_func = "register_sum_buffer_avg_spatial"

    buffer_sel_basis_gate = gloce_util.get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        updates,
        embeddings_update_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn,
        register_func,
    )

    Vh_gate_dict = {}
    gate_mean_dict = {}
    rel_gate_dict = {}
    for find_name in args.find_module_name:
        Vh_gate_dict[find_name] = {}
        gate_mean_dict[find_name] = {}
        rel_gate_dict[find_name] = {}

    for gloce_module in network.gloce_layers:
        n_forward = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]["n_forward"]
        n_sum_per_forward = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]["n_sum_per_forward"]
        n_sum = n_forward * n_sum_per_forward

        stacked_buffer_gate = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]["data"] / n_sum
        stacked_buffer_gate_mean = buffer_sel_basis_gate[gloce_module.find_name][gloce_module.gloce_org_name]["data_mean"] / n_sum
        stacked_buffer_gate_cov = stacked_buffer_gate - stacked_buffer_gate_mean.T @ stacked_buffer_gate_mean

        stacked_buffer_rel_mean = (
            target_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name]
            - stacked_buffer_gate_mean
        )
        stacked_buffer_rel_cov = (
            target_cov_dict[gloce_module.find_name][gloce_module.gloce_org_name]
            + stacked_buffer_rel_mean.T @ stacked_buffer_rel_mean
        )

        _, S_tar, Vh_gate = torch.linalg.svd(stacked_buffer_rel_cov, full_matrices=False)
        rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name] = Vh_gate[:gate_rank]
        gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name] = stacked_buffer_gate_mean

    register_buffer_fn = "norm_target.pt"
    register_func = "register_norm_buffer_avg_spatial"

    buffer_norm_basis_target = gloce_util.get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        targets,
        embeddings_target_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn,
        register_func,
        rel_gate_dict=rel_gate_dict,
        target_mean_dict=target_mean_dict,
        gate_mean_dict=gate_mean_dict,
    )

    register_buffer_fn = "norm_anchor.pt"
    register_func = "register_norm_buffer_avg_spatial"

    buffer_norm_basis_anchor = gloce_util.get_registered_buffer(
        args,
        module_name_list_all,
        org_modules_all,
        st_timestep,
        end_timestep,
        n_avail_tokens,
        anchors,
        embeddings_anchor_sel_base,
        embedding_unconditional,
        pipe,
        DEVICE_CUDA,
        register_buffer_path,
        register_buffer_fn,
        register_func,
        rel_gate_dict=rel_gate_dict,
        target_mean_dict=target_mean_dict,
        gate_mean_dict=gate_mean_dict,
    )

    for gloce_module in network.gloce_layers:
        n_forward_tar = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["n_forward"]
        n_forward_anc = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]["n_forward"]
        n_sum_tar = n_forward_tar
        n_sum_anc = n_forward_anc

        importance_tgt = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["data_max"] / n_sum_tar
        importance_anc = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]["data_max"] / n_sum_anc

        importance_tgt_stack = buffer_norm_basis_target[gloce_module.find_name][gloce_module.gloce_org_name]["data_stack"]
        importance_anc_stack = buffer_norm_basis_anchor[gloce_module.find_name][gloce_module.gloce_org_name]["data_stack"]
        importance_tgt_stack = torch.cat([imp.unsqueeze(0) for imp in importance_tgt_stack], dim=0)
        importance_anc_stack = torch.cat([imp.unsqueeze(0) for imp in importance_anc_stack], dim=0)

        print(gloce_module.gloce_org_name)
        print(f"Relative importance {(importance_tgt / importance_anc).item()}")

        tol1 = args.thresh
        x_center = importance_anc_stack.mean() + tol1 * importance_anc_stack.std()
        tol2 = 0.001 * tol1

        c_right = torch.tensor([0.99]).to(DEVICE_CUDA)
        C_right = torch.log(1 / (1 / c_right - 1))

        imp_center = x_center
        imp_slope = C_right / tol2

        print(
            f"{importance_anc_stack.max().item():10.5f}, "
            f"{imp_center.item():10.5f}, "
            f"{importance_tgt_stack.min().item():10.5f}, "
            f"{importance_tgt_stack.max().item():10.5f}"
        )

        Vh_gate = rel_gate_dict[gloce_module.find_name][gloce_module.gloce_org_name]
        gate_mean = gate_mean_dict[gloce_module.find_name][gloce_module.gloce_org_name]

        gloce_module.selector.select_weight.weight.data = Vh_gate.T.unsqueeze(0).clone().contiguous()
        gloce_module.selector.select_mean_diff.weight.data = gate_mean.clone().contiguous()

        gloce_module.selector.imp_center = imp_center
        gloce_module.selector.imp_slope = imp_slope
        print()

    print("[INFO] Saving quantized-point GLoCE parameters...")
    save_path = Path(f"{save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    network.save_weights(
        save_path / "ckpt.safetensors",
        dtype=save_weight_dtype,
        metadata=model_metadata,
    )

    flush()
    print("[INFO] Done.")


def main(args):
    config_file = args.config_file

    with open(config_file, "r") as f:
        raw_cfg = yaml.load(f, Loader=yaml.FullLoader)

    if "pretrained_model" not in raw_cfg:
        raw_cfg["pretrained_model"] = {
            "name_or_path": args.quantized_ckpt_path,
            "v2": False,
            "v_pred": False,
        }

    config = config_pkg.RootConfig(**raw_cfg)

    if config.train is None:
        config.train = config_pkg.TrainConfig()
    if config.save is None:
        config.save = config_pkg.SaveConfig()
    if config.logging is None:
        config.logging = config_pkg.LoggingConfig()
    if config.inference is None:
        config.inference = config_pkg.InferenceConfig()
    if config.other is None:
        config.other = config_pkg.OtherConfig()

    config.prompts_file_target = resolve_config_path(args.config_file, config.prompts_file_target)
    config.prompts_file_anchor = resolve_config_path(args.config_file, config.prompts_file_anchor)
    config.prompts_file_update = resolve_config_path(args.config_file, config.prompts_file_update)
    config.scripts_file = resolve_config_path(args.config_file, config.scripts_file)

    prompts_target = prompt_pkg.load_prompts_from_yaml(config.prompts_file_target)
    prompts_anchor = prompt_pkg.load_prompts_from_yaml(config.prompts_file_anchor)
    prompts_update = prompt_pkg.load_prompts_from_yaml(config.prompts_file_update)

    if args.gate_rank != -1:
        config.network.init_size = args.gate_rank
        config.network.hidden_size = args.gate_rank
        config.network.continual_rank = args.gate_rank

    if args.update_rank != -1:
        config.network.rank = args.update_rank

    base_logging_prompts = config.logging.prompts

    for p_idx, p in enumerate(prompts_target):
        config.logging.prompts = [
            prompt.replace("[target]", p.target) if "[target]" in prompt else prompt
            for prompt in base_logging_prompts
        ]

    args.find_module_name = args.find_module_name.split(",")
    if args.find_module_name.__class__ == str:
        args.find_module_name = [args.find_module_name]

    seed_everything(config.train.train_seed)
    train(config, prompts_target, prompts_anchor, prompts_update, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True, help="Config file for training.")
    parser.add_argument("--quantized_ckpt_path", required=True, type=str,
                        help="Path to PCR quantized-point SD checkpoint")

    parser.add_argument("--st_prompt_idx", type=int, default=-1)
    parser.add_argument("--end_prompt_idx", type=int, default=-1)
    parser.add_argument("--update_rank", type=int, default=-1)
    parser.add_argument("--degen_rank", type=int, default=-1)
    parser.add_argument("--gate_rank", type=int, default=-1)
    parser.add_argument("--n_tokens", type=int, default=-1)
    parser.add_argument("--eta", type=float, default=-1)
    parser.add_argument("--lamb", type=float, default=-1)
    parser.add_argument("--lamb2", type=float, default=-1)
    parser.add_argument("--p_val", type=float, default=-1)
    parser.add_argument("--find_module_name", type=str, default="unet_ca")

    parser.add_argument("--n_target_concepts", type=int, default=1, help="Number of target concepts")
    parser.add_argument("--n_anchor_concepts", type=int, default=5, help="Number of anchor concepts")
    parser.add_argument("--tar_concept_idx", type=int, default=0, help="Target concept index")
    parser.add_argument("--st_timestep", type=int, default=10, help="Start timestep")
    parser.add_argument("--end_timestep", type=int, default=20, help="End timestep")
    parser.add_argument("--n_generation_per_concept", type=int, default=3, help="Number of generations per concept")
    parser.add_argument("--sel_basis_buffer_fn", action="store_true", help="Select basis buffer function")

    parser.add_argument("--param_cache_path", type=str, default="./importance_cache/org_comps/sd_v1.4",
                        help="Path to parameter cache (kept for compatibility)")
    parser.add_argument("--emb_cache_path", type=str, default="./importance_cache/text_embs/sd_v1.4",
                        help="Path to embedding cache")
    parser.add_argument("--emb_cache_fn", type=str, default="text_emb_cache.pt",
                        help="Embedding cache file name")
    parser.add_argument("--buffer_path", type=str, default="./importance_cache/buffers")
    parser.add_argument("--use_emb_cache", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--last_layer", type=str, default="")
    parser.add_argument("--opposite_for_map", type=bool, default=False)
    parser.add_argument("--thresh", type=float, default=1.5)

    args = parser.parse_args()
    main(args)