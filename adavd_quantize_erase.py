#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import copy
import math
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

import torch
from torch import nn
torch.set_grad_enabled(False)

import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

# Important for torch.load of quantized ckpt
import quantization_tools.quantization.layers  # noqa: F401

from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub

from diffusers import DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor as DiffusersAttnProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATH_DTYPE = torch.float32
ORTHO_DECOMP_STORAGE = {}
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


def seed_everything_local(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_token(prompt, tokenizer=None):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return tokens


def get_textencoding(input_tokens, text_encoder):
    text_encoding = text_encoder(input_tokens.to(text_encoder.device))[0]
    return text_encoding


def get_eot_idx(tokens):
    return (tokens == 49407).nonzero(as_tuple=True)[1][0].item()


def get_spread_embedding(original_token, idx):
    spread_token = original_token.clone()
    spread_token[:, 1:, :] = original_token[:, idx - 1, :].unsqueeze(1)
    return spread_token


def process_img(decoded_image):
    decoded_image = decoded_image.squeeze(0)
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = (decoded_image * 255).byte()
    decoded_image = decoded_image.permute(1, 2, 0)
    decoded_image = decoded_image.cpu().numpy()
    return Image.fromarray(decoded_image)


def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r") as f:
        schema = json.load(f)

    required = [
        "schema_name",
        "erase_concepts",
        "preserve_concepts",
        "anchor_concepts",
        "guidance_scale",
        "num_inference_steps",
        "negative_prompt",
        "seeds",
        "prompt_templates",
    ]
    for k in required:
        if k not in schema:
            raise ValueError(f"Missing required schema key: {k}")

    if not isinstance(schema["erase_concepts"], list) or len(schema["erase_concepts"]) == 0:
        raise ValueError("erase_concepts must be a non-empty list")

    if not isinstance(schema["preserve_concepts"], list):
        raise ValueError("preserve_concepts must be a list")

    if not isinstance(schema["anchor_concepts"], list):
        raise ValueError("anchor_concepts must be a list")

    if not isinstance(schema["seeds"], list) or len(schema["seeds"]) == 0:
        raise ValueError("seeds must be a non-empty list")

    if not isinstance(schema["prompt_templates"], list) or len(schema["prompt_templates"]) == 0:
        raise ValueError("prompt_templates must be a non-empty list")

    if not isinstance(schema["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")

    return schema


def flatten_concepts(schema: Dict[str, Any], include_anchor: bool) -> Dict[str, List[str]]:
    return {
        "erase": list(schema["erase_concepts"]),
        "preserve": list(schema["preserve_concepts"]),
        "anchor": list(schema["anchor_concepts"]) if include_anchor else [],
    }


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
                    jobs.append({
                        "group": group_name,
                        "concept": concept,
                        "template": template,
                        "template_idx": template_idx,
                        "seed": int(seed),
                        "prompt": prompt,
                    })
    return jobs


def save_metadata(out_dir: str, schema: Dict[str, Any], jobs: List[Dict[str, Any]], args):
    meta = {
        "schema_path": args.schema_path,
        "quant_ckpt": args.quant_ckpt,
        "schema_name": schema.get("schema_name", "unknown"),
        "guidance_scale": schema["guidance_scale"],
        "num_inference_steps": schema["num_inference_steps"],
        "negative_prompt": schema["negative_prompt"],
        "num_jobs": len(jobs),
        "batch_size": args.batch_size,
        "include_anchor": args.include_anchor,
        "record_type": args.record_type,
        "sigmoid_a": args.sigmoid_a,
        "sigmoid_b": args.sigmoid_b,
        "sigmoid_c": args.sigmoid_c,
        "decomp_timestep": args.decomp_timestep,
        "device": args.device,
        "dtype": args.dtype,
        "mode": args.mode,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "resolved_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    with open(os.path.join(out_dir, "generation_manifest.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")


def save_image_with_sidecar(
    img: Image.Image,
    job: Dict[str, Any],
    out_dir: str,
    global_index: int,
    mode: str,
):
    group_dir = os.path.join(out_dir, mode, job["group"], safe_name(job["concept"]))
    ensure_dir(group_dir)

    base_name = (
        f"{global_index:05d}"
        f"_t{job['template_idx']:02d}"
        f"_s{job['seed']:02d}"
        f"_{safe_name(job['concept'])}"
    )
    img.save(os.path.join(group_dir, base_name + ".png"))


class VisualAttentionProcess(nn.Module):
    def __init__(
        self,
        module_name=None,
        atten_type='original',
        target_records=None,
        record=False,
        record_type=None,
        sigmoid_setting=None,
        decomp_timestep=0,
        **kwargs
    ):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = target_records
        self.record = record
        self.record_type = record_type
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor", None)
        attn.processor = AdaVDAttnProcessor(
            self.module_name,
            self.atten_type,
            self.target_records,
            self.record,
            self.record_type,
            self.sigmoid_setting,
            self.decomp_timestep,
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)


class AdaVDAttnProcessor:
    def __init__(
        self,
        module_name=None,
        atten_type='original',
        target_records=None,
        record=False,
        record_type=None,
        sigmoid_setting=None,
        decomp_timestep=0,
    ):
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = copy.copy(target_records) if target_records is not None else {}
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_ortho_decomp(self, target_value, pro_record, ortho_basis=None, project_matrix=None):
        orig_dtype = pro_record.dtype

        target_value = target_value.float()
        pro_record = pro_record.float()
        if ortho_basis is not None:
            ortho_basis = ortho_basis.float()
        if project_matrix is not None:
            project_matrix = project_matrix.float()

        if ortho_basis is None and project_matrix is None:
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1)
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1)

            dot1 = (tar_record_ * pro_record_).sum(-1)
            dot2 = (tar_record_ * tar_record_).sum(-1).clamp_min(1e-12)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)

            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)

            weight = torch.nan_to_num(cos_sim * (dot1 / dot2), nan=0.0)
            weight[0].fill_(0)

            era_record = (
                weight.unsqueeze(0).unsqueeze(-1)
                * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
            )
        else:
            tar_record_ = rearrange(target_value, 'b h l d -> l b (h d)')
            pro_record_ = rearrange(pro_record, 'h l d -> l (h d)').unsqueeze(1)

            dot1 = (ortho_basis * pro_record_).sum(-1)
            dot2 = (ortho_basis * ortho_basis).sum(-1).clamp_min(1e-12)
            weight = torch.nan_to_num((dot1 / dot2).unsqueeze(1), nan=0.0)
            weight[0].fill_(0)

            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)

            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            era_record = torch.bmm(weight, projected_basis).view((77, 16, -1)).permute(1, 0, 2)

        return era_record.to(orig_dtype)

    def gram_schmidt(self, V):
        V = V.float()
        n = len(V)
        project_matrix = torch.zeros((n, n), dtype=V.dtype, device=V.device)
        project_matrix += torch.diag(torch.ones(n, dtype=V.dtype, device=V.device))

        for i in range(1, n):
            vi = V[i:i+1, :]
            for j in range(i):
                qj = V[j:j+1, :]
                denom = torch.dot(qj.view(-1), qj.view(-1)).clamp_min(1e-12)
                project_matrix[i][j] = -torch.dot(qj.view(-1), vi.view(-1)) / denom

        ortho_basis = torch.matmul(project_matrix, V)
        return project_matrix, ortho_basis

    def cal_gram_schmidt(self, target_value):
        target_value = target_value.float()
        target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view(
            (target_value.size()[1], -1) + target_value.size()[-2:]
        )
        target_value_ = rearrange(target_value, 'b h l d -> b l (h d)')
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0)
        basis_ortho = torch.stack([result[1] for result in results], dim=0)
        return project_matrix, basis_ortho

    def record_ortho_decomp(self, target_record, current_record):
        current_name = next(k for k in target_record if k.endswith(self.module_name))
        current_timestep, current_block = current_name.split('.', 1)
        (target_value, project_matrix, ortho_basis) = target_record.pop(current_name)

        if int(current_timestep) <= self.decomp_timestep:
            return current_record, current_record

        if current_block not in ORTHO_DECOMP_STORAGE:
            target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
            target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view(
                (target_value.size()[1], -1) + target_value.size()[-2:]
            )

            current_record = current_record.view((2, int(len(current_record)//16), -1) + target_value.size()[-2:])
            current_record = current_record.permute(1, 0, 2, 3, 4).contiguous().view(
                (current_record.size()[1], -1) + target_value.size()[-2:]
            )

            erase_record, retain_record = [], []
            for pro_record in current_record:
                era_record = self.cal_ortho_decomp(target_value, pro_record, ortho_basis, project_matrix)
                ret_record = pro_record - era_record
                erase_record.append(era_record.view((2, -1) + era_record.size()[-2:]))
                retain_record.append(ret_record.view((2, -1) + ret_record.size()[-2:]))

            retain_record = rearrange(torch.stack(retain_record, dim=0), 'b n c l d -> (n b c) l d')
            erase_record = rearrange(torch.stack(erase_record, dim=0), 'b n c l d -> (n b c) l d')
            ORTHO_DECOMP_STORAGE[current_block] = (erase_record, retain_record)

        return ORTHO_DECOMP_STORAGE[current_block]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'queries' in self.target_records:
                erase_query, retain_query = self.record_ortho_decomp(
                    target_record=self.target_records['queries'],
                    current_record=query,
                )
                query = retain_query if self.atten_type == 'retain' else erase_query if self.atten_type == 'erase' else query

            if 'keys' in self.target_records:
                erase_key, retain_key = self.record_ortho_decomp(
                    target_record=self.target_records['keys'],
                    current_record=key,
                )
                key = retain_key if self.atten_type == 'retain' else erase_key if self.atten_type == 'erase' else key

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'attn_maps' in self.target_records:
                erase_attention_probs, retain_attention_probs = self.record_ortho_decomp(
                    target_record=self.target_records['attn_maps'],
                    current_record=attention_probs,
                )
                attention_probs = (
                    retain_attention_probs if self.atten_type == 'retain'
                    else erase_attention_probs if self.atten_type == 'erase'
                    else attention_probs
                )

        if encoder_hidden_states.shape[1] != 77:
            hidden_states = torch.bmm(attention_probs, value)
        else:
            if self.record:
                for kk, vv in {'queries': query, 'keys': key, 'values': value, 'attn_maps': attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1:
                            self.records[kk][self.module_name] = [vv, None, None]
                        else:
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif 'values' in self.target_records:
                erase_value, retain_value = self.record_ortho_decomp(
                    target_record=self.target_records['values'],
                    current_record=value,
                )
                value = retain_value if self.atten_type == 'retain' else erase_value if self.atten_type == 'erase' else value

            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def reset_attention_processors(unet):
    for name, m in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            try:
                m.set_processor(DiffusersAttnProcessor())
            except Exception:
                pass
    return unet


def set_attenprocessor(
    unet,
    atten_type='original',
    target_records=None,
    record=False,
    record_type=None,
    sigmoid_setting=None,
    decomp_timestep=0
):
    for name, m in unet.named_modules():
        if name.endswith('attn2') or name.endswith('attn1'):
            try:
                m.set_processor(
                    VisualAttentionProcess(
                        module_name=name,
                        atten_type=atten_type,
                        target_records=target_records,
                        record=record,
                        record_type=record_type,
                        sigmoid_setting=sigmoid_setting,
                        decomp_timestep=decomp_timestep,
                    )
                )
            except Exception as e:
                print(f"[WARN] Could not set processor for {name}: {e}")
    return unet


def diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    start_timesteps=0,
    guidance_scale=7.5,
    record=False,
    record_type=None,
    desc=None,
):
    visualize_map_withstep = {key: {} for key in record_type.strip().split(',')} if record_type is not None else {}

    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        if record:
            for t in record_type.strip().split(','):
                for value in unet.attn_processors.values():
                    if hasattr(value, "records"):
                        for k, v in value.records[t].items():
                            visualize_map_withstep[t][f'{timestep.item()}.{k}'] = v

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map_withstep) if record else latents


def load_quantized_pipe(quant_ckpt, device: str):
    print(f"[INFO] Loading quantized checkpoint: {quant_ckpt}")
    pipe = torch.load(quant_ckpt, map_location="cpu")
    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    return pipe


def build_target_concept_encoding(target_concepts: List[str], tokenizer, text_encoder):
    target_concept_encodings_ = [
        get_textencoding(get_token(concept, tokenizer), text_encoder)
        for concept in target_concepts
    ]
    target_eot_idxs = [get_eot_idx(get_token(concept, tokenizer)) for concept in target_concepts]
    target_concept_encoding = [
        get_spread_embedding(target_concept_encoding_, idx)
        for (target_concept_encoding_, idx) in zip(target_concept_encodings_, target_eot_idxs)
    ]
    return torch.concat(target_concept_encoding)


def compute_target_records(
    unet,
    scheduler,
    target_concept_encoding,
    uncond_encoding,
    guidance_scale,
    record_type,
    total_timesteps,
    device,
):
    reset_attention_processors(unet)
    unet = set_attenprocessor(
        unet,
        atten_type='original',
        record=True,
        record_type=record_type,
    )

    _, target_records = diffusion(
        unet=unet,
        scheduler=scheduler,
        latents=torch.zeros(
            len(target_concept_encoding), 4, 64, 64,
            device=device, dtype=target_concept_encoding.dtype
        ),
        text_embeddings=torch.cat(
            [uncond_encoding] * len(target_concept_encoding) + [target_concept_encoding],
            dim=0
        ),
        total_timesteps=1,
        start_timesteps=0,
        guidance_scale=guidance_scale,
        record=True,
        record_type=record_type,
        desc="Calculating target records on quantized model",
    )

    scheduler.set_timesteps(total_timesteps)
    original_keys = list(target_records[record_type].keys())
    target_records[record_type].update({
        f"{timestep}.{'.'.join(key.split('.')[1:])}": target_records[record_type][key]
        for timestep in scheduler.timesteps
        for key in original_keys
    })
    return target_records


def prepare_text_embeddings_for_batch(prompts, negative_prompt, tokenizer, text_encoder, device):
    cond_tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    neg_prompts = [negative_prompt] * len(prompts)
    uncond_tokens = tokenizer(
        neg_prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    cond_encoding = text_encoder(cond_tokens)[0]
    uncond_encoding = text_encoder(uncond_tokens)[0]
    text_embeddings = torch.cat([uncond_encoding, cond_encoding], dim=0)
    return text_embeddings


def decode_latents_to_pil(vae, latents):
    images = []
    for img in latents:
        decoded = vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]
        images.append(process_img(decoded))
    return images


def run_mode_batch(
    mode_name,
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    guidance_scale,
    target_records,
    sigmoid_setting,
    decomp_timestep,
):
    reset_attention_processors(unet)

    if mode_name == "original":
        pass
    else:
        unet = set_attenprocessor(
            unet,
            atten_type=mode_name,
            target_records=copy.deepcopy(target_records),
            sigmoid_setting=sigmoid_setting,
            decomp_timestep=decomp_timestep,
        )

    return diffusion(
        unet=unet,
        scheduler=scheduler,
        latents=latents,
        start_timesteps=0,
        text_embeddings=text_embeddings,
        total_timesteps=total_timesteps,
        guidance_scale=guidance_scale,
        desc=f"batch | {mode_name}",
    )


@torch.no_grad()
def main():
    global ORTHO_DECOMP_STORAGE
    global all_quant_layers

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_ckpt", required=True, type=str)
    parser.add_argument("--schema_path", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    parser.add_argument("--mode", type=str, default="original,erase,retain")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--include_anchor", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)

    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")

    parser.add_argument("--record_type", type=str, default="values")
    parser.add_argument("--decomp_timestep", type=int, default=0)
    parser.add_argument("--sigmoid_a", type=float, default=100.0)
    parser.add_argument("--sigmoid_b", type=float, default=0.93)
    parser.add_argument("--sigmoid_c", type=float, default=2.0)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    schema = load_schema(args.schema_path)
    jobs = build_generation_plan(schema, include_anchor=args.include_anchor)
    save_metadata(args.out_dir, schema, jobs, args)

    mode_list = [m.strip() for m in args.mode.split(",") if m.strip()]
    if len(mode_list) == 0:
        raise ValueError("mode must contain at least one of original, erase, retain")

    pipe = load_quantized_pipe(args.quant_ckpt, args.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    layers_linear = find_layers(pipe.unet, (LinearQuantHub,))
    layers_conv = find_layers(pipe.unet, (Conv2dQuantHub,))
    all_quant_layers = {**layers_linear, **layers_conv}

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    target_concepts = list(schema["erase_concepts"])
    target_concept_encoding = build_target_concept_encoding(target_concepts, tokenizer, text_encoder)

    base_uncond_encoding = get_textencoding(get_token(schema["negative_prompt"], tokenizer), text_encoder)

    target_records = None
    if "erase" in mode_list or "retain" in mode_list:
        target_records = compute_target_records(
            unet=unet,
            scheduler=pipe.scheduler,
            target_concept_encoding=target_concept_encoding,
            uncond_encoding=base_uncond_encoding,
            guidance_scale=float(schema["guidance_scale"]),
            record_type=args.record_type,
            total_timesteps=int(schema["num_inference_steps"]),
            device=args.device,
        )

    width = 512
    height = 512
    pipe.set_progress_bar_config(disable=False)

    global_idx = args.start_index
    guidance_scale = float(schema["guidance_scale"])
    num_inference_steps = int(schema["num_inference_steps"])
    negative_prompt = schema["negative_prompt"]

    print(f"[INFO] Total generations to run: {len(jobs)}")
    print(f"[INFO] Modes: {mode_list}")
    print(f"[INFO] Batch size: {args.batch_size}")

    start = 0
    while start < len(jobs):
        end = min(start + args.batch_size, len(jobs))
        batch_jobs = jobs[start:end]
        prompts = [j["prompt"] for j in batch_jobs]
        seeds = [int(j["seed"]) for j in batch_jobs]

        text_embeddings = prepare_text_embeddings_for_batch(
            prompts=prompts,
            negative_prompt=negative_prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=args.device,
        )

        latents_per_item = []
        latent_dtype = text_embeddings.dtype if args.dtype == "fp16" else torch.float32
        for seed in seeds:
            g = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu")
            g.manual_seed(seed)
            lat = torch.randn((1, 4, 64, 64), generator=g, device=args.device, dtype=latent_dtype)
            latents_per_item.append(lat)
        latents = torch.cat(latents_per_item, dim=0)

        if width != 512 or height != 512:
            raise ValueError("This script currently assumes 512x512 latent size (64x64 latent).")

        print(f"[INFO] Generating batch {start}:{end}")

        outputs_by_mode = {}
        for mode_name in mode_list:
            ORTHO_DECOMP_STORAGE = {}
            out_latents = run_mode_batch(
                mode_name=mode_name,
                unet=unet,
                scheduler=pipe.scheduler,
                latents=latents.clone(),
                text_embeddings=text_embeddings,
                total_timesteps=num_inference_steps,
                guidance_scale=guidance_scale,
                target_records=target_records,
                sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                decomp_timestep=args.decomp_timestep,
            )
            outputs_by_mode[mode_name] = decode_latents_to_pil(vae, out_latents)

        for batch_offset, job in enumerate(batch_jobs):
            per_mode_images = []
            for mode_name in mode_list:
                img = outputs_by_mode[mode_name][batch_offset]
                save_image_with_sidecar(
                    img=img,
                    job=job,
                    out_dir=args.out_dir,
                    global_index=global_idx,
                    mode=mode_name,
                )
                per_mode_images.append(img)

            if len(mode_list) > 1:
                combine_dir = os.path.join(
                    args.out_dir,
                    "combine",
                    job["group"],
                    safe_name(job["concept"]),
                )
                ensure_dir(combine_dir)

                widths, heights = zip(*(img.size for img in per_mode_images))
                combined = Image.new("RGB", (sum(widths), max(heights)))
                x = 0
                for img in per_mode_images:
                    combined.paste(img, (x, 0))
                    x += img.size[0]

                base_name = (
                    f"{global_idx:05d}"
                    f"_t{job['template_idx']:02d}"
                    f"_s{job['seed']:02d}"
                    f"_{safe_name(job['concept'])}"
                )
                combined.save(os.path.join(combine_dir, base_name + ".png"))

            global_idx += 1

        start = end

    print("[DONE] AdaVD quantized generation complete.")


if __name__ == "__main__":
    main()