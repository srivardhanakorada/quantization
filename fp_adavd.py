#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import copy
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


def seed_everything(seed, deterministic=False):
    import random
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
        return_tensors="pt"
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


def load_schema(schema_path):
    with open(schema_path, "r") as f:
        schema = json.load(f)

    required = [
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
    if not isinstance(schema["prompt_templates"], list) or len(schema["prompt_templates"]) == 0:
        raise ValueError("prompt_templates must be a non-empty list")
    if not isinstance(schema["seeds"], list) or len(schema["seeds"]) == 0:
        raise ValueError("seeds must be a non-empty list")
    if not isinstance(schema["negative_prompt"], str):
        raise ValueError("negative_prompt must be a string")

    return schema


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_name(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "", text)
    return text[:180]


def save_metadata(save_root, schema, args):
    os.makedirs(save_root, exist_ok=True)
    meta = {
        "schema_path": args.schema_path,
        "sd_ckpt": args.sd_ckpt,
        "include_anchor": args.include_anchor,
        "mode": args.mode,
        "record_type": args.record_type,
        "decomp_timestep": args.decomp_timestep,
        "sigmoid_a": args.sigmoid_a,
        "sigmoid_b": args.sigmoid_b,
        "sigmoid_c": args.sigmoid_c,
        "guidance_scale": float(schema["guidance_scale"]),
        "num_inference_steps": int(schema["num_inference_steps"]),
        "negative_prompt": schema["negative_prompt"],
        "schema_name": schema.get("schema_name", "unknown"),
        "seeds": schema["seeds"],
        "prompt_templates": schema["prompt_templates"],
    }
    with open(os.path.join(save_root, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(save_root, "resolved_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)


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
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(
            self.module_name,
            self.atten_type,
            self.target_records,
            self.record,
            self.record_type,
            self.sigmoid_setting,
            self.decomp_timestep
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)


class AttnProcessor():
    def __init__(
        self,
        module_name=None,
        atten_type='original',
        target_records=None,
        record=False,
        record_type=None,
        sigmoid_setting=None,
        decomp_timestep=0
    ) -> None:
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = copy.copy(target_records)
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_ortho_decomp(self, target_value, pro_record, ortho_basis=None, project_matrix=None):
        if ortho_basis is None and project_matrix is None:
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1)
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1)
            dot1 = (tar_record_ * pro_record_).sum(-1)
            dot2 = (tar_record_ * tar_record_).sum(-1)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            weight = torch.nan_to_num(cos_sim * (dot1 / dot2), nan=0.0)
            weight[0].fill_(0)
            era_record = weight.unsqueeze(0).unsqueeze(-1) * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
        else:
            tar_record_ = rearrange(target_value, 'b h l d -> l b (h d)')
            pro_record_ = rearrange(pro_record, 'h l d -> l (h d)').unsqueeze(1)
            dot1 = (ortho_basis * pro_record_).sum(-1)
            dot2 = (ortho_basis * ortho_basis).sum(-1)
            weight = torch.nan_to_num((dot1 / dot2).unsqueeze(1), nan=0.0)
            weight[0].fill_(0)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)
            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            era_record = torch.bmm(weight, projected_basis).view((77, 16, -1)).permute(1, 0, 2)
        return era_record

    def record_ortho_decomp(self, target_record, current_record):
        current_name = next(k for k in target_record if k.endswith(self.module_name))
        current_timestep, current_block = current_name.split('.', 1)
        (target_value, project_matrix, ortho_basis) = target_record.pop(current_name)

        if int(current_timestep) <= self.decomp_timestep:
            return current_record, current_record

        if current_block in ORTHO_DECOMP_STORAGE:
            pass
        else:
            target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
            target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
            current_record = current_record.view((2, int(len(current_record)//16), -1) + target_value.size()[-2:])
            current_record = current_record.permute(1, 0, 2, 3, 4).contiguous().view((current_record.size()[1], -1) + target_value.size()[-2:])
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

    def cal_gram_schmidt(self, target_value):
        target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        target_value_ = rearrange(target_value, 'b h l d -> b l (h d)')
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0)
        basis_ortho = torch.stack([result[1] for result in results], dim=0)
        return project_matrix, basis_ortho

    def gram_schmidt(self, V):
        n = len(V)
        project_matrix = torch.zeros((n, n), dtype=V.dtype).to(V.device) + torch.diag(torch.ones(n, dtype=V.dtype)).to(V.device)
        for i in range(1, n):
            vi = V[i:i+1, :]
            for j in range(i):
                qj = V[j:j+1, :]
                project_matrix[i][j] = -torch.dot(qj.view(-1), vi.view(-1)) / torch.dot(qj.view(-1), qj.view(-1))
        ortho_basis = torch.matmul(project_matrix.to(V.device), V)
        return project_matrix.to(V.device), ortho_basis

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
                erase_query, retain_query = self.record_ortho_decomp(self.target_records['queries'], query)
                query = retain_query if self.atten_type == 'retain' else erase_query if self.atten_type == 'erase' else query
            if 'keys' in self.target_records:
                erase_key, retain_key = self.record_ortho_decomp(self.target_records['keys'], key)
                key = retain_key if self.atten_type == 'retain' else erase_key if self.atten_type == 'erase' else key

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'attn_maps' in self.target_records:
                erase_attention_probs, retain_attention_probs = self.record_ortho_decomp(self.target_records['attn_maps'], attention_probs)
                attention_probs = retain_attention_probs if self.atten_type == 'retain' else erase_attention_probs if self.atten_type == 'erase' else attention_probs

        if encoder_hidden_states.shape[1] != 77:
            hidden_states = torch.bmm(attention_probs, value)
        else:
            if self.record:
                for kk, vv in {'queries': query, 'keys': key, 'values': value, 'attn_maps': attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1:
                            self.records[kk][self.module_name] = [vv] + [None, None]
                        else:
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif 'values' in self.target_records:
                erase_value, retain_value = self.record_ortho_decomp(self.target_records['values'], value)
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


def set_attenprocessor(unet, atten_type='original', target_records=None, record=False, record_type=None, sigmoid_setting=None, decomp_timestep=0):
    for name, m in unet.named_modules():
        if name.endswith('attn2') or name.endswith('attn1'):
            m.set_processor(VisualAttentionProcess(
                module_name=name,
                atten_type=atten_type,
                target_records=target_records,
                record=record,
                record_type=record_type,
                sigmoid_setting=sigmoid_setting,
                decomp_timestep=decomp_timestep
            ))
    return unet


def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, record=False, record_type=None, desc=None, **kwargs):
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
            for type_name in record_type.strip().split(','):
                for value in unet.attn_processors.values():
                    for k, v in value.records[type_name].items():
                        visualize_map_withstep[type_name][f'{timestep.item()}.{k}'] = v

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map_withstep) if record else latents


ORTHO_DECOMP_STORAGE = {}


@torch.no_grad()
def main():
    global ORTHO_DECOMP_STORAGE

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--sd_ckpt', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mode', type=str, default='retain', help='keep as retain')
    parser.add_argument('--decomp_timestep', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--sigmoid_a', type=float, default=100)
    parser.add_argument('--sigmoid_b', type=float, default=0.93)
    parser.add_argument('--sigmoid_c', type=float, default=2)
    parser.add_argument('--record_type', type=str, default='values')
    parser.add_argument('--schema_path', type=str, required=True)
    parser.add_argument('--include_anchor', action='store_true')
    args = parser.parse_args()

    schema = load_schema(args.schema_path)

    if schema["negative_prompt"] != "":
        raise ValueError("For faithful original-style AdaVD, schema['negative_prompt'] must be ''.")

    guidance_scale = float(schema["guidance_scale"])
    total_timesteps = int(schema["num_inference_steps"])

    target_concepts = [item.strip() for item in schema["erase_concepts"]]
    contents = list(schema["erase_concepts"]) + list(schema["preserve_concepts"])
    if args.include_anchor:
        contents += list(schema["anchor_concepts"])
    concept_list = unique_keep_order([item.strip() for item in contents])

    prompt_templates = list(schema["prompt_templates"])
    schema_seeds = [int(s) for s in schema["seeds"]]
    num_samples = len(schema_seeds)

    bs = args.batch_size if args.batch_size is not None else num_samples
    if bs != num_samples:
        raise ValueError(f"For seed-aligned comparison, batch_size must equal number of schema seeds ({num_samples}).")

    save_metadata(args.save_root, schema, args)

    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        safety_checker=None,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae

    unet_retain = copy.deepcopy(unet)

    target_concept_encodings_ = [get_textencoding(get_token(concept, tokenizer), text_encoder) for concept in target_concepts]
    target_eot_idxs = [get_eot_idx(get_token(concept, tokenizer)) for concept in target_concepts]
    target_concept_encoding = [
        get_spread_embedding(target_concept_encoding_, idx)
        for (target_concept_encoding_, idx) in zip(target_concept_encodings_, target_eot_idxs)
    ]
    target_concept_encoding = torch.concat(target_concept_encoding)

    uncond_encoding = get_textencoding(get_token('', tokenizer), text_encoder)

    unet = set_attenprocessor(unet, atten_type='original', record=True, record_type=args.record_type)
    _, target_records = diffusion(
        unet=unet,
        scheduler=pipe.scheduler,
        latents=torch.zeros(len(target_concept_encoding), 4, 64, 64).to(pipe.device, dtype=target_concept_encoding.dtype),
        text_embeddings=torch.cat([uncond_encoding] * len(target_concept_encoding) + [target_concept_encoding], dim=0),
        total_timesteps=1,
        start_timesteps=0,
        guidance_scale=guidance_scale,
        record=True,
        record_type=args.record_type,
        desc="Calculating target records",
    )
    pipe.scheduler.set_timesteps(total_timesteps)
    original_keys = target_records[args.record_type].keys()
    target_records[args.record_type].update({
        f"{timestep}.{'.'.join(key.split('.')[1:])}": target_records[args.record_type][key]
        for timestep in pipe.scheduler.timesteps
        for key in original_keys
    })
    del unet

    latent_bank = []
    for s in schema_seeds:
        seed_use = args.seed if args.seed is not None else s
        seed_everything(seed_use, True)
        latent = torch.randn(1, 4, 64, 64).to(pipe.device, dtype=target_concept_encoding.dtype)
        latent_bank.append(latent)
    latent_bank = torch.cat(latent_bank, dim=0)

    manifest_path = os.path.join(args.save_root, "generation_manifest.jsonl")
    mf = open(manifest_path, "w")

    global_prompt_idx = 0

    for concept in concept_list:
        prompts = [tmpl.format(concept) for tmpl in prompt_templates]

        for template_idx, prompt in enumerate(prompts):
            ORTHO_DECOMP_STORAGE, Images = {}, {}
            encoding = get_textencoding(get_token(prompt, tokenizer), text_encoder)
            batch_text_embeddings = torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0)

            unet_retain = set_attenprocessor(
                unet_retain,
                atten_type='retain',
                target_records=copy.deepcopy(target_records),
                sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                decomp_timestep=args.decomp_timestep,
            )
            Images['retain'] = diffusion(
                unet=unet_retain,
                scheduler=pipe.scheduler,
                latents=latent_bank.clone(),
                start_timesteps=0,
                text_embeddings=batch_text_embeddings,
                total_timesteps=total_timesteps,
                guidance_scale=guidance_scale,
                desc=f"{prompt} | retain"
            )

            decoded_imgs = {
                name: [process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]) for img in img_list]
                for name, img_list in Images.items()
            }

            if concept in schema["erase_concepts"]:
                split_name = "erase"
            elif concept in schema["preserve_concepts"]:
                split_name = "preserve"
            else:
                global_prompt_idx += 1
                continue

            save_dir = os.path.join(args.save_root, split_name, safe_name(concept))
            os.makedirs(save_dir, exist_ok=True)

            for idx, seed_val in enumerate(schema_seeds):
                filename = (
                    f"{global_prompt_idx:05d}"
                    f"_t{template_idx:02d}"
                    f"_s{seed_val:02d}"
                    f"_{safe_name(concept)}.png"
                )
                decoded_imgs['retain'][idx].save(os.path.join(save_dir, filename))

                mf.write(json.dumps({
                    "global_prompt_idx": global_prompt_idx,
                    "split": split_name,
                    "concept": concept,
                    "template_idx": template_idx,
                    "prompt": prompt,
                    "seed": seed_val,
                    "filename": filename,
                }) + "\n")

            global_prompt_idx += 1

    mf.close()


if __name__ == '__main__':
    main()