#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import copy
import argparse
import re
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
torch.backends.cudnn.enabled = False
torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
# HF compatibility patch BEFORE diffusers imports
# -------------------------------------------------------------------------
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from diffusers import DPMSolverMultistepScheduler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPEED_ROOT = Path(__file__).resolve().parent
QUANT_ROOT = PROJECT_ROOT / "PCR"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for p in [str(PROJECT_ROOT), str(SPEED_ROOT), str(QUANT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# PCR imports required before torch.load(...)
import quantization_tools.quantization.layers  # noqa: F401
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub

from speed.src.utils import *

all_quant_layers = {}


def sanitize_name(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[^\w\s-]", "", x)
    x = re.sub(r"\s+", "_", x)
    return x


def step_start_callback(step: int, timestep: int):
    for _, layer in all_quant_layers.items():
        for quantizer in layer.quantizer:
            quantizer.set_curr_step(step)


def load_quantized_pipeline(quant_ckpt_path: str):
    """
    Load quantized pipeline saved through torch.save(pipe).
    """
    print(f"Loading quantized checkpoint: {quant_ckpt_path}")
    pipe = torch.load(quant_ckpt_path, map_location="cpu")

    if isinstance(pipe, dict):
        sample_keys = list(pipe.keys())[:10]
        raise ValueError(
            "Expected --sd_ckpt to be a full quantized pipeline checkpoint, "
            "but got a plain dict/state-dict instead. "
            f"Sample keys: {sample_keys}. "
            "Pass the base quantized model to --sd_ckpt and the edited UNet/state-dict to --edit_ckpt."
        )

    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)
    unet = pipe.unet.to(device)

    text_encoder.eval()
    unet.eval()
    unet.requires_grad_(False)

    quant_layers = {
        **find_layers(unet, (LinearQuantHub,)),
        **find_layers(unet, (Conv2dQuantHub,)),
    }
    print(f"Found {len(quant_layers)} quantized layers.")

    return tokenizer, text_encoder, unet, pipe, quant_layers


def diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    start_timesteps=0,
    guidance_scale=7.5,
    desc=None,
):
    scheduler.set_timesteps(total_timesteps)

    for step_idx, timestep in enumerate(
        tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc)
    ):
        step_start_callback(step_idx, int(timestep))

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents



def load_edit_into_quantized_unet(unet, edit_ckpt_path):
    """
    Quantize-then-erase case: edited checkpoint is a UNet state_dict that can be
    loaded directly into a copy of the quantized UNet with strict=False.
    """
    unet_edit = copy.deepcopy(unet)
    state_dict = torch.load(edit_ckpt_path, map_location="cpu")

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Expected --edit_ckpt to be a state_dict/dict for quantize-then-erase, "
            f"but got {type(state_dict)}"
        )

    missing, unexpected = unet_edit.load_state_dict(state_dict, strict=False)
    print(f"Loaded edited UNet from: {edit_ckpt_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("First few missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First few unexpected keys:", unexpected[:10])

    unet_edit.eval()
    unet_edit.requires_grad_(False)
    return unet_edit


class SchemaDataset(Dataset):
    def __init__(self, schema: dict):
        self.schema_name = schema["schema_name"]
        self.guidance_scale = float(schema.get("guidance_scale", 7.5))
        self.num_inference_steps = int(schema.get("num_inference_steps", 50))
        self.negative_prompt = schema.get("negative_prompt", "")
        self.prompt_templates = schema["prompt_templates"]
        self.seeds = schema["seeds"]

        self.records = []

        def add_records(category, concepts):
            for concept in concepts:
                for template_idx, template in enumerate(self.prompt_templates):
                    prompt = template.format(concept)
                    for seed in self.seeds:
                        self.records.append({
                            "category": category,
                            "concept": concept,
                            "prompt_template": template,
                            "template_idx": template_idx,
                            "prompt": prompt,
                            "seed": int(seed),
                        })

        add_records("erase", schema.get("erase_concepts", []))
        add_records("preserve", schema.get("preserve_concepts", []))
        add_records("anchor", schema.get("anchor_concepts", []))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]



def collate_schema_batch(batch):
    return {
        "category": [x["category"] for x in batch],
        "concept": [x["concept"] for x in batch],
        "prompt_template": [x["prompt_template"] for x in batch],
        "template_idx": [x["template_idx"] for x in batch],
        "prompt": [x["prompt"] for x in batch],
        "seed": [x["seed"] for x in batch],
    }



def save_manifest_line(manifest_path, row: dict):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--schema_path", type=str, required=True)
    parser.add_argument("--sd_ckpt", type=str, required=True)
    parser.add_argument("--edit_ckpt", type=str, default=None)

    parser.add_argument("--mode", type=str, default="edit", help="original, edit or original,edit")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=1, help="images per prompt-seed pair")
    parser.add_argument("--write_manifest", action="store_true")

    args = parser.parse_args()

    mode_list = args.mode.replace(" ", "").split(",")

    with open(args.schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    dataset = SchemaDataset(schema)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_schema_batch,
    )

    global all_quant_layers
    tokenizer, text_encoder, unet, pipe, all_quant_layers = load_quantized_pipeline(
        quant_ckpt_path=args.sd_ckpt
    )

    if hasattr(pipe.scheduler, "config"):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        print("Warning: scheduler has no .config; using checkpoint scheduler as-is.")

    vae = pipe.vae

    unet_edit = None
    if "edit" in mode_list:
        if args.edit_ckpt is None:
            raise ValueError("--edit_ckpt is required when mode includes 'edit'")
        unet_edit = load_edit_into_quantized_unet(unet, args.edit_ckpt)

    schema_root = os.path.join(args.save_root, sanitize_name(schema["schema_name"]))
    os.makedirs(schema_root, exist_ok=True)

    manifest_path = os.path.join(schema_root, "generation_manifest.jsonl")
    if args.write_manifest and os.path.exists(manifest_path):
        os.remove(manifest_path)

    negative_prompt = schema.get("negative_prompt", "")
    uncond_embedding = get_textencoding(get_token(negative_prompt, tokenizer), text_encoder)

    total_timesteps = int(schema.get("num_inference_steps", 50))
    guidance_scale = float(schema.get("guidance_scale", 7.5))

    for batch_id, data in enumerate(dataloader):
        prompts = data["prompt"]
        batch_size_actual = len(prompts)

        embeddings = get_textencoding(get_token(prompts, tokenizer), text_encoder)
        text_embeddings = torch.cat([uncond_embedding] * batch_size_actual + [embeddings], dim=0)

        latents_list = []
        for seed in data["seed"]:
            g = torch.Generator("cpu").manual_seed(int(seed))
            lat = torch.randn((1, 4, 64, 64), generator=g)
            latents_list.append(lat)
        latent = torch.cat(latents_list, dim=0).to(device)

        save_images = {}

        if "original" in mode_list:
            save_images["original"] = diffusion(
                unet=unet,
                scheduler=pipe.scheduler,
                latents=latent.clone(),
                start_timesteps=0,
                text_embeddings=text_embeddings,
                total_timesteps=total_timesteps,
                guidance_scale=guidance_scale,
                desc=f"batch {batch_id} | original",
            )

        if "edit" in mode_list:
            save_images["edit"] = diffusion(
                unet=unet_edit,
                scheduler=pipe.scheduler,
                latents=latent.clone(),
                start_timesteps=0,
                text_embeddings=text_embeddings,
                total_timesteps=total_timesteps,
                guidance_scale=guidance_scale,
                desc=f"batch {batch_id} | edit",
            )

        decoded_imgs = {
            mode_name: [
                process_img(
                    vae.decode(
                        img.unsqueeze(0) / vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                )
                for img in img_list
            ]
            for mode_name, img_list in save_images.items()
        }

        def combine_images_horizontally(images):
            widths, heights = zip(*(img.size for img in images))
            new_img = Image.new("RGB", (sum(widths), max(heights)))
            for i, img in enumerate(images):
                new_img.paste(img, (sum(widths[:i]), 0))
            return new_img

        for idx in range(batch_size_actual):
            category = data["category"][idx]
            concept = data["concept"][idx]
            prompt = data["prompt"][idx]
            template_idx = int(data["template_idx"][idx])
            seed = int(data["seed"][idx])

            concept_dir = os.path.join(schema_root, category, sanitize_name(concept))

            for mode in mode_list:
                os.makedirs(os.path.join(concept_dir, mode), exist_ok=True)

            if len(mode_list) > 1:
                os.makedirs(os.path.join(concept_dir, "combine"), exist_ok=True)

            filename = (
                f"tpl{template_idx:02d}"
                f"_seed{seed:02d}"
                f"_{sanitize_name(prompt)[:120]}.png"
            )

            combined_images = []
            for mode in mode_list:
                out_path = os.path.join(concept_dir, mode, filename)
                decoded_imgs[mode][idx].save(out_path)
                combined_images.append(decoded_imgs[mode][idx])

                if args.write_manifest:
                    save_manifest_line(
                        manifest_path,
                        {
                            "schema_name": schema["schema_name"],
                            "category": category,
                            "concept": concept,
                            "prompt": prompt,
                            "prompt_template": data["prompt_template"][idx],
                            "template_idx": template_idx,
                            "seed": seed,
                            "mode": mode,
                            "file": out_path,
                            "guidance_scale": guidance_scale,
                            "num_inference_steps": total_timesteps,
                            "negative_prompt": negative_prompt,
                        },
                    )

            if len(mode_list) > 1:
                combined = combine_images_horizontally(combined_images)
                combined.save(os.path.join(concept_dir, "combine", filename.replace(".png", ".jpg")))

    print(f"\nDone. Outputs saved to: {schema_root}")


if __name__ == "__main__":
    main()
