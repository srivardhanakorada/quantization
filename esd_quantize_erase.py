#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import copy
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import huggingface_hub

# ---------------------------------------------------------------------
# Compatibility patch for environments with older hub API expectations
# ---------------------------------------------------------------------
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

# ---------------------------------------------------------------------
# Offline mode if needed
# ---------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------
# Quantization package imports
# ---------------------------------------------------------------------
import quantization_tools.quantization.layers  # noqa: F401
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub


# =====================================================================
# Config
# =====================================================================

@dataclass
class QuantESDConfig:
    quant_ckpt: str
    erase_concept: str
    erase_from: Optional[str]
    train_method: str
    iterations: int
    lr: float
    batch_size: int
    resolution: Optional[int]
    num_inference_steps: int
    guidance_scale: float
    negative_guidance: float
    save_path: str
    device: str
    torch_dtype: torch.dtype = torch.float32
    seed: int = 0

    @property
    def erase_from_effective(self) -> str:
        return self.erase_from if self.erase_from is not None else self.erase_concept


# =====================================================================
# General helpers
# =====================================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_lr_for_method(method: str) -> float:
    return 5e-5


def move_pipeline_to_device(pipe, device: str):
    if hasattr(pipe, "to"):
        try:
            return pipe.to(device)
        except Exception:
            pass

    for attr in ["unet", "text_encoder", "vae", "safety_checker"]:
        if hasattr(pipe, attr):
            mod = getattr(pipe, attr)
            if mod is not None and hasattr(mod, "to"):
                mod.to(device)
    return pipe


def freeze_all_params(module):
    if module is None:
        return
    if not hasattr(module, "parameters"):
        return
    for p in module.parameters():
        p.requires_grad = False


def freeze_pipeline_modules(pipe):
    for attr in ["unet", "text_encoder", "vae", "safety_checker"]:
        if hasattr(pipe, attr):
            freeze_all_params(getattr(pipe, attr))


def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def resolve_default_resolution(pipe) -> int:
    default_sample_size = getattr(pipe, "default_sample_size", None)
    if default_sample_size is None:
        default_sample_size = pipe.unet.config.sample_size

    if isinstance(default_sample_size, (tuple, list)):
        default_sample_size = default_sample_size[0]

    return int(default_sample_size) * pipe.vae_scale_factor


# =====================================================================
# Quantization helpers
# =====================================================================

def get_all_quant_layers(pipe):
    quant_layers = find_layers(pipe.unet, (LinearQuantHub,))
    names, mods = [], []
    for name, mod in pipe.unet.named_modules():
        if name in quant_layers:
            names.append(name)
            mods.append(quant_layers[name])
    return mods, names


def get_layer_qparams(layer):
    if not hasattr(layer, "quantizer") or len(layer.quantizer) == 0:
        raise RuntimeError("Layer has no quantizer attached")

    q = layer.quantizer[0]
    if not hasattr(q, "w_scale") or not hasattr(q, "w_zero_point"):
        raise RuntimeError("Could not find w_scale / w_zero_point on quantizer")

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
    n_bits: int = 8,
) -> torch.Tensor:
    qmin = 0
    qmax = (2 ** n_bits) - 1
    scale_b, zp_b = broadcast_qparams_like_weight(w_fp, scale, zero_point)
    q_idx = torch.round(w_fp / scale_b + zp_b)
    q_idx = torch.clamp(q_idx, qmin, qmax)
    w_q = (q_idx - zp_b) * scale_b
    return w_q


@torch.no_grad()
def project_quantized_layers_to_fixed_grid(pipe):
    quant_mods, quant_names = get_all_quant_layers(pipe)
    for mod, name in zip(quant_mods, quant_names):
        scale, zero_point = get_layer_qparams(mod)
        w = mod.core.weight.data.float()
        w_q = quantize_to_grid(w, scale.to(w.device), zero_point.to(w.device))
        mod.core.weight.data.copy_(w_q.to(mod.core.weight.device, dtype=mod.core.weight.dtype))


@torch.no_grad()
def clear_quant_layer_caches(pipe):
    """
    Only clear very specific known cache names. Do not aggressively detach
    arbitrary module attributes, because that can break the gradient path.
    """
    quant_mods, _ = get_all_quant_layers(pipe)
    for mod in quant_mods:
        for attr in [
            "cached_weight",
            "cached_qweight",
            "tmp_weight",
            "tmp_qweight",
            "weight_cache",
            "qweight_cache",
        ]:
            if hasattr(mod, attr):
                setattr(mod, attr, None)


# =====================================================================
# Trainable parameter selection
# =====================================================================

def choose_trainable_names_quantized(unet: torch.nn.Module, train_method: str) -> List[str]:
    train_method = train_method.lower()

    aliases = {
        "xattn": "esd-x",
        "noxattn": "esd-u",
        "full": "esd-all",
        "xattn-strict": "esd-x-strict",
        "esd-x": "esd-x",
        "esd-u": "esd-u",
        "esd-all": "esd-all",
        "esd-x-strict": "esd-x-strict",
        "selfattn": "selfattn",
    }

    if train_method not in aliases:
        raise ValueError(f"Unsupported train_method: {train_method}")
    train_method = aliases[train_method]

    quant_layers = find_layers(unet, (LinearQuantHub,))
    selected = []

    for module_name, module in unet.named_modules():
        if module_name not in quant_layers:
            continue

        use = False
        if train_method == "esd-x":
            use = "attn2" in module_name
        elif train_method == "esd-u":
            use = "attn2" not in module_name
        elif train_method == "esd-all":
            use = True
        elif train_method == "esd-x-strict":
            use = (
                module_name.endswith("attn2.to_k")
                or module_name.endswith("attn2.to_v")
                or ("attn2" in module_name and (module_name.endswith("to_k") or module_name.endswith("to_v")))
            )
        elif train_method == "selfattn":
            use = "attn1" in module_name

        if use:
            if hasattr(module, "core") and hasattr(module.core, "weight"):
                selected.append(f"{module_name}.core.weight")
            if hasattr(module, "core") and getattr(module.core, "bias", None) is not None:
                selected.append(f"{module_name}.core.bias")

    if len(selected) == 0:
        raise RuntimeError(f"No trainable quantized parameters found for method: {train_method}")

    return selected


def enable_named_trainable_params(module: torch.nn.Module, selected_names: List[str]):
    selected_names = set(selected_names)
    for name, param in module.named_parameters():
        param.requires_grad = name in selected_names


# =====================================================================
# Prompt encoding compatibility
# =====================================================================

@torch.no_grad()
def encode_prompt_compat(
    pipe,
    prompt,
    device: str,
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: str = "",
):
    if hasattr(pipe, "encode_prompt"):
        out = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        if isinstance(out, tuple):
            if len(out) >= 2:
                return out[0], out[1]
            if len(out) == 1:
                return out[0], None
        return out, None

    if hasattr(pipe, "_encode_prompt"):
        out = pipe._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        if do_classifier_free_guidance:
            if not torch.is_tensor(out):
                raise RuntimeError("_encode_prompt returned non-tensor output under CFG=True")
            half = out.shape[0] // 2
            negative_prompt_embeds = out[:half]
            prompt_embeds = out[half:]
            return prompt_embeds, negative_prompt_embeds
        return out, None

    raise AttributeError(
        f"{type(pipe).__name__} exposes neither encode_prompt nor _encode_prompt"
    )


# =====================================================================
# Prompt encoding / SD sampling helpers
# =====================================================================

@torch.no_grad()
def prepare_sd_context(pipe, config: QuantESDConfig) -> Dict[str, torch.Tensor]:
    resolution = config.resolution if config.resolution is not None else resolve_default_resolution(pipe)

    erase_embeds, null_embeds = encode_prompt_compat(
        pipe=pipe,
        prompt=config.erase_concept,
        device=config.device,
        num_images_per_prompt=config.batch_size,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )
    erase_embeds = erase_embeds.to(config.device).detach()
    null_embeds = null_embeds.to(config.device).detach()

    erase_from_embeds = None
    if config.erase_from is not None:
        erase_from_embeds, _ = encode_prompt_compat(
            pipe=pipe,
            prompt=config.erase_from,
            device=config.device,
            num_images_per_prompt=config.batch_size,
            do_classifier_free_guidance=False,
            negative_prompt="",
        )
        erase_from_embeds = erase_from_embeds.to(config.device).detach()

    timestep_cond = None
    if getattr(pipe.unet.config, "time_cond_proj_dim", None) is not None:
        guidance_scale_tensor = torch.tensor(
            config.guidance_scale - 1,
            device=config.device,
            dtype=torch.float32
        ).repeat(config.batch_size)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
        ).to(device=config.device, dtype=config.torch_dtype).detach()

    return {
        "resolution": resolution,
        "erase_embeds": erase_embeds,
        "null_embeds": null_embeds,
        "erase_from_embeds": erase_from_embeds,
        "sample_prompt_embeds": (erase_embeds if erase_from_embeds is None else erase_from_embeds).detach(),
        "sample_negative_prompt_embeds": null_embeds.detach(),
        "student_prompt_embeds": (erase_embeds if erase_from_embeds is None else erase_from_embeds).detach(),
        "timestep_cond": timestep_cond,
    }


def retrieve_timesteps(pipe, num_inference_steps: int, device: str):
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    return pipe.scheduler.timesteps


def prepare_latents_like_sd(
    pipe,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: str,
    generator: torch.Generator,
):
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return latents


@torch.no_grad()
def sample_until_timestep_sd(
    pipe,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    run_till_timestep: int,
    generator: torch.Generator,
    height: int,
    width: int,
):
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype
    batch_size = prompt_embeds.shape[0]

    timesteps = retrieve_timesteps(pipe, num_inference_steps, device)
    latents = prepare_latents_like_sd(
        pipe=pipe,
        batch_size=batch_size,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    extra_step_kwargs = {}
    if hasattr(pipe, "prepare_extra_step_kwargs"):
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)

    for t in timesteps[:run_till_timestep]:
        latent_model_input = latents
        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latents] * 2, dim=0)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        if guidance_scale > 1.0:
            encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            encoder_hidden_states = prompt_embeds

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

    return latents.detach(), timesteps[run_till_timestep].detach()


# =====================================================================
# One ESD training step on quantized checkpoint
# =====================================================================

def quantized_esd_training_step(
    pipe,
    teacher_pipe,
    context: Dict[str, torch.Tensor],
    config: QuantESDConfig,
):
    run_till_timestep = random.randint(0, config.num_inference_steps - 1)
    seed = random.randint(0, 2**15)

    with torch.no_grad():
        xt, timestep = sample_until_timestep_sd(
            pipe=teacher_pipe,
            prompt_embeds=context["sample_prompt_embeds"],
            negative_prompt_embeds=context["sample_negative_prompt_embeds"],
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            run_till_timestep=run_till_timestep,
            generator=make_sampling_generator(config.device, seed),
            height=context["resolution"],
            width=context["resolution"],
        )

        noise_pred_erase = teacher_pipe.unet(
            xt,
            timestep,
            encoder_hidden_states=context["erase_embeds"],
            timestep_cond=context["timestep_cond"],
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0].detach()

        noise_pred_null = teacher_pipe.unet(
            xt,
            timestep,
            encoder_hidden_states=context["null_embeds"],
            timestep_cond=context["timestep_cond"],
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0].detach()

        if context["erase_from_embeds"] is not None:
            noise_pred_erase_from = teacher_pipe.unet(
                xt,
                timestep,
                encoder_hidden_states=context["erase_from_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0].detach()
        else:
            noise_pred_erase_from = noise_pred_erase.detach()

    model_pred = pipe.unet(
        xt.detach(),
        timestep.detach(),
        encoder_hidden_states=context["student_prompt_embeds"].detach(),
        timestep_cond=(context["timestep_cond"].detach() if context["timestep_cond"] is not None else None),
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        return_dict=False,
    )[0]

    target = (
        noise_pred_erase_from
        - config.negative_guidance * (noise_pred_erase - noise_pred_null)
    ).detach()

    return model_pred, target, run_till_timestep


# =====================================================================
# Main training loop
# =====================================================================

def run_quantized_esd_training(config: QuantESDConfig) -> str:
    print(f"[INFO] Loading quantized checkpoint: {config.quant_ckpt}")

    pipe = torch.load(config.quant_ckpt, map_location="cpu")
    pipe = move_pipeline_to_device(pipe, config.device)

    teacher_pipe = copy.deepcopy(pipe)
    teacher_pipe = move_pipeline_to_device(teacher_pipe, config.device)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(teacher_pipe, "set_progress_bar_config"):
        teacher_pipe.set_progress_bar_config(disable=True)

    freeze_pipeline_modules(pipe)
    freeze_pipeline_modules(teacher_pipe)

    if hasattr(teacher_pipe, "unet") and teacher_pipe.unet is not None:
        for p in teacher_pipe.unet.parameters():
            p.requires_grad = False

    selected_names = choose_trainable_names_quantized(pipe.unet, config.train_method)
    enable_named_trainable_params(pipe.unet, selected_names)

    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters enabled.")

    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Train method     : {config.train_method}")
    print(f"[INFO] Trainable params : {num_trainable:,}")
    print(f"[INFO] Number tensors   : {len(trainable_params)}")

    pipe.unet.train()
    teacher_pipe.unet.eval()

    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.eval()
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.eval()

    if hasattr(teacher_pipe, "vae") and teacher_pipe.vae is not None:
        teacher_pipe.vae.eval()
    if hasattr(teacher_pipe, "text_encoder") and teacher_pipe.text_encoder is not None:
        teacher_pipe.text_encoder.eval()

    context = prepare_sd_context(pipe, config)

    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)

    start_time = time.time()
    pbar = tqdm(range(config.iterations), desc="Training Quantized ESD (SD)")

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        model_pred, target, timestep_index = quantized_esd_training_step(
            pipe=pipe,
            teacher_pipe=teacher_pipe,
            context=context,
            config=config,
        )

        loss = F.mse_loss(model_pred.float(), target.float())

        # Pragmatic workaround for cached autograd references inside custom quant wrappers
        loss.backward(retain_graph=True)
        optimizer.step()

        project_quantized_layers_to_fixed_grid(pipe)
        clear_quant_layer_caches(pipe)

        pbar.set_postfix({
            "esd_loss": f"{loss.item():.4f}",
            "timestep": timestep_index,
        })

    elapsed = time.time() - start_time
    print(f"[INFO] Training finished in {elapsed/60.0:.2f} minutes")

    save_dir = os.path.dirname(config.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(pipe, config.save_path)
    print(f"[INFO] Saved edited quantized checkpoint to: {config.save_path}")
    return config.save_path


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="QuantizedESD-SD",
        description="Run ESD directly on a quantized Stable Diffusion checkpoint.",
    )
    parser.add_argument("--quant_ckpt", type=str, required=True, help="Quantized checkpoint saved via torch.save(pipe, ...)")
    parser.add_argument("--erase_concept", type=str, required=True, help="Concept to erase")
    parser.add_argument("--erase_from", type=str, default=None, help="Concept to erase from")
    parser.add_argument(
        "--train_method",
        type=str,
        required=True,
        help="esd-x, esd-u, esd-all, esd-x-strict, selfattn (legacy aliases xattn, noxattn, full, xattn-strict also work)",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--negative_guidance", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()

    train_method = args.train_method.lower()
    aliases = {
        "xattn": "esd-x",
        "noxattn": "esd-u",
        "full": "esd-all",
        "xattn-strict": "esd-x-strict",
        "esd-x": "esd-x",
        "esd-u": "esd-u",
        "esd-all": "esd-all",
        "esd-x-strict": "esd-x-strict",
        "selfattn": "selfattn",
    }
    if train_method not in aliases:
        raise ValueError(f"Unsupported train_method: {args.train_method}")
    train_method = aliases[train_method]

    lr = args.lr if args.lr is not None else default_lr_for_method(train_method)

    config = QuantESDConfig(
        quant_ckpt=args.quant_ckpt,
        erase_concept=args.erase_concept,
        erase_from=args.erase_from,
        train_method=train_method,
        iterations=args.iterations,
        lr=lr,
        batch_size=args.batchsize,
        resolution=args.resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_guidance=args.negative_guidance,
        save_path=args.save_path,
        device=args.device,
        torch_dtype=torch.float32,
        seed=args.seed,
    )

    set_seed(config.seed)
    run_quantized_esd_training(config)


if __name__ == "__main__":
    main()