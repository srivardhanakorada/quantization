# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import sys
# import math
# import argparse
# from typing import List

# import torch
# import huggingface_hub

# if not hasattr(huggingface_hub, "cached_download"):
#     huggingface_hub.cached_download = huggingface_hub.hf_hub_download
#     sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["DIFFUSERS_OFFLINE"] = "1"

# from PIL import Image

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# OBAMA_PROMPTS = [
#     "a high quality portrait of Barack Obama",
#     "Barack Obama speaking at a podium, professional photograph",
#     "Barack Obama smiling, studio portrait",
#     "Barack Obama, detailed face, realistic photo",
# ]

# TRUMP_PROMPTS = [
#     "a high quality portrait of Donald Trump",
#     "Donald Trump speaking at a podium, professional photograph",
#     "Donald Trump smiling, studio portrait",
#     "Donald Trump, detailed face, realistic photo",
# ]


# def make_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
#     if len(images) == 0:
#         raise ValueError("No images provided for grid")

#     w, h = images[0].size
#     grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))

#     for idx, img in enumerate(images):
#         r = idx // cols
#         c = idx % cols
#         grid.paste(img, (c * w, r * h))

#     return grid


# def sanitize_filename(x: str) -> str:
#     keep = []
#     for ch in x:
#         if ch.isalnum() or ch in ("_", "-", "."):
#             keep.append(ch)
#         elif ch == " ":
#             keep.append("_")
#     return "".join(keep)


# def generate_set(
#     pipe,
#     prompts: List[str],
#     seeds: List[int],
#     out_dir: str,
#     tag: str,
#     num_inference_steps: int,
#     guidance_scale: float,
#     height: int,
#     width: int,
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     all_images = []

#     for p_idx, prompt in enumerate(prompts):
#         prompt_dir = os.path.join(out_dir, f"{tag}_prompt_{p_idx+1}")
#         os.makedirs(prompt_dir, exist_ok=True)

#         for seed in seeds:
#             generator = torch.Generator(device=DEVICE).manual_seed(seed)

#             result = pipe(
#                 prompt=prompt,
#                 num_inference_steps=num_inference_steps,
#                 guidance_scale=guidance_scale,
#                 height=height,
#                 width=width,
#                 generator=generator,
#             )

#             img = result.images[0]
#             all_images.append(img)

#             out_path = os.path.join(
#                 prompt_dir,
#                 f"seed_{seed}.png"
#             )
#             img.save(out_path)
#             print(f"[INFO] saved {out_path}")

#     cols = len(seeds)
#     rows = len(prompts)
#     grid = make_grid(all_images, rows=rows, cols=cols)
#     grid_path = os.path.join(out_dir, f"{tag}_grid.png")
#     grid.save(grid_path)
#     print(f"[INFO] saved grid {grid_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt", required=True, help="Quantized UCE checkpoint (.ckpt/.pt)")
#     parser.add_argument("--out_dir", default="smoke_test_outputs")
#     parser.add_argument("--steps", type=int, default=50)
#     parser.add_argument("--guidance_scale", type=float, default=7.5)
#     parser.add_argument("--height", type=int, default=512)
#     parser.add_argument("--width", type=int, default=512)
#     parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     print(f"[INFO] loading checkpoint from: {args.ckpt}")
#     pipe = torch.load(args.ckpt, map_location="cpu")

#     pipe = pipe.to(DEVICE)
#     pipe.unet.eval()
#     if hasattr(pipe, "vae"):
#         pipe.vae.eval()
#     if hasattr(pipe, "text_encoder"):
#         pipe.text_encoder.eval()

#     # keep inference lighter / cleaner
#     pipe.set_progress_bar_config(disable=False)

#     obama_dir = os.path.join(args.out_dir, "obama")
#     trump_dir = os.path.join(args.out_dir, "trump")

#     print("[INFO] generating Obama prompts")
#     generate_set(
#         pipe=pipe,
#         prompts=OBAMA_PROMPTS,
#         seeds=args.seeds,
#         out_dir=obama_dir,
#         tag="obama",
#         num_inference_steps=args.steps,
#         guidance_scale=args.guidance_scale,
#         height=args.height,
#         width=args.width,
#     )

#     print("[INFO] generating Trump prompts")
#     generate_set(
#         pipe=pipe,
#         prompts=TRUMP_PROMPTS,
#         seeds=args.seeds,
#         out_dir=trump_dir,
#         tag="trump",
#         num_inference_steps=args.steps,
#         guidance_scale=args.guidance_scale,
#         height=args.height,
#         width=args.width,
#     )

#     print("\n[INFO] done")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import os
# import sys
# import argparse
# from typing import List

# import torch
# import huggingface_hub

# if not hasattr(huggingface_hub, "cached_download"):
#     huggingface_hub.cached_download = huggingface_hub.hf_hub_download
#     sys.modules["huggingface_hub"].cached_download = huggingface_hub.hf_hub_download

# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["DIFFUSERS_OFFLINE"] = "1"

# from PIL import Image

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OBAMA_PROMPTS = [
#     "a high quality portrait of Barack Obama",
#     "Barack Obama speaking at a podium, professional photograph",
#     "Barack Obama smiling, studio portrait",
#     "Barack Obama, detailed face, realistic photo",
# ]

# TRUMP_PROMPTS = [
#     "a high quality portrait of Donald Trump",
#     "Donald Trump speaking at a podium, professional photograph",
#     "Donald Trump smiling, studio portrait",
#     "Donald Trump, detailed face, realistic photo",
# ]


# def make_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
#     if len(images) == 0:
#         raise ValueError("No images provided")
#     w, h = images[0].size
#     grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
#     for idx, img in enumerate(images):
#         r = idx // cols
#         c = idx % cols
#         grid.paste(img, (c * w, r * h))
#     return grid


# def generate_set(
#     pipe,
#     prompts: List[str],
#     seeds: List[int],
#     out_dir: str,
#     tag: str,
#     num_inference_steps: int,
#     guidance_scale: float,
#     height: int,
#     width: int,
# ):
#     os.makedirs(out_dir, exist_ok=True)
#     all_images = []

#     for p_idx, prompt in enumerate(prompts):
#         prompt_dir = os.path.join(out_dir, f"{tag}_prompt_{p_idx+1}")
#         os.makedirs(prompt_dir, exist_ok=True)

#         for seed in seeds:
#             generator = torch.Generator(device=DEVICE).manual_seed(seed)

#             result = pipe(
#                 prompt=prompt,
#                 num_inference_steps=num_inference_steps,
#                 guidance_scale=guidance_scale,
#                 height=height,
#                 width=width,
#                 generator=generator,
#             )

#             img = result.images[0]
#             all_images.append(img)

#             out_path = os.path.join(prompt_dir, f"seed_{seed}.png")
#             img.save(out_path)
#             print(f"[INFO] saved {out_path}")

#     grid = make_grid(all_images, rows=len(prompts), cols=len(seeds))
#     grid_path = os.path.join(out_dir, f"{tag}_grid.png")
#     grid.save(grid_path)
#     print(f"[INFO] saved grid {grid_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt", required=True, help="Base quantized checkpoint")
#     parser.add_argument("--out_dir", default="smoke_test_quant_base")
#     parser.add_argument("--steps", type=int, default=50)
#     parser.add_argument("--guidance_scale", type=float, default=7.5)
#     parser.add_argument("--height", type=int, default=512)
#     parser.add_argument("--width", type=int, default=512)
#     parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     print(f"[INFO] loading base quantized checkpoint from: {args.ckpt}")
#     pipe = torch.load(args.ckpt, map_location="cpu").to(DEVICE)
#     pipe.unet.eval()
#     if hasattr(pipe, "vae"):
#         pipe.vae.eval()
#     if hasattr(pipe, "text_encoder"):
#         pipe.text_encoder.eval()

#     obama_dir = os.path.join(args.out_dir, "obama")
#     trump_dir = os.path.join(args.out_dir, "trump")

#     print("[INFO] generating Obama prompts from base quantized model")
#     generate_set(
#         pipe=pipe,
#         prompts=OBAMA_PROMPTS,
#         seeds=args.seeds,
#         out_dir=obama_dir,
#         tag="obama",
#         num_inference_steps=args.steps,
#         guidance_scale=args.guidance_scale,
#         height=args.height,
#         width=args.width,
#     )

#     print("[INFO] generating Trump prompts from base quantized model")
#     generate_set(
#         pipe=pipe,
#         prompts=TRUMP_PROMPTS,
#         seeds=args.seeds,
#         out_dir=trump_dir,
#         tag="trump",
#         num_inference_steps=args.steps,
#         guidance_scale=args.guidance_scale,
#         height=args.height,
#         width=args.width,
#     )

#     print("\n[INFO] done")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image, ImageOps, ImageDraw


DEFAULT_CELEBS = [
    "Bill Clinton",
    # "Barack Obama",
    # "Leonardo DiCaprio",
    # "Brad Pitt",
    # "Lionel Messi",
    # "Cristiano Ronaldo",
]

DEFAULT_TEMPLATES = [
    "a photo of {}",
]

DEFAULT_SEEDS = [0, 1, 2, 3]


def make_grid(images: List[Image.Image], rows: int, cols: int, pad: int = 8) -> Image.Image:
    if len(images) == 0:
        raise ValueError("No images provided for grid")

    w, h = images[0].size
    grid_w = cols * w + (cols - 1) * pad
    grid_h = rows * h + (rows - 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), color="white")

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (w + pad)
        y = r * (h + pad)
        grid.paste(img, (x, y))

    return grid


def annotate_image(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 24), fill="white")
    draw.text((6, 4), text, fill="black")
    return img


def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace("'", "")
    )


def load_pipe(model_path: str, device: str, dtype: torch.dtype):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    pipe.set_progress_bar_config(disable=False)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out_dir", type=str, default="celebrity_probe_sd15")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--celebs",
        nargs="+",
        default=DEFAULT_CELEBS,
        help="List of celebrity names to probe",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=DEFAULT_TEMPLATES,
        help="Prompt templates with {} placeholder",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to probe per template",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch.float16 if args.dtype == "fp16" and args.device == "cuda" else torch.float32

    print(f"[INFO] Loading SD pipeline from: {args.model_path}")
    pipe = load_pipe(args.model_path, args.device, dtype)

    metadata_path = os.path.join(args.out_dir, "probe_config.txt")
    with open(metadata_path, "w") as f:
        f.write(f"model_path: {args.model_path}\n")
        f.write(f"guidance_scale: {args.guidance_scale}\n")
        f.write(f"num_inference_steps: {args.num_inference_steps}\n")
        f.write(f"height: {args.height}\n")
        f.write(f"width: {args.width}\n")
        f.write(f"negative_prompt: {args.negative_prompt}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"dtype: {args.dtype}\n")
        f.write(f"templates: {args.templates}\n")
        f.write(f"seeds: {args.seeds}\n")
        f.write(f"celebs: {args.celebs}\n")

    for celeb in args.celebs:
        celeb_dir = os.path.join(args.out_dir, sanitize_name(celeb))
        os.makedirs(celeb_dir, exist_ok=True)

        print(f"\n[INFO] Probing celebrity: {celeb}")
        all_images = []

        prompt_log_path = os.path.join(celeb_dir, "prompts.txt")
        with open(prompt_log_path, "w") as f:
            for template in args.templates:
                prompt = template.format(celeb)
                f.write(prompt + "\n")

        idx = 0
        for template in args.templates:
            prompt = template.format(celeb)

            for seed in args.seeds:
                generator = torch.Generator(device=args.device).manual_seed(seed)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                )

                img = result.images[0]
                filename = f"{idx:03d}_seed{seed}_{sanitize_name(template.replace('{}', celeb))}.png"
                img.save(os.path.join(celeb_dir, filename))

                label = f"{template} | seed={seed}"
                all_images.append(annotate_image(img, label))
                idx += 1

        rows = len(args.templates)
        cols = len(args.seeds)
        grid = make_grid(all_images, rows=rows, cols=cols, pad=8)
        grid.save(os.path.join(celeb_dir, "grid.png"))

        print(f"[INFO] Saved {len(all_images)} images for {celeb} to {celeb_dir}")

    print(f"\n[DONE] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()