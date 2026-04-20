#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import subprocess
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def parse_alpha_path_list(items):
    out = OrderedDict()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected alpha=path format, got: {item}")
        alpha_str, path = item.split("=", 1)
        alpha = float(alpha_str)
        out[alpha] = path
    return out


def get_dtype(dtype_str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def is_probably_directory_model(path):
    return os.path.isdir(path)


def load_pipeline_any(model_path, dtype=torch.float32, local_files_only=True):
    """
    Supports:
      1) Diffusers directory
      2) torch.load-able pipeline/checkpoint object with .unet
    """
    if is_probably_directory_model(model_path):
        return StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=local_files_only,
        )

    obj = torch.load(model_path, map_location="cpu")
    if hasattr(obj, "unet"):
        return obj

    raise ValueError(
        f"Could not load model from {model_path}. "
        f"Expected a diffusers directory or a torch checkpoint with `.unet`."
    )


def extract_unet_state(model_path, dtype=torch.float32, local_files_only=True):
    pipe = load_pipeline_any(
        model_path=model_path,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    state = {k: v.detach().cpu().float().clone() for k, v in pipe.unet.state_dict().items()}
    del pipe
    return state


def get_edited_param_names(state_dict):
    names = []
    for name in state_dict.keys():
        if "attn2" in name and (name.endswith("to_k.weight") or name.endswith("to_v.weight")):
            names.append(name)
    return sorted(names)


def compute_bucket_crossing_fraction(
    base_fp_sd,
    scaled_fp_sd,
    base_q_sd,
    scaled_q_sd,
    eps=1e-12,
    qeps=1e-12,
):
    """
    Computes:
      numerator   = number of changed edited weights whose quantized value differs from quantized base
      denominator = number of changed edited weights in FP
    """
    edited_names = get_edited_param_names(base_fp_sd)

    total_changed = 0
    total_crossed = 0
    per_layer = {}

    for name in edited_names:
        if name not in scaled_fp_sd or name not in base_q_sd or name not in scaled_q_sd:
            raise KeyError(f"Missing parameter {name} in one of the state dicts")

        base_fp = base_fp_sd[name]
        scaled_fp = scaled_fp_sd[name]
        base_q = base_q_sd[name]
        scaled_q = scaled_q_sd[name]

        changed_mask = (scaled_fp - base_fp).abs() > eps
        crossed_mask = (scaled_q - base_q).abs() > qeps

        changed_count = int(changed_mask.sum().item())
        crossed_count = int((crossed_mask & changed_mask).sum().item())

        total_changed += changed_count
        total_crossed += crossed_count

        per_layer[name] = {
            "changed_count": changed_count,
            "crossed_count": crossed_count,
            "fraction": (crossed_count / changed_count) if changed_count > 0 else 0.0,
        }

    frac = (total_crossed / total_changed) if total_changed > 0 else 0.0
    return frac, per_layer


def make_generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


@torch.no_grad()
def generate_images_for_model(
    model_path,
    out_dir,
    prompt,
    num_images=100,
    seeds=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,
    num_inference_steps=50,
    local_files_only=True,
):
    os.makedirs(out_dir, exist_ok=True)

    pipe = load_pipeline_any(
        model_path=model_path,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    if seeds is None:
        seeds = list(range(num_images))
    assert len(seeds) == num_images

    manifest = []
    for idx, seed in enumerate(seeds):
        generator = make_generator(device=device, seed=seed)
        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        filename = f"{idx:04d}_seed{seed}.png"
        save_path = os.path.join(out_dir, filename)
        image.save(save_path)

        manifest.append({
            "index": idx,
            "seed": seed,
            "prompt": prompt,
            "image_path": save_path,
        })

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    del pipe
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def run_external_gcd(gcd_cmd_template, image_dir, output_json):
    cmd = gcd_cmd_template.format(
        image_dir=shlex.quote(image_dir),
        output_json=shlex.quote(output_json),
    )
    subprocess.run(cmd, shell=True, check=True)

    if not os.path.exists(output_json):
        raise FileNotFoundError(f"GCD output JSON not found: {output_json}")

    with open(output_json, "r") as f:
        data = json.load(f)

    if "gcd_score" not in data:
        raise KeyError(
            f"GCD output JSON must contain key `gcd_score`. Got keys: {list(data.keys())}"
        )
    return float(data["gcd_score"])


def save_results_csv(rows, csv_path):
    fieldnames = [
        "alpha",
        "fp_model_path",
        "quant_model_path",
        "bucket_crossing_fraction",
        "gcd_fp",
        "gcd_quant",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plot(rows, out_png, out_pdf=None):
    rows = sorted(rows, key=lambda x: x["alpha"])
    alphas = [r["alpha"] for r in rows]
    gcd_quant = [r["gcd_quant"] for r in rows]
    bucket_frac = [r["bucket_crossing_fraction"] for r in rows]
    gcd_fp = [r["gcd_fp"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(alphas, gcd_quant, marker="o", label="Quantized GCD")
    if all(v is not None for v in gcd_fp):
        ax1.plot(alphas, gcd_fp, marker="x", linestyle="--", label="FP GCD")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("GCD-based ESR")
    ax1.set_xscale("log", base=2)
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    ax2.plot(alphas, bucket_frac, marker="s", linestyle="-.")
    ax2.set_ylabel("Fraction crossing buckets")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + ["Bucket crossing"], loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate update-scale ablation: image generation, GCD, bucket crossing, and plotting."
    )
    parser.add_argument("--base_model_fp", type=str, required=True)
    parser.add_argument("--base_model_quant", type=str, required=True)
    parser.add_argument(
        "--fp_models",
        type=str,
        nargs="+",
        required=True,
        help='List like: 1.0=/path/to/alpha_1p0 0.5=/path/to/alpha_0p5 ...',
    )
    parser.add_argument(
        "--quant_models",
        type=str,
        nargs="+",
        required=True,
        help='List like: 1.0=/path/to/q_alpha_1p0 0.5=/path/to/q_alpha_0p5 ...',
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a photo of Barack Obama")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--qeps", type=float, default=1e-12)
    parser.add_argument(
        "--gcd_cmd_template",
        type=str,
        default=None,
        help=(
            "Optional shell command template for external GCD evaluation. "
            "Must contain {image_dir} and {output_json}. "
            'Example: python gcd_eval.py --image_dir {image_dir} --output_json {output_json}'
        ),
    )
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    dtype = get_dtype(args.dtype)

    fp_models = parse_alpha_path_list(args.fp_models)
    quant_models = parse_alpha_path_list(args.quant_models)

    if set(fp_models.keys()) != set(quant_models.keys()):
        raise ValueError("Alpha keys in --fp_models and --quant_models must match exactly")

    alphas = sorted(fp_models.keys(), reverse=True)

    print("[INFO] Loading state dicts needed for bucket crossing...")
    base_fp_sd = extract_unet_state(
        args.base_model_fp, dtype=dtype, local_files_only=args.local_files_only
    )
    base_q_sd = extract_unet_state(
        args.base_model_quant, dtype=dtype, local_files_only=args.local_files_only
    )

    seeds = list(range(args.num_images))
    rows = []

    for alpha in alphas:
        alpha_tag = str(alpha).replace(".", "p")
        fp_model_path = fp_models[alpha]
        quant_model_path = quant_models[alpha]

        print(f"\n[INFO] Processing alpha={alpha}")

        fp_sd = extract_unet_state(fp_model_path, dtype=dtype, local_files_only=args.local_files_only)
        q_sd = extract_unet_state(quant_model_path, dtype=dtype, local_files_only=args.local_files_only)

        bucket_frac, per_layer = compute_bucket_crossing_fraction(
            base_fp_sd=base_fp_sd,
            scaled_fp_sd=fp_sd,
            base_q_sd=base_q_sd,
            scaled_q_sd=q_sd,
            eps=args.eps,
            qeps=args.qeps,
        )

        alpha_out = os.path.join(args.output_root, f"alpha_{alpha_tag}")
        os.makedirs(alpha_out, exist_ok=True)

        with open(os.path.join(alpha_out, "bucket_crossing.json"), "w") as f:
            json.dump(
                {
                    "alpha": alpha,
                    "bucket_crossing_fraction": bucket_frac,
                    "per_layer": per_layer,
                },
                f,
                indent=2,
            )

        fp_img_dir = os.path.join(alpha_out, "fp_images")
        q_img_dir = os.path.join(alpha_out, "quant_images")

        print(f"[INFO] Generating FP images for alpha={alpha}")
        generate_images_for_model(
            model_path=fp_model_path,
            out_dir=fp_img_dir,
            prompt=args.prompt,
            num_images=args.num_images,
            seeds=seeds,
            device=args.device,
            dtype=dtype,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            local_files_only=args.local_files_only,
        )

        print(f"[INFO] Generating quantized images for alpha={alpha}")
        generate_images_for_model(
            model_path=quant_model_path,
            out_dir=q_img_dir,
            prompt=args.prompt,
            num_images=args.num_images,
            seeds=seeds,
            device=args.device,
            dtype=dtype,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            local_files_only=args.local_files_only,
        )

        gcd_fp = None
        gcd_quant = None

        if args.gcd_cmd_template is not None:
            fp_gcd_json = os.path.join(alpha_out, "gcd_fp.json")
            q_gcd_json = os.path.join(alpha_out, "gcd_quant.json")

            print(f"[INFO] Running external GCD for FP alpha={alpha}")
            gcd_fp = run_external_gcd(args.gcd_cmd_template, fp_img_dir, fp_gcd_json)

            print(f"[INFO] Running external GCD for quantized alpha={alpha}")
            gcd_quant = run_external_gcd(args.gcd_cmd_template, q_img_dir, q_gcd_json)

        rows.append(
            {
                "alpha": alpha,
                "fp_model_path": fp_model_path,
                "quant_model_path": quant_model_path,
                "bucket_crossing_fraction": bucket_frac,
                "gcd_fp": gcd_fp,
                "gcd_quant": gcd_quant,
            }
        )

        del fp_sd, q_sd
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    csv_path = os.path.join(args.output_root, "update_scale_results.csv")
    save_results_csv(rows, csv_path)

    # Make plot only if gcd_quant values are available
    if all(r["gcd_quant"] is not None for r in rows):
        make_plot(
            rows,
            out_png=os.path.join(args.output_root, "update_scaling_curve.png"),
            out_pdf=os.path.join(args.output_root, "update_scaling_curve.pdf"),
        )

    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] Results CSV: {csv_path}")
    if all(r['gcd_quant'] is not None for r in rows):
        print(f"[INFO] Plot saved to: {os.path.join(args.output_root, 'update_scaling_curve.png')}")


if __name__ == "__main__":
    main()