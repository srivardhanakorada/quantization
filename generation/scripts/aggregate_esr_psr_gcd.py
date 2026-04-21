#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import subprocess
from statistics import mean


def folder_to_concept(folder_name: str) -> str:
    return folder_name.replace("_", " ")


def run_eval(eval_script: str, image_folder: str, target_name: str, save_json: str, save_excel: str = None):
    cmd = [
        "python", "-W", "ignore", eval_script,
        "--image_folder", image_folder,
        "--target_name", target_name,
        "--save_json", save_json,
    ]
    if save_excel is not None:
        cmd += ["--save_excel", save_excel]

    print("[CMD]", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {image_folder}")


def collect_group(run_dir: str, group_name: str, eval_script: str, results_dir: str, save_excels: bool):
    group_dir = os.path.join(run_dir, group_name)
    if not os.path.isdir(group_dir):
        return []

    concept_folders = [
        d for d in sorted(os.listdir(group_dir))
        if os.path.isdir(os.path.join(group_dir, d))
    ]

    out = []
    for concept_folder in concept_folders:
        image_folder = os.path.join(group_dir, concept_folder)
        target_name = folder_to_concept(concept_folder)

        json_path = os.path.join(results_dir, f"{group_name}__{concept_folder}.json")
        excel_path = os.path.join(results_dir, f"{group_name}__{concept_folder}.xlsx") if save_excels else None

        run_eval(
            eval_script=eval_script,
            image_folder=image_folder,
            target_name=target_name,
            save_json=json_path,
            save_excel=excel_path,
        )

        with open(json_path, "r") as f:
            data = json.load(f)

        out.append(data)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str,
                        help="Example: generation/outputs/8bit/erase_then_quantize/uce/single")
    parser.add_argument("--eval_script", default="generation/scripts/eval_one_folder_gcd.py", type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--save_excels", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip("/")
    results_dir = args.results_dir or os.path.join(run_dir, "gcd_results")
    os.makedirs(results_dir, exist_ok=True)

    erase_scores = collect_group(run_dir, "erase", args.eval_script, results_dir, args.save_excels)
    preserve_scores = collect_group(run_dir, "preserve", args.eval_script, results_dir, args.save_excels)

    esr = mean([x["avg_gcd"] for x in erase_scores]) if len(erase_scores) > 0 else 0.0
    psr = mean([x["avg_gcd"] for x in preserve_scores]) if len(preserve_scores) > 0 else 0.0

    summary = {
        "run_dir": run_dir,
        "ESR": esr,
        "PSR": psr,
        "erase_scores": erase_scores,
        "preserve_scores": preserve_scores,
    }

    out_path = os.path.join(run_dir, "esr_psr_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nUSE THESE NUMBERS IN THE TABLE\n")
    print(f"ESR = {esr:.6f}")
    print(f"PSR = {psr:.6f}")
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()