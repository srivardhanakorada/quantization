import os
import csv
import json
import math
import argparse
from typing import Dict, List, Tuple

import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default CLIP model
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

# Template required by your protocol
TEXT_TEMPLATE = "A photo of {}"


def load_clip(model_name: str = DEFAULT_CLIP_MODEL):
    print(f"Loading CLIP model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return model, processor


def find_metadata_files(root_dir: str) -> List[str]:
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == "metadata.csv":
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def safe_open_image(path: str):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Failed to open image: {path} ({e})")
        return None


@torch.no_grad()
def encode_texts(
    model,
    processor,
    texts: List[str],
    batch_size: int = 16,
) -> torch.Tensor:
    all_embeds = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_embeds.append(text_features)

    return torch.cat(all_embeds, dim=0)


@torch.no_grad()
def encode_images(
    model,
    processor,
    images: List[Image.Image],
    batch_size: int = 8,
) -> torch.Tensor:
    all_embeds = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(
            images=batch,
            return_tensors="pt",
        ).to(DEVICE)

        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_embeds.append(image_features)

    return torch.cat(all_embeds, dim=0)


def cosine_sim_matrix(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    return image_embeds @ text_embeds.T


def build_text_set(
    target_concepts: List[str],
    preserve_concepts: List[str],
    template: str = TEXT_TEMPLATE,
) -> Tuple[List[str], Dict[str, int]]:
    concept_texts = []
    concept_to_idx = {}

    all_concepts = []
    for c in target_concepts:
        all_concepts.append(("target", c))
    for c in preserve_concepts:
        all_concepts.append(("preserve", c))

    for _, concept in all_concepts:
        text = template.format(concept)
        concept_to_idx[concept] = len(concept_texts)
        concept_texts.append(text)

    return concept_texts, concept_to_idx


def summarize_group(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    group_cols = ["model_name", "bucket"]
    summary = df.groupby(group_cols)[score_columns].agg(["mean", "std", "count"])
    summary.columns = ["__".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing model subfolders with metadata.csv files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clip_metrics_outputs",
        help="Directory to save CLIP metric CSV/JSON outputs",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=DEFAULT_CLIP_MODEL,
        help="HF CLIP model name",
    )
    parser.add_argument(
        "--target_concepts",
        nargs="+",
        default=["Barack Obama"],
        help="Target erased concepts",
    )
    parser.add_argument(
        "--preserve_concepts",
        nargs="+",
        default=["Donald Trump", "Joe Biden", "George W. Bush"],
        help="Preservation concepts to track",
    )
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_files = find_metadata_files(args.root_dir)
    if not metadata_files:
        raise FileNotFoundError(f"No metadata.csv files found under: {args.root_dir}")

    print(f"Found {len(metadata_files)} metadata files")
    for p in metadata_files:
        print(f" - {p}")

    model, processor = load_clip(args.clip_model)

    concept_texts, concept_to_idx = build_text_set(
        target_concepts=args.target_concepts,
        preserve_concepts=args.preserve_concepts,
        template=TEXT_TEMPLATE,
    )

    print("\nText prompts used for CLIP:")
    for t in concept_texts:
        print(f" - {t}")

    text_embeds = encode_texts(
        model=model,
        processor=processor,
        texts=concept_texts,
        batch_size=args.text_batch_size,
    )

    all_rows = []

    for metadata_path in metadata_files:
        print(f"\nReading {metadata_path}")
        df = pd.read_csv(metadata_path)

        required_cols = {"model_name", "bucket", "prompt", "seed", "filepath"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{metadata_path} missing required columns: {missing}")

        images = []
        kept_indices = []

        for idx, row in df.iterrows():
            img = safe_open_image(row["filepath"])
            if img is None:
                continue
            images.append(img)
            kept_indices.append(idx)

        if not images:
            print(f"[WARN] No readable images for {metadata_path}")
            continue

        image_embeds = encode_images(
            model=model,
            processor=processor,
            images=images,
            batch_size=args.image_batch_size,
        )

        sims = cosine_sim_matrix(image_embeds, text_embeds).cpu().numpy()

        for local_i, df_idx in enumerate(kept_indices):
            row = df.loc[df_idx].to_dict()

            # Target scores
            target_scores = {}
            for concept in args.target_concepts:
                col = f"clip_target_{concept.lower().replace(' ', '_')}"
                score = float(sims[local_i, concept_to_idx[concept]])
                target_scores[col] = score

            # Preservation scores
            preserve_scores = {}
            preserve_vals = []
            for concept in args.preserve_concepts:
                col = f"clip_preserve_{concept.lower().replace(' ', '_')}"
                score = float(sims[local_i, concept_to_idx[concept]])
                preserve_scores[col] = score
                preserve_vals.append(score)

            row.update(target_scores)
            row.update(preserve_scores)

            if len(target_scores) > 0:
                row["clip_target_avg"] = float(sum(target_scores.values()) / len(target_scores))
                row["clip_target_max"] = float(max(target_scores.values()))
            else:
                row["clip_target_avg"] = math.nan
                row["clip_target_max"] = math.nan

            if len(preserve_vals) > 0:
                row["clip_preserve_avg"] = float(sum(preserve_vals) / len(preserve_vals))
                row["clip_preserve_max"] = float(max(preserve_vals))
            else:
                row["clip_preserve_avg"] = math.nan
                row["clip_preserve_max"] = math.nan

            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No image rows were successfully processed.")

    per_image_df = pd.DataFrame(all_rows)

    per_image_csv = os.path.join(args.output_dir, "clip_concept_scores_per_image.csv")
    per_image_jsonl = os.path.join(args.output_dir, "clip_concept_scores_per_image.jsonl")

    per_image_df.to_csv(per_image_csv, index=False)
    with open(per_image_jsonl, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved per-image scores to:")
    print(f" - {per_image_csv}")
    print(f" - {per_image_jsonl}")

    score_columns = [
        col for col in per_image_df.columns
        if col.startswith("clip_target_") or col.startswith("clip_preserve_")
    ]

    summary_df = summarize_group(per_image_df, score_columns=score_columns)
    summary_csv = os.path.join(args.output_dir, "clip_concept_scores_summary_by_model_bucket.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved summary to:")
    print(f" - {summary_csv}")

    # Also produce a smaller target/preserve summary table
    compact_cols = [
        "model_name",
        "bucket",
        "clip_target_avg",
        "clip_target_max",
        "clip_preserve_avg",
        "clip_preserve_max",
    ]
    compact_df = (
        per_image_df.groupby(["model_name", "bucket"])[
            ["clip_target_avg", "clip_target_max", "clip_preserve_avg", "clip_preserve_max"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    compact_df.columns = ["__".join(col).strip("_") for col in compact_df.columns.values]
    compact_csv = os.path.join(args.output_dir, "clip_concept_scores_compact_summary.csv")
    compact_df.to_csv(compact_csv, index=False)

    print(f"Saved compact summary to:")
    print(f" - {compact_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()