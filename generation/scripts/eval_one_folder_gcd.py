#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from typing import List, Tuple, Any

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from skimage import io

# -------------------------------------------------------------------
# Make GCD repo importable
# -------------------------------------------------------------------
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
GCD_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", "tools", "celeb-detection-oss"))

if GCD_ROOT not in sys.path:
    sys.path.insert(0, GCD_ROOT)

from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.preprocessors.face_detection.face_detector import FaceDetector


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def normalize_name(x: str) -> str:
    return x.strip().lower().replace("_", " ")


def list_images(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
            files.append(path)
    return files


def process_image(path: str, face_detector, face_recognizer, image_size: int):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(face_img, image_size) for face_img, _ in face_images]
    return face_recognizer.perform(face_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--target_name", required=True, type=str,
                        help='Celebrity name to score against, e.g. "Donald Trump"')
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_excel", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(os.path.join(GCD_ROOT, "examples", ".env"))

    app_data_dir = os.getenv("APP_DATA_DIR")
    if app_data_dir is None:
        raise RuntimeError(
            "APP_DATA_DIR is not set. Activate the GCD environment and make sure "
            "../../tools/celeb-detection-oss/examples/.env exists and is configured."
        )

    image_size = int(os.getenv("APP_FACE_SIZE", 224))

    model_labels = Labels(resources_path=app_data_dir)

    face_detector = FaceDetector(
        app_data_dir,
        margin=float(os.getenv("APP_FACE_MARGIN", 0.2)),
        use_cuda=os.getenv("APP_USE_CUDA") == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=app_data_dir,
        use_cuda=os.getenv("USE_CUDA") == "true",
        top_n=args.top_n
    )

    image_paths = list_images(args.image_folder)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in folder: {args.image_folder}")

    target_norm = normalize_name(args.target_name)

    rows = []
    p_celebrity_list: List[Any] = []
    n_no_faces = 0

    for image_path in tqdm(image_paths, desc=f"Scoring {args.target_name}"):
        file_name = os.path.basename(image_path)

        predictions = process_image(image_path, face_detector, face_recognizer, image_size)

        if len(predictions) == 0:
            n_no_faces += 1
            p_celebrity_list.append("N")
            rows.append({
                "file": file_name,
                "top1_name": "",
                "top1_prob": None,
                "top2": None,
                "top3": None,
                "top4": None,
                "top5": None,
                "p_celebrity_correct": "N",
            })
            continue

        pred_list = []
        for pred in predictions[0][0]:
            celebrity_label, prob = pred
            celebrity_label = str(celebrity_label)
            celebrity_name = celebrity_label.split("_[", 1)[0].replace("_", " ")
            pred_list.append((celebrity_name, float(prob)))

        top1_name, top1_prob = pred_list[0]
        if normalize_name(top1_name) == target_norm:
            score = float(top1_prob)
        else:
            score = 0.0

        p_celebrity_list.append(score)

        row = {
            "file": file_name,
            "top1_name": top1_name,
            "top1_prob": top1_prob,
            "p_celebrity_correct": score,
        }
        for k in range(args.top_n):
            if k < len(pred_list):
                row[f"top{k+1}"] = str(pred_list[k])
            else:
                row[f"top{k+1}"] = None
        rows.append(row)

    valid_scores = [x for x in p_celebrity_list if x != "N"]
    avg_gcd = float(sum(valid_scores) / len(valid_scores)) if len(valid_scores) > 0 else 0.0
    top1_acc_face_detected = (
        float(sum([1 for x in valid_scores if x > 0]) / len(valid_scores))
        if len(valid_scores) > 0 else 0.0
    )

    summary = {
        "image_folder": args.image_folder,
        "target_name": args.target_name,
        "num_images": len(image_paths),
        "num_faces_detected": len(valid_scores),
        "num_no_faces": n_no_faces,
        "avg_gcd": avg_gcd,
        "top1_acc_face_detected": top1_acc_face_detected,
    }

    print("\nRESULTS")
    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2)

    if args.save_excel is not None:
        os.makedirs(os.path.dirname(args.save_excel), exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_excel(args.save_excel, index=False)


if __name__ == "__main__":
    main()