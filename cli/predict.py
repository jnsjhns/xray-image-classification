# cli/predict.py
from __future__ import annotations

import argparse
import random

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from tensorflow import keras

from src.config import PipelineConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict chest X-ray classes for one image, multiple images, or folders."
    )

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=None,
        help="One or more image paths.",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image_dir recursively.",
    )

    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Randomly sample N images from the selected input images.",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=["NORMAL", "PNEUMONIA"],
        help="Class names in the same order as during training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random sampling.",
    )

    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(args: argparse.Namespace) -> list[Path]:
    image_paths: list[Path] = []

    if args.images:
        image_paths.extend(Path(p) for p in args.images)

    if args.image_dir:
        image_dir = Path(args.image_dir)

        if args.recursive:
            candidates = image_dir.rglob("*")
        else:
            candidates = image_dir.glob("*")

        image_paths.extend(p for p in candidates if is_image_file(p))

    image_paths = [p for p in image_paths if is_image_file(p)]

    if not image_paths:
        raise ValueError(
            "No valid images found. Use --images, --image_dir, or both."
        )

    image_paths = sorted(set(image_paths))

    if args.random is not None:
        if args.random <= 0:
            raise ValueError("--random must be greater than 0.")

        random.seed(args.seed)
        image_paths = random.sample(
            image_paths,
            k=min(args.random, len(image_paths)),
        )

    return image_paths


def load_image(image_path: Path, img_size: int) -> np.ndarray:
    img = keras.utils.load_img(
        image_path,
        target_size=(img_size, img_size),
        color_mode="rgb",
    )
    img_array = keras.utils.img_to_array(img)
    return img_array


def predict_images(
    model: keras.Model,
    image_paths: list[Path],
    img_size: int,
    class_names: list[str],
) -> None:
    images = np.stack(
        [load_image(path, img_size) for path in image_paths],
        axis=0,
    )

    predictions = model.predict(images, verbose=0)

    print()
    print("Predictions")
    print("=" * 100)

    for image_path, probs in zip(image_paths, predictions):
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        probs_text = " | ".join(
            f"{class_name}: {float(prob):.4f}"
            for class_name, prob in zip(class_names, probs)
        )

        print(f"{image_path}")
        print(f"  → Prediction : {pred_class}")
        print(f"  → Confidence : {confidence:.4f}")
        print(f"  → Probabilities: {probs_text}")
        print()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    model = keras.models.load_model(model_path, compile=False)

    img_size = int(model.input_shape[1])

    image_paths = collect_image_paths(args)

    if len(args.class_names) != model.output_shape[-1]:
        raise ValueError(
            f"Number of class names ({len(args.class_names)}) does not match "
            f"model output size ({model.output_shape[-1]})."
        )

    print(f"Loaded model : {model_path}")
    print(f"Images found : {len(image_paths)}")

    predict_images(
        model=model,
        image_paths=image_paths,
        img_size=img_size,
        class_names=args.class_names,
    )

    print(f"Model input shape: {model.input_shape}")


if __name__ == "__main__":
    main()