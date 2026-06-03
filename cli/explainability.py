# cli/explainability.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import PipelineConfig
from src.data_loader import build_datasets
from src.helpers import print_section, print_kv, ensure_dir
from src.model import get_preprocess_fn


# ---------- ARGUMENTS ---------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM explainability for chest X-ray models",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_per_class", type=int, default=2)
    return parser.parse_args()


# ---------- DATA ---------- #

def collect_images(
    config: PipelineConfig,
    num_per_class: int,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    data = build_datasets(config)

    class_names = data.class_names
    num_classes = data.num_classes

    imgs_by_class: list[list[np.ndarray]] = [[] for _ in range(num_classes)]

    for imgs, labels in data.test_ds:
        imgs = imgs.numpy().astype("uint8")
        labels = labels.numpy()
        for img, lbl in zip(imgs, labels):
            imgs_by_class[int(lbl)].append(img)

    out_imgs: list[np.ndarray] = []
    out_labels: list[int] = []

    for c in range(num_classes):
        if not imgs_by_class[c]:
            continue
        random.shuffle(imgs_by_class[c])
        chosen = imgs_by_class[c][:num_per_class]
        out_imgs.extend(chosen)
        out_labels.extend([c] * len(chosen))

    return out_imgs, out_labels, class_names


# ---------- GRAD-CAM ---------- #

def make_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    class_idx: int | None = None,
) -> np.ndarray:
    """
    img_array: shape (1, H, W, 3), already preprocessed.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    last_conv_layer = model.get_layer("gradcam_features")
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds_for_grad = grad_model(img_tensor, training=False)

        if class_idx is None:
            class_idx = int(tf.argmax(preds_for_grad[0]))

        loss = preds_for_grad[:, class_idx]

    grads = tape.gradient(loss, conv_out)

    if grads is None:
        raise ValueError(
            "Gradients are None. The gradient flow from the output to "
            "the 'gradcam_features' layer seems to be broken."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # (batch, C)
    conv_out = conv_out[0]                             # (H, W, C)
    pooled_grads = pooled_grads[0]                     # (C,)

    conv_out = tf.multiply(conv_out, pooled_grads)

    heatmap = tf.reduce_sum(conv_out, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


def overlay(img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    heatmap_resized = tf.image.resize(
        heatmap[..., None],
        (h, w),
    ).numpy().squeeze()
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_resized)[..., :3]
    colored = (colored * 255).astype("uint8")

    out = img.astype("float32") * 0.6 + colored.astype("float32") * 0.4
    out = np.clip(out, 0, 255).astype("uint8")
    return out


# ---------- MAIN ---------- #

def main() -> None:
    args = parse_args()
    config = PipelineConfig()

    print_section("Loading model")
    model_path = Path(args.model_path)
    model = keras.models.load_model(model_path, compile=False)
    print_kv("Model", model.name)

    preprocess_fn = get_preprocess_fn(config.model_name)

    run_dir = model_path.parents[1]  # experiment_outputs/<timestamp_run_name>/
    fig_dir = run_dir / "reports" / "figures" / "grad_cam"
    ensure_dir(fig_dir)

    print_section("Loading test data")
    imgs, labels, class_names = collect_images(config, args.num_per_class)
    print_kv("Classes", class_names)
    print_kv("Images", len(imgs))

    print_section("Grad-CAM")

    for idx, (img, label) in enumerate(zip(imgs, labels)):
        class_idx = int(label)
        class_name = class_names[class_idx]

        x = img.astype("float32")
        x = preprocess_fn(x)
        x = np.expand_dims(x, axis=0)

        heatmap = make_gradcam_heatmap(model, x, class_idx)
        overlay_img = overlay(img, heatmap)

        out_path = fig_dir / f"gradcam_{idx:03d}_{class_name}.png"

        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{class_name} (true)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlay_img)
        plt.title("Grad-CAM")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print_kv("Saved", out_path)


if __name__ == "__main__":
    main()