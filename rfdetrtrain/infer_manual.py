#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.lines import Line2D
from rfdetr import RFDETRSegLarge


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on one image."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to checkpoint (.pth)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Image path for inference.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("../data/manual_dataset/valid/_annotations.medium.coco.json"),
        help="Annotation json for category names/colors.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for model.predict",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save visualization (png). If omitted, shows window.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable drawing class legend on the image.",
    )
    return parser.parse_args()


def ensure_existing_file(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def load_categories(annotation_path: Path):
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {annotation_path}")
    return categories


def get_class_colors(categories):
    cmap = plt.get_cmap("tab20")
    cat_ids = sorted(c["id"] for c in categories)
    return {cid: cmap(i % 20) for i, cid in enumerate(cat_ids)}


def draw_contours(ax, masks, labels, colors):
    for mask, class_id in zip(masks, labels):
        color = colors.get(class_id, (1.0, 0.0, 0.0, 1.0))
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=2)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = ensure_existing_file(args.weights, "Weights")
    annotation_path = ensure_existing_file(args.annotation, "Annotation file")
    image_path = ensure_existing_file(args.image, "Image")

    categories = load_categories(annotation_path)
    colors = get_class_colors(categories)
    model_label_to_cat_id = {i: c["id"] for i, c in enumerate(sorted(categories, key=lambda x: x["id"]))}

    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(f"Annotation: {annotation_path}")
    print(f"Image: {image_path}")
    print(f"Threshold: {args.threshold}")

    model = RFDETRSegLarge(pretrain_weights=str(weights_path), device=device)
    model.optimize_for_inference()

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    dets = model.predict(image, threshold=args.threshold)
    if isinstance(dets, tuple):
        dets = dets[0]

    pred_masks = []
    pred_labels = []
    pred_scores = []

    if dets is not None and len(dets.xyxy) > 0:
        for i in range(len(dets.xyxy)):
            pred_masks.append(dets.mask[i])
            mapped_label = model_label_to_cat_id.get(int(dets.class_id[i]), int(dets.class_id[i]) + 1)
            pred_labels.append(mapped_label)
            pred_scores.append(float(dets.confidence[i]))

    print(f"Detections: {len(pred_masks)}")
    if pred_scores:
        print("Scores:", ", ".join(f"{s:.3f}" for s in pred_scores))

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(image_np)
    draw_contours(ax, pred_masks, pred_labels, colors)
    ax.set_title("Prediction")
    ax.axis("off")

    if not args.no_legend:
        legend_elements = [
            Line2D([0], [0], color=colors[c["id"]], lw=5, label=c["name"]) for c in categories
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            framealpha=0.85,
            fontsize=8,
            title="Classes",
            title_fontsize=9,
        )
    plt.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
