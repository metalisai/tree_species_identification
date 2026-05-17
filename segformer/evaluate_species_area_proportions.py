#!/usr/bin/env python3
"""Evaluate tree-species area proportions from a trained SegFormer checkpoint.
This script runs semantic inference on a split (default: test) of the 3_fold dataset,
computes per-class pixel area/proportions (excluding background), and reports
classification metrics (precision/recall/F1/IoU) per class.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

try:
    import tifffile
except ImportError:
    tifffile = None


@dataclass
class Sample:
    image_path: Path
    width: int
    height: int
    annotations: List[dict]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate species area proportions from SegFormer checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint directory")
    p.add_argument(
        "--pretrained-model",
        type=str,
        default="nvidia/segformer-b2-finetuned-ade-512-512",
        help="Fallback image processor source when checkpoint lacks preprocessor config",
    )
    p.add_argument("--dataset-root", type=Path, default=Path("../data/compiled_datasets_3_fold"))
    p.add_argument("--gsd", type=str, required=True, help="Single gsd folder, e.g. gsd_0p05")
    p.add_argument("--split-root-name", type=str, default="dataset", help="Split root under gsd folder")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--annotation-file", type=str, default="_annotations.fine.coco.json")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp-dtype", choices=["none", "fp16", "bf16"], default="bf16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-json", type=Path, default=Path("species_area_eval.json"))
    p.add_argument("--output-csv", type=Path, default=Path("species_area_eval.csv"))
    return p.parse_args()


def load_image_rgb(image_path: Path) -> np.ndarray:
    suffix = image_path.suffix.lower()
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        image = tifffile.imread(str(image_path))
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape {image.shape} for {image_path}")
        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_rle(segmentation: dict) -> np.ndarray:
    if mask_utils is None:
        raise RuntimeError("RLE segmentation found but pycocotools is not installed")
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, segmentation["size"][0], segmentation["size"][1])
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def rasterize_semantic_mask(height: int, width: int, anns: List[dict], cat_to_train: Dict[int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    anns_sorted = sorted(anns, key=lambda x: float(x.get("area", 0.0)), reverse=True)
    for ann in anns_sorted:
        train_id = cat_to_train.get(int(ann.get("category_id")))
        if train_id is None:
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            polys = []
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                polys.append(np.round(pts).astype(np.int32))
            if polys:
                cv2.fillPoly(mask, polys, int(train_id))
        elif isinstance(seg, dict):
            bin_mask = decode_rle(seg)
            mask[bin_mask] = int(train_id)
    return mask


def build_label_maps(categories: List[dict]) -> Tuple[Dict[int, int], Dict[int, str]]:
    categories = sorted(categories, key=lambda x: int(x["id"]))
    cat_to_train = {int(c["id"]): i + 1 for i, c in enumerate(categories)}
    id2label = {0: "background"}
    for c in categories:
        id2label[cat_to_train[int(c["id"])] ] = str(c["name"])
    return cat_to_train, id2label


def load_split_samples(split_dir: Path, ann_file: str, cat_to_train: Dict[int, int]) -> List[Sample]:
    ann_path = split_dir / ann_file
    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {int(img["id"]): img for img in coco["images"]}
    ann_by_image: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        ann_by_image.setdefault(image_id, []).append(ann)

    samples: List[Sample] = []
    for image_id, img in img_map.items():
        if image_id not in ann_by_image:
            continue
        samples.append(
            Sample(
                image_path=split_dir / img["file_name"],
                width=int(img["width"]),
                height=int(img["height"]),
                annotations=ann_by_image[image_id],
            )
        )
    samples.sort(key=lambda s: str(s.image_path))
    return samples


def compute_metrics(conf: np.ndarray, id2label: Dict[int, str]) -> Tuple[dict, dict]:
    num_classes = conf.shape[0]
    per_class = {}
    f1_list = []
    iou_list = []
    precision_list = []
    recall_list = []

    for c in range(1, num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = (2 * p * r) / max(p + r, 1e-8)
        iou = tp / max(tp + fp + fn, 1)
        per_class[id2label[c]] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "iou": float(iou),
            "gt_pixels": int(conf[c, :].sum()),
            "pred_pixels": int(conf[:, c].sum()),
        }
        f1_list.append(float(f1))
        iou_list.append(float(iou))
        precision_list.append(float(p))
        recall_list.append(float(r))

    overall = {
        "macro_precision": float(np.mean(precision_list)) if precision_list else float("nan"),
        "macro_recall": float(np.mean(recall_list)) if recall_list else float("nan"),
        "macro_f1": float(np.mean(f1_list)) if f1_list else float("nan"),
        "mean_iou": float(np.mean(iou_list)) if iou_list else float("nan"),
    }
    return overall, per_class


def to_comma(v: float) -> str:
    return f"{v:.6f}".replace(".", ",")


def main() -> None:
    args = parse_args()

    split_dir = args.dataset_root / args.gsd / args.split_root_name / args.split
    ann_path = split_dir / args.annotation_file
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = coco["categories"]
    cat_to_train, id2label = build_label_maps(categories)
    num_classes = len(id2label)

    samples = load_split_samples(split_dir, args.annotation_file, cat_to_train)
    if not samples:
        raise RuntimeError("No samples found in requested split")

    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
    try:
        processor = SegformerImageProcessor.from_pretrained(args.checkpoint)
    except OSError:
        processor = SegformerImageProcessor.from_pretrained(args.pretrained_model)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tfm = A.Compose(
        [
            A.Resize(args.image_size, args.image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=processor.image_mean, std=processor.image_std, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),
        ]
    )

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i : i + args.batch_size]
            px = []
            gt = []
            for s in batch:
                image = load_image_rgb(s.image_path)
                gt_mask = rasterize_semantic_mask(s.height, s.width, s.annotations, cat_to_train)
                out = tfm(image=image, mask=gt_mask)
                px.append(out["image"])
                gt.append(out["mask"].to(torch.int64))

            pixel_values = torch.stack(px).to(device, non_blocking=True)
            labels = torch.stack(gt).to(device, non_blocking=True)

            if device.type == "cuda" and args.amp_dtype in {"fp16", "bf16"}:
                dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(pixel_values=pixel_values).logits
            else:
                logits = model(pixel_values=pixel_values).logits

            up = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1)

            y_true = labels.view(-1).cpu().numpy()
            y_pred = pred.view(-1).cpu().numpy()
            idx = y_true * num_classes + y_pred
            binc = np.bincount(idx, minlength=num_classes * num_classes)
            conf += binc.reshape(num_classes, num_classes)

    overall, per_class = compute_metrics(conf, id2label)

    gt_pixels = {id2label[c]: int(conf[c, :].sum()) for c in range(1, num_classes)}
    pred_pixels = {id2label[c]: int(conf[:, c].sum()) for c in range(1, num_classes)}
    gt_total = sum(gt_pixels.values())
    pred_total = sum(pred_pixels.values())

    gt_props = {k: (v / gt_total if gt_total > 0 else 0.0) for k, v in gt_pixels.items()}
    pred_props = {k: (v / pred_total if pred_total > 0 else 0.0) for k, v in pred_pixels.items()}

    prop_abs_err = {k: abs(pred_props[k] - gt_props[k]) for k in gt_props}
    prop_rel_err = {
        k: (prop_abs_err[k] / gt_props[k] if gt_props[k] > 0 else float("nan"))
        for k in gt_props
    }
    l1 = float(sum(prop_abs_err.values()))
    proportion_similarity = float(1.0 - 0.5 * l1)

    result = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "gsd": args.gsd,
            "split_root_name": args.split_root_name,
            "split": args.split,
            "annotation_file": args.annotation_file,
            "image_size": args.image_size,
            "num_samples": len(samples),
            "num_classes_with_background": num_classes,
        },
        "overall_metrics_fg": overall,
        "area_proportions": {
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "gt_proportions": gt_props,
            "pred_proportions": pred_props,
            "abs_error": prop_abs_err,
            "relative_error": prop_rel_err,
            "l1_distance": l1,
            "proportion_similarity": proportion_similarity,
        },
        "per_class_metrics": per_class,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class",
                "gt_pixels",
                "pred_pixels",
                "gt_proportion",
                "pred_proportion",
                "abs_error",
                "relative_error",
                "precision",
                "recall",
                "f1",
                "iou",
            ]
        )
        for cls in sorted(per_class.keys()):
            m = per_class[cls]
            w.writerow(
                [
                    cls,
                    gt_pixels[cls],
                    pred_pixels[cls],
                    to_comma(gt_props[cls]),
                    to_comma(pred_props[cls]),
                    to_comma(prop_abs_err[cls]),
                    "" if np.isnan(prop_rel_err[cls]) else to_comma(prop_rel_err[cls]),
                    to_comma(m["precision"]),
                    to_comma(m["recall"]),
                    to_comma(m["f1"]),
                    to_comma(m["iou"]),
                ]
            )

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(
        "FG overall: "
        f"macro_f1={overall['macro_f1']:.4f}, mean_iou={overall['mean_iou']:.4f}, "
        f"macro_precision={overall['macro_precision']:.4f}, macro_recall={overall['macro_recall']:.4f}"
    )
    print(
        "Proportion quality: "
        f"L1={l1:.6f}, similarity(1-0.5*L1)={proportion_similarity:.6f}"
    )


if __name__ == "__main__":
    main()
