#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    get_cosine_schedule_with_warmup,
)

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

try:
    import tifffile
except ImportError:
    tifffile = None


def suppress_opencv_tiff_warnings() -> None:
    try:
        if hasattr(cv2, "setLogLevel"):
            if hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
            elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                cv2.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HuggingFace SegFormer for semantic segmentation from COCO instance annotations"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../0818_clipped_forai/compiled_datasets"),
        help="Directory containing gsd_* folders",
    )
    parser.add_argument(
        "--gsd",
        nargs="+",
        default=None,
        help="Subset of gsd folders to use (default: all under dataset-root)",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="_annotations.fine.coco.json",
        help="COCO annotation filename located in each split folder",
    )
    parser.add_argument(
        "--separate-gsds",
        action="store_true",
        help="Train one model per GSD folder instead of combining all selected GSDs",
    )
    parser.add_argument("--pretrained-model", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TensorBoard logging under each run directory (default: enabled)",
    )
    parser.add_argument(
        "--tb-flush-secs",
        type=int,
        default=30,
        help="TensorBoard flush interval in seconds",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-split",
        choices=["valid", "test"],
        default="valid",
        help="Split used for --eval-only evaluation (default: valid)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to saved model checkpoint directory")
    parser.add_argument(
        "--split-root-name",
        type=str,
        default="dataset",
        help="Split root folder under each gsd (default: dataset, e.g. datasets_learning_curve)",
    )
    parser.add_argument(
        "--train-split-name",
        type=str,
        default="train",
        help="Training split folder name (default: train, e.g. train_40)",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Optional suffix appended to experiment directory name",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="ce",
        help="Training loss function (default: ce)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma value for focal loss when --loss focal",
    )
    parser.add_argument(
        "--use-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable training augmentations (default: enabled)",
    )
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use inverse-frequency class weights for CE/focal loss",
    )
    parser.add_argument(
        "--ignore-background-in-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore background GT pixels when computing metrics (default: disabled)",
    )
    parser.add_argument(
        "--surnud-weight-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier for class 'surnud' weight when class weights are enabled",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_albumentations_version() -> None:
    expected = "1.4.24"
    if A.__version__ != expected:
        raise RuntimeError(
            f"albumentations=={expected} is required, found {A.__version__}. "
            "Install with: pip install albumentations==1.4.24"
        )


def get_autocast_context(device: torch.device, amp_dtype: str):
    if device.type == "cuda" and amp_dtype in {"fp16", "bf16"}:
        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@dataclass
class SemanticSample:
    image_path: Path
    width: int
    height: int
    annotations: List[dict]


def decode_rle(segmentation: dict) -> np.ndarray:
    if mask_utils is None:
        raise RuntimeError(
            "RLE segmentation found but pycocotools is not installed. "
            "Install with: pip install pycocotools"
        )
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, segmentation["size"][0], segmentation["size"][1])
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def rasterize_semantic_mask(
    height: int,
    width: int,
    anns: List[dict],
    cat_id_to_train_id: Dict[int, int],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    anns_sorted = sorted(anns, key=lambda x: float(x.get("area", 0.0)), reverse=True)

    for ann in anns_sorted:
        cat_id = ann.get("category_id")
        train_id = cat_id_to_train_id.get(cat_id)
        if train_id is None:
            continue

        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            polygons: List[np.ndarray] = []
            for poly in segmentation:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                polygons.append(pts)
            if polygons:
                cv2.fillPoly(mask, polygons, int(train_id))
        elif isinstance(segmentation, dict):
            bin_mask = decode_rle(segmentation)
            mask[bin_mask] = int(train_id)

    return mask


class CocoInstanceToSemanticDataset(Dataset):
    def __init__(
        self,
        split_dirs: List[Path],
        annotation_file: str,
        transform: A.Compose,
        cat_id_to_train_id: Dict[int, int],
        max_samples: int = None,
    ) -> None:
        self.transform = transform
        self.cat_id_to_train_id = cat_id_to_train_id
        self.samples: List[SemanticSample] = []

        for split_dir in split_dirs:
            ann_path = split_dir / annotation_file
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing annotation file: {ann_path}")

            with ann_path.open("r", encoding="utf-8") as f:
                coco = json.load(f)

            image_id_to_data = {img["id"]: img for img in coco["images"]}
            anns_by_image: Dict[int, List[dict]] = {}
            for ann in coco["annotations"]:
                anns_by_image.setdefault(ann["image_id"], []).append(ann)

            for image_id, img in image_id_to_data.items():
                self.samples.append(
                    SemanticSample(
                        image_path=split_dir / img["file_name"],
                        width=int(img["width"]),
                        height=int(img["height"]),
                        annotations=anns_by_image.get(image_id, []),
                    )
                )

        self.samples.sort(key=lambda s: str(s.image_path))
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = load_image_rgb(sample.image_path)

        semantic_mask = rasterize_semantic_mask(
            height=sample.height,
            width=sample.width,
            anns=sample.annotations,
            cat_id_to_train_id=self.cat_id_to_train_id,
        )

        transformed = self.transform(image=image, mask=semantic_mask)
        return {
            "pixel_values": transformed["image"],
            "labels": transformed["mask"].long(),
        }


def build_transforms(image_size: int, mean: Sequence[float], std: Sequence[float]) -> Tuple[A.Compose, A.Compose]:
    train_tfms = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(90, 90), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),
        ]
    )
    val_tfms = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),
        ]
    )
    return train_tfms, val_tfms


def build_transforms_configurable(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    use_augmentation: bool,
) -> Tuple[A.Compose, A.Compose]:
    if use_augmentation:
        return build_transforms(image_size, mean, std)
    train_tfms = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),
        ]
    )
    val_tfms = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(transpose_mask=False),
        ]
    )
    return train_tfms, val_tfms


def build_label_maps(categories: List[dict]) -> Tuple[Dict[int, int], Dict[int, str], Dict[str, int]]:
    categories = sorted(categories, key=lambda x: int(x["id"]))
    cat_id_to_train_id = {int(cat["id"]): i + 1 for i, cat in enumerate(categories)}

    id2label = {0: "background"}
    label2id = {"background": 0}
    for cat in categories:
        train_id = cat_id_to_train_id[int(cat["id"])]
        name = str(cat["name"])
        id2label[train_id] = name
        label2id[name] = train_id
    return cat_id_to_train_id, id2label, label2id


def categories_signature(categories: List[dict]) -> List[Tuple[int, str]]:
    return sorted([(int(cat["id"]), str(cat["name"])) for cat in categories], key=lambda x: x[0])


def compute_metrics_from_confusion(conf_mat: np.ndarray, id2label: Dict[int, str]) -> dict:
    num_classes = conf_mat.shape[0]
    class_metrics = {}

    f1_values = []
    iou_values = []
    dice_values = []
    precision_values = []
    recall_values = []

    fg_total = conf_mat[1:, :].sum()
    fg_correct = np.trace(conf_mat[1:, 1:])
    pixel_accuracy_fg = float(fg_correct / max(fg_total, 1))

    for c in range(1, num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp

        precision_denom = tp + fp
        recall_denom = tp + fn
        f1_denom = 2 * tp + fp + fn
        iou_denom = tp + fp + fn

        precision = float(tp / precision_denom) if precision_denom > 0 else float("nan")
        recall = float(tp / recall_denom) if recall_denom > 0 else float("nan")
        f1 = float((2 * tp) / f1_denom) if f1_denom > 0 else float("nan")
        iou = float(tp / iou_denom) if iou_denom > 0 else float("nan")
        dice = float((2 * tp) / f1_denom) if f1_denom > 0 else float("nan")

        class_metrics[id2label[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "dice": dice,
            "support_pixels": int(conf_mat[c, :].sum()),
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        iou_values.append(iou)
        dice_values.append(dice)

    metrics = {
        "macro_precision": float(np.nanmean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.nanmean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.nanmean(f1_values)) if f1_values else 0.0,
        "mean_iou": float(np.nanmean(iou_values)) if iou_values else 0.0,
        "mean_dice": float(np.nanmean(dice_values)) if dice_values else 0.0,
        "pixel_accuracy_fg": pixel_accuracy_fg,
        "class_metrics": class_metrics,
    }
    return metrics


def compute_class_weights_from_coco(
    split_dirs: List[Path],
    annotation_file: str,
    cat_id_to_train_id: Dict[int, int],
    num_classes: int,
    label2id: Dict[str, int],
    surnud_weight_multiplier: float,
) -> torch.Tensor:
    class_area = np.zeros((num_classes,), dtype=np.float64)
    class_area[0] = 1.0

    for split_dir in split_dirs:
        ann_path = split_dir / annotation_file
        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        for ann in coco.get("annotations", []):
            cat_id = int(ann.get("category_id"))
            train_id = cat_id_to_train_id.get(cat_id)
            if train_id is None:
                continue
            class_area[train_id] += float(ann.get("area", 0.0))

    fg_area = class_area[1:]
    fg_area = np.maximum(fg_area, 1.0)
    fg_freq = fg_area / fg_area.sum()
    fg_weights = 1.0 / fg_freq
    fg_weights = fg_weights / np.mean(fg_weights)

    weights = np.ones((num_classes,), dtype=np.float32)
    weights[1:] = fg_weights.astype(np.float32)

    surnud_id = label2id.get("surnud")
    if surnud_id is not None and 0 <= surnud_id < num_classes:
        weights[surnud_id] *= float(surnud_weight_multiplier)

    return torch.tensor(weights, dtype=torch.float32)


def compute_seg_loss(
    outputs,
    labels: torch.Tensor,
    loss_name: str,
    focal_gamma: float,
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    weight = class_weights
    if loss_name == "ce":
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(logits, labels, weight=weight)

    logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** focal_gamma) * ce
    if weight is not None:
        alpha_t = weight[labels]
        focal = alpha_t * focal

    return focal.mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    id2label: Dict[int, str],
    amp_dtype: str,
    loss_name: str,
    focal_gamma: float,
    class_weights: Optional[torch.Tensor],
    ignore_background_in_metrics: bool,
) -> dict:
    model.eval()
    conf_mat_fg = np.zeros((num_classes, num_classes), dtype=np.int64)
    conf_mat_full = np.zeros((num_classes, num_classes), dtype=np.int64)
    running_loss = 0.0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with get_autocast_context(device, amp_dtype):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = compute_seg_loss(outputs, labels, loss_name, focal_gamma, class_weights)

        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * pixel_values.size(0)

        preds_np = preds.cpu().numpy().astype(np.int64)
        labels_np = labels.cpu().numpy().astype(np.int64)

        # Full confusion matrix (includes background)
        gt_full = labels_np.reshape(-1)
        pd_full = preds_np.reshape(-1)
        bincount_full = np.bincount(gt_full * num_classes + pd_full, minlength=num_classes * num_classes)
        conf_mat_full += bincount_full.reshape(num_classes, num_classes)

        # Foreground-only confusion matrix for metric computation
        valid = labels_np != 0
        if np.any(valid):
            gt_fg = labels_np[valid]
            pd_fg = preds_np[valid]
            bincount_fg = np.bincount(gt_fg * num_classes + pd_fg, minlength=num_classes * num_classes)
            conf_mat_fg += bincount_fg.reshape(num_classes, num_classes)

    metrics_conf = conf_mat_fg if ignore_background_in_metrics else conf_mat_full
    metrics = compute_metrics_from_confusion(metrics_conf, id2label)
    metrics["val_loss"] = running_loss / max(len(loader.dataset), 1)
    metrics["confusion_matrix_fg"] = conf_mat_fg.tolist()
    metrics["confusion_matrix_full"] = conf_mat_full.tolist()
    # Backward compatibility: keep confusion_matrix key, now pointing to full matrix
    metrics["confusion_matrix"] = conf_mat_full.tolist()
    return metrics


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_confusion_matrix_csv(path: Path, conf_mat: np.ndarray, id2label: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [id2label[i] for i in range(len(id2label))]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *labels])
        for i, row in enumerate(conf_mat):
            writer.writerow([labels[i], *row.tolist()])


def log_metrics_to_tensorboard(writer, split: str, metrics: dict, step: int) -> None:
    writer.add_scalar(f"{split}/loss", metrics[f"{split}_loss"], step)
    writer.add_scalar(f"{split}/macro_precision", metrics["macro_precision"], step)
    writer.add_scalar(f"{split}/macro_recall", metrics["macro_recall"], step)
    writer.add_scalar(f"{split}/macro_f1", metrics["macro_f1"], step)
    writer.add_scalar(f"{split}/mean_iou", metrics["mean_iou"], step)
    writer.add_scalar(f"{split}/mean_dice", metrics["mean_dice"], step)
    writer.add_scalar(f"{split}/pixel_accuracy_fg", metrics["pixel_accuracy_fg"], step)

    for class_name, class_data in metrics["class_metrics"].items():
        for metric_name in ["precision", "recall", "f1", "iou", "dice"]:
            value = class_data[metric_name]
            if isinstance(value, float) and not np.isnan(value):
                writer.add_scalar(f"{split}_class/{class_name}/{metric_name}", value, step)


def load_categories(annotation_path: Path) -> List[dict]:
    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco["categories"]


def build_split_dirs(dataset_root: Path, gsd_folders: List[str], split_root_name: str, split_name: str) -> List[Path]:
    split_dirs = []
    for gsd in gsd_folders:
        split_dir = dataset_root / gsd / split_root_name / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        split_dirs.append(split_dir)
    return split_dirs


def run_experiment(args: argparse.Namespace, gsd_folders: List[str]) -> None:
    train_dirs = build_split_dirs(args.dataset_root, gsd_folders, args.split_root_name, args.train_split_name)
    val_dirs = build_split_dirs(args.dataset_root, gsd_folders, args.split_root_name, "valid")
    eval_dirs = build_split_dirs(args.dataset_root, gsd_folders, args.split_root_name, args.eval_split)

    category_signatures = []
    categories_ref = None
    for train_dir in train_dirs:
        ann_path = train_dir / args.annotation_file
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")
        categories = load_categories(ann_path)
        category_signatures.append(categories_signature(categories))
        if categories_ref is None:
            categories_ref = categories

    if len({tuple(sig) for sig in category_signatures}) != 1:
        raise RuntimeError("Category IDs/names differ across selected GSD datasets. Use consistent annotation sets.")

    cat_id_to_train_id, id2label, label2id = build_label_maps(categories_ref)
    num_classes = len(id2label)

    processor = SegformerImageProcessor.from_pretrained(args.pretrained_model)
    train_tfms, val_tfms = build_transforms_configurable(
        args.image_size,
        processor.image_mean,
        processor.image_std,
        args.use_augmentation,
    )

    train_ds = CocoInstanceToSemanticDataset(
        split_dirs=train_dirs,
        annotation_file=args.annotation_file,
        transform=train_tfms,
        cat_id_to_train_id=cat_id_to_train_id,
        max_samples=args.max_train_samples,
    )
    val_ds = CocoInstanceToSemanticDataset(
        split_dirs=eval_dirs if args.eval_only else val_dirs,
        annotation_file=args.annotation_file,
        transform=val_tfms,
        cat_id_to_train_id=cat_id_to_train_id,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    gsd_slug = "all_gsds" if len(gsd_folders) > 1 else gsd_folders[0]
    exp_name = f"{gsd_slug}__{args.annotation_file.replace('.json', '')}"
    if args.run_suffix:
        exp_name = f"{exp_name}__{args.run_suffix}"
    out_dir = args.output_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard support requires the tensorboard package. "
                "Install with: pip install tensorboard"
            ) from exc
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"), flush_secs=args.tb_flush_secs)

    save_json(
        out_dir / "data_config.json",
        {
            "dataset_root": str(args.dataset_root),
            "gsd_folders": gsd_folders,
            "annotation_file": args.annotation_file,
            "split_root_name": args.split_root_name,
            "train_split_name": args.train_split_name,
            "eval_split": args.eval_split,
            "use_augmentation": args.use_augmentation,
            "ignore_background_in_metrics": args.ignore_background_in_metrics,
            "num_classes": num_classes,
            "id2label": id2label,
            "label2id": label2id,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        },
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.checkpoint is not None:
        model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            args.pretrained_model,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    model.to(device)

    if args.eval_only:
        class_weights = None
        if args.use_class_weights:
            class_weights = compute_class_weights_from_coco(
                train_dirs,
                args.annotation_file,
                cat_id_to_train_id,
                num_classes,
                label2id,
                args.surnud_weight_multiplier,
            ).to(device)
        metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            id2label,
            args.amp_dtype,
            args.loss,
            args.focal_gamma,
            class_weights,
            args.ignore_background_in_metrics,
        )
        save_json(out_dir / "eval_metrics.json", metrics)
        conf_full = np.asarray(metrics["confusion_matrix_full"], dtype=np.int64)
        conf_fg = np.asarray(metrics["confusion_matrix_fg"], dtype=np.int64)
        save_confusion_matrix_csv(out_dir / "confusion_matrix.csv", conf_full, id2label)
        save_confusion_matrix_csv(out_dir / "confusion_matrix_full.csv", conf_full, id2label)
        save_confusion_matrix_csv(out_dir / "confusion_matrix_fg.csv", conf_fg, id2label)
        if tb_writer is not None:
            log_metrics_to_tensorboard(tb_writer, "val", metrics, step=0)
            tb_writer.flush()
            tb_writer.close()
        print(f"[{exp_name}] macro_f1={metrics['macro_f1']:.4f} mean_iou={metrics['mean_iou']:.4f}")
        return

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp_dtype == "fp16"))
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights_from_coco(
            train_dirs,
            args.annotation_file,
            cat_id_to_train_id,
            num_classes,
            label2id,
            args.surnud_weight_multiplier,
        ).to(device)
        print(f"[{exp_name}] class_weights={class_weights.detach().cpu().tolist()}")
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen_samples = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with get_autocast_context(device, args.amp_dtype):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = (
                    compute_seg_loss(outputs, labels, args.loss, args.focal_gamma, class_weights)
                    / args.grad_accum_steps
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.grad_accum_steps * pixel_values.size(0)
            seen_samples += pixel_values.size(0)

            if step % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            pbar.set_postfix(loss=f"{(epoch_loss / max(seen_samples, 1)):.4f}")

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            id2label,
            args.amp_dtype,
            args.loss,
            args.focal_gamma,
            class_weights,
            args.ignore_background_in_metrics,
        )
        train_loss = epoch_loss / max(len(train_loader.dataset), 1)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        save_json(out_dir / "metrics_history.json", {"history": history})
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", train_loss, epoch)
            tb_writer.add_scalar("train/lr", record["lr"], epoch)
            log_metrics_to_tensorboard(tb_writer, "val", val_metrics, epoch)

        print(
            f"[{exp_name}] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} macro_f1={val_metrics['macro_f1']:.4f} "
            f"mean_iou={val_metrics['mean_iou']:.4f} mean_dice={val_metrics['mean_dice']:.4f}"
        )

        if epoch % args.save_every == 0:
            latest_dir = out_dir / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(latest_dir)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            save_json(out_dir / "best_metrics.json", record)
            conf_full = np.asarray(val_metrics["confusion_matrix_full"], dtype=np.int64)
            conf_fg = np.asarray(val_metrics["confusion_matrix_fg"], dtype=np.int64)
            save_confusion_matrix_csv(out_dir / "confusion_matrix.csv", conf_full, id2label)
            save_confusion_matrix_csv(out_dir / "confusion_matrix_full.csv", conf_full, id2label)
            save_confusion_matrix_csv(out_dir / "confusion_matrix_fg.csv", conf_fg, id2label)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


def main() -> None:
    args = parse_args()
    suppress_opencv_tiff_warnings()
    ensure_albumentations_version()
    set_seed(args.seed)

    if args.gsd is None:
        args.gsd = sorted([p.name for p in args.dataset_root.glob("gsd_*") if p.is_dir()])
        if not args.gsd:
            raise RuntimeError(f"No gsd_* folders found under {args.dataset_root}")

    start = time.time()
    if args.separate_gsds:
        for gsd_folder in args.gsd:
            run_experiment(args, [gsd_folder])
    else:
        run_experiment(args, args.gsd)

    elapsed = time.time() - start
    print(f"Done. Total elapsed: {elapsed / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
