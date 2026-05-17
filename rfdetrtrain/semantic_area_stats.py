import csv
import os
import math
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from tqdm import tqdm
from rfdetr import RFDETRSegLarge, RFDETRSeg2XLarge, RFDETRSegDinov3


# ---- Config ----
DATASET_DIR = "../data/compiled_datasets_3_fold/gsd_0p05/dataset"
SPLIT = "test"
MODEL_PATH = "./train_output_lc_gsd_0p05_train_40_fine_a/checkpoint_best_total.pth"
MODEL_NAME = "segl"  # segl | segxxl | segd3
THRESHOLD = 0.3
DEVICE = "cuda"  # cuda | cpu
OUTPUT_CSV = "./results/species_area_proportions_test.csv"


def load_model(model_path, model_name="segl", device="cuda"):
    if model_name == "segl":
        model = RFDETRSegLarge(pretrain_weights=model_path, device=device)
    elif model_name == "segxxl":
        model = RFDETRSeg2XLarge(pretrain_weights=model_path, device=device)
    elif model_name == "segd3":
        model = RFDETRSegDinov3(pretrain_weights=model_path, freeze_encoder=False, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.optimize_for_inference()
    return model


def build_pred_map(detections, height, width, threshold=0.3):
    pred_map = np.full((height, width), fill_value=-1, dtype=np.int32)
    pred_score = np.full((height, width), fill_value=-np.inf, dtype=np.float32)

    for i in range(len(detections.xyxy)):
        score = float(detections.confidence[i])
        if score < threshold:
            continue

        cls = int(detections.class_id[i])
        mask = np.asarray(detections.mask[i]).astype(bool)

        update = mask & (score > pred_score)
        pred_map[update] = cls
        pred_score[update] = score

    return pred_map


def decode_rle(segmentation):
    if isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, segmentation["size"][0], segmentation["size"][1])
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def build_gt_map(coco, img_id, height, width):
    gt_map = np.full((height, width), fill_value=-1, dtype=np.int32)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    anns_sorted = sorted(anns, key=lambda x: float(x.get("area", 0.0)), reverse=True)

    class_mask = np.zeros((height, width), dtype=np.uint8)
    class_mask.fill(0)

    for ann in anns_sorted:
        cls = int(ann["category_id"]) - 1
        seg = ann.get("segmentation")

        if isinstance(seg, list):
            polys = []
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                polys.append(np.round(pts).astype(np.int32))
            if polys:
                cv2.fillPoly(class_mask, polys, int(cls + 1))
        elif isinstance(seg, dict):
            bin_mask = decode_rle(seg)
            class_mask[bin_mask] = int(cls + 1)

    fg = class_mask > 0
    gt_map[fg] = class_mask[fg].astype(np.int32) - 1

    return gt_map


def main():
    split_dir = os.path.join(DATASET_DIR, SPLIT)
    annotation_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    print(f"Loading annotations: {annotation_path}")
    coco = COCO(annotation_path)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    class_names = {int(cat["id"]) - 1: cat["name"] for cat in cats}

    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH, MODEL_NAME, DEVICE)

    img_ids = coco.getImgIds()
    img_datas = coco.loadImgs(img_ids)

    gt_pixels = {cls: 0 for cls in class_names.keys()}
    pred_pixels = {cls: 0 for cls in class_names.keys()}

    for img_id, img_data in tqdm(list(zip(img_ids, img_datas)), total=len(img_ids)):
        img_path = os.path.join(split_dir, img_data["file_name"])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        detections = model.predict(image, threshold=THRESHOLD)
        if isinstance(detections, tuple):
            detections = detections[0]

        pred_map = build_pred_map(detections, height, width, threshold=THRESHOLD)
        gt_map = build_gt_map(coco, img_id, height, width)

        for cls in class_names.keys():
            gt_pixels[cls] += int((gt_map == cls).sum())
            pred_pixels[cls] += int((pred_map == cls).sum())

    total_gt = sum(gt_pixels.values())
    total_pred = sum(pred_pixels.values())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "gt_pixels",
            "pred_pixels",
            "gt_proportion",
            "pred_proportion",
            "relative_error",
        ])

        for cls in sorted(class_names.keys()):
            gt = gt_pixels[cls]
            pred = pred_pixels[cls]
            gt_prop = (gt / total_gt) if total_gt > 0 else 0.0
            pred_prop = (pred / total_pred) if total_pred > 0 else 0.0

            if gt_prop == 0.0:
                rel_error = math.nan
            else:
                rel_error = (pred_prop - gt_prop) / gt_prop

            writer.writerow([
                class_names[cls],
                gt,
                pred,
                gt_prop,
                pred_prop,
                rel_error,
            ])

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
