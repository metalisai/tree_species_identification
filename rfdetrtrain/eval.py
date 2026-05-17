import os
import re
import csv
import json
import argparse
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib.pyplot as plt
from rfdetr import RFDETRSegLarge, RFDETRSeg2XLarge, RFDETRSegDinov3


def load_model(model_dir, model_name="segl", device="cuda"):
    if model_name == "segl":
        model = RFDETRSegLarge(pretrain_weights=model_dir, device=device)
    elif model_name == "segxxl":
        model = RFDETRSeg2XLarge(pretrain_weights=model_dir, device=device)
    elif model_name == "segd3":
        model = RFDETRSegDinov3(pretrain_weights=model_dir, freeze_encoder=False, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    trainable_params = sum(
        p.numel() for p in model.model.model.parameters() if p.requires_grad
    )

    model.optimize_for_inference()
    return model, trainable_params


def mask_to_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def instance_map(scores, masks, H, W):
    order = np.array(scores).argsort()[::-1]
    omasks = np.array(masks)[order]

    imap = np.zeros((H, W), dtype=np.int32) - 1
    current_id = 0

    for i in range(len(omasks)):
        mask = omasks[i] > 0.5
        new_pixels = mask & (imap == -1)
        imap[new_pixels] = order[current_id]
        current_id += 1

    return imap


def get_coco_results(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    masks, scores, cat_ids = [], [], []

    for ann in anns:
        masks.append(coco.annToMask(ann))
        scores.append(1.0)
        cat_ids.append(ann["category_id"] - 1)

    return {"masks": masks, "scores": scores, "class_ids": cat_ids}


def calc_pq(img_results, gt_results, class_agonistic=False):
    H = img_results["height"]
    W = img_results["width"]

    imap_pred = instance_map(img_results["scores"], img_results["masks"], H, W)
    imap_gt = instance_map(gt_results["scores"], gt_results["masks"], H, W)

    all_pred_ids = set(range(len(img_results["masks"])))
    all_gt_ids = set(range(len(gt_results["masks"])))

    unique_pred, counts_pred = np.unique(imap_pred, return_counts=True)
    unique_gt, counts_gt = np.unique(imap_gt, return_counts=True)

    unique_pred, counts_pred = unique_pred[unique_pred != -1], counts_pred[unique_pred != -1]
    unique_gt, counts_gt = unique_gt[unique_gt != -1], counts_gt[unique_gt != -1]

    pred_to_count = dict(zip(unique_pred, counts_pred))
    gt_to_count = dict(zip(unique_gt, counts_gt))

    imap_intr = (imap_pred.astype(np.uint32) << 16) | imap_gt.astype(np.uint32)
    unique_intr, counts_intr = np.unique(imap_intr, return_counts=True)

    matched_preds, matched_gts = set(), set()
    tp_ious = []

    for val, count in zip(unique_intr, counts_intr):
        p = val >> 16
        g = val & 0xFFFF

        if p >= 0xFFFF or g >= 0xFFFF:
            continue

        iou = count / (pred_to_count[p] + gt_to_count[g] - count)

        if iou > 0.5:
            if class_agonistic or img_results["class_ids"][p] == gt_results["class_ids"][g]:
                tp_ious.append(iou)
                matched_preds.add(p)
                matched_gts.add(g)

    num_tp = len(tp_ious)
    num_fp = len(all_pred_ids - matched_preds)
    num_fn = len(all_gt_ids - matched_gts)

    sq = sum(tp_ious) / num_tp if num_tp > 0 else 0.0
    fq = num_tp / (num_tp + 0.5*num_fp + 0.5*num_fn) if num_tp > 0 else 0.0
    pq = sq * fq

    return {"PQ": pq, "SQ": sq, "FQ": fq}


def calc_f1(img_results, gt_results, num_classes=None):
    H = img_results["height"]
    W = img_results["width"]

    pred_map = np.full((H, W), fill_value=-1, dtype=np.int32)
    pred_score = np.full((H, W), fill_value=-np.inf, dtype=np.float32)

    for mask, score, cls in zip(
        img_results["masks"],
        img_results["scores"],
        img_results["class_ids"],
    ):
        mask = mask.astype(bool)
        update = mask & (score > pred_score)
        pred_map[update] = cls
        pred_score[update] = score

    gt_map = np.full((H, W), fill_value=-1, dtype=np.int32)
    for mask, cls in zip(gt_results["masks"], gt_results["class_ids"]):
        mask = mask.astype(bool)
        gt_map[mask] = cls

    if num_classes is None:
        classes = set(img_results["class_ids"]) | set(gt_results["class_ids"])
    else:
        classes = range(num_classes)

    stats = {}
    for cls in classes:
        pred_c = pred_map == cls
        gt_c = gt_map == cls

        tp = np.logical_and(pred_c, gt_c).sum()
        fp = np.logical_and(pred_c, ~gt_c).sum()
        fn = np.logical_and(~pred_c, gt_c).sum()

        stats[cls] = {"tp": int(tp), "fp": int(fp), "fn": int(fn)}

    return stats


def aggregate_stats(all_stats):
    total_stats = {}
    for img_stats in all_stats:
        for cls, s in img_stats.items():
            if cls not in total_stats:
                total_stats[cls] = {"tp": 0, "fp": 0, "fn": 0}
            total_stats[cls]["tp"] += s["tp"]
            total_stats[cls]["fp"] += s["fp"]
            total_stats[cls]["fn"] += s["fn"]
    return total_stats


def compute_f1_from_stats(stats):
    f1_per_class = {}
    for cls, s in stats.items():
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        denom = 2 * tp + fp + fn
        f1_per_class[cls] = (2 * tp) / denom if denom > 0 else 0.0

    mean_f1 = np.mean(list(f1_per_class.values())) if f1_per_class else 0.0
    return f1_per_class, mean_f1


def to_float(value):
    return float(value) if value is not None else None


def parse_checkpoint_metadata(checkpoint_path):
    dirname = os.path.basename(os.path.dirname(checkpoint_path))
    match = re.match(r"train_output_lc_gsd_0p05_train_(\d+)_(coarse|medium|fine)_a", dirname)
    if match is None:
        return {"variant": "unknown", "train_percent": None, "run_name": dirname}
    return {
        "variant": match.group(2),
        "train_percent": int(match.group(1)),
        "run_name": dirname,
    }

def run_inference(model, coco, split_dir, batch_size=1, threshold=0.05):
    img_ids = coco.getImgIds()
    img_datas = coco.loadImgs(img_ids)

    results = []
    pq_results = []
    f1_stats = []
    last_img_size = None

    for i in tqdm(range(0, len(img_datas), batch_size)):
        batch = img_datas[i:i + batch_size]

        images = []
        meta = []

        for img_data in batch:
            img_path = os.path.join(split_dir, img_data["file_name"])
            image = Image.open(img_path).convert("RGB")

            images.append(image)
            meta.append((img_data["id"], image.size))

        # Safe inference (no batching assumption)
        detections_batch = []
        for img in images:
            out = model.predict(img, threshold=threshold)
            if isinstance(out, tuple):
                out = out[0]
            detections_batch.append(out)

        for detections, (img_id, img_size) in zip(detections_batch, meta):
            last_img_size = img_size

            # PIL gives (W, H)
            W, H = img_size

            img_results = {
                "masks": [],
                "scores": [],
                "class_ids": [],
                "height": H,
                "width": W
            }

            img_f1_results = {
                "masks": [],
                "scores": [],
                "class_ids": [],
                "height": H,
                "width": W,
            }

            for j in range(len(detections.xyxy)):
                score = float(detections.confidence[j])
                if score is None:
                    continue

                mask = detections.mask[j]

                # --- COCO results ---
                results.append({
                    "image_id": img_id,
                    "category_id": int(detections.class_id[j]) + 1,
                    "bbox": xyxy_to_xywh(detections.xyxy[j]),
                    "segmentation": mask_to_rle(mask),
                    "score": score,
                })

                # --- PQ inputs ---
                # use higher threshold for PQ to focus on better quality masks
                # PQ is sensitive to false positives, so we use more realistic threshold here
                if score >= 0.3:
                    img_results["masks"].append(mask)
                    img_results["scores"].append(score)
                    img_results["class_ids"].append(detections.class_id[j])

                if score >= 0.25:
                    img_f1_results["masks"].append(mask)
                    img_f1_results["scores"].append(score)
                    img_f1_results["class_ids"].append(detections.class_id[j])

            # --- compute PQ per image ---
            gt_results = get_coco_results(coco, img_id)
            pq_score = calc_pq(img_results, gt_results)
            pq_results.append(pq_score)

            f1_score_stats = calc_f1(img_f1_results, gt_results)
            f1_stats.append(f1_score_stats)

    return results, last_img_size, pq_results, f1_stats

def evaluate(coco, results, iou_type):
    coco_dt = coco.loadRes(results)
    evaluator = COCOeval(coco, coco_dt, iouType=iou_type)

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return evaluator.stats


def convert_to_single_class(coco, results):
    coco.dataset["categories"] = [{"id": 1, "name": "object"}]
    coco.createIndex()

    for ann in coco.dataset["annotations"]:
        ann["category_id"] = 1

    for res in results:
        res["category_id"] = 1

    return coco, results


def get_annotation_path(split_dir, variant=None):
    if variant in {"fine", "medium", "coarse"}:
        candidate = os.path.join(split_dir, f"_annotations.{variant}.coco.json")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Missing annotation file: {candidate}")

    default_candidate = os.path.join(split_dir, "_annotations.coco.json")
    if os.path.exists(default_candidate):
        return default_candidate
    raise FileNotFoundError(f"Missing annotation file: {default_candidate}")


def evaluate_model(dataset_dir, split, model_dir, model_name="segl", device="cuda", variant_override=None):
    if device == "cuda":
        torch.cuda.empty_cache()

    split_dir = os.path.join(dataset_dir, split)
    metadata = parse_checkpoint_metadata(model_dir)
    variant_for_annotations = variant_override if variant_override is not None else metadata.get("variant")
    annotation_path = get_annotation_path(split_dir, variant_for_annotations)

    print(f"Loading dataset from: {split_dir}")
    print(f"Using annotations: {os.path.basename(annotation_path)}")
    coco = COCO(annotation_path)

    print("Loading model...")
    model, trainable_params = load_model(model_dir, model_name, device=device)

    print("Running inference...")
    results, img_size, pq_results, f1_stats = run_inference(model, coco, split_dir)

    print("\n=== Segmentation Metrics ===")
    seg_stats = evaluate(coco, results, "segm")

    print("\n=== Bounding Box Metrics (Class-Agnostic) ===")
    coco_box = COCO(annotation_path)
    coco_box, box_results = convert_to_single_class(coco_box, list(results))
    box_stats = evaluate(coco_box, box_results, "bbox")

    print("\n=== Panoptic Quality Metrics ===")
    avg_pq = float(np.mean([r["PQ"] for r in pq_results])) if pq_results else 0.0
    avg_sq = float(np.mean([r["SQ"] for r in pq_results])) if pq_results else 0.0
    avg_fq = float(np.mean([r["FQ"] for r in pq_results])) if pq_results else 0.0
    print(f"PQ: {avg_pq}")
    print(f"SQ: {avg_sq}")
    print(f"RQ (FQ): {avg_fq}")

    print("\n=== Semantic F1 Metrics ===")
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    id_to_name = {cat["id"] - 1: cat["name"] for cat in cats}

    f1_per_class_idx, mean_f1 = compute_f1_from_stats(aggregate_stats(f1_stats))
    f1_per_class_named = {}
    for cls, cls_f1 in sorted(f1_per_class_idx.items()):
        cls_name = id_to_name.get(cls, f"class_{cls}")
        f1_per_class_named[cls_name] = float(cls_f1)
        print(f"{cls_name}: {cls_f1}")
    print(f"mean_f1: {mean_f1}")

    metrics = {
        "model_path": model_dir,
        "run_name": metadata["run_name"],
        "variant": variant_for_annotations,
        "train_percent": metadata["train_percent"],
        "trainable_params": int(trainable_params),
        "resolution": f"{img_size[0]}x{img_size[1]}" if img_size else None,
        "segm_ap": to_float(seg_stats[0]),
        "segm_ap50": to_float(seg_stats[1]),
        "segm_ap75": to_float(seg_stats[2]),
        "segm_ar100": to_float(seg_stats[8]),
        "bbox_ap": to_float(box_stats[0]),
        "bbox_ap50": to_float(box_stats[1]),
        "pq": avg_pq,
        "sq": avg_sq,
        "rq": avg_fq,
        "macro_f1": float(mean_f1),
        "f1_per_class": f1_per_class_named,
    }

    print("\n=== Final Summary ===")
    print("params, res, ca mAP, ca mAP50, mAP, AP50, AP75, mAR100, PQ, SQ, FQ, mean_f1")
    print(
        f"{trainable_params}, "
        f"{metrics['resolution']}, "
        f"{metrics['bbox_ap']}, {metrics['bbox_ap50']}, "
        f"{metrics['segm_ap']}, {metrics['segm_ap50']}, {metrics['segm_ap75']}, {metrics['segm_ar100']} "
        f"{metrics['pq']} {metrics['sq']} {metrics['rq']} {metrics['macro_f1']}"
    )

    return metrics


def save_metrics(outputs_dir, metrics):
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, f"{metrics['run_name']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def write_aggregate_files(outputs_dir, all_metrics):
    os.makedirs(outputs_dir, exist_ok=True)
    table_json_path = os.path.join(outputs_dir, "learning_curve_metrics.json")
    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)

    csv_fields = [
        "run_name",
        "variant",
        "train_percent",
        "model_path",
        "trainable_params",
        "resolution",
        "segm_ap",
        "segm_ap50",
        "segm_ap75",
        "segm_ar100",
        "bbox_ap",
        "bbox_ap50",
        "pq",
        "sq",
        "rq",
        "macro_f1",
    ]
    csv_path = os.path.join(outputs_dir, "learning_curve_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for m in all_metrics:
            row = {k: m.get(k) for k in csv_fields}
            writer.writerow(row)


def plot_learning_curves(outputs_dir, all_metrics):
    variants = ["fine", "medium", "coarse"]
    colors = {"fine": "#1b9e77", "medium": "#d95f02", "coarse": "#7570b3"}
    labels_et = {"fine": "peen", "medium": "keskmine", "coarse": "jäme"}
    f1_field = "macro_f1_no_saar_lehis" if any("macro_f1_no_saar_lehis" in m for m in all_metrics) else "macro_f1"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_f1, ax_ap50 = axes

    for variant in variants:
        rows = [m for m in all_metrics if m.get("variant") == variant and m.get("train_percent") is not None]
        rows = sorted(rows, key=lambda x: x["train_percent"])
        if not rows:
            continue

        x = np.array([r["train_percent"] for r in rows], dtype=float)
        y_f1 = np.array([r[f1_field] for r in rows], dtype=float)
        y_ap50 = np.array([r["segm_ap50"] for r in rows], dtype=float)

        X = np.vstack([np.ones_like(x), np.log(x)]).T
        beta_f1 = np.linalg.lstsq(X, y_f1, rcond=None)[0]
        beta_ap50 = np.linalg.lstsq(X, y_ap50, rcond=None)[0]
        x_fit = np.linspace(float(x.min()), float(x.max()), 200)
        y_fit_f1 = beta_f1[0] + beta_f1[1] * np.log(x_fit)
        y_fit_ap50 = beta_ap50[0] + beta_ap50[1] * np.log(x_fit)

        ax_f1.plot(x, y_f1, marker="o", color=colors[variant], label=labels_et[variant])
        ax_ap50.plot(x, y_ap50, marker="o", color=colors[variant], label=labels_et[variant])
        ax_f1.plot(x_fit, y_fit_f1, linestyle="--", color=colors[variant], label="_nolegend_")
        ax_ap50.plot(x_fit, y_fit_ap50, linestyle="--", color=colors[variant], label="_nolegend_")

    ax_f1.set_xlabel("Andmestiku suurus (%)")
    ax_f1.set_ylabel("F1 skoor")
    ax_f1.grid(True, alpha=0.3)

    ax_ap50.set_xlabel("Andmestiku suurus (%)")
    ax_ap50.set_ylabel("AP50")
    ax_ap50.grid(True, alpha=0.3)

    ax_f1.legend(loc="lower right", ncol=1)
    ax_ap50.legend(loc="lower right", ncol=1)

    plot_path = os.path.join(outputs_dir, "learning_curve.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    fig_f1, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for variant in variants:
        rows = [m for m in all_metrics if m.get("variant") == variant and m.get("train_percent") is not None]
        rows = sorted(rows, key=lambda x: x["train_percent"])
        if not rows:
            continue
        x = np.array([r["train_percent"] for r in rows], dtype=float)
        y = np.array([r[f1_field] for r in rows], dtype=float)
        X = np.vstack([np.ones_like(x), np.log(x)]).T
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        x_fit = np.linspace(float(x.min()), float(x.max()), 200)
        y_fit = beta[0] + beta[1] * np.log(x_fit)
        ax.plot(x, y, marker="o", color=colors[variant], label=labels_et[variant])
        ax.plot(x_fit, y_fit, linestyle="--", color=colors[variant], label="_nolegend_")
    ax.set_xlabel("Andmestiku suurus (%)")
    ax.set_ylabel("F1 skoor")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_f1.savefig(os.path.join(outputs_dir, "learning_curve_macro_f1.png"), dpi=180)
    plt.close(fig_f1)

    fig_ap50, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for variant in variants:
        rows = [m for m in all_metrics if m.get("variant") == variant and m.get("train_percent") is not None]
        rows = sorted(rows, key=lambda x: x["train_percent"])
        if not rows:
            continue
        x = np.array([r["train_percent"] for r in rows], dtype=float)
        y = np.array([r["segm_ap50"] for r in rows], dtype=float)
        X = np.vstack([np.ones_like(x), np.log(x)]).T
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        x_fit = np.linspace(float(x.min()), float(x.max()), 200)
        y_fit = beta[0] + beta[1] * np.log(x_fit)
        ax.plot(x, y, marker="o", color=colors[variant], label=labels_et[variant])
        ax.plot(x_fit, y_fit, linestyle="--", color=colors[variant], label="_nolegend_")
    ax.set_xlabel("Andmestiku suurus (%)")
    ax.set_ylabel("AP50")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_ap50.savefig(os.path.join(outputs_dir, "learning_curve_ap50.png"), dpi=180)
    plt.close(fig_ap50)


def load_metrics_table(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = {"variant", "train_percent", "macro_f1", "segm_ap50"}
    for idx, row in enumerate(data):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Row {idx} missing keys: {missing}")

    return data


def regenerate_plots_from_metrics(metrics_path, outputs_dir=None):
    if outputs_dir is None:
        outputs_dir = os.path.dirname(metrics_path)
    all_metrics = load_metrics_table(metrics_path)
    plot_learning_curves(outputs_dir, all_metrics)
    print(f"Regenerated plots in: {outputs_dir}")


def discover_checkpoints(base_dir):
    checkpoints = []
    for name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, name, "checkpoint_best_total.pth")
        if os.path.isfile(model_path):
            checkpoints.append(model_path)

    def sort_key(path):
        meta = parse_checkpoint_metadata(path)
        variant_order = {"coarse": 0, "medium": 1, "fine": 2}
        return (
            variant_order.get(meta["variant"], 99),
            meta["train_percent"] if meta["train_percent"] is not None else 10**9,
        )

    return sorted(checkpoints, key=sort_key)


def run_learning_curve(
    dataset_dir,
    split,
    model_name="segl",
    outputs_dir=None,
    base_dir=".",
    device="cuda",
    variant_override=None,
):
    checkpoints = discover_checkpoints(base_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {base_dir}")

    if outputs_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_dir = os.path.join(base_dir, "results", f"learning_curve_{stamp}")
    os.makedirs(outputs_dir, exist_ok=True)

    all_metrics = []
    for checkpoint_path in checkpoints:
        print("\n" + "=" * 80)
        print(f"Evaluating: {checkpoint_path}")
        metrics = evaluate_model(
            dataset_dir=dataset_dir,
            split=split,
            model_dir=checkpoint_path,
            model_name=model_name,
            device=device,
            variant_override=variant_override,
        )
        save_metrics(outputs_dir, metrics)
        all_metrics.append(metrics)

    write_aggregate_files(outputs_dir, all_metrics)
    plot_learning_curves(outputs_dir, all_metrics)
    print(f"\nSaved results to: {outputs_dir}")
    return outputs_dir


def main(dataset_dir, split="val", model_dir=None, model_name="segl", device="cuda", variant_override=None):
    return evaluate_model(
        dataset_dir=dataset_dir,
        split=split,
        model_dir=model_dir,
        model_name=model_name,
        device=device,
        variant_override=variant_override,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and plot learning curves")
    parser.add_argument("--mode", choices=["eval", "plot"], default="eval")
    parser.add_argument(
        "--dataset-dir",
        default="../data/compiled_datasets_3_fold/gsd_0p05/dataset",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-name", default="segl")
    parser.add_argument("--base-dir", default=os.path.dirname(__file__))
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU")
    parser.add_argument("--variant", choices=["fine", "medium", "coarse"], default=None, help="Override variant for annotation file selection")
    parser.add_argument(
        "--outputs-dir",
        default=os.path.join(os.path.dirname(__file__), "results", "learning_curve"),
    )
    parser.add_argument(
        "--metrics-path",
        default=os.path.join(os.path.dirname(__file__), "results", "learning_curve", "learning_curve_metrics.json"),
    )
    args = parser.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "eval":
        run_learning_curve(
            dataset_dir=args.dataset_dir,
            split=args.split,
            model_name=args.model_name,
            outputs_dir=args.outputs_dir,
            base_dir=args.base_dir,
            device=device,
            variant_override=args.variant,
        )
    else:
        regenerate_plots_from_metrics(args.metrics_path, args.outputs_dir)
