#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from pathlib import Path


PERCENTAGES = [5, 10, 20, 40, 80, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nested COCO training subsets for learning-curve experiments."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to dataset root containing train/ valid/ test/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory (default: <dataset-root-parent>/datasets_learning_curve)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Shuffle seed")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_images(split_src_dir: Path, split_dst_dir: Path, images: list[dict]) -> None:
    split_dst_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        file_name = str(image["file_name"])
        src = split_src_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing source image for copy: {src}")
        dst = split_dst_dir / file_name
        shutil.copy2(src, dst)


def variant_name(path: Path) -> str | None:
    match = re.fullmatch(r"_annotations\.(.+)\.coco\.json", path.name)
    if not match:
        return None
    return match.group(1)


def collect_variant_files(split_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for file_path in sorted(split_dir.glob("_annotations.*.coco.json")):
        variant = variant_name(file_path)
        if variant is None:
            continue
        result[variant] = file_path
    if not result:
        raise FileNotFoundError(f"No variant annotations found in {split_dir}")
    return result


def check_references(coco: dict) -> None:
    image_ids = [int(img["id"]) for img in coco.get("images", [])]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate image IDs found")

    image_id_set = set(image_ids)
    category_ids = {int(cat["id"]) for cat in coco.get("categories", [])}
    for ann in coco.get("annotations", []):
        ann_img_id = int(ann["image_id"])
        ann_cat_id = int(ann["category_id"])
        if ann_img_id not in image_id_set:
            raise ValueError(f"Annotation {ann.get('id')} references missing image_id {ann_img_id}")
        if ann_cat_id not in category_ids:
            raise ValueError(f"Annotation {ann.get('id')} references missing category_id {ann_cat_id}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_root = args.output_root or (dataset_root.parent / "datasets_learning_curve")

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    test_dir = dataset_root / "test"
    for split_dir in (train_dir, valid_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")

    train_variants = collect_variant_files(train_dir)
    valid_variants = collect_variant_files(valid_dir)
    test_variants = collect_variant_files(test_dir)

    variant_keys = sorted(train_variants.keys())
    if sorted(valid_variants.keys()) != variant_keys or sorted(test_variants.keys()) != variant_keys:
        raise ValueError("Variant mismatch across train/valid/test")

    canonical_variant = variant_keys[0]
    canonical_train = read_json(train_variants[canonical_variant])
    image_ids = [int(img["id"]) for img in canonical_train.get("images", [])]
    if not image_ids:
        raise ValueError("Training set has zero images")

    rng = random.Random(args.seed)
    shuffled_ids = image_ids[:]
    rng.shuffle(shuffled_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"shuffle_order_seed_{args.seed}.txt").write_text(
        "\n".join(str(i) for i in shuffled_ids) + "\n",
        encoding="utf-8",
    )

    # Verify all train variants share same image id universe
    target_id_set = set(image_ids)
    for variant, path in train_variants.items():
        payload = read_json(path)
        ids = {int(img["id"]) for img in payload.get("images", [])}
        if ids != target_id_set:
            raise ValueError(f"Train image-id mismatch for variant '{variant}'")

    total_images = len(shuffled_ids)
    subset_id_lists: dict[int, list[int]] = {}
    for pct in PERCENTAGES:
        if pct == 100:
            k = total_images
        else:
            k = math.floor((pct / 100.0) * total_images)
            k = max(1, k)
        subset_id_lists[pct] = shuffled_ids[:k]

    # Build subsets for each variant
    for pct, subset_ids in subset_id_lists.items():
        subset_set = set(subset_ids)
        subset_dir = output_root / f"train_{pct}"
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        subset_dir.mkdir(parents=True, exist_ok=True)

        (output_root / f"train_{pct}_image_ids.txt").write_text(
            "\n".join(str(i) for i in subset_ids) + "\n", encoding="utf-8"
        )

        for variant, path in train_variants.items():
            payload = read_json(path)
            filtered_images = [img for img in payload.get("images", []) if int(img["id"]) in subset_set]
            filtered_annotations = [
                ann for ann in payload.get("annotations", []) if int(ann["image_id"]) in subset_set
            ]

            subset_payload = {
                "images": filtered_images,
                "annotations": filtered_annotations,
                "categories": payload.get("categories", []),
            }
            check_references(subset_payload)
            out_path = subset_dir / f"_annotations.{variant}.coco.json"
            write_json(out_path, subset_payload)

            # Copy relevant images once per subset (using first variant only)
            if variant == canonical_variant:
                copy_images(train_dir, subset_dir, filtered_images)

    # Copy valid/test annotations unchanged
    out_valid = output_root / "valid"
    out_test = output_root / "test"
    if out_valid.exists():
        shutil.rmtree(out_valid)
    if out_test.exists():
        shutil.rmtree(out_test)
    out_valid.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)
    for variant, path in valid_variants.items():
        shutil.copy2(path, out_valid / f"_annotations.{variant}.coco.json")
    for variant, path in test_variants.items():
        shutil.copy2(path, out_test / f"_annotations.{variant}.coco.json")

    # Copy valid/test images referenced by canonical variant annotations
    valid_payload = read_json(valid_variants[canonical_variant])
    test_payload = read_json(test_variants[canonical_variant])
    copy_images(valid_dir, out_valid, valid_payload.get("images", []))
    copy_images(test_dir, out_test, test_payload.get("images", []))

    print(f"Done: {output_root}")


if __name__ == "__main__":
    main()
