#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geodataset.tilerize import LabeledRasterTilerizer


@dataclass(frozen=True)
class ZoneConfig:
    name: str
    raster_path: Path
    labels_path: Path


@dataclass
class ZoneOutput:
    zone_name: str
    output_root: Path
    coco_json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile per-GSD train/valid instance-segmentation datasets."
    )
    parser.add_argument(
        "--gsd",
        type=float,
        nargs="+",
        required=True,
        help="List of ground sample distance values, e.g. --gsd 0.05 0.1 0.2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("compiled_datasets"),
        help="Root directory where compiled datasets are written.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("_tiler_work"),
        help="Intermediate directory used for per-zone tiler outputs.",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument("--min-intersection-ratio", type=float, default=0.05)
    parser.add_argument(
        "--ignore-tiles-without-labels",
        dest="ignore_tiles_without_labels",
        action="store_true",
        default=True,
        help="Ignore tiles without labels (default: true).",
    )
    parser.add_argument(
        "--keep-tiles-without-labels",
        dest="ignore_tiles_without_labels",
        action="store_false",
        help="Keep tiles without labels.",
    )
    parser.add_argument(
        "--clean-work-dir",
        action="store_true",
        help="Delete temporary per-zone outputs after each GSD build.",
    )
    return parser.parse_args()


def gsd_tag(gsd: float) -> str:
    return f"{gsd:.5f}".rstrip("0").rstrip(".").replace(".", "p")


def load_merge_map(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"merge_map\s*=\s*(\{.*\})", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse merge_map from {path}")
    parsed = ast.literal_eval(match.group(1))
    if not isinstance(parsed, dict):
        raise ValueError(f"merge_map in {path} is not a dict")
    result: dict[str, str] = {}
    for key, value in parsed.items():
        result[str(key)] = str(value)
    return result


def find_coco_json(zone_output_root: Path) -> Path:
    candidates = sorted(zone_output_root.rglob("*.json"))
    coco_candidates = [p for p in candidates if "coco" in p.name.lower()]
    if not coco_candidates:
        raise FileNotFoundError(f"No COCO JSON found under {zone_output_root}")

    preferred = [p for p in coco_candidates if p.name.endswith("_all.json")]
    if len(preferred) == 1:
        return preferred[0]
    if len(coco_candidates) == 1:
        return coco_candidates[0]

    names = "\n".join(str(p) for p in coco_candidates)
    raise RuntimeError(
        f"Ambiguous COCO JSON under {zone_output_root}. Found:\n{names}\n"
        "Please adjust find_coco_json selection logic."
    )


def build_basename_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name
        index.setdefault(name, []).append(file_path)
    return index


def resolve_image_path(
    file_name: str,
    json_parent: Path,
    zone_output_root: Path,
    basename_index: dict[str, list[Path]],
) -> Path:
    candidates = [
        (json_parent / file_name),
        (zone_output_root / file_name),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    fallback = basename_index.get(Path(file_name).name, [])
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) > 1:
        joined = "\n".join(str(p) for p in fallback)
        raise RuntimeError(
            f"Image path '{file_name}' is ambiguous under {zone_output_root}:\n{joined}"
        )
    raise FileNotFoundError(
        f"Could not resolve image path '{file_name}' under {zone_output_root}"
    )


def tile_zone(
    zone: ZoneConfig,
    gsd: float,
    output_root: Path,
    tile_size: int,
    tile_overlap: float,
    min_intersection_ratio: float,
    ignore_tiles_without_labels: bool,
) -> ZoneOutput:
    zone_output_root = output_root / zone.name
    zone_output_root.mkdir(parents=True, exist_ok=True)

    tilerizer = LabeledRasterTilerizer(
        raster_path=str(zone.raster_path),
        labels_path=str(zone.labels_path),
        output_path=str(zone_output_root),
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        labels_gdf=None,
        ground_resolution=gsd,
        use_rle_for_labels=False,
        min_intersection_ratio=min_intersection_ratio,
        ignore_tiles_without_labels=ignore_tiles_without_labels,
        ignore_black_white_alpha_tiles_threshold=0.8,
        main_label_category_column_name="Label",
        other_labels_attributes_column_names=None,
    )
    tilerizer.generate_coco_dataset()

    coco_json_path = find_coco_json(zone_output_root)
    return ZoneOutput(
        zone_name=zone.name,
        output_root=zone_output_root,
        coco_json_path=coco_json_path,
    )


def merge_split_zone_outputs(
    zone_outputs: list[ZoneOutput],
    split_dir: Path,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)

    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    seen_dest_names: set[str] = set()

    next_image_id = 1
    next_annotation_id = 1

    for zone_output in zone_outputs:
        data = json.loads(zone_output.coco_json_path.read_text(encoding="utf-8"))
        categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
        basename_index = build_basename_index(zone_output.output_root)

        image_id_map: dict[int, int] = {}
        for image in data.get("images", []):
            old_image_id = int(image["id"])
            src_path = resolve_image_path(
                file_name=str(image["file_name"]),
                json_parent=zone_output.coco_json_path.parent,
                zone_output_root=zone_output.output_root,
                basename_index=basename_index,
            )

            base_name = Path(str(image["file_name"])).name
            dest_name = f"{zone_output.zone_name}__{base_name}"
            if dest_name in seen_dest_names:
                stem = Path(base_name).stem
                suffix = Path(base_name).suffix
                counter = 2
                while True:
                    candidate = f"{zone_output.zone_name}__{stem}_{counter}{suffix}"
                    if candidate not in seen_dest_names:
                        dest_name = candidate
                        break
                    counter += 1
            seen_dest_names.add(dest_name)

            dest_path = split_dir / dest_name
            shutil.copy2(src_path, dest_path)

            new_image_id = next_image_id
            next_image_id += 1
            image_id_map[old_image_id] = new_image_id

            new_image = dict(image)
            new_image["id"] = new_image_id
            new_image["file_name"] = dest_name
            merged_images.append(new_image)

        for annotation in data.get("annotations", []):
            old_category_id = int(annotation["category_id"])
            if old_category_id not in categories:
                raise KeyError(
                    f"Category id {old_category_id} not found in {zone_output.coco_json_path}"
                )
            old_image_id = int(annotation["image_id"])
            if old_image_id not in image_id_map:
                raise KeyError(
                    f"Image id {old_image_id} not found in merged image map for {zone_output.coco_json_path}"
                )

            new_annotation = dict(annotation)
            new_annotation["id"] = next_annotation_id
            next_annotation_id += 1
            new_annotation["image_id"] = image_id_map[old_image_id]
            new_annotation["source_label"] = categories[old_category_id]
            merged_annotations.append(new_annotation)

    return {
        "images": merged_images,
        "annotations": merged_annotations,
    }


def build_category_index(label_map: dict[str, str]) -> dict[str, int]:
    labels = sorted(set(label_map.values()))
    return {label: idx for idx, label in enumerate(labels, start=1)}


def remap_split_for_fineness(
    split_data: dict[str, Any],
    label_map: dict[str, str],
    category_id_by_name: dict[str, int],
) -> dict[str, Any]:
    missing = sorted(
        {
            ann["source_label"]
            for ann in split_data["annotations"]
            if ann["source_label"] not in label_map
        }
    )
    if missing:
        raise ValueError(
            "Missing source labels in mapping: " + ", ".join(missing)
        )

    categories = [
        {"id": category_id_by_name[name], "name": name, "supercategory": ""}
        for name in sorted(category_id_by_name, key=lambda x: category_id_by_name[x])
    ]

    annotations = []
    for ann in split_data["annotations"]:
        mapped_name = label_map[ann["source_label"]]
        mapped_id = category_id_by_name[mapped_name]
        new_ann = {k: v for k, v in ann.items() if k != "source_label"}
        new_ann["category_id"] = mapped_id
        annotations.append(new_ann)

    return {
        "images": split_data["images"],
        "annotations": annotations,
        "categories": categories,
    }


def validate_split(coco_data: dict[str, Any], split_dir: Path) -> None:
    image_ids = {int(img["id"]) for img in coco_data["images"]}
    category_ids = {int(cat["id"]) for cat in coco_data["categories"]}
    for ann in coco_data["annotations"]:
        if int(ann["image_id"]) not in image_ids:
            raise ValueError(f"Annotation {ann['id']} references unknown image_id")
        if int(ann["category_id"]) not in category_ids:
            raise ValueError(f"Annotation {ann['id']} references unknown category_id")
    for img in coco_data["images"]:
        image_path = split_dir / str(img["file_name"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file referenced by COCO: {image_path}")


def write_coco(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def split_images_train_valid(
    images: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    if not images:
        raise ValueError("Cannot split empty image set")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    image_ids = [int(img["id"]) for img in images]
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    train_count = int(len(image_ids) * train_fraction)
    train_count = max(1, min(train_count, len(image_ids) - 1))

    train_ids = set(image_ids[:train_count])
    valid_ids = set(image_ids[train_count:])
    return train_ids, valid_ids


def subset_split_data(
    split_data: dict[str, Any],
    selected_image_ids: set[int],
) -> dict[str, Any]:
    images = [img for img in split_data["images"] if int(img["id"]) in selected_image_ids]
    annotations = [
        ann for ann in split_data["annotations"] if int(ann["image_id"]) in selected_image_ids
    ]
    return {
        "images": images,
        "annotations": annotations,
    }


def copy_split_images(split_data: dict[str, Any], src_dir: Path, dst_dir: Path) -> None:
    for img in split_data["images"]:
        file_name = str(img["file_name"])
        src = src_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing merged image for split copy: {src}")
        shutil.copy2(src, dst_dir / file_name)


def build_for_gsd(
    gsd: float,
    zones: list[ZoneConfig],
    mapping_files: dict[str, Path],
    output_root: Path,
    work_root: Path,
    tile_size: int,
    tile_overlap: float,
    min_intersection_ratio: float,
    ignore_tiles_without_labels: bool,
    clean_work_dir: bool,
) -> None:
    tag = gsd_tag(gsd)
    gsd_root = output_root / f"gsd_{tag}" / "dataset"
    train_dir = gsd_root / "train"
    valid_dir = gsd_root / "valid"
    test_dir = gsd_root / "test"
    combined_dir = gsd_root / "_combined_z1_z2"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if valid_dir.exists():
        shutil.rmtree(valid_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    gsd_work_root = work_root / f"gsd_{tag}"
    gsd_work_root.mkdir(parents=True, exist_ok=True)

    zone_outputs: dict[str, ZoneOutput] = {}
    for zone in zones:
        print(f"[{tag}] Tiling {zone.name} ...")
        zone_outputs[zone.name] = tile_zone(
            zone=zone,
            gsd=gsd,
            output_root=gsd_work_root,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            min_intersection_ratio=min_intersection_ratio,
            ignore_tiles_without_labels=ignore_tiles_without_labels,
        )

    print(f"[{tag}] Merging combined split (Z1+Z2) ...")
    combined_data = merge_split_zone_outputs(
        [zone_outputs["Z1"], zone_outputs["Z2"]],
        split_dir=combined_dir,
    )

    train_ids, valid_ids = split_images_train_valid(
        images=combined_data["images"],
        train_fraction=0.85,
        seed=42,
    )
    train_data = subset_split_data(combined_data, train_ids)
    valid_data = subset_split_data(combined_data, valid_ids)
    copy_split_images(train_data, combined_dir, train_dir)
    copy_split_images(valid_data, combined_dir, valid_dir)

    print(
        f"[{tag}] Train/valid split from Z1+Z2: "
        f"{len(train_data['images'])} train images, {len(valid_data['images'])} valid images"
    )

    print(f"[{tag}] Merging test split (Z3) ...")
    test_data = merge_split_zone_outputs(
        [zone_outputs["Z3"]],
        split_dir=test_dir,
    )

    mapping_dicts = {
        name: load_merge_map(path)
        for name, path in mapping_files.items()
    }
    category_indices = {
        name: build_category_index(mapping)
        for name, mapping in mapping_dicts.items()
    }

    for fineness in ("fine", "medium", "coarse", "original", "cloutier"):
        train_coco = remap_split_for_fineness(
            split_data=train_data,
            label_map=mapping_dicts[fineness],
            category_id_by_name=category_indices[fineness],
        )
        valid_coco = remap_split_for_fineness(
            split_data=valid_data,
            label_map=mapping_dicts[fineness],
            category_id_by_name=category_indices[fineness],
        )
        test_coco = remap_split_for_fineness(
            split_data=test_data,
            label_map=mapping_dicts[fineness],
            category_id_by_name=category_indices[fineness],
        )

        if train_coco["categories"] != valid_coco["categories"]:
            raise RuntimeError(
                f"Category tables differ between splits for {fineness}."
            )
        if train_coco["categories"] != test_coco["categories"]:
            raise RuntimeError(
                f"Category tables differ between splits for {fineness}."
            )

        validate_split(train_coco, train_dir)
        validate_split(valid_coco, valid_dir)
        validate_split(test_coco, test_dir)

        train_out = train_dir / f"_annotations.{fineness}.coco.json"
        valid_out = valid_dir / f"_annotations.{fineness}.coco.json"
        test_out = test_dir / f"_annotations.{fineness}.coco.json"
        write_coco(train_out, train_coco)
        write_coco(valid_out, valid_coco)
        write_coco(test_out, test_coco)

    shutil.copy2(
        train_dir / "_annotations.fine.coco.json",
        train_dir / "_annotations.coco.json",
    )
    shutil.copy2(
        valid_dir / "_annotations.fine.coco.json",
        valid_dir / "_annotations.coco.json",
    )
    shutil.copy2(
        test_dir / "_annotations.fine.coco.json",
        test_dir / "_annotations.coco.json",
    )

    if clean_work_dir:
        shutil.rmtree(gsd_work_root)

    shutil.rmtree(combined_dir)

    print(f"[{tag}] Done: {gsd_root}")


def main() -> None:
    args = parse_args()

    zones = [
        ZoneConfig("Z1", Path("rgb_z1.tif"), Path("Z1_polygons.gpkg")),
        ZoneConfig("Z2", Path("rgb_z2.tif"), Path("Z2_polygons.gpkg")),
        ZoneConfig("Z3", Path("rgb_z3.tif"), Path("Z3_polygons.gpkg")),
    ]
    mapping_files = {
        "fine": Path("classes_fine.txt"),
        "medium": Path("classes_medium.txt"),
        "coarse": Path("classes_coarse.txt"),
        "original": Path("classes_original.txt"),
        "cloutier": Path("classes_cloutier.txt"),
    }

    missing_paths = [
        p
        for p in [*(z.raster_path for z in zones), *(z.labels_path for z in zones), *mapping_files.values()]
        if not p.exists()
    ]
    if missing_paths:
        joined = "\n".join(str(p) for p in missing_paths)
        raise FileNotFoundError(f"Missing required input files:\n{joined}")

    for gsd in args.gsd:
        build_for_gsd(
            gsd=gsd,
            zones=zones,
            mapping_files=mapping_files,
            output_root=args.output_root,
            work_root=args.work_root,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            min_intersection_ratio=args.min_intersection_ratio,
            ignore_tiles_without_labels=args.ignore_tiles_without_labels,
            clean_work_dir=args.clean_work_dir,
        )


if __name__ == "__main__":
    main()
