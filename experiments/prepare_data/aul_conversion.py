"""Convert AUL source images and polygon annotations to nnU-Net
 format.

Source images are RGB JPEGs. Each is converted to 8-bit grayscale (Pillow
convert("L")) and re-encoded as PNG at its original size; nothing is resized,
cropped or rescaled. Segmentation labels are rendered from expert-annotated
polygons as pixel masks with three classes: background (0), liver (1), mass (2).
Every polygon file must hold exactly one polygon, every Benign and Malignant
image must have a mass polygon, no Normal image may have one, and every image
must have a liver polygon except those listed in
KNOWN_MISSING_LIVER_POLYGONS. For the images in LIVER_POLYGON_IN_OUTLINE, AUL
files the liver polygon as the scan outline (segmentation/outline/), so that
file is used as the liver. A conversion_report.json of per-category label pixel
counts is written beside the dataset.

Dataset: Annotated Ultrasound Liver (AUL) images
DOI: 10.5281/zenodo.7272660 (https://doi.org/10.5281/zenodo.7272660)
Citation: Xu, Y., Zheng, B., Liu, X., Wu, T., Ju, J., Wang, S., Lian, Y.,
          Zhang, H., Liang, T., Sang, Y., Jiang, R., Wang, G., Ren, J., &
          Chen, T. (2022). Annotated Ultrasound Liver images [Data set].
          Zenodo. https://doi.org/10.5281/zenodo.7272660
This is a versioned Zenodo record (not the floating concept DOI
10.5281/zenodo.7272659), so it resolves to the same fixed archive files
regardless of any future dataset versions published under the same
concept DOI. File checksums as of this record:
  Benign.zip    md5:c37fef0cb2730236a79ef57e5315995e
  Malignant.zip md5:63894a9e5654a69c3b94bda84071dfb0
  Normal.zip    md5:a7e16299b2cf12ca4a6c3468d2e4978f

Expected input structure:
    AUL/
        Benign/image/*.jpg, Benign/segmentation/{liver,mass,outline}/*.json
        Malignant/image/*.jpg, Malignant/segmentation/{liver,mass,outline}/*.json
        Normal/image/*.jpg, Normal/segmentation/{liver,outline}/*.json

Outputs a 588/147 train/test split, with UltraBench's AUL test set as the 147
test cases (see aul_splits.py), nested training subsets and five-fold
assignments into the nnU-Net raw dataset directory.
"""

import os
import json

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from aul_splits import build_mapping


CATEGORIES = ["Benign", "Malignant", "Normal"]

# AUL record 7272660 has no liver polygon for this Malignant image, so its label
# holds the mass only (case_mapping.json places it in the training pool).
KNOWN_MISSING_LIVER_POLYGONS = {("Malignant", "374.jpg")}

# For these Malignant images, AUL has no file in segmentation/liver/ and the
# polygon in segmentation/outline/ traces the liver rather than the scan region
# (as also corrected in UltraBench). Both are in the test set.
LIVER_POLYGON_IN_OUTLINE = {("Malignant", "229.jpg"), ("Malignant", "306.jpg")}


def load_polygon(json_path):
    with open(json_path) as f:
        points = json.load(f)
    if (not isinstance(points, list) or len(points) < 3 or
            not all(isinstance(p, list) and len(p) == 2 and
                    all(isinstance(v, (int, float)) for v in p) for p in points)):
        raise ValueError(f"Expected one polygon [[x, y], ...] with at least 3 points: {json_path}")
    return [(p[0], p[1]) for p in points]


def render_mask(polygons_by_label, image_size):
    """Render segmentation mask from polygons.

    Polygons are drawn in label order (1=liver first, 2=mass second)
    so that mass pixels overwrite liver pixels where they overlap.
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for label_val in sorted(polygons_by_label.keys()):
        poly = polygons_by_label[label_val]
        if poly is not None:
            draw.polygon(poly, fill=label_val)
    return np.array(mask, dtype=np.uint8)


def gather_cases(raw_data_dir):
    cases = []
    for category in CATEGORIES:
        img_dir = raw_data_dir / category / "image"
        liver_dir = raw_data_dir / category / "segmentation" / "liver"
        mass_dir = raw_data_dir / category / "segmentation" / "mass"
        outline_dir = raw_data_dir / category / "segmentation" / "outline"
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = Path(img_file).stem
            liver_source = (outline_dir if (category, img_file) in LIVER_POLYGON_IN_OUTLINE
                            else liver_dir)
            cases.append({
                "category": category,
                "original_file": img_file,
                "image": img_dir / img_file,
                "liver_json": liver_source / f"{stem}.json",
                "liver_dir_json": liver_dir / f"{stem}.json",
                "mass_json": mass_dir / f"{stem}.json",
            })
    return cases


def check_annotations(cases, raw_data_dir):
    """Fail on any annotation layout that would silently give a wrong label."""
    problems = []
    for case in cases:
        key = (case["category"], case["original_file"])
        has_liver = case["liver_dir_json"].exists()
        if key in KNOWN_MISSING_LIVER_POLYGONS or key in LIVER_POLYGON_IN_OUTLINE:
            if has_liver:
                problems.append(f"{key[0]}/{key[1]}: liver polygon present but expected missing")
            if key in LIVER_POLYGON_IN_OUTLINE and not case["liver_json"].exists():
                problems.append(f"{key[0]}/{key[1]}: no outline polygon to use as liver")
        elif not has_liver:
            problems.append(f"{key[0]}/{key[1]}: no liver polygon")
        has_mass = case["mass_json"].exists()
        if has_mass != (case["category"] != "Normal"):
            problems.append(f"{key[0]}/{key[1]}: {'unexpected' if has_mass else 'missing'} mass polygon")
    for category in CATEGORIES:
        stems = {Path(case["original_file"]).stem for case in cases if case["category"] == category}
        for label in ("liver", "mass"):
            folder = raw_data_dir / category / "segmentation" / label
            if folder.is_dir():
                problems += [f"{folder / entry.name}: not a polygon file for any image"
                             for entry in sorted(folder.iterdir())
                             if entry.suffix != ".json" or entry.stem not in stems]
    if problems:
        raise ValueError("Unexpected AUL annotations:\n  " + "\n  ".join(problems))


def main():
    parser = argparse.ArgumentParser(description="Convert AUL to nnU-Net format")
    parser.add_argument("--raw-data-dir", type=Path, required=True,
                        help="Path to AUL raw data (contains Benign/, Malignant/, Normal/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="nnU-Net raw dataset output path")
    parser.add_argument("--reference-mapping", type=Path, required=True,
                        help="Committed case_mapping.json the assignments must match exactly "
                             "(experiments/prepare_data/reference/case_mapping.json)")
    args = parser.parse_args()

    cases = gather_cases(args.raw_data_dir)
    by_source = {(case["category"], case["original_file"]): case for case in cases}
    if len(by_source) != len(cases):
        parser.error("Duplicate AUL source image")
    files_by_category = {category: [case["original_file"] for case in cases
                                    if case["category"] == category] for category in CATEGORIES}
    try:
        mapping = build_mapping(files_by_category)
    except ValueError as error:
        parser.error(str(error))
    if json.loads(args.reference_mapping.read_text()) != mapping:
        parser.error("AUL split differs from the reference mapping")
    try:
        check_annotations(cases, args.raw_data_dir)
    except ValueError as error:
        parser.error(str(error))
    print(f"Total cases found: {len(cases)}")

    for subdir in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    report = {category: {"cases": 0, "liver_pixels": 0, "mass_pixels": 0,
                         "cases_without_liver_pixels": 0, "cases_without_mass_pixels": 0}
              for category in CATEGORIES}

    def write_cases(entries, img_subdir, lbl_subdir):
        for entry in entries:
            case = by_source[(entry["category"], entry["original_file"])]
            case_name = entry["case_name"]

            img = Image.open(case["image"])
            img_gray = np.array(img.convert("L"), dtype=np.uint8)

            Image.fromarray(img_gray).save(
                args.output_dir / img_subdir / f"{case_name}_0000.png")

            polygons = {}
            if case["liver_json"].exists():
                polygons[1] = load_polygon(case["liver_json"])
            if case["mass_json"].exists():
                polygons[2] = load_polygon(case["mass_json"])

            label = render_mask(polygons, img.size)
            Image.fromarray(label).save(
                args.output_dir / lbl_subdir / f"{case_name}.png")
            stats = report[entry["category"]]
            liver_pixels, mass_pixels = int((label == 1).sum()), int((label == 2).sum())
            stats["cases"] += 1
            stats["liver_pixels"] += liver_pixels
            stats["mass_pixels"] += mass_pixels
            stats["cases_without_liver_pixels"] += liver_pixels == 0
            stats["cases_without_mass_pixels"] += mass_pixels == 0

    train_cases = [entry for entry in mapping if entry["split"] == "train"]
    test_cases = [entry for entry in mapping if entry["split"] == "test"]
    try:
        write_cases(train_cases, "imagesTr", "labelsTr")
        write_cases(test_cases, "imagesTs", "labelsTs")
    except ValueError as error:
        parser.error(str(error))

    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": {"background": 0, "liver": 1, "mass": 2},
        "numTraining": len(train_cases),
        "file_ending": ".png",
    }
    with open(args.output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    with open(args.output_dir / "case_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    with open(args.output_dir / "conversion_report.json", "w") as f:
        json.dump({"known_missing_liver_polygons": sorted(f"{c}/{n}" for c, n in KNOWN_MISSING_LIVER_POLYGONS),
                   "liver_polygons_from_outline": sorted(f"{c}/{n}" for c, n in LIVER_POLYGON_IN_OUTLINE),
                   "categories": report}, f, indent=2)
    for category, stats in report.items():
        print(f"{category}: {stats['cases']} cases, {stats['liver_pixels']} liver pixels, "
              f"{stats['mass_pixels']} mass pixels, {stats['cases_without_liver_pixels']} without liver, "
              f"{stats['cases_without_mass_pixels']} without mass")

    print(f"\nSplit: {len(train_cases)} training, {len(test_cases)} test")
    print(f"Case mapping saved to {args.output_dir / 'case_mapping.json'}")
    print(f"Dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
