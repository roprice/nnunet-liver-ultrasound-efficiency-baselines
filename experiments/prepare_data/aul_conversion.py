"""Convert AUL source images and polygon annotations to nnU-Net
 format.

Images are passed through without modification. Segmentation labels are
rendered from expert-annotated polygons as pixel masks with three
classes: background (0), liver (1), mass (2).

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
        Benign/image/*.jpg, Benign/segmentation/{liver,mass}/*.json
        Malignant/image/*.jpg, Malignant/segmentation/{liver,mass}/*.json
        Normal/image/*.jpg, Normal/segmentation/{liver}/*.json

Outputs a stratified 588/147 train/test split (seed 42) with nested training
subsets and five-fold assignments into the nnU-Net raw dataset directory.
"""

import os
import json

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from aul_splits import build_mapping


CATEGORIES = ["Benign", "Malignant", "Normal"]


def load_polygon(json_path):
    with open(json_path) as f:
        points = json.load(f)
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
        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = Path(img_file).stem
            cases.append({
                "category": category,
                "original_file": img_file,
                "image": img_dir / img_file,
                "liver_json": liver_dir / f"{stem}.json",
                "mass_json": mass_dir / f"{stem}.json",
            })
    return cases


def main():
    parser = argparse.ArgumentParser(description="Convert AUL to nnU-Net format")
    parser.add_argument("--raw-data-dir", type=Path, required=True,
                        help="Path to AUL raw data (contains Benign/, Malignant/, Normal/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="nnU-Net raw dataset output path")
    parser.add_argument("--reference-mapping", type=Path,
                        help="Require exactly the same assignments as an existing case_mapping.json")
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
    if args.reference_mapping and json.loads(args.reference_mapping.read_text()) != mapping:
        parser.error("AUL split differs from the reference mapping")
    print(f"Total cases found: {len(cases)}")

    for subdir in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

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

    train_cases = [entry for entry in mapping if entry["split"] == "train"]
    test_cases = [entry for entry in mapping if entry["split"] == "test"]
    write_cases(train_cases, "imagesTr", "labelsTr")
    write_cases(test_cases, "imagesTs", "labelsTs")

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

    print(f"\nSplit: {len(train_cases)} training, {len(test_cases)} test")
    print(f"Case mapping saved to {args.output_dir / 'case_mapping.json'}")
    print(f"Dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
