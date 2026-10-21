#!/usr/bin/env python3
"""Build the committed SMC-LUD reference manifest from the local extracted dataset.

This is a maintenance tool: it walks the extracted original_data tree, validates the
published structure and counts, joins the patient tables, recovers the dataset's own
split assignment per image by content hash, and writes reference/smc_lud_reference.json.
The converter (prepare_external_data.py) later checks any copy of the dataset against
that committed reference instead of trusting the folder layout.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SOURCE = HERE / "SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)"
DEFAULT_REFERENCE = HERE / "reference" / "smc_lud_reference.json"
IMG_EXTS = {".png", ".jpg", ".jpeg"}
CLASSES = {
    "HCC": {"clean_dir": "HCC_clean (wo caliper)", "caliper_dir": "HCC_caliper",
            "table": "HCC_patient_table.csv", "pathology": "malignant"},
    "Hemangioma": {"clean_dir": "Hem_clean (wo caliper)", "caliper_dir": "Hem_caliper",
                   "table": "Hem_patient_table.csv", "pathology": "benign"},
}
# Verified against the SMC-LUD data descriptor (Tak et al., Scientific Data 13:649, 2026)
# and this extracted copy on 2026-10-15.
EXPECTED_COUNTS = {
    ("HCC", "clean"): {"patients": 421, "images": 1389},
    ("HCC", "caliper"): {"patients": 557, "images": 1327},
    ("Hemangioma", "clean"): {"patients": 323, "images": 1267},
    ("Hemangioma", "caliper"): {"patients": 374, "images": 1402},
}
ZIP_MD5 = "03b28af94e37c4ae9b27024654aaf91e"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hcc_table(path):
    """Return {subject_id: max_diameter_cm or None} for HCC."""
    diameters = {}
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        require(header[1].strip() == "Subject_id", f"Unexpected HCC table header: {path}")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            subject = row[1].strip()
            require(subject not in diameters, f"Duplicate HCC Subject_id: {subject}")
            value = row[9].strip() if len(row) > 9 else ""
            try:
                diameters[subject] = float(value) if value else None
            except ValueError:
                diameters[subject] = None
    return diameters


def hemangioma_table(path):
    """Return the set of Subject_id values; the multi-line header varies."""
    with path.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if len(row) >= 2 and row[0].strip().isdigit()
                and row[1].strip().startswith("Hem_")]
    ids = [row[1].strip() for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate Hemangioma Subject_id values")
    return set(ids)


def content_index(source):
    """Map sha256 to the split assignment for both split trees."""
    assignments = {}
    for top, splits in (("split_for_train", ("train", "val", "test")),
                        ("clean_ver_for_train", ("train_clean", "val_clean", "test_clean"))):
        field = "split" if top == "split_for_train" else "clean_split"
        for split in splits:
            folder = source / top / split
            if not folder.exists():
                continue
            label = split.removesuffix("_clean")
            for class_name in CLASSES:
                class_folder = folder / class_name
                require(class_folder.is_dir(), f"Missing split folder: {class_folder}")
                for file in sorted(class_folder.iterdir()):
                    if not (file.is_file() and file.suffix.lower() in IMG_EXTS):
                        continue
                    digest = sha256(file)
                    require(assignments.get(digest, {}).get(field) in (None, label),
                            f"Conflicting {field} for content {digest}")
                    assignments.setdefault(digest, {})[field] = label
    return assignments


def build(source):
    require(source.is_dir(), f"Missing extracted SMC-LUD dataset: {source}")
    original = source / "original_data"
    split_assignments = content_index(source)
    hcc_diameters = hcc_table(original / "HCC" / CLASSES["HCC"]["table"])
    hem_ids = hemangioma_table(original / "Hemangioma" / CLASSES["Hemangioma"]["table"])

    images = []
    anomalies = {}
    for class_name, info in CLASSES.items():
        for subset, dir_name in (("clean", info["clean_dir"]), ("caliper", info["caliper_dir"])):
            subset_root = original / class_name / dir_name
            require(subset_root.is_dir(), f"Missing subset folder: {subset_root}")
            patients = sorted(p.name for p in subset_root.iterdir()
                              if p.is_dir() and p.name.startswith(("HCC_", "Hem_")))
            key = (class_name, subset)
            expected = EXPECTED_COUNTS[key]
            require(len(patients) == expected["patients"],
                    f"Patient-count mismatch for {key}: {len(patients)} != {expected['patients']}")
            count = 0
            for patient in patients:
                subject = patient.split("_", 1)[1] if class_name == "HCC" else patient
                if class_name == "HCC":
                    require(subject in hcc_diameters, f"HCC patient folder missing from table: {patient}")
                    diameter = hcc_diameters[subject]
                else:
                    diameter = None
                patient_dir = subset_root / patient
                files = sorted(f for f in patient_dir.iterdir()
                               if f.is_file() and f.suffix.lower() in IMG_EXTS)
                require(files, f"Empty patient folder: {patient_dir}")
                for file in files:
                    digest = sha256(file)
                    with Image.open(file) as image:
                        width, height = image.size
                        mode = image.mode
                        image.verify()
                    stem8 = file.name.split(".")[0][:8]
                    images.append({
                        "case": f"smc_{patient}_{stem8}",
                        "patient": patient,
                        "class": class_name,
                        "pathology": CLASSES[class_name]["pathology"],
                        "subset": subset,
                        "split": split_assignments.get(digest, {}).get("split"),
                        "clean_split": split_assignments.get(digest, {}).get("clean_split"),
                        "source_relative_path": str(file.relative_to(source)),
                        "sha256": digest,
                        "bytes": file.stat().st_size,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "ext": file.suffix.lower(),
                        "tumor_max_diameter_cm": diameter,
                    })
                    count += 1
            require(count == expected["images"],
                    f"Image-count mismatch for {key}: {count} != {expected['images']}")

    cases = [image["case"] for image in images]
    require(len(cases) == len(set(cases)), "Case-name collision; increase stem length")
    hcc_folder_subjects = {image["patient"].split("_", 1)[1]
                           for image in images if image["class"] == "HCC"}
    require(set(hcc_diameters) == hcc_folder_subjects,
            "HCC patient-table and folder subjects differ")
    hem_folder_ids = {image["patient"] for image in images if image["class"] == "Hemangioma"}
    anomalies["hemangioma_table_without_images"] = sorted(hem_ids - hem_folder_ids)
    anomalies["hemangioma_folders_without_table_row"] = sorted(hem_folder_ids - hem_ids)
    clean_hcc = [image for image in images if image["class"] == "HCC" and image["subset"] == "clean"]
    clean_with_size = [image for image in clean_hcc if image["tumor_max_diameter_cm"] is not None]
    anomalies["clean_hcc_images_without_tumor_size"] = len(clean_hcc) - len(clean_with_size)

    reference = {
        "schema_version": 1,
        "dataset": "SMC-LUD (figshare 31112716, v3)",
        "dataset_doi": "10.6084/m9.figshare.31112716",
        "source_zip_md5": ZIP_MD5,
        "built_from": str(source),
        "published_source": "Tak et al. SMC-LUD, Scientific Data 13:649 (2026)",
        "expected_counts": {f"{c}/{s}": v for (c, s), v in sorted(EXPECTED_COUNTS.items())},
        "anomalies_noted": anomalies,
        "images": sorted(images, key=lambda image: image["case"]),
    }
    return reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--force", action="store_true", help="Replace an existing reference")
    args = parser.parse_args(argv)
    require(args.source.resolve() == DEFAULT_SOURCE.resolve(),
            "The reference must be built from the canonical extracted dataset, not a copy")
    reference = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        require(args.output.read_text() == json.dumps(reference, indent=2) + "\n",
                f"Existing reference differs; inspect before replacing: {args.output}")
        print(f"Reference already up to date: {args.output}")
        return
    args.output.write_text(json.dumps(reference, indent=2) + "\n")
    total = len(reference["images"])
    print(f"Wrote reference with {total} images: {args.output}")


if __name__ == "__main__":
    main()
