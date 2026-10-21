#!/usr/bin/env python3
"""Convert the extracted SMC-LUD dataset into the experiment-06 inference cohort.

Re-verifies the source tree against the committed reference (every image's SHA256 and
case identity), then writes 8-bit grayscale ``<case>_0000.png`` inputs for nnUNetv2_predict
plus a manifest and conversion_report.json. The primary cohort is the Clean subset;
``--include-caliper`` additionally produces the caliper inputs for the optional
secondary run. No segmentation references exist for this dataset; classification labels
and patient identity come from the dataset's folder structure and patient tables.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SOURCE = HERE / "SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)"
DEFAULT_REFERENCE = HERE / "reference" / "smc_lud_reference.json"
DEFAULT_OUTPUT = REPO / "archives_staging/06_external_validation/smc_lud_prepared"
LUMA_NOTE = ("RGB inputs converted to 8-bit grayscale with PIL Image.convert('L') "
             "(ITU-R 601-2 luma), then saved as PNG. No resizing or intensity rescaling.")
MANIFEST_FIELDS = ("case", "patient", "class", "pathology", "subset", "split",
                   "clean_split", "source_relative_path", "sha256", "width", "height",
                   "tumor_max_diameter_cm")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path):
    require(path.is_file(), f"Missing reference manifest: {path}; run build_smc_lud_reference.py first")
    reference = json.loads(path.read_text())
    require(reference.get("schema_version") == 1, "Unsupported reference schema")
    cases = [image["case"] for image in reference["images"]]
    require(len(cases) == len(set(cases)), "Reference contains duplicate case ids")
    ontology = {(image["pathology"], image["subset"]) for image in reference["images"]}
    require(ontology == {("malignant", "clean"), ("malignant", "caliper"),
                         ("benign", "clean"), ("benign", "caliper")},
            "Reference pathology/subset ontology mismatch")
    return reference


def verify_source(source, reference):
    """Fail loudly if the extracted tree does not match the committed reference."""
    mismatches = {}
    for image in reference["images"]:
        path = source / image["source_relative_path"]
        if not path.is_file():
            mismatches.setdefault("missing", []).append(image["case"])
            continue
        if sha256(path) != image["sha256"]:
            mismatches.setdefault("hash", []).append(image["case"])
    require(not mismatches,
            f"Source tree differs from committed reference; run build_smc_lud_reference.py to investigate: "
            f"{json.dumps({k: (len(v), v[:5]) for k, v in mismatches.items()})}")
    extra = {str(path.relative_to(source))
             for path in (source / "original_data").rglob("*")
             if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")} - \
            {image["source_relative_path"] for image in reference["images"]}
    require(not extra, f"Unexpected image files in source tree: {sorted(extra)[:5]}")
    totals = Counter((image["class"], image["subset"]) for image in reference["images"])
    for key, expected in reference["expected_counts"].items():
        class_name, subset = key.split("/")
        require(totals[class_name, subset] == expected["images"],
                f"Reference count mismatch for {key}")


def prepare(source, reference, output, subsets):
    check_destinations(source, output)
    selected = [image for image in reference["images"] if image["subset"] in subsets]
    manifest_cases = []
    for subset in subsets:
        if sum(1 for image in selected if image["subset"] == subset) == 0:
            continue
        folder = output / ("images" if subset == "clean" else f"images_{subset}")
        folder.mkdir(parents=True, exist_ok=True)
        for image in [image for image in selected if image["subset"] == subset]:
            source_file = source / image["source_relative_path"]
            out = folder / f"{image['case']}_0000.png"
            require(not out.exists(), f"Refusing to overwrite existing case: {out}")
            with Image.open(source_file) as original:
                grayscale = original.convert("L")
                grayscale.save(out, format="PNG")
            with Image.open(out) as check:
                require(check.mode == "L" and check.size == (image["width"], image["height"]),
                        f"Grayscale conversion check failed: {out}")
            manifest_cases.append({field: image[field] for field in MANIFEST_FIELDS})
    return manifest_cases


def check_destinations(source, output):
    require(not output.resolve().is_relative_to(source.resolve()),
            "Output must not be inside the SMC-LUD source tree")


def write_reports(output, reference, manifest_cases, subsets):
    pathology_counts = Counter(case["pathology"] for case in manifest_cases)
    patients = {case["patient"] for case in manifest_cases}
    manifest = {
        "schema_version": 1,
        "prepared_from": "reference/smc_lud_reference.json",
        "subsets": sorted(subsets),
        "conversion": LUMA_NOTE,
        "case_count": len(manifest_cases),
        "patient_count": len(patients),
        "cases": manifest_cases,
    }
    report = {
        "status": "converted",
        "case_count": len(manifest_cases),
        "patient_count": len(patients),
        "class_counts": dict(sorted(Counter(case["class"] for case in manifest_cases).items())),
        "pathology_counts": dict(sorted(pathology_counts.items())),
        "hcc_tumor_size_known": sum(1 for c in manifest_cases
                                   if c["class"] == "HCC" and c["tumor_max_diameter_cm"] is not None),
        "reference_anomalies_noted": reference["anomalies_noted"],
        "expected_counts": reference["expected_counts"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smc_lud_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-caliper", action="store_true",
                        help="Also convert the caliper subset (secondary analysis cohort)")
    args = parser.parse_args(argv)
    try:
        reference = load_reference(args.reference)
        verify_source(args.source, reference)
        require(not (args.output_dir / "smc_lud_manifest.json").exists(),
                f"Manifest already exists; delete or choose a new --output-dir: {args.output_dir}")
        subsets = ("clean", "caliper") if args.include_caliper else ("clean",)
        manifest_cases = prepare(args.source, reference, args.output_dir, subsets)
        report = write_reports(args.output_dir, reference, manifest_cases, subsets)
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    print(f"Converted {report['case_count']} cases from {report['patient_count']} patients: "
          f"{report['pathology_counts']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
