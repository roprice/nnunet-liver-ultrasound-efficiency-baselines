#!/usr/bin/env python3
"""Extract caliper-derived proxy reference masks from SMC-LUD twin pairs.

For hemangioma images with a pixel-identical clean/caliper twin, subtracts the
pair to isolate caliper overlay marks, then derives the lesion's approximate
position and diameter from the caliper endpoints.  Writes a filled-circle proxy
mask (label 2 on a zero background) for each usable pair.

For HCC caliper images (no reliable twins), detects colored caliper marks
directly via chroma thresholding.  HCC proxies are keyed to the caliper case
name because the clean and caliper frames are different views; use them with
the runner's --subset caliper flag.

Run after prepare_external_data.py and before external_validation_runner.py.
The runner consumes proxy masks via its --proxy-masks flag.

Usage:
    python3 -I experiments/06_external_validation/extract_caliper_proxy.py \
        --source "path/to/SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)" \
        --reference experiments/06_external_validation/reference/smc_lud_reference.json \
        --output-dir archives/06_external_validation/caliper_annotation_tests/proxy_masks
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import label as label_components
from skimage.morphology import dilation, square


THUMBNAIL_SIZE = 48
TWIN_CORRELATION_THRESHOLD = 0.95
DIFF_THRESHOLD = 40
DILATION_SIZE = 5
MIN_BLOB_PIXELS = 15
CORNER_MARGIN_FRACTION = 0.15
PROXY_LABEL = 2
CHROMA_THRESHOLD = 40
HCC_MIN_CALIPER_BLOB = 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def thumbnail_correlation(path_a, path_b):
    size = THUMBNAIL_SIZE
    with Image.open(path_a) as img:
        a = np.asarray(
            img.convert("L").resize((size, size), Image.BILINEAR),
            dtype=np.float64).ravel()
    with Image.open(path_b) as img:
        b = np.asarray(
            img.convert("L").resize((size, size), Image.BILINEAR),
            dtype=np.float64).ravel()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom == 0:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def extract_marks_from_diff(clean_path, caliper_path):
    """Subtract twin pair, threshold, cluster into mark blobs.

    Returns (marks, mark_in_clean).  mark_in_clean is True when the brighter
    difference is in the clean image (contaminated pair).
    """
    with Image.open(clean_path) as img:
        clean = np.asarray(img.convert("L"), dtype=np.int16)
    with Image.open(caliper_path) as img:
        caliper = np.asarray(img.convert("L"), dtype=np.int16)

    diff = caliper - clean
    bright_in_caliper = int((diff > DIFF_THRESHOLD).sum())
    bright_in_clean = int((diff < -DIFF_THRESHOLD).sum())
    mark_in_clean = bright_in_clean > bright_in_caliper

    binary = (np.abs(diff) > DIFF_THRESHOLD).astype(np.uint8)
    binary = dilation(binary, square(DILATION_SIZE))

    labeled = label_components(binary, connectivity=2)
    marks = []
    for comp in range(1, labeled.max() + 1):
        mask = labeled == comp
        area = int(mask.sum())
        if area < MIN_BLOB_PIXELS:
            continue
        ys, xs = np.nonzero(mask)
        marks.append({
            "centroid": (float(ys.mean()), float(xs.mean())),
            "area": area,
            "bbox": (int(ys.min()), int(xs.min()),
                     int(ys.max()), int(xs.max())),
        })

    return marks, mark_in_clean


def extract_marks_from_chroma(image_path):
    """Detect colored caliper marks in a single image via chroma thresholding."""
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
        height, width = rgb.shape[:2]

    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    colored = (chroma > CHROMA_THRESHOLD).astype(np.uint8)
    colored = dilation(colored, square(3))

    labeled = label_components(colored, connectivity=2)
    marks = []
    for comp in range(1, labeled.max() + 1):
        mask = labeled == comp
        area = int(mask.sum())
        if area < HCC_MIN_CALIPER_BLOB:
            continue
        ys, xs = np.nonzero(mask)
        marks.append({
            "centroid": (float(ys.mean()), float(xs.mean())),
            "area": area,
            "bbox": (int(ys.min()), int(xs.min()),
                     int(ys.max()), int(xs.max())),
        })

    return marks, height, width


def filter_corner_blobs(marks, height, width):
    """Remove blobs whose centroid falls in a corner region."""
    margin_y = height * CORNER_MARGIN_FRACTION
    margin_x = width * CORNER_MARGIN_FRACTION
    filtered = []
    for mark in marks:
        cy, cx = mark["centroid"]
        at_y_edge = cy < margin_y or cy > height - margin_y
        at_x_edge = cx < margin_x or cx > width - margin_x
        if at_y_edge and at_x_edge:
            continue
        filtered.append(mark)
    return filtered


def caliper_segment(marks):
    """From exactly two marks, return (center_y, center_x, diameter_px)."""
    if len(marks) != 2:
        return None
    cy0, cx0 = marks[0]["centroid"]
    cy1, cx1 = marks[1]["centroid"]
    center_y = (cy0 + cy1) / 2
    center_x = (cx0 + cx1) / 2
    diameter = np.sqrt((cy1 - cy0) ** 2 + (cx1 - cx0) ** 2)
    return center_y, center_x, diameter


def render_proxy_mask(center_y, center_x, diameter, height, width):
    """Filled circle at PROXY_LABEL on a zero background."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    r = diameter / 2
    draw.ellipse(
        (center_x - r, center_y - r, center_x + r, center_y + r),
        fill=PROXY_LABEL)
    return np.array(mask, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Hemangioma twin-pair extraction
# ---------------------------------------------------------------------------

def find_hemangioma_twins(source, reference):
    """Match hemangioma clean/caliper images by patient + thumbnail correlation.

    Returns list of (clean_image_info, caliper_image_info, correlation).
    Greedy 1:1 matching (highest correlation first) prevents duplicate use.
    """
    hem_images = [img for img in reference["images"]
                  if img["class"] == "Hemangioma"]
    by_patient = defaultdict(lambda: {"clean": [], "caliper": []})
    for img in hem_images:
        by_patient[img["patient"]][img["subset"]].append(img)

    pairs = []
    for patient in sorted(by_patient):
        clean_list = by_patient[patient]["clean"]
        caliper_list = by_patient[patient]["caliper"]
        if not clean_list or not caliper_list:
            continue

        scored = []
        for ci, clean_img in enumerate(clean_list):
            clean_path = source / clean_img["source_relative_path"]
            if not clean_path.is_file():
                continue
            for ki, cal_img in enumerate(caliper_list):
                cal_path = source / cal_img["source_relative_path"]
                if not cal_path.is_file():
                    continue
                corr = thumbnail_correlation(clean_path, cal_path)
                if corr >= TWIN_CORRELATION_THRESHOLD:
                    scored.append((corr, ci, ki))

        scored.sort(reverse=True)
        used_clean = set()
        used_caliper = set()
        for corr, ci, ki in scored:
            if ci in used_clean or ki in used_caliper:
                continue
            used_clean.add(ci)
            used_caliper.add(ki)
            pairs.append((clean_list[ci], caliper_list[ki], corr))

    return pairs


def process_hemangioma(source, reference, output):
    """Extract proxy masks from hemangioma twin pairs.

    Proxy masks are keyed to the clean case name (the frames are identical).
    """
    pairs = find_hemangioma_twins(source, reference)
    print(f"Found {len(pairs)} hemangioma twin pairs")

    results = []
    contaminated = []
    skipped_counts = defaultdict(int)

    for clean_img, cal_img, corr in pairs:
        clean_path = source / clean_img["source_relative_path"]
        cal_path = source / cal_img["source_relative_path"]

        marks, mark_in_clean = extract_marks_from_diff(
            clean_path, cal_path)

        if mark_in_clean:
            contaminated.append(clean_img["case"])
            continue

        filtered = filter_corner_blobs(
            marks, clean_img["height"], clean_img["width"])
        seg = caliper_segment(filtered)

        if seg is None:
            skipped_counts[len(filtered)] += 1
            continue

        center_y, center_x, diameter = seg
        mask = render_proxy_mask(
            center_y, center_x, diameter,
            clean_img["height"], clean_img["width"])
        out_path = output / f"{clean_img['case']}.png"
        Image.fromarray(mask).save(out_path)

        results.append({
            "case": clean_img["case"],
            "patient": clean_img["patient"],
            "class": "Hemangioma",
            "method": "twin_diff",
            "n_raw_marks": len(marks),
            "n_filtered_marks": len(filtered),
            "center_y": round(center_y, 1),
            "center_x": round(center_x, 1),
            "diameter_px": round(diameter, 1),
            "correlation": round(corr, 4),
        })

    return results, contaminated, dict(skipped_counts)


# ---------------------------------------------------------------------------
# HCC chroma-based extraction
# ---------------------------------------------------------------------------

def process_hcc(source, reference, output):
    """Extract proxy masks from HCC caliper images via chroma detection.

    Proxies are keyed to the caliper case name because HCC clean and caliper
    images are different frames.  Use with the runner's --subset caliper flag.
    """
    hcc_caliper = [img for img in reference["images"]
                   if img["class"] == "HCC" and img["subset"] == "caliper"]

    results = []
    skipped_counts = defaultdict(int)

    for cal_img in hcc_caliper:
        cal_path = source / cal_img["source_relative_path"]
        if not cal_path.is_file():
            continue
        marks, height, width = extract_marks_from_chroma(cal_path)
        filtered = filter_corner_blobs(marks, height, width)
        seg = caliper_segment(filtered)

        if seg is None:
            skipped_counts[len(filtered)] += 1
            continue

        center_y, center_x, diameter = seg
        mask = render_proxy_mask(
            center_y, center_x, diameter, height, width)
        out_path = output / f"{cal_img['case']}.png"
        Image.fromarray(mask).save(out_path)

        results.append({
            "case": cal_img["case"],
            "patient": cal_img["patient"],
            "class": "HCC",
            "method": "chroma",
            "n_raw_marks": len(marks),
            "n_filtered_marks": len(filtered),
            "center_y": round(center_y, 1),
            "center_x": round(center_x, 1),
            "diameter_px": round(diameter, 1),
        })

    return results, dict(skipped_counts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", type=Path,
        default=(Path(__file__).resolve().parent
                 / "SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)"),
        help="Extracted SMC-LUD dataset root")
    parser.add_argument(
        "--reference", type=Path,
        default=(Path(__file__).resolve().parent
                 / "reference" / "smc_lud_reference.json"),
        help="Committed reference manifest")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for proxy mask PNGs and extraction report")
    parser.add_argument(
        "--skip-hcc", action="store_true",
        help="Only process hemangioma twins (skip HCC chroma detection)")
    args = parser.parse_args(argv)

    require(args.reference.is_file(),
            f"Missing reference: {args.reference}")
    reference = json.loads(args.reference.read_text())
    require(reference.get("schema_version") == 1,
            "Unsupported reference schema")
    require(args.source.is_dir(),
            f"Missing source dataset: {args.source}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    hem_results, contaminated, hem_skipped = process_hemangioma(
        args.source, reference, output)
    print(f"Hemangioma: {len(hem_results)} proxy masks, "
          f"{len(contaminated)} contaminated, "
          f"skipped by mark count: {hem_skipped}")

    hcc_results, hcc_skipped = [], {}
    if not args.skip_hcc:
        hcc_results, hcc_skipped = process_hcc(
            args.source, reference, output)
        print(f"HCC: {len(hcc_results)} proxy masks, "
              f"skipped by mark count: {hcc_skipped}")

    report = {
        "hemangioma": {
            "method": "twin_diff",
            "twin_pairs_found": (len(hem_results) + len(contaminated)
                                 + sum(hem_skipped.values())),
            "masks_written": len(hem_results),
            "contaminated_excluded": len(contaminated),
            "contaminated_cases": contaminated,
            "skipped_by_filtered_mark_count": hem_skipped,
        },
        "hcc": {
            "method": "chroma",
            "masks_written": len(hcc_results),
            "skipped_by_filtered_mark_count": hcc_skipped,
        },
        "proxy_cases": hem_results + hcc_results,
        "total_masks": len(hem_results) + len(hcc_results),
    }
    report_path = output / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"Report: {report_path}")
    print(f"Total: {report['total_masks']} proxy masks in {output}")


if __name__ == "__main__":
    main()
