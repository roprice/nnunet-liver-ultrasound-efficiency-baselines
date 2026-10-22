"""Do caliper marks change predictions? Compare each hemangioma Clean frame with its marked twin.

The twin frames are identical to the Clean frames except for the caliper marks, so any
difference between the two predictions is caused by the marks. Prepare and predict the twins with
experiments/06_external_validation/prepare_caliper_cohort.py and the runner's --subset caliper
--experiments data_efficiency, then run from the repository root:
    python -m analysis.06_external_validation.twin_mark_effect.twin_mark_effect
"""

import argparse
import importlib
import json
import statistics
from pathlib import Path

from analysis.predictions_dataset import build_predictions_dataset as builder
from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

external = importlib.import_module("analysis.06_external_validation.detection_rates.detection_rates")
proxy_detection = importlib.import_module("analysis.06_external_validation.proxy_detection")

PROJECT_ROOT = external.PROJECT_ROOT
FOLDS = external.FOLDS
PRIMARY_MODEL = "data_efficiency"
OUTPUT_PATH = Path(__file__).with_suffix(".md")
DEFAULT_PAIRS = PROJECT_ROOT / "archives/06_external_validation/caliper/prepared/twin_pairs.json"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "archives/06_external_validation/caliper/logs/predictions"
RATE_METRICS = ("presence", "overlap_iou_00", "centroid_deq_100")


def read_retained(path, shape):
    prediction = builder.read_mask(path)
    if prediction.shape != shape:
        raise RuntimeError(f"{path.name}: prediction dimensions differ from the proxy")
    return builder.retained_mass_mask(prediction == builder.MASS_VALUE)


def detected(result, metric):
    return result.presence if metric == "presence" else result.detected[metric]


def compare_pair(clean_retained, twin_retained, proxy):
    clean = proxy_detection.evaluate_proxy_mask(clean_retained, proxy)
    twin = proxy_detection.evaluate_proxy_mask(twin_retained, proxy)
    return {
        "clean": clean,
        "twin": twin,
        "twin_only_components": sum(iou == 0 for iou in builder.component_ious(twin_retained, clean_retained)),
        "clean_only_components": sum(iou == 0 for iou in builder.component_ious(clean_retained, twin_retained)),
        "identical_masks": bool((clean_retained == twin_retained).all()),
    }


def summarize(comparisons):
    n = len(comparisons)
    summary = {"pairs": n}
    for metric in RATE_METRICS:
        summary[f"{metric}_clean"] = sum(detected(c["clean"], metric) for c in comparisons) / n
        summary[f"{metric}_twin"] = sum(detected(c["twin"], metric) for c in comparisons) / n
    summary["presence_clean_only"] = sum(c["clean"].presence and not c["twin"].presence for c in comparisons)
    summary["presence_twin_only"] = sum(c["twin"].presence and not c["clean"].presence for c in comparisons)
    summary["identical_masks"] = sum(c["identical_masks"] for c in comparisons) / n
    summary["twin_only_components"] = sum(c["twin_only_components"] for c in comparisons) / n
    summary["clean_only_components"] = sum(c["clean_only_components"] for c in comparisons) / n
    return summary


def build_summaries(data, manifest, pair_records, predictions_dir):
    groups = external.select_groups(data.rows)
    external.validate_report_inputs(data, manifest, groups)
    proxy_groups = proxy_detection.select_proxy_groups(groups)
    twin_of = {record["clean_case"]: record["caliper_case"] for record in pair_records}
    reference_rows = proxy_groups[PRIMARY_MODEL, 0]
    if {row.image_id for row in reference_rows} != set(twin_of):
        raise RuntimeError("Twin pairs do not match the proxy-bearing hemangioma Clean cohort")
    masks = proxy_detection.load_proxy_masks(reference_rows)
    summaries = {}
    for (model, fold), rows in proxy_groups.items():
        if model != PRIMARY_MODEL:
            continue
        config_id = f"{model}_588images_seed42_fold{fold}_final"
        comparisons = []
        for row in rows:
            proxy = masks[row.image_id]
            clean_path = proxy_detection.case_mask_path(row.prediction_path, row.image_id)
            twin_path = predictions_dir / config_id / f"{twin_of[row.image_id]}.png"
            if not twin_path.is_file():
                raise RuntimeError(f"Missing twin prediction: {twin_path}")
            comparisons.append(compare_pair(read_retained(clean_path, proxy.shape),
                                            read_retained(twin_path, proxy.shape), proxy))
        summaries[model, fold] = summarize(comparisons)
        print(f"Compared {model}, fold {fold} ({len(rows)} pairs)", flush=True)
    return summaries


def mean_sd(values, digits=3):
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def build_report(data, pair_records, summaries):
    pairs = summaries[PRIMARY_MODEL, 0]["pairs"]
    patients = len({record["patient"] for record in pair_records})
    lines = [
        "# Effect of caliper marks on predictions",
        "",
        f"Prediction snapshot `{data.snapshot_id}`. {pairs} hemangioma Clean frames ({patients} patients) "
        "with usable twin-derived proxies, each paired with its caliper twin: the same pixels plus the "
        "caliper marks. Both frames go through the same 5 final nnU-Net default-augmentation models "
        "(588-image AUL training pool, seed 42, 150 epochs, folds 0–4), so any difference between the "
        "two predictions is caused by the marks.",
        "",
        "Rates are mean ± sample SD across the five fold models on the same pairs; "
        "differences are marked twin minus Clean frame, as proportions, computed per fold. The study's "
        "relevance threshold is 0.03. Overlap and centroid detection use the same circular proxy for "
        "both frames of a pair. These are hemangioma marks only; HCC frames come from several scanners "
        "with other mark styles and are not tested here.",
        "",
        "## Detection with and without marks",
        "",
        "| metric | frame | value |",
        "|---|---|---|",
    ]
    for metric in RATE_METRICS:
        for frame, suffix in (("Clean", "clean"), ("marked twin", "twin")):
            cell = mean_sd([summaries[PRIMARY_MODEL, fold][f"{metric}_{suffix}"] for fold in FOLDS])
            lines.append(f"| {metric} | {frame} | {cell} |")
        cell = mean_sd([summaries[PRIMARY_MODEL, fold][f"{metric}_twin"]
                        - summaries[PRIMARY_MODEL, fold][f"{metric}_clean"] for fold in FOLDS])
        lines.append(f"| {metric} | difference | {cell} |")
    lines.extend([
        "",
        "## Pair-level agreement",
        "",
        "Components are retained predicted components (same size filter as everywhere else). A component "
        "is only in one frame's prediction when it has no pixel overlap with the other frame's retained mask.",
        "",
        "| measure | value |",
        "|---|---|",
    ])
    for label, key in (("pairs with identical retained masks", "identical_masks"),
                       ("components only in the marked twin, per pair", "twin_only_components"),
                       ("components only in the Clean frame, per pair", "clean_only_components"),
                       ("presence only on the Clean frame (pairs)", "presence_clean_only"),
                       ("presence only on the marked twin (pairs)", "presence_twin_only")):
        cell = mean_sd([summaries[PRIMARY_MODEL, fold][key] for fold in FOLDS],
                       digits=1 if key.startswith("presence") else 3)
        lines.append(f"| {label} | {cell} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS,
                        help="Runner predictions directory for the twin run")
    args = parser.parse_args(argv)
    data = load_predictions_dataset()
    manifest = json.loads(external.MANIFEST_PATH.read_text())
    pair_records = json.loads(args.pairs.read_text())
    summaries = build_summaries(data, manifest, pair_records, args.predictions)
    OUTPUT_PATH.write_text(build_report(data, pair_records, summaries))
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
