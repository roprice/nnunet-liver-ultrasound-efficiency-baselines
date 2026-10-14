"""Run: python -m analysis.03_augmentation_ablation.augmentation_ablation_report.augmentation_ablation_report."""

import importlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.predictions_dataset.load_predictions_dataset import load_predictions_dataset

rates = importlib.import_module("analysis.03_augmentation_ablation.detection_rates.detection_rates")
HERE = Path(__file__).resolve().parent
CHECKPOINTS = ["final", "best"]
SERIES = [
    ("Malignant triage", "triage_detection_flag", "#27ae60", "o", "-"),
    ("Malignant IoU > 0", "overlap_detection_iou_00_flag", "#2a78d6", "s", "-"),
    ("Malignant IoU > 0.2", "overlap_detection_iou_02_flag", "#eb6834", "^", "-"),
    ("FP/image (IoU > 0)", "overlap_detection_iou_00_fp_count", "#898781", "D", "--"),
    ("Normal-image flagged fraction", "triage_detection_flag", "#898781", "D", "--"),
]


def summarize(rows, manifest, snapshot):
    if manifest["snapshot_id"] != snapshot:
        raise RuntimeError("Dataset snapshot does not match build manifest")
    selected = [r for r in rows if r.experiment_name == "augmentation_ablation"
                and r.evaluation_split == "test" and r.checkpoint in CHECKPOINTS]
    if {r.training_set_size for r in selected} != set(rates.SIZES):
        raise RuntimeError(f"Expected training-set sizes {rates.SIZES}")
    results = {}
    cohort = None
    for checkpoint in CHECKPOINTS:
        for size in rates.SIZES:
            group = [r for r in selected if r.checkpoint == checkpoint and r.training_set_size == size]
            if {r.fold for r in group} != set(rates.FOLDS):
                raise RuntimeError(f"{checkpoint}/{size}: expected folds 0-4")
            expected_epoch = 150 if checkpoint == "final" else None
            values = [[] for _ in SERIES]
            epochs = []
            for fold in rates.FOLDS:
                fold_rows = [r for r in group if r.fold == fold]
                current = {(r.image_id, r.pathology, r.reference_path) for r in fold_rows}
                if len({r.image_id for r in fold_rows}) != len(fold_rows) or (
                        cohort is not None and current != cohort):
                    raise RuntimeError("Duplicate images or inconsistent test cohort")
                cohort = current
                if {(r.seed, r.encoder) for r in fold_rows} != {(42, group[0].encoder)}:
                    raise RuntimeError("Inconsistent seed or encoder")
                epoch_set = {r.epoch for r in fold_rows}
                if (len(epoch_set) != 1 or any(type(epoch) is not int or epoch <= 0 for epoch in epoch_set)
                        or (expected_epoch is not None and epoch_set != {expected_epoch})):
                    raise RuntimeError("Unexpected checkpoint epoch")
                epochs.append(next(iter(epoch_set)))
                malignant = [r for r in fold_rows if r.pathology == "malignant"]
                normal = [r for r in fold_rows if r.pathology == "normal"]
                if not malignant or not normal:
                    raise RuntimeError("Missing malignant or normal cohort")
                for index, (_, field, *_) in enumerate(SERIES):
                    population = malignant if index < 3 else (fold_rows if index == 3 else normal)
                    values[index].append(statistics.mean(getattr(r, field) for r in population))
            results[checkpoint, size] = {
                "stats": [(statistics.mean(v), statistics.stdev(v)) for v in values],
                "epochs": epochs,
            }
    counts = {pathology: sum(p == pathology for _, p, _ in cohort)
              for pathology in ("malignant", "benign", "normal")}
    return results, counts


def curves(ax, x, results, indices):
    for index in indices:
        label, _, color, marker, style = SERIES[index]
        means = [r["stats"][index][0] for r in results]
        sds = [r["stats"][index][1] for r in results]
        ax.plot(x, means, label=label, color=color, marker=marker, linestyle=style, linewidth=2)
        ax.fill_between(x, [m-s for m, s in zip(means, sds)],
                        [m+s for m, s in zip(means, sds)], color=color, alpha=.2, linewidth=0)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=.25)


def charts(results):
    final = [results["final", size] for size in rates.SIZES]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    curves(axes[0], rates.SIZES, final, range(3))
    curves(axes[1], rates.SIZES, final, [3])
    axes[0].set(ylabel="Malignant detection rate", ylim=(0, 1.05),
                title="Augmentation ablation: final checkpoint (150 epochs)")
    axes[1].set(ylabel="FP per evaluated image", xlabel="Training-pool size (images)", ylim=(0, 1.05))
    for ax in axes:
        ax.legend(fontsize=9)
    fig.savefig(HERE / "augmentation_ablation_report.png", dpi=200)
    plt.close(fig)


def table(results, checkpoint):
    lines = ["| Training pool | Malignant triage | Malignant IoU > 0 | Malignant IoU > 0.2 | FP/image (IoU > 0) | Normal flagged fraction |",
             "|---|---|---|---|---|---|"]
    for size in rates.SIZES:
        cells = [f"{mean:.3f} ± {sd:.3f}" for mean, sd in results[checkpoint, size]["stats"]]
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    return lines


def main():
    data = load_predictions_dataset()
    manifest = json.loads(rates.MANIFEST_PATH.read_text())
    results, counts = summarize(data.rows, manifest, data.snapshot_id)
    charts(results)
    lines = ["# Augmentation-ablation report", "",
             f"Dataset snapshot `{data.snapshot_id}` (schema {manifest['schema_version']}); test split, seed 42, five fold models per training-pool size.", "",
             "## Definitions and cohort", "",
             "The brightness-augmentation application probability is increased from 0.15 to 0.5; the multiplier range remains 0.75–1.25.", "",
             f"The same {sum(counts.values())} unique test images ({counts['malignant']} malignant, {counts['benign']} benign, {counts['normal']} normal) are evaluated by every model. Repeated model evaluations are not additional unique cases.", "",
             "Training-pool sizes are 74 and 588 images; each fold trains on its training portion of that pool. Detection rates use only malignant images. Triage means any retained predicted mass component, regardless of location. Overlap detection requires at least one retained component with IoU strictly greater than 0 or 0.2 against the full reference mass mask.", "",
             "FP/image (IoU > 0) counts retained components with no reference-mass overlap, divided by all evaluated images (malignant, benign and normal). Every retained component on a normal image is an FP. The normal-image flagged fraction is the fraction of normal images with any retained component; it is distinct from component FP/image.", "",
             "Predictions use the shared 8-connected component filter with minimum area int(0.0003 × image area). Every table cell and shaded band is mean ± sample SD across five fold-model rates (SD denominator n−1). Models share the same test images and overlapping training data: this SD describes model variability, not independent patient uncertainty or a confidence interval. Bands are untruncated mean ± SD; axes start at zero.", "",
             "## Final150 training-pool sweep", "",
             "Fixed 150-epoch checkpoints provide the main size comparison. No winning-size selection rule is imposed.", "",
             "![Final150 aligned detection and FP panels](augmentation_ablation_report.png)", ""]
    lines.extend(table(results, "final"))
    lines.extend(["", "## Best checkpoints (separate descriptive comparison)", "",
                  "Best checkpoints are selected by training-time validation, with potentially different epochs; they are not fixed-budget substitutes for final150.", ""])
    lines.extend(table(results, "best"))
    lines.extend(["", "Best checkpoint epochs (folds 0–4): " + "; ".join(
        f"pool {size}: " + ", ".join(map(str, results["best", size]["epochs"])) for size in rates.SIZES) + ".", "",
        "Companion reports: [malignant detection](../malignant_detection_rates/malignant_detection_rates.md), [benign detection](../benign_detection_rates/benign_detection_rates.md), [segmentation](../segmentation_rates/segmentation_rates.md), and [full detection criteria](../detection_rates/detection_rates.md).", ""])
    (HERE / "augmentation_ablation_report.md").write_text("\n".join(lines))
    print(f"Generated augmentation-ablation report: snapshot {data.snapshot_id}, cohort {counts}")


if __name__ == "__main__":
    main()
