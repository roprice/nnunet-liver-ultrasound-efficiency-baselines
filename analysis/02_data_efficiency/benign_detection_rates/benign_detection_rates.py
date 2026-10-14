"""Report benign-mass detection rates by training-set size.

Run from the repository root:
    python3 -m analysis.02_data_efficiency.benign_detection_rates.benign_detection_rates
"""

import importlib
import json
from pathlib import Path

report = importlib.import_module(
    "analysis.02_data_efficiency.detection_rates.detection_rates"
)
OUTPUT_PATH = Path(__file__).resolve().with_suffix(".md")


def main():
    data = report.load_predictions_dataset()
    with report.MANIFEST_PATH.open() as file:
        manifest = json.load(file)
    OUTPUT_PATH.write_text(report.build_report(data, manifest, pathology="benign"))
    print(f"Wrote {OUTPUT_PATH.relative_to(report.PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
