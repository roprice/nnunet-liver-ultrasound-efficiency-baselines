#!/usr/bin/env python3
"""Prepare the caliper-subset inference cohort for external_validation_runner.py --subset caliper.

Holds the marked twin of every hemangioma Clean frame with a twin_diff proxy mask (for the
twin-pair test of whether marks change predictions) and every HCC caliper frame with a
transcribed proxy mask (see caliper_transcription_runbook.md). Also writes twin_pairs.json,
mapping each hemangioma Clean case to its marked twin.
Predict with the runner's --experiments data_efficiency: the analyses use only the default models.

Run after extract_caliper_proxy.py and transcribe_calipers.py masks:
    python3 -I experiments/06_external_validation/prepare_caliper_cohort.py
"""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE = REPO / "archives/06_external_validation"


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_external_data = load_sibling("prepare_external_data")
extract_caliper_proxy = load_sibling("extract_caliper_proxy")
require = prepare_external_data.require


def select_twin_pairs(source, reference, proxy_cases):
    clean_cases = {case["case"] for case in proxy_cases if case["class"] == "Hemangioma"}
    pairs = [(clean, caliper) for clean, caliper, _ in
             extract_caliper_proxy.find_hemangioma_twins(source, reference)
             if clean["case"] in clean_cases]
    require({clean["case"] for clean, _ in pairs} == clean_cases,
            "Not every proxy-bearing hemangioma Clean case has a twin")
    require(len({caliper["case"] for _, caliper in pairs}) == len(pairs),
            "A marked twin is used by more than one Clean case")
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=prepare_external_data.DEFAULT_SOURCE)
    parser.add_argument("--reference", type=Path, default=prepare_external_data.DEFAULT_REFERENCE)
    parser.add_argument("--proxy-masks", type=Path, default=ARCHIVE / "proxy_masks/automatic",
                        help="extract_caliper_proxy.py output; supplies the hemangioma twins")
    parser.add_argument("--hcc-proxy-masks", type=Path, default=ARCHIVE / "proxy_masks/hcc_transcribed",
                        help="transcribe_calipers.py masks output; supplies the HCC frames")
    parser.add_argument("--output-dir", type=Path, default=ARCHIVE / "caliper/prepared")
    args = parser.parse_args(argv)

    reference = prepare_external_data.load_reference(args.reference)
    prepare_external_data.verify_source(args.source, reference)
    require(not (args.output_dir / "smc_lud_manifest.json").exists(),
            f"Manifest already exists; delete or choose a new --output-dir: {args.output_dir}")
    proxy_cases = json.loads((args.proxy_masks / "extraction_report.json").read_text())["proxy_cases"]

    pairs = select_twin_pairs(args.source, reference, proxy_cases)
    hcc_report = args.hcc_proxy_masks / "extraction_report.json"
    hcc_cases = {case["case"] for case in json.loads(hcc_report.read_text())["proxy_cases"]
                 if case["class"] == "HCC"}
    hcc_images = [image for image in reference["images"]
                  if image["case"] in hcc_cases and image["subset"] == "caliper"]
    require(len(hcc_images) == len(hcc_cases), "HCC proxy case is not a caliper-subset frame")
    selected = {**reference, "images": [caliper for _, caliper in pairs] + hcc_images}

    manifest_cases = prepare_external_data.prepare(args.source, selected, args.output_dir, ("caliper",))
    report = prepare_external_data.write_reports(args.output_dir, selected, manifest_cases, ("caliper",))
    pairs_path = args.output_dir / "twin_pairs.json"
    pairs_path.write_text(json.dumps(
        [{"clean_case": clean["case"], "caliper_case": caliper["case"], "patient": clean["patient"]}
         for clean, caliper in pairs], indent=2) + "\n")
    print(f"{len(pairs)} hemangioma twin pairs + {len(hcc_images)} HCC caliper frames "
          f"({report['case_count']} cases) -> {args.output_dir}")


if __name__ == "__main__":
    main()
