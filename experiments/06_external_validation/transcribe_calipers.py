#!/usr/bin/env python3
"""Transcribe HCC caliper endpoints by hand, then write circular proxy masks from them.

The person transcribing only records where the sonographer's two caliper marks are; they do not
judge the lesion. Frames without exactly one pair of marks on one lesion are excluded.
See caliper_transcription_runbook.md.

    python3 experiments/06_external_validation/transcribe_calipers.py transcribe
    python3 experiments/06_external_validation/transcribe_calipers.py masks
"""

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SOURCE = HERE / "SMC-LUD Liver Ultrasound Dataset (HCC vs Hemangioma)"
DEFAULT_REFERENCE = HERE / "reference" / "smc_lud_reference.json"
DEFAULT_TRANSCRIPTIONS = REPO / "archives_staging/06_external_validation/caliper_transcriptions.json"
DEFAULT_MASKS = REPO / "archives_staging/06_external_validation/proxy_masks_transcribed"
MIN_FRAMES_PER_PATIENT = 4
MIN_ENDPOINT_DISTANCE_PX = 3
INSTRUCTIONS = ("click both caliper centers, Enter = save | x = exclude | u = undo click | q = quit\n"
                "Exclude unless there is exactly one pair of marks on one lesion.")


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_frames(reference, min_frames_per_patient):
    frames = [image for image in reference["images"]
              if image["class"] == "HCC" and image["subset"] == "caliper"]
    per_patient = Counter(image["patient"] for image in frames)
    selected = [image for image in frames if per_patient[image["patient"]] >= min_frames_per_patient]
    return sorted(selected, key=lambda image: (image["patient"], image["case"]))


def load_transcriptions(path, min_frames_per_patient):
    if not path.is_file():
        return {"schema_version": 1, "class": "HCC", "subset": "caliper",
                "min_frames_per_patient": min_frames_per_patient, "records": []}
    transcriptions = json.loads(path.read_text())
    require(transcriptions["schema_version"] == 1, "Unsupported transcription schema")
    require(transcriptions["min_frames_per_patient"] == min_frames_per_patient,
            f"{path} was started with --min-frames-per-patient "
            f"{transcriptions['min_frames_per_patient']}")
    return transcriptions


def save_transcriptions(path, transcriptions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(transcriptions, indent=2) + "\n")
    temporary.replace(path)


class Session:
    """Decision state for one frame at a time; saves after every accepted or excluded frame."""

    def __init__(self, queue, transcriptions, save):
        self.queue = list(queue)
        self.transcriptions = transcriptions
        self.save = save
        self.points = []

    @property
    def current(self):
        return self.queue[0] if self.queue else None

    def click(self, y, x):
        if self.current is not None and len(self.points) < 2:
            self.points.append([round(float(y), 1), round(float(x), 1)])

    def undo(self):
        if self.points:
            self.points.pop()

    def accept(self):
        if len(self.points) != 2:
            return False
        (y0, x0), (y1, x1) = self.points
        if math.hypot(y1 - y0, x1 - x0) < MIN_ENDPOINT_DISTANCE_PX:
            return False
        self.record("pair", self.points)
        return True

    def exclude(self):
        self.record("excluded", None)

    def record(self, status, points):
        image = self.queue.pop(0)
        self.transcriptions["records"].append(
            {"case": image["case"], "patient": image["patient"], "status": status, "points": points})
        self.save(self.transcriptions)
        self.points = []


def transcribe(args):
    import matplotlib.pyplot as plt

    reference = json.loads(args.reference.read_text())
    transcriptions = load_transcriptions(args.transcriptions, args.min_frames_per_patient)
    done = {record["case"] for record in transcriptions["records"]}
    frames = select_frames(reference, args.min_frames_per_patient)
    queue = [image for image in frames if image["case"] not in done]
    print(f"{len(frames)} frames from {len({i['patient'] for i in frames})} patients; "
          f"{len(done)} done, {len(queue)} to go")
    if not queue:
        return
    session = Session(queue, transcriptions, lambda t: save_transcriptions(args.transcriptions, t))
    plt.rcParams["keymap.quit"] = []
    figure, axes = plt.subplots(figsize=(9, 9))

    def draw():
        axes.clear()
        image = session.current
        if image is None:
            plt.close(figure)
            return
        path = args.source / image["source_relative_path"]
        require(sha256(path) == image["sha256"], f"Source image differs from reference: {path}")
        with Image.open(path) as frame:
            axes.imshow(frame.convert("RGB"))
        for y, x in session.points:
            axes.plot(x, y, marker="o", markersize=10, markerfacecolor="none", color="red")
        position = len(frames) - len(session.queue) + 1
        axes.set_title(f"{position}/{len(frames)}  {image['patient']}  {image['case']}\n{INSTRUCTIONS}",
                       fontsize=9)
        axes.set_axis_off()
        figure.canvas.draw_idle()

    def on_click(event):
        toolbar = figure.canvas.toolbar
        if event.inaxes is axes and event.button == 1 and not (toolbar is not None and toolbar.mode):
            session.click(event.ydata, event.xdata)
            draw()

    def on_key(event):
        if event.key == "enter":
            session.accept()
        elif event.key == "x":
            session.exclude()
        elif event.key == "u":
            session.undo()
        elif event.key == "q":
            plt.close(figure)
            return
        draw()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
    remaining = len(session.queue)
    print(f"{remaining} frames left; transcriptions in {args.transcriptions}")


def write_masks(transcriptions, reference, output):
    render_proxy_mask = load_sibling("extract_caliper_proxy").render_proxy_mask
    require(not (output / "extraction_report.json").exists(),
            f"Masks already written; delete or choose a new --output-dir: {output}")
    images = {image["case"]: image for image in reference["images"]}
    records = transcriptions["records"]
    require(len({record["case"] for record in records}) == len(records), "Duplicate transcribed cases")
    output.mkdir(parents=True, exist_ok=True)
    proxy_cases = []
    for record in records:
        if record["status"] != "pair":
            continue
        image = images[record["case"]]
        require(image["class"] == "HCC" and image["subset"] == "caliper",
                f"Not an HCC caliper frame: {record['case']}")
        (y0, x0), (y1, x1) = record["points"]
        center_y, center_x = (y0 + y1) / 2, (x0 + x1) / 2
        diameter = math.hypot(y1 - y0, x1 - x0)
        mask = render_proxy_mask(center_y, center_x, diameter, image["height"], image["width"])
        Image.fromarray(mask).save(output / f"{record['case']}.png")
        proxy_cases.append({"case": record["case"], "patient": record["patient"], "class": "HCC",
                            "method": "transcribed", "center_y": round(center_y, 1),
                            "center_x": round(center_x, 1), "diameter_px": round(diameter, 1)})
    report = {
        "hcc": {"method": "transcribed", "masks_written": len(proxy_cases),
                "excluded": sum(record["status"] == "excluded" for record in records),
                "min_frames_per_patient": transcriptions["min_frames_per_patient"]},
        "proxy_cases": proxy_cases,
        "total_masks": len(proxy_cases),
    }
    (output / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("transcribe", "masks"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--transcriptions", type=Path, default=DEFAULT_TRANSCRIPTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MASKS,
                        help="masks: directory for proxy PNGs and extraction_report.json")
    parser.add_argument("--min-frames-per-patient", type=int, default=MIN_FRAMES_PER_PATIENT,
                        help="Only queue HCC patients with at least this many caliper frames")
    args = parser.parse_args(argv)
    if args.command == "transcribe":
        transcribe(args)
        return
    transcriptions = load_transcriptions(args.transcriptions, args.min_frames_per_patient)
    report = write_masks(transcriptions, json.loads(args.reference.read_text()), args.output_dir)
    print(f"{report['total_masks']} masks, {report['hcc']['excluded']} excluded -> {args.output_dir}")


if __name__ == "__main__":
    main()
