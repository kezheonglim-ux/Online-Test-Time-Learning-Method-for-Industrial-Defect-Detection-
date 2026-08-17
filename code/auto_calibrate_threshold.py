"""Trusted-normal threshold calibration for the rev1.8 patch detector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_ROOT = Path(r"C:\cira_ttl_model")
CALIBRATION_ROOT = Path(r"C:\cira_batch_test\normal_trusted")
YOLO_PATH = MODEL_ROOT / "yolo26n-cls.pt"

GOOD_PREFIX = "good_"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

ANOMALY_QUANTILE = 0.995
UPDATE_QUANTILE = 0.90
SAFETY_MARGIN_STD_FACTOR = 0.10
MINIMUM_IMAGES = 3
RECOMMENDED_IMAGES = 20


class Tee:
    """Write output to console and report file."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def dry_run_report_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CALIBRATION_ROOT / f"auto_calibration_dry_run_{stamp}.txt"


sys.path.insert(0, str(MODEL_ROOT))
from cira_ttl_anomaly import TTLAnomalyDetector  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate rev1.8 patch thresholds from trusted good_ images."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--category")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_category_name(category: str) -> str:
    value = str(category).strip().lower()
    if not value or value in {".", ".."}:
        raise ValueError("Invalid category.")
    if any(c in value for c in ("/", "\\")) or ".." in value:
        raise ValueError(f"Unsafe category: {category!r}")
    return value


def required_category_files(category: str) -> dict[str, Path]:
    category_dir = MODEL_ROOT / category
    return {
        "category_dir": category_dir,
        "threshold": category_dir / "threshold.json",
        "patch_memory_bank": category_dir / "patch_memory_bank.pt",
        "patch_adapter": category_dir / "patch_adapter.pt",
        "yolo": YOLO_PATH,
        "anomaly_module": MODEL_ROOT / "cira_ttl_anomaly.py",
        "images": CALIBRATION_ROOT / category,
    }


def validate_paths(category: str) -> dict[str, Path]:
    paths = required_category_files(category)
    required = [
        paths["threshold"],
        paths["patch_memory_bank"],
        paths["yolo"],
        paths["anomaly_module"],
        paths["images"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required path(s) not found:\n  - " + "\n  - ".join(missing)
        )
    return paths


def find_good_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower().startswith(GOOD_PREFIX)
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def build_detector(paths, cfg):
    adapter_path = (
        str(paths["patch_adapter"])
        if paths["patch_adapter"].exists()
        else None
    )

    return TTLAnomalyDetector(
        patch_memory_bank_path=str(paths["patch_memory_bank"]),
        patch_adapter_path=adapter_path,
        threshold=cfg.get("threshold", 999.0),
        model_name=str(paths["yolo"]),
        img_size=cfg.get("img_size", 384),
        feature_choice=cfg.get("feature_choice", "last2"),
        patch_grid=cfg.get("patch_grid", 14),
        patch_top_fraction=cfg.get("patch_top_fraction", 0.05),
        update_threshold=cfg.get("update_threshold"),
        accept_margin=cfg.get("accept_margin", 0.95),
        online_lr=cfg.get("online_lr", 1e-4),
        online_steps=cfg.get("online_steps", 1),
        max_patch_memory=cfg.get("max_patch_memory", 16000),
        consistency_weight=cfg.get("consistency_weight", 1.0),
        anchor_weight=cfg.get("anchor_weight", 0.1),
    )


def calculate_scores(detector, image_paths):
    scores = []
    records = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Cannot read {image_path}; skipped.")
            continue

        score = float(detector.score_only(image))
        scores.append(score)
        records.append((image_path.name, score))

    return np.asarray(scores, dtype=float), records


def backup_threshold(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"threshold_before_calibration_{stamp}.json")
    shutil.copy2(path, backup)
    return backup


def calibrate_category(category: str, dry_run=False):
    category = safe_category_name(category)
    paths = validate_paths(category)
    images = find_good_images(paths["images"])

    if len(images) < MINIMUM_IMAGES:
        raise ValueError(
            f"{category}: only {len(images)} trusted good image(s)."
        )

    if len(images) < RECOMMENDED_IMAGES:
        print(
            f"WARNING: '{category}' has {len(images)} good images; "
            f"{RECOMMENDED_IMAGES}+ is recommended."
        )

    cfg = load_json(paths["threshold"])
    detector = build_detector(paths, cfg)
    scores, records = calculate_scores(detector, images)

    if len(scores) < MINIMUM_IMAGES:
        raise ValueError(f"{category}: not enough readable good images.")

    q_anomaly = float(np.quantile(scores, ANOMALY_QUANTILE))
    q_update = float(np.quantile(scores, UPDATE_QUANTILE))
    safety_margin = float(SAFETY_MARGIN_STD_FACTOR * np.std(scores))

    anomaly_threshold = q_anomaly + safety_margin
    update_threshold = min(q_update, anomaly_threshold)

    print("\n" + "=" * 72)
    print(f"CATEGORY: {category}")
    print(f"Trusted image folder: {paths['images']}")
    print("Only good_* files are used.")
    print("-" * 72)

    for name, score in records:
        print(f"{name:<32} {score:.8f}")

    print("-" * 72)
    print(f"Readable good images : {len(scores)}")
    print(f"Minimum score        : {scores.min():.8f}")
    print(f"Mean score           : {scores.mean():.8f}")
    print(f"Standard deviation   : {scores.std():.8f}")
    print(f"Maximum score        : {scores.max():.8f}")
    print(f"Anomaly quantile     : {ANOMALY_QUANTILE}")
    print(f"Update quantile      : {UPDATE_QUANTILE}")
    print(f"Safety margin        : {safety_margin:.8f}")

    old_threshold = cfg.get("threshold")
    old_update = cfg.get("update_threshold")

    cfg.update(
        {
            "threshold_original": cfg.get(
                "threshold_original", old_threshold
            ),
            "threshold_before_last_calibration": old_threshold,
            "update_threshold_before_last_calibration": old_update,
            "threshold": float(anomaly_threshold),
            "update_threshold": float(update_threshold),
            "threshold_method": "rev18_trusted_normal_dual_threshold",
            "score_method": "patch_nearest_normal",
            "calibration_category": category,
            "calibration_source_folder": str(paths["images"]),
            "calibration_filename_rule": "good_* only",
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "anomaly_quantile": ANOMALY_QUANTILE,
            "update_quantile": UPDATE_QUANTILE,
            "safety_margin_std_factor": SAFETY_MARGIN_STD_FACTOR,
            "safety_margin": safety_margin,
            "normal_score_count": int(len(scores)),
            "normal_score_min": float(scores.min()),
            "normal_score_mean": float(scores.mean()),
            "normal_score_std": float(scores.std()),
            "normal_score_max": float(scores.max()),
            "calibration_images": [name for name, _ in records],
        }
    )

    if dry_run:
        print("\n" + "=" * 72)
        print("DRY-RUN NOTICE")
        print("threshold.json has not been changed.")
        print(f"Current anomaly threshold : {old_threshold!r}")
        print(f"Proposed anomaly threshold: {anomaly_threshold:.8f}")
        print(f"Current update threshold  : {old_update!r}")
        print(f"Proposed update threshold : {update_threshold:.8f}")

        if isinstance(old_threshold, (int, float)) and old_threshold != 0:
            pct = (
                (anomaly_threshold - float(old_threshold))
                / abs(float(old_threshold))
                * 100.0
            )
            direction = "increase" if pct > 0 else "decrease"
            print(f"Anomaly-threshold change  : {pct:+.2f}% ({direction})")

        print("\nNext command to update threshold.json:")
        print(
            f"python auto_calibrate_threshold.py --category {category}"
        )
        backup = None
    else:
        backup = backup_threshold(paths["threshold"])
        temp = paths["threshold"].with_suffix(".json.tmp")

        with temp.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")

        os.replace(temp, paths["threshold"])
        print(f"Updated: {paths['threshold']}")
        print(f"Backup : {backup}")

    return {
        "category": category,
        "image_count": len(scores),
        "threshold": anomaly_threshold,
        "update_threshold": update_threshold,
        "dry_run": dry_run,
    }


def discover_categories():
    if not CALIBRATION_ROOT.exists():
        return []

    categories = []

    for folder in sorted(CALIBRATION_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        category = safe_category_name(folder.name)
        paths = required_category_files(category)

        required = [
            paths["threshold"],
            paths["patch_memory_bank"],
        ]

        if all(p.exists() for p in required) and find_good_images(folder):
            categories.append(category)

    return categories


def run_calibration(args):
    if args.category:
        categories = [safe_category_name(args.category)]
    else:
        categories = discover_categories()

    if not categories:
        raise RuntimeError("No rev1.8 calibratable categories found.")

    print("Categories selected:", ", ".join(categories))
    failures = []
    completed = 0

    for category in categories:
        try:
            calibrate_category(category, dry_run=args.dry_run)
            completed += 1
        except Exception as exc:
            failures.append((category, str(exc)))
            print(f"\nERROR [{category}]: {exc}")

    print("\n" + "=" * 72)
    print(f"Completed: {completed} category/categories")
    print(f"Failed   : {len(failures)} category/categories")

    for category, message in failures:
        print(f"  - {category}: {message}")

    return 1 if failures else 0


def main():
    args = parse_args()

    if not args.dry_run:
        return run_calibration(args)

    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = dry_run_report_path()

    with report_path.open("w", encoding="utf-8") as report_file:
        tee_out = Tee(sys.stdout, report_file)
        tee_err = Tee(sys.stderr, report_file)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"Dry-run report: {report_path}")
            code = run_calibration(args)
            print(f"\nSaved dry-run report: {report_path}")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
