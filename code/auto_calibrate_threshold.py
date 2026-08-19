# Calibrate category thresholds from trusted normal deployment images.

Only files starting with ``good_`` are used. The script reads the current
patch model for each category, calculates normal anomaly-score statistics,
and updates ``threshold.json`` with a backup.
# 

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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

sys.path.insert(0, str(MODEL_ROOT))
from cira_ttl_anomaly import TTLAnomalyDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    # Read category and dry-run options from the command line 
    parser = argparse.ArgumentParser(
        description="Calibrate CTTA thresholds from trusted good_ images."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--category",
        help="Calibrate one category, for example: --category bottle",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Calibrate every valid category under the trusted-normal folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and print thresholds without modifying threshold.json.",
    )
    return parser.parse_args()


def safe_category_name(category: str) -> str:
    # Normalize a category name and reject path-like input 
    value = str(category).strip().lower()
    if not value or value in {".", ".."}:
        raise ValueError("Invalid empty category name.")
    if any(char in value for char in ("/", "\\")) or ".." in value:
        raise ValueError(f"Unsafe category name: {category!r}")
    return value


def required_category_files(category: str) -> dict[str, Path]:
    # Return model and calibration paths for one category 
    category_dir = MODEL_ROOT / category
    return {
        "category_dir": category_dir,
        "threshold": category_dir / "threshold.json",
        "patch_adapter": category_dir / "patch_adapter.pt",
        "patch_memory_bank": category_dir / "patch_memory_bank.pt",
        "yolo": YOLO_PATH,
        "anomaly_module": MODEL_ROOT / "cira_ttl_anomaly.py",
        "images": CALIBRATION_ROOT / category,
    }


def validate_paths(category: str) -> dict[str, Path]:
    # Check that the files needed for calibration are available 
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
            "Required path(s) not found for category '{}':\n  - {}".format(
                category, "\n  - ".join(missing)
            )
        )
    return paths


def find_good_images(folder: Path) -> list[Path]:
    # Return trusted normal images whose names start with good_ 
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower().startswith(GOOD_PREFIX)
    )


def load_json(path: Path) -> dict[str, Any]:
    # Load and validate a JSON configuration object 
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def build_detector(paths: dict[str, Path], cfg: dict[str, Any]) -> TTLAnomalyDetector:
    # Build a detector for score-only calibration 
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
        consistency_threshold=cfg.get("consistency_threshold", 0.002),
        adapter_update_enabled=False,
        memory_update_enabled=False,
    )


def calculate_scores(
    detector: TTLAnomalyDetector, image_paths: list[Path]
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    # Score trusted normal images without changing online state 
    scores: list[float] = []
    records: list[tuple[str, float]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: OpenCV cannot read {image_path}-- skipped.")
            continue

        # Score the image without updating the adapter or memory bank.
        score = float(detector.score_only(image))
        scores.append(score)
        records.append((image_path.name, score))

    return np.asarray(scores, dtype=float), records


def backup_threshold(path: Path) -> Path:
    # Create a timestamped copy before changing threshold.json 
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"threshold_before_calibration_{stamp}.json")
    shutil.copy2(path, backup_path)
    return backup_path


def calibrate_category(category: str, dry_run: bool = False) -> dict[str, Any]:
    # Calculate and optionally save thresholds for one category 
    category = safe_category_name(category)
    paths = validate_paths(category)
    image_paths = find_good_images(paths["images"])

    if len(image_paths) < MINIMUM_IMAGES:
        raise ValueError(
            f"Category '{category}' has only {len(image_paths)} trusted good_ image(s). "
            f"At least {MINIMUM_IMAGES} are required."
        )

    if len(image_paths) < RECOMMENDED_IMAGES:
        print(
            f"WARNING: '{category}' has {len(image_paths)} good images. "
            f"Calibration works, but {RECOMMENDED_IMAGES}+ is recommended for a stable high quantile."
        )

    cfg = load_json(paths["threshold"])
    detector = build_detector(paths, cfg)
    normal_scores, records = calculate_scores(detector, image_paths)

    if len(normal_scores) < MINIMUM_IMAGES:
        raise ValueError(
            f"Only {len(normal_scores)} readable good image(s) remained for '{category}'."
        )

    # High quantile sets the anomaly boundary, q0.90 is stricter for online updates.
    anomaly_before_margin = float(np.quantile(normal_scores, ANOMALY_QUANTILE))
    update_threshold = float(np.quantile(normal_scores, UPDATE_QUANTILE))
    safety_margin = float(SAFETY_MARGIN_STD_FACTOR * np.std(normal_scores))
    anomaly_threshold = float(anomaly_before_margin + safety_margin)

    # Keep the update threshold at or below the anomaly threshold.
    update_threshold = min(update_threshold, anomaly_threshold)

    print("\n" + "=" * 72)
    print(f"CATEGORY: {category}")
    print(f"Trusted image folder: {paths['images']}")
    print("Only good_* files are included, bad_* files are excluded.")
    print("-" * 72)
    for name, score in records:
        print(f"{name:<32} {score:.8f}")

    print("-" * 72)
    print(f"Readable good images : {len(normal_scores)}")
    print(f"Minimum score        : {normal_scores.min():.8f}")
    print(f"Mean score           : {normal_scores.mean():.8f}")
    print(f"Standard deviation   : {normal_scores.std():.8f}")
    print(f"Maximum score        : {normal_scores.max():.8f}")
    print(f"Anomaly quantile     : {ANOMALY_QUANTILE}")
    print(f"Update quantile      : {UPDATE_QUANTILE}")
    print(f"Safety margin        : {safety_margin:.8f}")

    previous_threshold = cfg.get("threshold")
    previous_update_threshold = cfg.get("update_threshold")

    # Preserve the first threshold for baseline comparison, then store current calibration.
    cfg.update(
        {
            "threshold_original": cfg.get("threshold_original", previous_threshold),
            "threshold_before_last_calibration": previous_threshold,
            "update_threshold_before_last_calibration": previous_update_threshold,
            "threshold": anomaly_threshold,
            "update_threshold": update_threshold,
            "threshold_method": "deployment_auto_calibrated_dual_threshold",
            "calibration_category": category,
            "calibration_source_folder": str(paths["images"]),
            "calibration_filename_rule": "good_* only",
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "anomaly_quantile": ANOMALY_QUANTILE,
            "update_quantile": UPDATE_QUANTILE,
            "safety_margin_std_factor": SAFETY_MARGIN_STD_FACTOR,
            "safety_margin": safety_margin,
            "normal_score_count": int(len(normal_scores)),
            "normal_score_min": float(normal_scores.min()),
            "normal_score_mean": float(normal_scores.mean()),
            "normal_score_std": float(normal_scores.std()),
            "normal_score_max": float(normal_scores.max()),
            "calibration_images": [name for name, _ in records],
        }
    )

    backup_path: Path | None = None

    # Dry run: show the proposed values without changing threshold.json.
    if dry_run:
        print("\n" + "=" * 72)
        print("DRY-RUN NOTICE")
        print("threshold.json has not been changed.")
        print(f"Current anomaly threshold : {previous_threshold!r}")
        print(f"Proposed anomaly threshold: {anomaly_threshold:.8f}")
        print(f"Current update threshold  : {previous_update_threshold!r}")
        print(f"Proposed update threshold : {update_threshold:.8f}")

        if isinstance(previous_threshold, (int, float)) and previous_threshold != 0:
            threshold_change_pct = (
                (anomaly_threshold - float(previous_threshold))
                / abs(float(previous_threshold))
                * 100.0
            )
            direction = "increase" if threshold_change_pct > 0 else "decrease"
            print(
                f"Anomaly-threshold change  : "
                f"{threshold_change_pct:+.2f}% ({direction})"
            )
        else:
            print(
                "Anomaly-threshold change  : "
                "unavailable (current value missing or zero)"
            )

        print("\nNext command to update threshold.json:")
        print(f"python auto_calibrate_threshold.py --category {category}")
    else:
        # Update mode: back up threshold.json, then save the calibrated values.
        backup_path = backup_threshold(paths["threshold"])
        temp_path = paths["threshold"].with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(cfg, file, indent=4)
            file.write("\n")
        os.replace(temp_path, paths["threshold"])
        print(f"Updated: {paths['threshold']}")
        print(f"Backup : {backup_path}")
        print(
            "A running updated app_ctta.py will detect this file change and apply "
            "the new thresholds on the next request."
        )

    return {
        "category": category,
        "image_count": int(len(normal_scores)),
        "threshold": anomaly_threshold,
        "update_threshold": update_threshold,
        "threshold_path": str(paths["threshold"]),
        "backup_path": str(backup_path) if backup_path else None,
        "dry_run": dry_run,
    }


def discover_categories() -> list[str]:
    # Find categories that have model files and trusted normal images 
    if not CALIBRATION_ROOT.exists():
        return []

    categories: list[str] = []
    for folder in sorted(CALIBRATION_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        category = safe_category_name(folder.name)
        model_files = required_category_files(category)
        required = [
            model_files["threshold"],
            model_files["patch_memory_bank"],
        ]
        if all(path.exists() for path in required) and find_good_images(folder):
            categories.append(category)
    return categories


def main() -> int:
    # Run calibration for one category or all available categories 
    args = parse_args()

    if not MODEL_ROOT.exists():
        raise FileNotFoundError(f"MODEL_ROOT does not exist: {MODEL_ROOT}")
    if not CALIBRATION_ROOT.exists():
        raise FileNotFoundError(
            f"Calibration/test root does not exist: {CALIBRATION_ROOT}"
        )

    if args.category:
        categories = [safe_category_name(args.category)]
    else:
        # Use all valid categories when no category is specified.
        categories = discover_categories()

    if not categories:
        raise RuntimeError(
            "No calibratable categories found. Each category needs model files and "
            "trusted good_* images in the category calibration folder."
        )

    print("Categories selected:", ", ".join(categories))
    failures: list[tuple[str, str]] = []
    results: list[dict[str, Any]] = []

    for category in categories:
        try:
            results.append(calibrate_category(category, dry_run=args.dry_run))
        except Exception as exc:  # Continue with the remaining categories.
            failures.append((category, str(exc)))
            print(f"\nERROR [{category}]: {exc}")

    print("\n" + "=" * 72)
    print(f"Completed: {len(results)} category/categories")
    print(f"Failed   : {len(failures)} category/categories")
    for category, message in failures:
        print(f"  - {category}: {message}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
