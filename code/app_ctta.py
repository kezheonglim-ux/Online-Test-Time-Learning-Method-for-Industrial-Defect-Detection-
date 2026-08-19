# Flask API for category-based anomaly detection and online adaptation 

# Handles experiment selection, category loading, prediction, logging, and checkpoints.
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, request

MODEL_ROOT = Path(r"C:\cira_ttl_model")
YOLO_PATH = MODEL_ROOT / "yolo26n-cls.pt"

LOG_DIR = Path(r"C:\cira_ttl_logs")
LOG_FILE = LOG_DIR / "prediction_log.csv"
SERVER_EVENT_LOG = LOG_DIR / "server_event_log.txt"

CHECKPOINT_ROOT = Path(r"C:\cira_ttl_checkpoints")
SAVE_STATE_EVERY_N_UPDATES = 10

# Experiment selector:
# "baseline", "calibration", "memory_ctta", "adapter_ctta", "full_ctta"
EXPERIMENT = "full_ctta"

VALID_EXPERIMENTS = {
    "baseline",
    "calibration",
    "memory_ctta",
    "adapter_ctta",
    "full_ctta",
}


sys.path.insert(0, str(MODEL_ROOT))
from cira_ttl_anomaly import TTLAnomalyDetector  # noqa: E402

app = Flask(__name__)

DETECTORS: dict[str, TTLAnomalyDetector] = {}
MODEL_CONFIGS: dict[str, dict[str, Any]] = {}
THRESHOLD_FILE_MTIMES: dict[str, int] = {}
UPDATE_COUNTERS: dict[str, int] = {}



def get_experiment_settings():
    # Map EXPERIMENT to scoring mode and enabled online updates 
    experiment = str(EXPERIMENT).strip().lower()

    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(f"Unsupported EXPERIMENT: {EXPERIMENT}")

    return {
        "experiment": experiment,
        "mode": "evaluate" if experiment in {"baseline", "calibration"} else "monitor",
        "adapter_update_enabled": experiment in {"adapter_ctta", "full_ctta"},
        "memory_update_enabled": experiment in {"memory_ctta", "full_ctta"},
    }


def get_threshold_for_experiment(cfg, fallback):
    # Use the original threshold for baseline, calibrated threshold otherwise 
    if get_experiment_settings()["experiment"] == "baseline":
        return float(
            cfg.get(
                "threshold_original",
                cfg.get("threshold_before_last_calibration", fallback),
            )
        )

    return float(cfg.get("threshold", fallback))


def ensure_folders():
    # Create log and checkpoint folders when missing 
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


def event_log(message: str):
    # Write server-side diagnostics without changing the API payload 
    ensure_folders()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    line = f"[{stamp}] {message}"
    print(line)
    with SERVER_EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def normalize_path(path):
    # Normalize incoming Windows paths for consistent handling 
    if path is None:
        return ""
    return str(path).strip().strip('"').replace("\\", "/")


def safe_category_name(category):
    # Remove path characters from a category value 
    value = str(category or "").strip().lower()
    return value.replace("\\", "").replace("/", "").replace("..", "")


def get_json_data():
    # Return a JSON request body for POST or PUT requests 
    if request.method in {"POST", "PUT"}:
        data = request.get_json(silent=True) or {}
        return data if isinstance(data, dict) else {}
    return {}


def get_image_path_from_request():
    # Read image_path from query, JSON body, or CiRA-style URL input 
    data = get_json_data()
    image_path = request.args.get("image_path", "") or data.get("image_path", "")

    if not image_path:
        raw_url = request.full_path
        if ";image_path=" in raw_url:
            image_path = raw_url.split(";image_path=", 1)[1]
            image_path = image_path.split("&", 1)[0].split("?", 1)[0]

    return normalize_path(image_path)


def get_category_from_request(image_path=""):
    # Read category from the request or infer it from the image folder 
    data = get_json_data()
    category = request.args.get("category", "") or data.get("category", "")

    if not category and image_path:
        category = os.path.basename(os.path.dirname(image_path))

    return safe_category_name(category)


def get_mode_from_request():
    # Return the mode controlled by the selected experiment 
    # EXPERIMENT controls the mode used by the service.
    return get_experiment_settings()["mode"]


def category_paths(category):
    # Return model files used by one category 
    category_dir = MODEL_ROOT / category
    return {
        "category_dir": category_dir,
        "threshold": category_dir / "threshold.json",
        "patch_memory_bank": category_dir / "patch_memory_bank.pt",
        "patch_adapter": category_dir / "patch_adapter.pt",
        "yolo": YOLO_PATH,
    }


def get_available_categories():
    # List categories with a threshold and patch memory available 
    categories = []
    if not MODEL_ROOT.exists():
        return categories

    for folder in sorted(MODEL_ROOT.iterdir()):
        if not folder.is_dir():
            continue

        paths = category_paths(folder.name)
        if paths["threshold"].exists() and paths["patch_memory_bank"].exists():
            categories.append(folder.name)

    return categories


def read_threshold_config(path):
    # Load one category threshold configuration 
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return cfg


def apply_threshold_config(category, detector, cfg, mtime_ns):
    # Apply threshold values and cache the config file time 
    detector.threshold = get_threshold_for_experiment(
        cfg,
        detector.threshold,
    )

    update_threshold = cfg.get("update_threshold")
    if update_threshold is None:
        accept_margin = float(
            cfg.get("accept_margin", getattr(detector, "accept_margin", 0.95))
        )
        update_threshold = detector.threshold * accept_margin

    detector.update_threshold = float(update_threshold)
    MODEL_CONFIGS[category] = cfg
    THRESHOLD_FILE_MTIMES[category] = mtime_ns


def refresh_threshold_if_changed(category, detector, force=False):
    # Reload threshold.json only when the file changes 
    path = category_paths(category)["threshold"]
    current_mtime = path.stat().st_mtime_ns
    previous_mtime = THRESHOLD_FILE_MTIMES.get(category)

    # Skip JSON reload when the file has not changed.
    if not force and previous_mtime == current_mtime:
        return False

    cfg = read_threshold_config(path)
    apply_threshold_config(category, detector, cfg, current_mtime)
    event_log(f"threshold_reload category={category}")
    return True


def load_detector_for_category(category):
    # Load and cache one category detector with current experiment settings 
    category = safe_category_name(category)

    if not category:
        raise ValueError("Empty category received.")

    if category in DETECTORS:
        detector = DETECTORS[category]
        refresh_threshold_if_changed(category, detector)
        return detector

    paths = category_paths(category)

    for key in ("threshold", "patch_memory_bank", "yolo"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Required file not found: {paths[key]}")

    cfg = read_threshold_config(paths["threshold"])

    # A saved adapter is optional, otherwise the detector starts from identity.
    adapter_path = str(paths["patch_adapter"]) if paths["patch_adapter"].exists() else None

    settings = get_experiment_settings()

    detector = TTLAnomalyDetector(
        patch_memory_bank_path=str(paths["patch_memory_bank"]),
        patch_adapter_path=adapter_path,
        threshold=get_threshold_for_experiment(cfg, 999.0),
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
        adapter_update_enabled=settings["adapter_update_enabled"],
        memory_update_enabled=settings["memory_update_enabled"],
    )

    DETECTORS[category] = detector
    UPDATE_COUNTERS[category] = 0

    apply_threshold_config(
        category,
        detector,
        cfg,
        paths["threshold"].stat().st_mtime_ns,
    )

    event_log(
        "detector_loaded "
        f"experiment={settings['experiment']} "
        f"category={category} "
        f"threshold={detector.threshold} "
        f"update_threshold={detector.update_threshold} "
        f"memory_size={detector.patch_memory_bank.shape[0]}"
    )

    return detector


def log_prediction(record):
    # Append prediction and adaptation diagnostics to the CSV log 
    ensure_folders()
    file_exists = LOG_FILE.exists()

    fieldnames = [
        "timestamp", "experiment", "category", "mode", "image_path", "file_name",
        "label", "is_anomaly", "score_before", "anomaly_score",
        "threshold", "anomaly_threshold", "update_threshold",
        "threshold_reloaded", "updated_memory", "memory_size",
        "update_allowed", "update_loss", "adapter_updated",
        "adapter_delta_norm", "device", "score_method", "feature_choice",
        "patch_grid", "patch_top_fraction", "checkpoint_saved",
        "checkpoint_memory_path", "checkpoint_adapter_path",
        "consistency_error", "consistency_threshold", "score_gate_pass",
        "consistency_gate_pass", "adapter_update_enabled",
        "memory_update_enabled",
    ]

    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fieldnames})


def save_state_if_needed(category, detector, result):
    # Save adapted state periodically and return server-side checkpoint paths 
    if not (
        result.get("updated_memory", False)
        or result.get("adapter_updated", False)
    ):
        return False, "", ""

    UPDATE_COUNTERS[category] = UPDATE_COUNTERS.get(category, 0) + 1
    counter = UPDATE_COUNTERS[category]

    # Save periodically instead of writing a checkpoint after every update.
    if counter % SAVE_STATE_EVERY_N_UPDATES != 0:
        return False, "", ""

    checkpoint_dir = CHECKPOINT_ROOT / category
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    memory_path = checkpoint_dir / f"patch_memory_bank_update_{counter}.pt"
    adapter_path = checkpoint_dir / f"patch_adapter_update_{counter}.pt"

    event_log(
        f"checkpoint_begin category={category} counter={counter} "
        f"memory={memory_path.name} adapter={adapter_path.name}"
    )

    detector.save_patch_memory_bank(str(memory_path))
    detector.save_patch_adapter(str(adapter_path))

    event_log(
        f"checkpoint_done category={category} counter={counter} "
        f"memory_exists={memory_path.exists()} adapter_exists={adapter_path.exists()}"
    )

    return True, str(memory_path), str(adapter_path)


def run_prediction(detector, image, mode):
    # Run static scoring in evaluate mode or CTTA in monitor mode 
    if mode in {"evaluate", "calibrate"}:
        score = float(detector.score_only(image))
        is_anomaly = score >= detector.threshold

        return {
            "label": "anomaly" if is_anomaly else "normal",
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": score,
            "score_before": score,
            "threshold": detector.threshold,
            "anomaly_threshold": detector.threshold,
            "update_threshold": detector.update_threshold,
            "update_allowed": False,
            "updated_memory": False,
            "memory_size": int(detector.patch_memory_bank.shape[0]),
            "update_loss": None,
            "adapter_updated": False,
            "adapter_delta_norm": 0.0,
            "consistency_error": 0.0,
            "consistency_threshold": float(detector.consistency_threshold),
            "score_gate_pass": False,
            "consistency_gate_pass": False,
            "adapter_update_enabled": bool(detector.adapter_update_enabled),
            "memory_update_enabled": bool(detector.memory_update_enabled),
            "device": detector.device,
            "score_method": "patch_nearest_normal",
            "feature_choice": detector.feature_choice,
            "patch_grid": detector.patch_grid,
            "patch_top_fraction": detector.patch_top_fraction,
        }

    return detector.predict(image, allow_update=True)


@app.route("/", methods=["GET"])
def home():
    # Return service status and category availability 
    return jsonify({
        "status": "OK",
        "message": "patch CTTA service is running",
        "experiment": get_experiment_settings()["experiment"],
        "available_categories": get_available_categories(),
        "loaded_categories": sorted(DETECTORS),
        "threshold_hot_reload": True,
    })


@app.route("/categories", methods=["GET"])
def categories():
    # Return available and already-loaded categories 
    return jsonify({
        "status": "OK",
        "available_categories": get_available_categories(),
        "loaded_categories": sorted(DETECTORS),
    })


@app.route("/config", methods=["GET"])
def config():
    # Return the active configuration for one category 
    try:
        category = safe_category_name(request.args.get("category", ""))

        if not category:
            return jsonify({
                "status": "OK",
                "available_categories": get_available_categories(),
                "loaded_categories": sorted(DETECTORS),
            })

        detector = load_detector_for_category(category)
        refreshed = refresh_threshold_if_changed(category, detector)

        return jsonify({
            "status": "OK",
            "experiment": get_experiment_settings()["experiment"],
            "category": category,
            "threshold": detector.threshold,
            "update_threshold": detector.update_threshold,
            "img_size": detector.img_size,
            "memory_size": int(detector.patch_memory_bank.shape[0]),
            "feature_choice": detector.feature_choice,
            "patch_grid": detector.patch_grid,
            "patch_top_fraction": detector.patch_top_fraction,
            "threshold_reloaded": refreshed,
        })
    except Exception as exc:
        return jsonify({"status": "ERROR", "message": str(exc)}), 500


@app.route("/predict", methods=["GET", "POST", "PUT"])
def predict():
    # Process one image and return the fixed CiRA-facing JSON schema 
    try:
        image_path = get_image_path_from_request()
        category = get_category_from_request(image_path)
        mode = get_mode_from_request()

        if not image_path:
            return jsonify({"status": "ERROR", "message": "No image_path received"}), 400

        if not category:
            return jsonify({"status": "ERROR", "message": "No category received"}), 400

        if not os.path.exists(image_path):
            return jsonify({
                "status": "ERROR",
                "message": "Image path does not exist",
                "image_path": image_path,
            }), 404

        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"status": "ERROR", "message": "OpenCV cannot read image"}), 400

        detector = load_detector_for_category(category)
        threshold_reloaded = refresh_threshold_if_changed(category, detector)

        event_log(
            f"predict_begin category={category} file={os.path.basename(image_path)} mode={mode}"
        )

        result = run_prediction(detector, image, mode)

        checkpoint_saved, checkpoint_memory_path, checkpoint_adapter_path = (
            save_state_if_needed(category, detector, result)
        )

        # Keep this schema stable for CiRA, local checkpoint paths stay in server logs.
        response = {
            "status": "OK",
            "message": "prediction completed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": get_experiment_settings()["experiment"],
            "category": category,
            "mode": mode,
            "image_path": image_path,
            "file_name": os.path.basename(image_path),
            "label": result.get("label", "unknown"),
            "prediction": result.get("label", "unknown"),
            "is_anomaly": bool(result.get("is_anomaly", False)),
            "score_before": float(result.get("score_before", -1)),
            "anomaly_score": float(result.get("anomaly_score", -1)),
            "threshold": float(result.get("threshold", detector.threshold)),
            "anomaly_threshold": float(
                result.get("anomaly_threshold", detector.threshold)
            ),
            "update_threshold": float(
                result.get("update_threshold", detector.update_threshold)
            ),
            "threshold_reloaded": bool(threshold_reloaded),
            "update_allowed": bool(result.get("update_allowed", False)),
            "updated_memory": bool(result.get("updated_memory", False)),
            "memory_size": int(result.get("memory_size", -1)),
            "update_loss": result.get("update_loss"),
            "adapter_updated": bool(result.get("adapter_updated", False)),
            "adapter_delta_norm": float(result.get("adapter_delta_norm", 0.0)),
            "device": str(result.get("device", "unknown")),
            "score_method": str(result.get("score_method", "")),
            "feature_choice": str(result.get("feature_choice", "")),
            "patch_grid": int(result.get("patch_grid", 0)),
            "patch_top_fraction": float(result.get("patch_top_fraction", 0.0)),
            "checkpoint_saved": bool(checkpoint_saved),
            "consistency_error": float(result.get("consistency_error", 0.0)),
            "consistency_threshold": float(result.get("consistency_threshold", 0.0)),
            "score_gate_pass": bool(result.get("score_gate_pass", False)),
            "consistency_gate_pass": bool(result.get("consistency_gate_pass", False)),
            "adapter_update_enabled": bool(
                result.get("adapter_update_enabled", detector.adapter_update_enabled)
            ),
            "memory_update_enabled": bool(
                result.get("memory_update_enabled", detector.memory_update_enabled)
            ),
        }

        # Keep checkpoint paths in the server log for traceability.
        log_record = dict(response)
        log_record["checkpoint_memory_path"] = checkpoint_memory_path
        log_record["checkpoint_adapter_path"] = checkpoint_adapter_path
        log_prediction(log_record)

        event_log(
            f"predict_response_ready category={category} "
            f"file={response['file_name']} "
            f"checkpoint_saved={checkpoint_saved} "
            f"updated_memory={response['updated_memory']} "
            f"adapter_updated={response['adapter_updated']} "
            f"adapter_delta_norm={response['adapter_delta_norm']:.8f} "
            f"update_loss={response['update_loss']}"
        )

        return jsonify(response)

    except Exception as exc:
        tb = traceback.format_exc()
        event_log("predict_error\n" + tb)
        return jsonify({
            "status": "ERROR",
            "message": str(exc),
            "traceback": tb,
        }), 500


if __name__ == "__main__":
    ensure_folders()
    event_log("service_start")
    print("=" * 60)
    print("Starting patch CTTA Flask service")
    print("EXPERIMENT:", get_experiment_settings()["experiment"])
    print("MODEL_ROOT:", MODEL_ROOT)
    print("Available categories:", get_available_categories())
    print("Server diagnostic log:", SERVER_EVENT_LOG)
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
