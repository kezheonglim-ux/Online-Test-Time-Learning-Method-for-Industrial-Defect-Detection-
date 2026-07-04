# ============================================================
# Code Flow Summary: app_ctta.py
# ===============================================================================================================================================================
# | Part | Main Function            | Description                                                                                                               |
# |------|--------------------------|---------------------------------------------------------------------------------------------------------------------------|
# | 1    | Library Import           | Import Flask, OpenCV, PyTorch, JSON, CSV and system utilities used by the deployment API.                                 |
# | 2    | Configuration            | Define model folder, YOLO26 path, log folder, checkpoint folder and memory checkpoint interval.                           |
# | 3    | Model Import             | Add the model folder to Python path and import TTLAnomalyDetector from cira_ttl_anomaly.py.                               |
# | 4    | Flask Setup and Cache    | Create the Flask app and cache loaded category detectors to avoid reloading models for every image.                       |
# | 5    | Utility Functions        | Handle folder creation, path normalization, safe category naming, JSON reading, image path extraction and mode selection. |
# | 6    | Category-Aware Loading   | Load category-specific memory_bank.pt, ttl_adapter.pt and threshold.json based on the requested category.                 |
# | 7    | Logging and Checkpoint   | Save prediction records to CSV and optionally save memory-bank checkpoints after online updates.                          |
# | 8    | Prediction Mode Control  | Use evaluate/calibrate mode for score-only prediction and monitor mode for online test-time updating.                     |
# | 9    | API Routes               | Provide home, categories, config and predict endpoints for CiRA CORE and manual testing.                                  |
# | 10   | Main Runner              | Start the Flask service on 127.0.0.1:5000 for local CiRA CORE connection.                                                 |
# ===============================================================================================================================================================

from flask import Flask, jsonify, request
import os
import json
import cv2
import sys
import traceback
import csv
import torch
from datetime import datetime

# ============================================================
# Flask API for CiRA CORE + CTTA Anomaly Detection
# Category-aware version
#
# Model folder structure expected:
#
# C:\cira_ttl_model
#   ├── yolo26n-cls.pt
#   ├── cira_ttl_anomaly.py
#   ├── bottle
#   │   ├── memory_bank.pt
#   │   ├── threshold.json
#   │   └── ttl_adapter.pt
#   ├── cable
#   │   ├── memory_bank.pt
#   │   ├── threshold.json
#   │   └── ttl_adapter.pt
#   └── ...
#
# CiRA should send:
# {
#   "image_path": "C:/cira_batch_test/bottle/001.png",
#   "category": "bottle",
#   "mode": "evaluate"
# }
#
# mode:
# - evaluate  = score only, no online memory update
# - monitor   = normal deployment, allows online update
# - calibrate = score only, same as evaluate but named for calibration
# ============================================================


# ============================================================
# Configuration
# ============================================================

MODEL_ROOT = r"C:\cira_ttl_model"
YOLO_PATH = os.path.join(MODEL_ROOT, "yolo26n-cls.pt")

LOG_DIR = r"C:\cira_ttl_logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")

CHECKPOINT_ROOT = r"C:\cira_ttl_checkpoints"
SAVE_MEMORY_EVERY_N_UPDATES = 10

# Add model root so Python can import cira_ttl_anomaly.py.
# The main anomaly detection logic is kept in cira_ttl_anomaly.py.
sys.path.append(MODEL_ROOT)

from cira_ttl_anomaly import TTLAnomalyDetector


app = Flask(__name__)

# Cache loaded detectors so the same category model is not reloaded for every image.
# This improves deployment speed during batch image testing.
DETECTORS = {}
MODEL_CONFIGS = {}
MEMORY_UPDATE_COUNTERS = {}


# ============================================================
# Utility functions
# ============================================================

def ensure_folders():
    """Create log and checkpoint folders if not exist."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)


def normalize_path(path):
    """Normalize Windows path for OpenCV and Flask."""
    if path is None:
        return ""
    return str(path).strip().replace("\\", "/")


def safe_category_name(category):
    """
    Clean category name from request.
    This prevents path injection and keeps category folder name safe.
    """
    category = str(category).strip().lower()
    category = category.replace("\\", "").replace("/", "")
    category = category.replace("..", "")
    return category


def get_json_data():
    """
    Read JSON body from POST / PUT request.
    If request is GET, return empty dict.
    """
    if request.method in ["POST", "PUT"]:
        data = request.get_json(silent=True) or {}
        return data
    return {}


def get_image_path_from_request():
    """
    Supports:
    1. GET  /predict?image_path=C:/...
    2. POST/PUT JSON {"image_path": "C:/..."}
    3. CiRA strange URL style: /predict;image_path=C:/...
    """
    data = get_json_data()

    image_path = request.args.get("image_path", "")
    if not image_path:
        image_path = data.get("image_path", "")

    # Some CiRA versions may convert ? into ;
    if not image_path:
        raw_url = request.full_path
        if ";image_path=" in raw_url:
            image_path = raw_url.split(";image_path=", 1)[1]
            image_path = image_path.split("&", 1)[0]
            image_path = image_path.split("?", 1)[0]

    return normalize_path(image_path)


def get_category_from_request(image_path=""):
    """
    Get category from JSON/query first.
    If missing, try to infer from parent folder name.

    Example:
    C:/cira_batch_test/bottle/001.png
    → category = bottle
    """
    data = get_json_data()

    category = request.args.get("category", "")
    if not category:
        category = data.get("category", "")

    if not category and image_path:
        try:
            parent_folder = os.path.basename(os.path.dirname(image_path))
            category = parent_folder
        except:
            category = ""

    return safe_category_name(category)


def get_mode_from_request():
    """
    mode:
    - evaluate: no memory update, useful for debug checking
    - monitor: allows online update, useful for real deployment
    - calibrate: score only, useful for threshold calibration
    """
    data = get_json_data()

    mode = request.args.get("mode", "")
    if not mode:
        mode = data.get("mode", "evaluate")

    mode = str(mode).lower().strip()

    if mode not in ["evaluate", "monitor", "calibrate"]:
        mode = "evaluate"

    return mode


def get_available_categories():
    """Return category folders that contain required model files."""
    categories = []

    if not os.path.exists(MODEL_ROOT):
        return categories

    for name in sorted(os.listdir(MODEL_ROOT)):
        category_dir = os.path.join(MODEL_ROOT, name)

        if not os.path.isdir(category_dir):
            continue

        required = [
            os.path.join(category_dir, "threshold.json"),
            os.path.join(category_dir, "ttl_adapter.pt"),
            os.path.join(category_dir, "memory_bank.pt")
        ]

        if all(os.path.exists(p) for p in required):
            categories.append(name)

    return categories


# ============================================================
# Category-aware model loading
# ============================================================

def load_detector_for_category(category):
    """
    Load detector for a specific category.
    Example:
    category = bottle
    load:
    C:\cira_ttl_model\bottle\threshold.json
    C:\cira_ttl_model\bottle\ttl_adapter.pt
    C:\cira_ttl_model\bottle\memory_bank.pt
    """
    category = safe_category_name(category)

    if not category:
        raise ValueError("Empty category received.")

    if category in DETECTORS:
        return DETECTORS[category]

    category_dir = os.path.join(MODEL_ROOT, category)

    threshold_path = os.path.join(category_dir, "threshold.json")
    adapter_path = os.path.join(category_dir, "ttl_adapter.pt")
    memory_bank_path = os.path.join(category_dir, "memory_bank.pt")
    yolo_path = YOLO_PATH

    required_files = [
        threshold_path,
        adapter_path,
        memory_bank_path,
        yolo_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError("Required file not found: {}".format(file_path))

    with open(threshold_path, "r") as f:
        cfg = json.load(f)

    detector = TTLAnomalyDetector(
        adapter_path=adapter_path,
        memory_bank_path=memory_bank_path,
        threshold=cfg.get("threshold", 999.0),
        model_name=yolo_path,
        img_size=cfg.get("img_size", 224),

        top_k_references=cfg.get("top_k_references", 5),
        reference_weight=cfg.get("reference_weight", 0.7),
        global_weight=cfg.get("global_weight", 0.3),
        accept_margin=cfg.get("accept_margin", 0.95),
        update_threshold=cfg.get("update_threshold", None),

        online_lr=cfg.get("online_lr", 1e-4),
        max_memory_bank=cfg.get("max_memory_bank", 4000),
        online_steps=cfg.get("online_steps", 1),
        consistency_weight=cfg.get("consistency_weight", 1.0),
        anchor_weight=cfg.get("anchor_weight", 0.1)
    )

    DETECTORS[category] = detector
    MODEL_CONFIGS[category] = cfg
    MEMORY_UPDATE_COUNTERS[category] = 0

    print("=" * 60)
    print("Loaded detector for category:", category)
    print("Category folder:", category_dir)
    print("Threshold:", cfg.get("threshold", "not found"))
    print("Update threshold:", cfg.get("update_threshold", "not found"))
    print("Image size:", cfg.get("img_size", 224))
    print("=" * 60)

    return detector


# ============================================================
# Logging and checkpoint
# ============================================================

def log_prediction(record):
    """Save prediction record to CSV for debugging and thesis evidence."""
    ensure_folders()

    file_exists = os.path.exists(LOG_FILE)

    fieldnames = [
        "timestamp",
        "category",
        "mode",
        "image_path",
        "file_name",
        "label",
        "is_anomaly",
        "score_before",
        "anomaly_score",
        "threshold",
        "anomaly_threshold",
        "update_threshold",
        "updated_memory",
        "memory_size",
        "update_loss",
        "device"
    ]

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {}
        for key in fieldnames:
            row[key] = record.get(key, "")

        writer.writerow(row)


def save_memory_checkpoint_if_needed(category, detector, result):
    """
    Save memory bank checkpoint after every N memory updates.
    This is optional but useful for deployment traceability.
    """
    if not result.get("updated_memory", False):
        return None

    if category not in MEMORY_UPDATE_COUNTERS:
        MEMORY_UPDATE_COUNTERS[category] = 0

    MEMORY_UPDATE_COUNTERS[category] += 1
    counter = MEMORY_UPDATE_COUNTERS[category]

    if counter % SAVE_MEMORY_EVERY_N_UPDATES != 0:
        return None

    ensure_folders()

    category_checkpoint_dir = os.path.join(CHECKPOINT_ROOT, category)
    os.makedirs(category_checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        category_checkpoint_dir,
        "memory_bank_update_{}.pt".format(counter)
    )

    # If cira_ttl_anomaly.py has save_memory_bank(), use it.
    # If not, save memory_bank directly.
    if hasattr(detector, "save_memory_bank"):
        detector.save_memory_bank(checkpoint_path)
    else:
        torch.save(detector.memory_bank.cpu(), checkpoint_path)

    return checkpoint_path


# ============================================================
# Prediction helper
# ============================================================

def run_prediction(detector, img, mode):
    """
    Run prediction according to mode.

    evaluate:
        no online update. Uses score_only().
    calibrate:
        same as evaluate. Useful for collecting normal scores.
    monitor:
        uses detector.predict(), which may update adapter and memory bank
        if the image is normal-like.
    """
    if mode in ["evaluate", "calibrate"]:
        score = float(detector.score_only(img))
        is_anomaly = score >= float(detector.threshold)

        result = {
            "label": "anomaly" if is_anomaly else "normal",
            "is_anomaly": bool(is_anomaly),

            "anomaly_score": score,
            "score_before": score,

            "threshold": float(detector.threshold),
            "anomaly_threshold": float(detector.threshold),
            "update_threshold": float(getattr(detector, "update_threshold", -1)),

            "updated_memory": False,
            "memory_size": int(detector.memory_bank.shape[0]),
            "update_loss": None,

            "device": str(getattr(detector, "device", "unknown")),
            "top_k_references": int(getattr(detector, "top_k_references", -1)),
            "reference_weight": float(getattr(detector, "reference_weight", -1)),
            "global_weight": float(getattr(detector, "global_weight", -1)),
            "accept_margin": float(getattr(detector, "accept_margin", -1))
        }

        return result

    # monitor mode: allow online test-time update
    # This uses your detector's existing predict() function.
    result = detector.predict(img)

    return result


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "message": "Category-aware CTTA Flask service is running",
        "model_root": MODEL_ROOT,
        "yolo_path": YOLO_PATH,
        "available_categories": get_available_categories(),
        "loaded_categories": sorted(list(DETECTORS.keys()))
    })


@app.route("/categories", methods=["GET"])
def categories():
    return jsonify({
        "status": "OK",
        "available_categories": get_available_categories(),
        "loaded_categories": sorted(list(DETECTORS.keys()))
    })


@app.route("/config", methods=["GET"])
def config():
    category = request.args.get("category", "")

    if category:
        category = safe_category_name(category)

        if category not in DETECTORS:
            detector = load_detector_for_category(category)
        else:
            detector = DETECTORS[category]

        cfg = MODEL_CONFIGS.get(category, {})

        return jsonify({
            "status": "OK",
            "category": category,
            "threshold": float(getattr(detector, "threshold", -1)),
            "update_threshold": float(getattr(detector, "update_threshold", -1)),
            "img_size": int(getattr(detector, "img_size", -1)),
            "memory_size": int(detector.memory_bank.shape[0]),
            "raw_threshold_json": cfg
        })

    return jsonify({
        "status": "OK",
        "message": "No category specified. Showing service summary.",
        "available_categories": get_available_categories(),
        "loaded_categories": sorted(list(DETECTORS.keys()))
    })


@app.route("/predict", methods=["GET", "POST", "PUT"])
def predict():
    try:
        image_path = get_image_path_from_request()
        category = get_category_from_request(image_path)
        mode = get_mode_from_request()

        if not image_path:
            return jsonify({
                "status": "ERROR",
                "message": "No image_path received",
                "full_path": request.full_path
            }), 400

        if not category:
            return jsonify({
                "status": "ERROR",
                "message": "No category received and cannot infer from image_path",
                "image_path": image_path
            }), 400

        if not os.path.exists(image_path):
            return jsonify({
                "status": "ERROR",
                "message": "Image path does not exist",
                "image_path": image_path
            }), 404

        img = cv2.imread(image_path)

        if img is None:
            return jsonify({
                "status": "ERROR",
                "message": "OpenCV cannot read image",
                "image_path": image_path
            }), 400

        # Load the correct category detector based on the folder/category name.
        # This loads the matching memory bank, adapter and threshold files.
        detector = load_detector_for_category(category)

        # Run anomaly detection according to the selected mode.
        # evaluate/calibrate = score only; monitor = allow online update.
        result = run_prediction(detector, img, mode)

        file_name = os.path.basename(image_path)

        response = {
            "status": "OK",
            "message": "CTTA prediction completed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "category": category,
            "mode": mode,

            "image_path": image_path,
            "file_name": file_name,
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),

            "label": result.get("label", "unknown"),
            "prediction": result.get("label", "unknown"),
            "is_anomaly": bool(result.get("is_anomaly", False)),

            "score_before": float(result.get("score_before", -1)),
            "anomaly_score": float(result.get("anomaly_score", -1)),

            "threshold": float(result.get("threshold", -1)),
            "anomaly_threshold": float(result.get("anomaly_threshold", result.get("threshold", -1))),
            "update_threshold": float(result.get("update_threshold", -1)),

            "updated_memory": bool(result.get("updated_memory", False)),
            "memory_size": int(result.get("memory_size", -1)),
            "update_loss": result.get("update_loss", None),

            "device": result.get("device", "unknown"),

            "top_k_references": result.get("top_k_references", None),
            "reference_weight": result.get("reference_weight", None),
            "global_weight": result.get("global_weight", None),
            "accept_margin": result.get("accept_margin", None)
        }

        checkpoint_path = save_memory_checkpoint_if_needed(category, detector, response)
        response["checkpoint_saved"] = checkpoint_path is not None
        response["checkpoint_path"] = checkpoint_path

        log_prediction(response)

        return jsonify(response)

    except Exception as e:
        error_traceback = traceback.format_exc()

        print("ERROR during prediction:")
        print(error_traceback)

        return jsonify({
            "status": "ERROR",
            "message": str(e),
            "traceback": error_traceback,
            "full_path": request.full_path
        }), 500


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ensure_folders()

    print("=" * 60)
    print("Starting category-aware CTTA Flask service")
    print("MODEL_ROOT:", MODEL_ROOT)
    print("YOLO_PATH:", YOLO_PATH)
    print("Available categories:", get_available_categories())
    print("=" * 60)

    app.run(host="127.0.0.1", port=5000, debug=False)