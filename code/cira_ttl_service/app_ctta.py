from flask import Flask, jsonify, request
import os
import json
import cv2
import sys
import traceback
import csv
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

MODEL_DIR = r"C:\cira_ttl_model"

sys.path.append(MODEL_DIR)

from cira_ttl_anomaly import TTLAnomalyDetector


app = Flask(__name__)

detector = None
MODEL_CONFIG = {}

LOG_DIR = r"C:\cira_ttl_logs"
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.csv")
CHECKPOINT_DIR = r"C:\cira_ttl_checkpoints"

SAVE_MEMORY_EVERY_N_UPDATES = 10
memory_update_counter = 0

# ============================================================
# Helper function
# ============================================================

def ensure_deployment_folders():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log_prediction(record):
    ensure_deployment_folders()

    file_exists = os.path.exists(LOG_FILE)

    fieldnames = [
        "timestamp",
        "mode",
        "image_path",
        "label",
        "is_anomaly",
        "score_before",
        "anomaly_score",
        "anomaly_threshold",
        "update_threshold",
        "allow_update",
        "updated_memory",
        "memory_size",
        "update_loss",
        "device"
    ]

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": record.get("timestamp"),
            "mode": record.get("mode"),
            "image_path": record.get("image_path"),
            "label": record.get("label"),
            "is_anomaly": record.get("is_anomaly"),
            "score_before": record.get("score_before"),
            "anomaly_score": record.get("anomaly_score"),
            "anomaly_threshold": record.get("anomaly_threshold"),
            "update_threshold": record.get("update_threshold"),
            "allow_update": record.get("allow_update"),
            "updated_memory": record.get("updated_memory"),
            "memory_size": record.get("memory_size"),
            "update_loss": record.get("update_loss"),
            "device": record.get("device")
        })


def save_memory_checkpoint_if_needed(result):
    global memory_update_counter

    if not result.get("updated_memory", False):
        return None

    memory_update_counter += 1

    if memory_update_counter % SAVE_MEMORY_EVERY_N_UPDATES != 0:
        return None

    ensure_deployment_folders()

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"memory_bank_update_{memory_update_counter}.pt"
    )

    detector.save_memory_bank(checkpoint_path)

    return checkpoint_path

# ============================================================
# Load CTTA detector
# ============================================================

def load_detector():
    global detector

    threshold_path = os.path.join(MODEL_DIR, "threshold.json")
    adapter_path = os.path.join(MODEL_DIR, "ttl_adapter.pt")
    memory_bank_path = os.path.join(MODEL_DIR, "memory_bank.pt")
    yolo_path = os.path.join(MODEL_DIR, "yolo26n-cls.pt")

    required_files = [
        threshold_path,
        adapter_path,
        memory_bank_path,
        yolo_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    global MODEL_CONFIG
    
    with open(threshold_path, "r") as f:
        cfg = json.load(f)
    
    MODEL_CONFIG = cfg
    
    print("Loaded category:", cfg.get("category", "not_saved"))
    print("Loaded threshold:", cfg.get("threshold"))
    print("Loaded update threshold:", cfg.get("update_threshold", "not_saved"))
    print("Loaded img_size:", cfg.get("img_size", 224))
    print("Top-K references:", cfg.get("top_k_references", 5))
    print("Reference weight:", cfg.get("reference_weight", 0.7))
    print("Global weight:", cfg.get("global_weight", 0.3))
    print("Accept margin:", cfg.get("accept_margin", 0.95))

    detector = TTLAnomalyDetector(
        adapter_path=adapter_path,
        memory_bank_path=memory_bank_path,
        threshold=cfg["threshold"],
        model_name=yolo_path,
        img_size=cfg.get("img_size", 224),
    
        top_k_references=cfg.get("top_k_references", 5),
        reference_weight=cfg.get("reference_weight", 0.7),
        global_weight=cfg.get("global_weight", 0.3),
        accept_margin=cfg.get("accept_margin", 0.95),
        update_threshold=cfg.get("update_threshold", None),
    
        online_lr=cfg.get("online_lr", 1e-4),
        max_memory_bank=cfg.get("max_memory_bank", 4000)
    )

    print("CTTA detector loaded successfully.")


# ============================================================
# Read image_path from request
# Supports:
# 1. /predict?image_path=C:/...
# 2. /predict;image_path=C:/...
# 3. POST JSON {"image_path": "C:/..."}
# ============================================================

def get_image_path_from_request():
    image_path = ""

    # Normal GET query format:
    # /predict?image_path=C:/...
    image_path = request.args.get("image_path", "")

    # POST JSON format:
    # {"image_path": "C:/..."}
    if not image_path and request.method == "POST":
        data = request.get_json(silent=True) or {}
        image_path = data.get("image_path", "")

    # CiRA may convert ? into ; in the URL:
    # /predict;image_path=C:/...
    if not image_path:
        raw_url = request.full_path

        if ";image_path=" in raw_url:
            image_path = raw_url.split(";image_path=", 1)[1]
            image_path = image_path.split("&", 1)[0]
            image_path = image_path.split("?", 1)[0]

    image_path = image_path.strip()
    image_path = image_path.replace("\\", "/")

    return image_path

def get_mode_from_request():
    mode = request.args.get("mode", "monitor")

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", mode)

    mode = str(mode).lower().strip()

    if mode not in ["monitor", "evaluate", "calibrate"]:
        mode = "monitor"

    return mode

# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "OK",
        "message": "CTTA Flask service is running",
        "detector_loaded": detector is not None,
        "model_dir": MODEL_DIR
    })

@app.route("/config", methods=["GET"])
def config():
    if detector is None:
        return jsonify({
            "status": "ERROR",
            "message": "Detector is not loaded"
        }), 500

    return jsonify({
        "status": "OK",
        "message": "Current Flask CTTA configuration",
        "model_dir": MODEL_DIR,
        "category": MODEL_CONFIG.get("category", "not_saved"),
        "threshold": float(getattr(detector, "threshold", -1)),
        "update_threshold": float(getattr(detector, "update_threshold", -1)),
        "log_file": LOG_FILE,
        "checkpoint_dir": CHECKPOINT_DIR,
        "save_memory_every_n_updates": SAVE_MEMORY_EVERY_N_UPDATES,
        "img_size": int(getattr(detector, "img_size", -1)),
        "top_k_references": int(getattr(detector, "top_k_references", -1)),
        "reference_weight": float(getattr(detector, "reference_weight", -1)),
        "global_weight": float(getattr(detector, "global_weight", -1)),
        "accept_margin": float(getattr(detector, "accept_margin", -1)),
        "online_lr": MODEL_CONFIG.get("online_lr", "not_saved"),
        "max_memory_bank": int(getattr(detector, "max_memory_bank", -1)),
        "memory_size": int(detector.memory_bank.shape[0]),
        "raw_threshold_json": MODEL_CONFIG
    })

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if detector is None:
            return jsonify({
                "status": "ERROR",
                "message": "Detector is not loaded"
            }), 500

        image_path = get_image_path_from_request()
        mode = get_mode_from_request()

        if not image_path:
            return jsonify({
                "status": "ERROR",
                "message": "No image_path received",
                "full_path": request.full_path
            }), 400

        if not os.path.exists(image_path):
            return jsonify({
                "status": "ERROR",
                "message": "Image path does not exist",
                "image_path": image_path,
                "full_path": request.full_path
            }), 404

        img = cv2.imread(image_path)

        if img is None:
            return jsonify({
                "status": "ERROR",
                "message": "OpenCV cannot read image",
                "image_path": image_path
            }), 400

        if mode == "monitor":
            result = detector.predict(img, allow_update=True)

        elif mode == "evaluate":
            result = detector.predict(img, allow_update=False)

        elif mode == "calibrate":
            score = detector.score_only(img)
            is_anomaly = score >= detector.threshold

            result = {
                "label": "anomaly" if is_anomaly else "normal",
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": score,
                "score_before": score,
                "threshold": detector.threshold,
                "anomaly_threshold": detector.threshold,
                "update_threshold": detector.update_threshold,
                "allow_update": False,
                "updated_memory": False,
                "memory_size": int(detector.memory_bank.shape[0]),
                "update_loss": None,
                "device": "calibration_score_only",
                "top_k_references": detector.top_k_references,
                "reference_weight": detector.reference_weight,
                "global_weight": detector.global_weight,
                "accept_margin": detector.accept_margin
            }

        response = {
            "status": "OK",
            "message": "CTTA prediction completed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "mode": mode,
            "image_path": image_path,
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),

            "label": result.get("label", "unknown"),
            "is_anomaly": bool(result.get("is_anomaly", False)),

            "score_before": float(result.get("score_before", -1)),
            "anomaly_score": float(result.get("anomaly_score", -1)),

            "threshold": float(result.get("threshold", -1)),
            "anomaly_threshold": float(result.get("anomaly_threshold", result.get("threshold", -1))),
            "update_threshold": float(result.get("update_threshold", -1)),

            "allow_update": bool(result.get("allow_update", False)),
            "updated_memory": bool(result.get("updated_memory", False)),
            "memory_size": int(result.get("memory_size", -1)),
            "update_loss": result.get("update_loss", None),

            "device": result.get("device", "unknown"),

            "top_k_references": result.get("top_k_references", None),
            "reference_weight": result.get("reference_weight", None),
            "global_weight": result.get("global_weight", None),
            "accept_margin": result.get("accept_margin", None)
        }

        checkpoint_path = save_memory_checkpoint_if_needed(response)
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
    load_detector()
    app.run(host="127.0.0.1", port=5000, debug=False)