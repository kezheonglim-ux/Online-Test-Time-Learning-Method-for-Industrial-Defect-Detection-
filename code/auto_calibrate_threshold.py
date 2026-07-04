# ============================================================
# Code Flow Summary: auto_calibrate_threshold.py
# ============================================================
# | Part | Main Function | Description |
# |---|---|---|
# | 1 | Library Import | Import file handling, JSON, OpenCV, NumPy and system path utilities. |
# | 2 | Path Configuration | Define the deployed model folder and the trusted normal calibration image folder. |
# | 3 | Model Import | Import TTLAnomalyDetector from cira_ttl_anomaly.py for score-only anomaly scoring. |
# | 4 | Score Calculation Function | Read each normal calibration image and calculate its anomaly score without online update. |
# | 5 | Load Existing Threshold Config | Read threshold.json to reuse the existing model parameters and scoring settings. |
# | 6 | Detector Initialization | Load YOLO26 feature extractor, adapter, memory bank and current threshold settings. |
# | 7 | Normal Score Collection | Calculate anomaly scores from trusted normal deployment images. |
# | 8 | Dual Threshold Calibration | Calculate a stricter update threshold and a high-percentile anomaly threshold. |
# | 9 | Safety Margin | Add a small margin to the anomaly threshold based on normal score variation. |
# | 10 | Save Updated Config | Write the calibrated threshold, update threshold and calibration statistics back to threshold.json. |
# ============================================================

import os
import json
import cv2
import sys
import numpy as np

# ============================================================
# Path configuration
# ============================================================

# Folder containing the deployed YOLO26 model, adapter, memory bank and threshold.json.
MODEL_DIR = r"C:\cira_ttl_model"

# Folder containing trusted normal images collected from the deployment condition.
# These images are used only for threshold calibration, not for defect training.
NORMAL_DIR = r"C:\cira_ttl_calibration\bottle\normal"

# Allow Python to import cira_ttl_anomaly.py from the model folder.
sys.path.append(MODEL_DIR)

from cira_ttl_anomaly import TTLAnomalyDetector


# ============================================================
# Calculate normal calibration scores
# ============================================================

def calculate_scores(detector, folder):
    """
    Calculate anomaly scores for trusted normal calibration images.

    The detector uses score_only(), so no online adapter update and no
    memory-bank update are performed during calibration.
    """
    scores = []
    records = []

    for name in os.listdir(folder):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(folder, name)
            img = cv2.imread(image_path)

            if img is None:
                print("Cannot read:", image_path)
                continue

            # Score-only mode keeps the detector fixed during calibration.
            score = detector.score_only(img)
            scores.append(score)
            records.append((name, score))

    return np.array(scores, dtype=float), records


# ============================================================
# Load existing threshold configuration
# ============================================================

# Read the existing threshold.json so the calibration uses the same
# image size, Top-K, score weights and online-update settings as deployment.
with open(os.path.join(MODEL_DIR, "threshold.json"), "r") as f:
    cfg = json.load(f)


# ============================================================
# Initialize detector
# ============================================================

# Load the same detector components used during deployment:
# frozen YOLO26 feature extractor, online adapter, memory bank and threshold.
detector = TTLAnomalyDetector(
    adapter_path=os.path.join(MODEL_DIR, "ttl_adapter.pt"),
    memory_bank_path=os.path.join(MODEL_DIR, "memory_bank.pt"),
    threshold=cfg.get("threshold", 999),
    model_name=os.path.join(MODEL_DIR, "yolo26n-cls.pt"),
    img_size=cfg.get("img_size", 224),

    top_k_references=cfg.get("top_k_references", 5),
    reference_weight=cfg.get("reference_weight", 0.7),
    global_weight=cfg.get("global_weight", 0.3),
    accept_margin=cfg.get("accept_margin", 0.95),

    update_threshold=cfg.get("update_threshold", None),

    online_lr=cfg.get("online_lr", 1e-4),
    max_memory_bank=cfg.get("max_memory_bank", 4000)
)


# ============================================================
# Collect scores from trusted normal images
# ============================================================

normal_scores, records = calculate_scores(detector, NORMAL_DIR)

if len(normal_scores) < 3:
    raise ValueError("Too few normal calibration images. Please add more normal images.")


# ============================================================
# Calculate calibrated thresholds
# ============================================================

# Anomaly threshold:
# high percentile of normal deployment scores.
# Images with scores above this threshold are treated as anomalies.
ANOMALY_QUANTILE = 0.995

# Update threshold:
# lower percentile used as a stricter boundary for online memory-bank updates.
# Only very normal-like samples should be allowed to update the memory bank.
UPDATE_QUANTILE = 0.95

anomaly_threshold = float(np.quantile(normal_scores, ANOMALY_QUANTILE))
update_threshold = float(np.quantile(normal_scores, UPDATE_QUANTILE))

# Add a small safety margin to reduce false alarms caused by normal score variation.
safety_margin = float(0.1 * np.std(normal_scores))
anomaly_threshold_with_margin = anomaly_threshold + safety_margin


# ============================================================
# Print calibration result for checking
# ============================================================

print("\n===== Normal calibration scores =====")
for name, score in records:
    print(f"{name}: {score:.6f}")

print("\n===== Calibration summary =====")
print("Count:", len(normal_scores))
print("Min:", float(normal_scores.min()))
print("Mean:", float(normal_scores.mean()))
print("Std:", float(normal_scores.std()))
print("Max:", float(normal_scores.max()))
print("Anomaly threshold before margin:", anomaly_threshold)
print("Safety margin:", safety_margin)
print("Final anomaly threshold:", anomaly_threshold_with_margin)
print("Update threshold:", update_threshold)


# ============================================================
# Save updated threshold.json
# ============================================================

# Keep the original threshold for traceability and save the calibrated values.
cfg["threshold_original"] = cfg.get("threshold")
cfg["threshold"] = anomaly_threshold_with_margin
cfg["update_threshold"] = update_threshold

# Store calibration method and statistics for deployment record.
cfg["threshold_method"] = "deployment_auto_calibrated_dual_threshold"
cfg["anomaly_quantile"] = ANOMALY_QUANTILE
cfg["update_quantile"] = UPDATE_QUANTILE
cfg["safety_margin"] = safety_margin

cfg["normal_score_count"] = int(len(normal_scores))
cfg["normal_score_min"] = float(normal_scores.min())
cfg["normal_score_mean"] = float(normal_scores.mean())
cfg["normal_score_std"] = float(normal_scores.std())
cfg["normal_score_max"] = float(normal_scores.max())

with open(os.path.join(MODEL_DIR, "threshold.json"), "w") as f:
    json.dump(cfg, f, indent=4)

print("\nUpdated threshold.json saved.")
