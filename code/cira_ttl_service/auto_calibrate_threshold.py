import os
import json
import cv2
import sys
import numpy as np

MODEL_DIR = r"C:\cira_ttl_model"
NORMAL_DIR = r"C:\cira_ttl_calibration\bottle\normal"

sys.path.append(MODEL_DIR)

from cira_ttl_anomaly import TTLAnomalyDetector


def calculate_scores(detector, folder):
    scores = []
    records = []

    for name in os.listdir(folder):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(folder, name)
            img = cv2.imread(image_path)

            if img is None:
                print("Cannot read:", image_path)
                continue

            score = detector.score_only(img)
            scores.append(score)
            records.append((name, score))

    return np.array(scores, dtype=float), records


with open(os.path.join(MODEL_DIR, "threshold.json"), "r") as f:
    cfg = json.load(f)

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

normal_scores, records = calculate_scores(detector, NORMAL_DIR)

if len(normal_scores) < 3:
    raise ValueError("Too few normal calibration images. Please add more normal images.")

ANOMALY_QUANTILE = 0.995
UPDATE_QUANTILE = 0.95

anomaly_threshold = float(np.quantile(normal_scores, ANOMALY_QUANTILE))
update_threshold = float(np.quantile(normal_scores, UPDATE_QUANTILE))

safety_margin = float(0.1 * np.std(normal_scores))
anomaly_threshold_with_margin = anomaly_threshold + safety_margin

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

cfg["threshold_original"] = cfg.get("threshold")
cfg["threshold"] = anomaly_threshold_with_margin
cfg["update_threshold"] = update_threshold

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