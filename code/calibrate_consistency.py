import pandas as pd
import numpy as np
import json
from pathlib import Path

LOG_FILE = Path(r"C:\cira_ttl_logs\prediction_log.csv")
MODEL_ROOT = Path(r"C:\cira_ttl_model")

df = pd.read_csv(LOG_FILE)

# Keep trusted-good images only
df = df[df["file_name"].str.startswith("good_", na=False)]

for category, group in df.groupby("category"):

    errors = group["consistency_error"].dropna().astype(float)

    if len(errors) == 0:
        print(category, "NO DATA")
        continue

    threshold = float(np.quantile(errors, 0.95))

    json_path = MODEL_ROOT / category / "threshold.json"

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["consistency_threshold"] = threshold

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    print(
        category,
        "samples =", len(errors),
        "consistency_threshold =", threshold
    )