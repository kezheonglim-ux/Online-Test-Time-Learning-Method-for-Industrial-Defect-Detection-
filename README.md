# Online-Test-Time-Learning-Method-for-Industrial-Defect-Detection-
Modern manufacturing requires defect detection systems that are efficient, adaptive and capable of identifying unknown defects. This research proposes an online test-time learning method using a YOLO26 nano classification model within an open-ended anomaly detection pipeline. A low-code platform is also developed to support easier implementation and practical use.

DATASET
--------
https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library/data
- Total of 5354 images (Train: 3629; Test: 1725). About 70:30 ratio. 
- Having 15 types material including
    - Hazelnut: 501
    - Screw: 439
    - Pill: 434
    - Carpet: 397
    - Grid: 385
    - Zipper: 379
    - Cable: 374
    - Wood: 362
    - Capsule: 351
    - Tile: 350
    - Leather: 337
    - Metal_nut: 313
    - Transistor: 313
    - Bottle: 292
    - Toothbrush: 102

            
## WORKFLOWS
## Overall Workflow

### Process Flow

<pre>
Stage 1: Offline Preparation in Notebook
        ↓
Stage 2: Deployment Auto-Calibration
        ↓
Stage 3: CiRA CORE + Flask CTTA Deployment Workflow
</pre>

| Stage | Name | Main Purpose |
|-----|-----|-----|
| Stage 1 | Offline Preparation in Notebook | Prepare the initial normal representation and export model files |
| Stage 2 | Deployment Auto-Calibration | Adjust the anomaly and update thresholds using trusted normal deployment images |
| Stage 3 | CiRA CORE + Flask CTTA Deployment Workflow | Run the defect detection system through CiRA CORE and Flask API |


## Stage 1 — Offline Preparation in Notebook

### Purpose

Prepare the initial normal representation and model files before deployment.

### Process Flow

```text
MVTec AD train/good images
        ↓
Image preprocessing
        ↓
Frozen YOLO26 feature extraction
        ↓
Online adapter with identity initialization
        ↓
Build normal memory bank
        ↓
Use validation normal scores to calculate initial threshold
        ↓
Export deployment files:
    - yolo26n-cls.pt
    - ttl_adapter.pt
    - memory_bank.pt
    - threshold.json


### Explanation

The offline preparation from notebook here is to develop and prepare the initial anomaly detection model before deployment. A YOLO-based model is used as a frozen feature extractor, only a small adapter layer and the stored normal reference features are updated during online testing. This makes the system more stable and avoids the risk of changing the entire model during deployment.

Normal training images are passed through the YOLO feature extractor to obtain visual feature embeddings and these embeddings are stored in a normal memory bank as reference patterns of normal products. During validation, normal validation images are compared with this memory bank to calculate a distribution of normal anomaly scores. The initial anomaly threshold is then calculated from this normal-score distribution rather than being directly predicted by the model. 
End of this stage, four deployment files are generated and saved for later testing and deployment. 


| File | Purpose |
|---|---|
| `yolo26n-cls.pt` | Frozen YOLO feature extractor used to convert images into visual feature embeddings. |
| `ttl_adapter.pt` | Lightweight online adapter used to adjust the extracted feature embedding before comparison. |
| `memory_bank.pt` | Normal reference feature bank used to calculate the anomaly score. |
| `threshold.json` | Stores anomaly threshold and update threshold for decision-making and online update control. |
