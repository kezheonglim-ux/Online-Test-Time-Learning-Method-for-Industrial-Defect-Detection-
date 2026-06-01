# Online-Test-Time-Learning-Method-for-Industrial-Defect-Detection-
Modern manufacturing requires defect detection systems that are efficient, adaptive and capable of identifying unknown defects. This research proposes an online test-time learning method using a YOLO26 nano classification model within an open-ended anomaly detection pipeline. A low-code platform is also developed to support easier implementation and practical use.

## 1.0 DATASET
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

            
## 2.0 WORKFLOWS
## 2.1 Overall Workflow

### 2.1.1 Process Flow

<pre>
Stage 1: Offline Preparation in Notebook
        ↓
Stage 2: Deployment Auto-Calibration
        ↓
Stage 3: CiRA CORE + Flask CTTA Deployment Workflow
</pre>

### 2.1.2 Purpose

| Stage | Name | Main Purpose |
|---|---|---|
| Stage&nbsp;1 | Offline Preparation in Notebook | Prepare the initial normal representation and export model files |
| Stage&nbsp;2 | Deployment Auto-Calibration | Adjust the anomaly and update thresholds using trusted normal deployment images |
| Stage&nbsp;3 | CiRA CORE + Flask CTTA Deployment Workflow | Run the defect detection system through CiRA CORE and Flask API |


## 2.2 Stage 1: Offline Preparation in Notebook

### 2.2.1 Purpose

Prepare the initial normal representation and model files before deployment.

### 2.2.2 Process Flow

<pre>
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
</pre>

### 2.2.3 Description

The offline preparation from notebook here is to develop and prepare the initial anomaly detection model before deployment. A YOLO-based model is used as a frozen feature extractor, only a small adapter layer and the stored normal reference features are updated during online testing. This makes the system more stable and avoids the risk of changing the entire model during deployment.

Normal training images are passed through the YOLO feature extractor to obtain visual feature embeddings and these embeddings are stored in a normal memory bank as reference patterns of normal products. During validation, normal validation images are compared with this memory bank to calculate a distribution of normal anomaly scores. The initial anomaly threshold is then calculated from this normal-score distribution rather than being directly predicted by the model. 
End of this stage, four deployment files are generated and saved for later testing and deployment. 

### 2.2.4 Files in use

| File | Purpose |
|---|---|
| `train_rev1.3.ipynb` | Dataset preparation, feature extraction, memory bank construction, initial threshold calibration, evalution and export of model files |

### 2.2.5 Exported Files

| File | Purpose |
|---|---|
| `yolo26n-cls.pt` | Provides the frozen YOLO feature extractor. It converts each incoming image into a visual feature embedding without retraining the YOLO model. |
| `ttl_adapter.pt` | Stores the lightweight online adapter. The extracted feature embedding is passed through this adapter before comparison with the memory bank. |
| `memory_bank.pt` | Stores the normal reference feature embeddings generated from normal training images. Incoming images are compared with this memory bank to calculate the anomaly score. |
| `threshold.json` | Stores the calibrated decision settings, including the anomaly threshold and update threshold. These settings are used to classify the image and control online updating. |

## 2.3 Stage 2: Deployment Auto-Calibration

### 2.3.1 Purpose

Adjust the decision thresholds to match the deployment image condition.

### 2.3.2 Process Flow

<pre>
Trusted normal deployment images
        ↓
score_only() calculates normal deployment scores
        ↓
Calculate anomaly_threshold from high normal-score quantile
        ↓
Calculate update_threshold from stricter normal-score quantile
        ↓
Update threshold.json automatically
</pre>

### 2.3.3 Description

Deployment auto-calibration is included to adjust the decision thresholds before the system is used in the CiRA CORE workflow. Although the notebook generates an initial threshold from validation normal images, the actual deployment images may have different lighting, image quality, background, capture angle or image source. Therefore, a small set of trusted normal images from the target deployment category is used to calculate a more suitable `anomaly_threshold` and `update_threshold`. 

The `score_only()` function calculates anomaly scores without updating the adapter or memory bank. These scores are then used to generate the `anomaly_threshold` and `update_threshold`. The `anomaly_threshold` is used to decide whether an image should be classified as normal or anomalous. The `update_threshold` is used to decide whether an image is confidently normal and suitable for online updating. After calibration, the updated threshold values are saved into `threshold.json`.

In the current implementation, about 5 normal bottle images from the testing folder are selected and treated as deployment-condition normal samples for demonstration. The bottle category is used as the example category in this stage. If additional categories are deployed in the future, the same calibration process should be repeated separately for each category because each product category has its own normal feature distribution and decision boundary.

### 2.3.4 Files in use

| File | Purpose |
|---|---|
| `auto_calibrate_threshold.py` | Deployment threshold calibration. Calibrated anomaly threshold and update threshold. Trusted normal deployment images used to calibrate threshold under testing folder stored in `workdir\cira_ttl_calibration` |

### 2.3.5 Output

| Output File | Purpose |
|---|---|
| `threshold.json` | Stores the updated `anomaly_threshold`, `update_threshold` and calibration settings for deployment |


## 2.4 Stage 3: CiRA CORE + Flask CTTA Deployment Workflow

### 2.4.1 Purpose

Run the defect detection system through a low-code workflow.

### 2.4.2 Process Flow

<pre>
CiRA CORE triggers inspection
        ↓
RestGetJson sends image path to Flask CTTA API
        ↓
Flask loads the image and calls the CTTA detector
        ↓
CTTA detector extracts YOLO26 features and compares with memory bank
        ↓
Calculate anomaly score and apply dual-threshold decision
        ↓
If confidently normal:
    update adapter and memory bank
Else:
    no online update
        ↓
Return JSON result to CiRA CORE
        ↓
Display result and save prediction log / memory checkpoint
</pre>

###  2.4.3 Description

In this stage, the prepared anomaly detection workflow is deployed through CiRA CORE and the Flask CTTA service. CiRA CORE acts as the low-code workflow interface, while Flask serves as the bridge to the Python-based CTTA detector by loading the exported model files, running the detection process, and returning the prediction result to CiRA CORE.

The CiRA CORE `RestGetJson` node sends the image path to the Flask API. Flask then reads the image and calls the CTTA detector. The detector extracts YOLO26 visual features, compares them with the normal memory bank, calculates the anomaly score and applies the dual-threshold decision logic.

If the image is confidently normal, the lightweight online adapter and memory bank are updated during online test-time learning. If the image is not confidently normal, no online update is performed. The final prediction result is returned to CiRA CORE in JSON format and can be displayed through the Debug node or dashboard. Prediction logs and memory checkpoints are also saved for traceability.

### 2.4.4 Files in use

| File | Purpose |
|---|---|
| `app_ctta.py` | Flask service file, bridge between CiRA CORE and the Python-based CTTA detector. It load the export files output from `train.ipynb`, receiving image oath from CiRA CORE, running prediction through the CTTA detector, supporting `monitor`/`evalute`/`calibrate` modes, logging predictions to CSV, saving memory checkpoints and returning JSON result to CiRA CORE|
| `cira_ttl_anomaly.py` | CTTA detector, contain real detection logic used during testing and deployment. It performs two key functions: feature extraction and anomaly score calculation. |
|| Feature extraction converts the input image into a numerical feature embedding so that it can be compared with normal reference features. |
|| Anomaly score calculation used to measures how different the incoming image is from the stored normal references. If the image is less similar to the memory bank, the anomaly score becomes higher. |

### 2.4.5 Output

| Output File | Purpose |
|---|---|
| JSON prediction result | Returned to CiRA CORE and shows prediction result for each image displayed in the Debug node or dashboard  |
| `prediction_log.csv` | Stores prediction history, including image path, score, label, update status and memory size |
| `memory_bank_update_*.pt`| Saves updated memory bank after selected online updates |

### 2.4.6 CiRA CORE overview
In CiRA CORE, a low-code workflow is built using AutoRun, RestGetJson and Debug nodes. 

