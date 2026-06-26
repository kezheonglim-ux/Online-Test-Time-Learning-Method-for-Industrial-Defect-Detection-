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

            
## 2.0 Project Workflow 

### 2.1 Project Purpose 

The purpose of this project is to develop a working online test-time learning method for industrial defect detection. The system is designed to identify abnormal product images by comparing incoming image features with stored normal reference features. Instead of retraining the entire model during deployment, this project uses a frozen YOLO26 feature extractor, a lightweight online adapter, a normal memory bank, and calibrated decision thresholds. This allows the system to keep the main feature extractor stable while still adapting to trusted normal images during testing. The project also connects the Python-based defect detection pipeline with Flask API and CiRA CORE. Through this integration, the system can display the current image, image number, total image count, category, prediction result, anomaly status, anomaly score, and threshold in a low-code workflow interface. 

### 2.2 Article Review <!-- Leave this section blank first. This part will later review related studies and methods that support the proposed system. --> 

#### 2.2.1 Overview of Industrial Defect Detection 
<!-- Leave blank first. --> 

#### 2.2.2 Literature Review of Existing Methods 
<!-- Leave blank first. --> 

#### 2.2.3 Comparison of Different Methods 

| Research Direction | Main Technical Idea | Typical Technical Components | Strength | Limitation / Trade-Off | Relevance to This Project |
|---|---|---|---|---|---|
| Traditional Image Processing | Detect defects using manually designed visual rules. | Edge detection, thresholding, texture features, morphology, color/shape rules. | Simple, fast, and easy to explain. | Weak when defects are complex, small, irregular, or affected by lighting changes. | Used as background only. Not selected because the project needs adaptive feature-based detection. |
| Supervised Deep Learning | Train a model using labelled normal and defect samples. | CNN, ResNet, Faster R-CNN, YOLO, DETR, labelled defect datasets. | High accuracy when enough labelled defect data are available. | Requires sufficient labelled defect samples; performance may drop when unseen defect types appear. | Not fully suitable because industrial defect samples may be limited or unknown. |
| YOLO-Based Detection | Detect object/defect regions using a fast one-stage detection model. | YOLO backbone, neck, detection head, bounding box prediction, feature maps. | Fast inference, practical for real-time inspection, and suitable as a feature extractor. | Usually needs labelled bounding boxes if used as a detector; may not directly solve unknown anomaly detection. | Selected as the frozen feature extractor because it provides useful visual features without retraining the full model. |
| Unsupervised Anomaly Detection | Learn normal patterns and detect deviations from normal data. | Normal training images, anomaly score, threshold, distance-based comparison. | Suitable when defect labels are limited or unavailable. | Performance depends on the quality of normal representation and threshold selection. | Highly relevant because the project compares test image features with stored normal references. |
| Memory Bank-Based Anomaly Detection | Store normal feature embeddings as reference and compare test features against them. | Pre-trained backbone, feature embedding, normal memory bank, nearest-neighbor distance, anomaly score. | Effective for detecting unknown defects without requiring labelled defect samples. | Memory size, feature quality, and threshold setting affect detection stability. | Core idea of this project: normal image embeddings are stored as a reference memory bank. |
| Incremental Learning | Update the model when new defect categories or new data become available. | Knowledge distillation, replay buffer, teacher-student model, feature alignment. | Helps reduce catastrophic forgetting when learning new classes. | Usually requires staged training, labelled new data, or offline update sessions. | Related but not selected as the main method because the project focuses on adaptation during testing. |
| Few-Shot / Unknown Defect Detection | Detect or learn new defects using very limited samples. | Few-shot learning, contrastive learning, meta-learning, unknown-class detection. | Useful when new defect types have only a few examples. | Often still requires labelled support samples or special training setup. | Supports the motivation, but the project aims to reduce dependence on labelled defect samples. |
| Test-Time Adaptation / Test-Time Learning | Adapt the model or part of the system during inference using incoming test data. | Online update, pseudo-label, entropy minimization, adapter update, trusted normal sample update. | Allows adaptation to deployment data without full retraining. | Risk of error accumulation if abnormal or wrongly predicted samples are used for updating. | Selected because the project needs online adaptation during industrial deployment. |
| Low-Code Implementation | Deploy the detection workflow using visual or low-code components. | CiRA CORE nodes, Flask API, image display, result display, button trigger, JSON communication. | Easier for engineers or non-AI users to operate and monitor. | Less flexible than full custom coding for complex model design. | Selected because the project aims to connect the AI backend with a usable industrial workflow interface. |

#### 2.2.4 Selected Method for This Project 
<!-- Leave blank first. --> 

#### 2.2.5 Research Gap and Project Positioning 
<!-- Leave blank first. -->

### 2.3 Proposed System Workflow

The proposed system is arranged into three main stages:

<pre>
Stage 1: Offline Preparation in Notebook
        ↓
Stage 2: Deployment Auto-Calibration
        ↓
Stage 3: CiRA CORE + Flask CTTA Deployment Workflow
</pre>

| Stage | Name | Main Usage |
|---|---|---|
| Stage&nbsp;1 | Offline Preparation in Notebook | Prepare the initial normal representation and export model files |
| Stage&nbsp;2 | Deployment Auto-Calibration | Adjust the anomaly and update thresholds using trusted normal deployment images |
| Stage&nbsp;3 | CiRA CORE + Flask CTTA Deployment Workflow | Run the defect detection system through CiRA CORE and Flask API |


## 2.4 Stage 1: Offline Preparation in Notebook

### 2.4.1 Purpose

Prepare the initial normal representation and model files before deployment.

### 2.4.2 Process Flow

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

### 2.4.3 Description

The offline preparation from notebook here is to develop and prepare the initial anomaly detection model before deployment. A YOLO-based model is used as a frozen feature extractor, only a small adapter layer and the stored normal reference features are updated during online testing. This makes the system more stable and avoids the risk of changing the entire model during deployment.

Normal training images are passed through the YOLO feature extractor to obtain visual feature embeddings and these embeddings are stored in a normal memory bank as reference patterns of normal products. During validation, normal validation images are compared with this memory bank to calculate a distribution of normal anomaly scores. The initial anomaly threshold is then calculated from this normal-score distribution rather than being directly predicted by the model. 
End of this stage, four deployment files are generated and saved for later testing and deployment. 

### 2.4.4 Files in use

| File | Usage |
|---|---|
| `train_rev1.3.ipynb` | Dataset preparation, feature extraction, memory bank construction, initial threshold calibration, evalution and export of model files |

### 2.4.5 Exported Files

| File | Usage |
|---|---|
| `yolo26n-cls.pt` | Provides the frozen YOLO feature extractor. It converts each incoming image into a visual feature embedding without retraining the YOLO model. |
| `ttl_adapter.pt` | Stores the lightweight online adapter. The extracted feature embedding is passed through this adapter before comparison with the memory bank. |
| `memory_bank.pt` | Stores the normal reference feature embeddings generated from normal training images. Incoming images are compared with this memory bank to calculate the anomaly score. |
| `threshold.json` | Stores the calibrated decision settings, including the anomaly threshold and update threshold. These settings are used to classify the image and control online updating. |

## 2.5 Stage 2: Deployment Auto-Calibration

### 2.5.1 Purpose

Adjust the decision thresholds to match the deployment image condition.

### 2.5.2 Process Flow

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

### 2.5.3 Description

Deployment auto-calibration is included to adjust the decision thresholds before the system is used in the CiRA CORE workflow. Although the notebook generates an initial threshold from validation normal images, the actual deployment images may have different lighting, image quality, background, capture angle or image source. Therefore, a small set of trusted normal images from the target deployment category is used to calculate a more suitable `anomaly_threshold` and `update_threshold`. 

The `score_only()` function calculates anomaly scores without updating the adapter or memory bank. These scores are then used to generate the `anomaly_threshold` and `update_threshold`. The `anomaly_threshold` is used to decide whether an image should be classified as normal or anomalous. The `update_threshold` is used to decide whether an image is confidently normal and suitable for online updating. After calibration, the updated threshold values are saved into `threshold.json`.

In the current implementation, about 5 normal bottle images from the testing folder are selected and treated as deployment-condition normal samples for demonstration. The bottle category is used as the example category in this stage. If additional categories are deployed in the future, the same calibration process should be repeated separately for each category because each product category has its own normal feature distribution and decision boundary.

### 2.5.4 Files in use

| File | Usage |
|---|---|
| `auto_calibrate_threshold.py` | Deployment threshold calibration. Calibrated anomaly threshold and update threshold. Trusted normal deployment images used to calibrate threshold under testing folder stored in `workdir\cira_ttl_calibration` |

### 2.5.5 Output

| Output File | Usage |
|---|---|
| `threshold.json` | Stores the updated `anomaly_threshold`, `update_threshold` and calibration settings for deployment |


## 2.6 Stage 3: CiRA CORE + Flask CTTA Deployment Workflow

### 2.6.1 Purpose

Run the defect detection system through a low-code workflow.

### 2.6.2 Process Flow

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

###  2.6.3 Description

In this stage, the prepared anomaly detection workflow is deployed through CiRA CORE and the Flask CTTA service. CiRA CORE acts as the low-code workflow interface, while Flask serves as the bridge to the Python-based CTTA detector by loading the exported model files, running the detection process, and returning the prediction result to CiRA CORE.

The CiRA CORE `RestGetJson` node sends the image path to the Flask API. Flask then reads the image and calls the CTTA detector. The detector extracts YOLO26 visual features, compares them with the normal memory bank, calculates the anomaly score and applies the dual-threshold decision logic.

If the image is confidently normal, the lightweight online adapter and memory bank are updated during online test-time learning. If the image is not confidently normal, no online update is performed. The final prediction result is returned to CiRA CORE in JSON format and can be displayed through the Debug node or dashboard. Prediction logs and memory checkpoints are also saved for traceability.

### 2.6.4 Files in use

| File | Usage |
|---|---|
| `app_ctta.py` | Flask service file, bridge between CiRA CORE and the Python-based CTTA detector. It load the export files output from `train.ipynb`, receiving image oath from CiRA CORE, running prediction through the CTTA detector, supporting `monitor`/`evalute`/`calibrate` modes, logging predictions to CSV, saving memory checkpoints and returning JSON result to CiRA CORE|
| `cira_ttl_anomaly.py` | CTTA detector, contain real detection logic used during testing and deployment. It performs two key functions: feature extraction and anomaly score calculation.<br><br>Feature extraction converts the input image into a numerical feature embedding so that it can be compared with normal reference features.<br><br>Anomaly score calculation used to measures how different the incoming image is from the stored normal references. If the image is less similar to the memory bank, the anomaly score becomes higher. |

### 2.6.5 Output

| Output File | Usage |
|---|---|
| JSON prediction result | Returned to CiRA CORE and shows prediction result for each image displayed in the Debug node or dashboard  |
| `prediction_log.csv` | Stores prediction history, including image path, score, label, update status and memory size |
| `memory_bank_update_*.pt`| Saves updated memory bank after selected online updates |

### 2.7 CiRA CORE Overview

In CiRA CORE, a low-code workflow is built with three flows: Main Flow, Stop Flow, and Reset Flow.

#### 2.7.1 Main Flow

![CiRA CORE Main Flow](write-up/images/cira_run_flow.PNG)

#### 2.7.2 Stop Flow

![CiRA CORE Stop Flow](write-up/images/cira_stop_flow.PNG)

#### 2.7.3 Reset Flow

![CiRA CORE Reset Flow](write-up/images/cira_reset_flow.PNG)


#### 2.7.4 Feature Nodes

| Feature Node | Usage in Workflow |
|---|---|
| `Button Run` | Used as the main execution trigger for each workflow.<br><br>**In use:** The Run `Button Run` starts the batch image testing loop. |
| `Python`-Batch Image Loader | Python node used to control the batch image input process.<br><br>***In use:** It checks `stop.txt`, reads `batch_index.txt`, loads the next valid image from `C:\cira_batch_test`, and prepares `image_path`, `category`, and `mode` for Flask prediction. |
| `Python`-Prediction Result Parser | Python node used to process the prediction result and prepare UI output.<br><br>***In use:** It parses the CTTA Flask API response, extracts the prediction result, anomaly score, threshold, and image information, then prepares `display_text`, `led_color`, and the current image object for UI display. |
| `Python`-Batch Index Reset | Python node used to reset the batch image sequence.<br><br>***In use:** It resets `batch_index.txt` to `0`, allowing the next Run Flow to restart from the first image. |
| `Python`-Stop File Gen | Python node used to request a safe stop for the Run Flow.<br><br>***In use:** It creates `stop.txt` in `C:\cira_batch_test`. The Run Flow will detect this file before loading the next image and stop after the current image is completed. |
| `Debug` | Used to monitor internal payloads and troubleshoot the workflow.<br><br>***In use:** It verifies image loading, Flask API responses, prediction results, stop requests, reset status, and UI output values. |
| `IfElse` | Used to decide whether the Run Flow should continue.<br><br>***In use:** If `have_img = true`, the image is sent to the Flask API for prediction. If `have_img = false`, the workflow stops because the batch is completed, stopped, or has an error. |
| `RestPutJson` | Used to send image information to the CTTA Flask API.<br><br>***In use:** It sends `image_path`, `category`, and `mode` to the Flask `/predict` endpoint and receives the anomaly prediction result. |
| `Set` | Used to publish Python output to selected UI display channels.<br><br>***In use:** `Set(text_status)` updates the result Text display, `Set(led_status)` updates the LED anomaly status, and `Set(image_status)` updates the current image display. |
| `Get` | Used to retrieve selected flow data when required.<br><br>***In use:** It is mainly used for testing or retrieving UI-related data. The main Run / Stop / Reset execution is still controlled by `Button Run` for stability. |
| `Delay` | Used to control the interval before the next image is loaded.<br><br>***In use:** After each prediction is completed and the UI is updated, the Delay block triggers the Run Flow again to process the next image automatically. |
| `Text` | Used to display the parsed prediction result.<br><br>***In use:** It displays progress, category, result, anomaly status, file name, anomaly score, and threshold through `Set(text_status)`. |
| `Label` | Used as a static explanation block on the CiRA CORE canvas.<br><br>***In use:** It describes the purpose and logic of the Run Flow, Stop Flow, and Reset Flow for better readability. |
| `LED` | Used to show anomaly status visually.<br><br>***In use:** It receives data through `Set(led_status)`. The LED turns red when an anomaly is detected and green when the image is classified as normal. |
| `Image` | Used to display the current image being analyzed.<br><br>***In use:** It receives the actual image object through `Set(image_status)`, allowing the UI to show the image processed in the current prediction cycle. |






