# Online-Test-Time-Learning-Method-for-Industrial-Defect-Detection-
Current manufacturing industries on defect detection not only require efficiency work but also able to learn new information with unknown defect detection. Here was introducing an online test-time learning method to support the need and a low-code platform developed for easy use.

**Dataset** 
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
 
**Code test flow**
TRAIN PHASE
------------
Normal Images

    ↓
YOLO Feature Extractor

    ↓
Online Adapter (initialized as identity)

    ↓
Memory Bank of Normal Features

TEST-TIME PHASE
---------------
Incoming Image
    ↓
YOLO Feature Extractor
    ↓
Online Adapter
    ↓
Compare with Memory Bank
    ↓
Anomaly Score
    ↓
Normal-like sample?
   ├─ No  → Output anomaly result
   └─ Yes → Online Adapter Update
                ↓
            Update Memory Bank
                ↓
            Output updated result
