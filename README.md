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

            
## Code Test Flow

```mermaid
flowchart TD

    subgraph TRAIN[Train Phase]
        A[Normal Images] --> B[YOLO Feature Extractor]
        B --> C[Online Adapter<br/>Initialized as Identity]
        C --> D[Memory Bank<br/>Normal Feature Storage]
    end

    subgraph TEST[Test-Time Phase]
        E[Incoming Image] --> F[YOLO Feature Extractor]
        F --> G[Online Adapter]
        G --> H[Test Feature Embedding]
        H --> I[Top-K Normal Reference Selection]
        I --> J[Feature Comparison]
        J --> K[Anomaly Score Calculation]
        K --> L[Result Generation]
    end

    L --> M{Normal-like Sample?}

    M -->|No| N[Output:<br/>Anomaly Result]
    M -->|Yes| O[Update Online Adapter]
    O --> P[Add Feature to Memory Bank]
    P --> Q[Update Normal Representation]
    Q --> R[Output:<br/>Normal Result]
```
### Explanation

During the training phase, normal images are passed through the YOLO feature extractor and online adapter. The extracted normal features are stored in a memory bank.

During the test-time phase, each incoming image is converted into a feature embedding and compared with the most similar normal reference features from the memory bank. An anomaly score is calculated to classify the image as normal or anomalous.

If the sample is considered normal-like, the online adapter and memory bank are updated to improve future adaptation without full model retraining.
