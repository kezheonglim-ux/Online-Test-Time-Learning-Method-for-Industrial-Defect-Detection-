# Online Test-Time Learning Method for Industrial Defect Detection

This repository presents a practical industrial anomaly-detection system built around a frozen YOLO26 feature extractor, local patch features, category-specific normal memory, deployment calibration, and online test-time adaptation (CTTA).

The final system is deployed through Flask and connected to CiRA CORE for low-code batch testing.

> **Final practical result:** calibration gives the largest deployment gain, while memory adaptation provides the strongest additional CTTA improvement.

---

## Table of Contents

- [1. Project Summary](#1-project-summary)
- [2. System Integration](#2-system-integration)
- [3. Final Method](#3-final-method)
- [4. Final Result](#4-final-result)
- [5. Main Findings](#5-main-findings)
- [6. Deployment](#6-deployment)
- [7. Limitations](#7-limitations)
- [8. More Technical Detail](#8-more-technical-detail)

---

# 1. Project Summary

| Item | Final setup |
|---|---|
| Dataset | MVTec AD, 15 categories |
| Backbone | Frozen YOLO26 |
| Representation | Local patch features |
| Normal reference | Category-specific patch memory |
| Deployment calibration | Trusted normal images |
| Safe update | Score gate + consistency gate |
| Online adaptation | PatchAdapter and/or patch memory |
| API | Flask |
| Low-code integration | CiRA CORE |
| Final validation | 600 images, seeds 42 / 130 / 2030 |

The detector learns the normal reference from normal images and uses defect labels only for evaluation.

---

# 2. System Integration

```text
Offline preparation
      ↓
Frozen YOLO26
      ↓
Local patch features
      ↓
Normal patch memory + threshold
      ↓
Deployment calibration
      ↓
Flask CTTA service
      ↓
Safe online update
      ↓
CiRA CORE batch workflow
```

> **Figure placeholder — Overall integration**
>
> Add one figure showing offline preparation, Flask inference, and CiRA CORE.

---

# 3. Final Method

The image anomaly score is built from the most abnormal local patches:

\[
d_i = \min_{m \in M}
\left(
1-\frac{z_i^\top m}{\|z_i\|\|m\|}
\right)
\]

\[
S(x)=\frac{1}{K}\sum_{i\in TopK(d)} d_i
\]

Deployment calibration uses trusted-normal scores:

\[
T_{anom}=Q_{0.995}(S_{normal})+\lambda\sigma
\]

\[
T_{update}=Q_{0.90}(S_{normal})
\]

Online updates are allowed only when both confidence and consistency checks pass.

Final main settings:

```text
img_size           = 384
feature_choice     = last2
patch_grid         = 14
patch_top_fraction = 0.05
max_patch_memory   = 16000
online_lr          = 1e-4
online_steps       = 1
```

---

# 4. Final Result

## Final Ablation

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% | 71.33% | 87.67% | 79.36% | 94.70% |
| Calibration | 84.17% | 77.33% | **91.00%** | 84.09% | 94.70% |
| **Memory CTTA** | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | 94.70% ± 0.02% |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

Accuracy progression:

```text
Baseline      79.50%
Calibration   84.17%
Memory CTTA   85.22%
Full CTTA     85.17%
```

---

# 5. Main Findings

- Local patch features clearly outperformed the earlier global-feature design.
- Calibration improved deployment accuracy by **4.67 percentage points** over baseline.
- Memory CTTA achieved the best overall accuracy and Macro F1.
- Full CTTA achieved the best normal recall, but did not clearly outperform Memory CTTA.
- Adapter learning was verified technically, but its standalone accuracy contribution was limited.
- Safer update selection reduced bad updates while keeping anomaly recall stable.

> Representation-stage AUROC and final deployment-ablation metrics come from different evaluation stages and should not be treated as one continuous benchmark.

---

# 6. Deployment

```text
C:\cira_ttl_model\
├── yolo26n-cls.pt
├── cira_ttl_anomaly.py
└── <category>\
    ├── patch_memory_bank.pt
    └── threshold.json

C:\cira_ttl_service\
└── app_ctta.py
```

CiRA CORE handles the batch sequence and result display:

```text
python1.py
   ↓
Flask /predict
   ↓
CTTA detector
   ↓
python2.py
   ↓
Text + LED + image
```

> **Figure placeholder — CiRA CORE flow**
>
> Add the full workflow screenshot here.

---

# 7. Limitations

- Some defective samples are still accepted for online updates.
- The current PatchAdapter gives limited standalone gain.
- Memory adaptation is the dominant online component in the present setup.
- Online adaptation slightly trades anomaly recall for better normal recall.
- Only three shuffled sequences were tested.
- Validation is limited to MVTec AD and the current 600-image deployment stream.
- Long-duration drift and cumulative memory contamination remain untested.

---

# 8. More Technical Detail

For code purpose, execution flow, formulas, revision-by-revision experiments, discussion, and next-step reasoning, see:

[`code/README.md`](code/README.md)
