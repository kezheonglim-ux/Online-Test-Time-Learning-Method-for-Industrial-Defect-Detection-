# Code, Method and Experiment Notes

This file documents the implementation and the experiment path behind the final CTTA system.

The root [`README.md`](../README.md) gives the overall integration and final result. This file focuses on code responsibilities, internal flow, formulas, and what each revision taught us.

---

## Table of Contents

- [1. Code Map](#1-code-map)
- [2. Execution Flow](#2-execution-flow)
- [3. Core Method](#3-core-method)
- [4. Experiment History](#4-experiment-history)
  - [4.1 rev1.0](#41-rev10)
  - [4.2 rev1.1](#42-rev11)
  - [4.3 rev1.2](#43-rev12)
  - [4.4 rev1.3](#44-rev13)
  - [4.5 rev1.4](#45-rev14)
  - [4.6 rev1.5](#46-rev15)
  - [4.7 rev1.6](#47-rev16)
  - [4.8 rev1.7](#48-rev17)
  - [4.9 rev1.8](#49-rev18)
  - [4.10 rev1.9](#410-rev19)
  - [4.11 rev1.10](#411-rev110)
  - [4.12 rev1.11](#412-rev111)
  - [4.13 rev1.12](#413-rev112)
- [5. Final Configuration](#5-final-configuration)
- [6. Reproducibility](#6-reproducibility)

---

# 1. Code Map

| File | Purpose | Usage | Main internal flow |
|---|---|---|---|
| [`train_rev1.8.ipynb`](train_rev1.8.ipynb) | Build the offline patch representation. | Run before deployment export. | Prepare data → extract frozen YOLO features → compare patch settings → validate all categories → export memory + threshold. |
| [`cira_ttl_anomaly.py`](cira_ttl_anomaly.py) | Core anomaly detector and CTTA logic. | Imported by Flask. | Preprocess → patch features → anomaly score → safe gate → optional adapter/memory update → final result. |
| [`app_ctta.py`](app_ctta.py) | Flask API and experiment controller. | Start before batch testing. | Read request → load category model → select experiment → call detector → log → return JSON. |
| [`auto_calibrate_threshold.py`](auto_calibrate_threshold.py) | Calibrate anomaly and update thresholds. | Run with trusted normal images. | Score `good_*` images → q0.995 anomaly threshold → q0.90 update threshold → update JSON. |
| [`calibrate_consistency.py`](calibrate_consistency.py) | Set category consistency thresholds. | Run after trusted-normal consistency logging. | Read normal log → q0.95 consistency error → save threshold. |
| [`python1.py`](python1.py) | CiRA batch loader. | First Python block in the loop. | Collect images → fixed-seed shuffle → read index → return current image/category. |
| [`python2.py`](python2.py) | CiRA result parser. | Run after REST response. | Validate JSON → parse result → set LED → load image → return display values. |
| `test_flow_rev1.3.flow` | CiRA workflow. | Import into CiRA CORE. | Run / Stop / Reset → Python1 → REST → Python2 → display → loop. |

---

# 2. Execution Flow

## Offline

```text
MVTec AD
   ↓
train_rev1.8.ipynb
   ↓
Frozen YOLO26
   ↓
Local patch embeddings
   ↓
Normal patch memory
   ↓
threshold.json
```

## Calibration

```text
Trusted normal images
   ↓
auto_calibrate_threshold.py
   ↓
anomaly threshold + update threshold

Trusted consistency log
   ↓
calibrate_consistency.py
   ↓
consistency threshold
```

## Online inference

```text
python1.py
   ↓
image + category
   ↓
app_ctta.py
   ↓
cira_ttl_anomaly.py
   ↓
score
   ↓
safe gate
   ↓
optional online update
   ↓
CSV + JSON
   ↓
python2.py
   ↓
CiRA display
```

> **Figure placeholder — Code integration**
>
> Add one diagram showing how the offline export feeds the Flask/CiRA runtime.

---

# 3. Core Method

## Patch score

\[
d_i =
\min_{m\in M}
\left(
1-\frac{z_i^\top m}{\|z_i\|\|m\|}
\right)
\]

\[
S(x)
=
\frac{1}{K}
\sum_{i\in TopK(d)}
d_i
\]

The top patch fraction is used so small local defects are not averaged away.

## Deployment thresholds

\[
T_{anom}
=
Q_{0.995}(S_{normal})+\lambda\sigma
\]

\[
T_{update}
=
Q_{0.90}(S_{normal})
\]

`T_update` is stricter than `T_anom`, so not every normal prediction is accepted for learning.

## Safe update gate

\[
G_{update}
=
[S(x)<T_{update}]
\land
[E_{cons}<T_{cons}]
\]

## Adapter loss

\[
L_{online}
=
\alpha L_{cons}
+
\beta L_{anchor}
\]

with:

```text
consistency_weight = 1.0
anchor_weight      = 0.1
online_lr          = 1e-4
online_steps       = 1
```

YOLO26 remains frozen.

---

# 4. Experiment History

Each revision below uses the same four-part format:

- **Purpose / idea**
- **Result / finding**
- **Discussion**
- **Next**

## 4.1 rev1.0

**Purpose / idea**  
Build the first end-to-end test-time workflow.

**Result / finding**  
The basic inference path worked.

**Discussion**  
This revision proved the pipeline could run, but did not yet solve open-ended industrial anomaly detection.

**Next**  
Move toward normal-reference anomaly detection.

---

## 4.2 rev1.1

**Purpose / idea**  
Use normal behaviour as the reference instead of known defect classes.

**Result / finding**  
The project direction shifted to open-ended anomaly detection.

**Discussion**  
This is better suited to industrial use because future defects may not exist in the training set.

**Next**  
Prepare model outputs for deployment.

---

## 4.3 rev1.2

**Purpose / idea**  
Make the trained representation reusable outside the notebook.

**Result / finding**  
Model loading and export issues were resolved.

**Discussion**  
The detector could now be connected to Flask and CiRA instead of remaining notebook-only.

**Next**  
Add deployment-specific calibration.

---

## 4.4 rev1.3

**Purpose / idea**  
Recalculate the operating threshold from trusted normal deployment images.

**Result / finding**  
Automatic threshold calibration was added.

**Discussion**  
This addressed the gap between offline validation and deployment conditions.

**Next**  
Expand testing across all categories.

---

## 4.5 rev1.4

**Purpose / idea**  
Evaluate the workflow across all 15 MVTec AD categories.

**Result / finding**

```text
Overall category accuracy:
62.53% → 79.81%
```

**Discussion**  
The broader evaluation exposed stronger category variation and made one shared configuration more important.

**Next**  
Clean the implementation before testing a new representation.

---

## 4.6 rev1.5

**Purpose / idea**  
Simplify code and comments before the representation experiments.

**Result / finding**  
No new performance claim.

**Discussion**  
This was a maintenance revision.

**Next**  
Test a global YOLO feature baseline.

---

## 4.7 rev1.6

**Purpose / idea**  
Use a global YOLO embedding for normal-reference anomaly scoring.

**Result / finding**

```text
Mean AUROC ≈ 84.84%
```

**Discussion**  
Global features worked as a baseline but were weak for small or localized defects.

**Next**  
Use local patch features.

---

## 4.8 rev1.7

**Purpose / idea**  
Compare local YOLO patches with a normal patch memory.

**Result / finding**

```text
Mean AUROC ≈ 93.02%
```

Improvement:

\[
93.02-84.84=8.18
\]

**Discussion**  
Local patches gave the largest representation-level gain because abnormal regions no longer had to dominate a whole-image embedding.

**Next**  
Tune patch settings and freeze one shared offline configuration.

---

## 4.9 rev1.8

**Purpose / idea**  
Choose one patch setup that works consistently across categories.

**Final setting**

```text
img_size           = 384
feature_choice     = last2
patch_grid         = 14
patch_top_fraction = 0.05
max_patch_memory   = 16000
```

**Result / finding**

| Metric | Result |
|---|---:|
| Mean AUROC | **94.49%** |
| Accuracy | **82.36%** |
| Normal Recall | **84.59%** |
| Anomaly Recall | **80.86%** |
| Macro F1 | **78.42%** |

**Discussion**  
The offline representation was strong enough to freeze. Later work focused on deployment adaptation instead of changing YOLO features.

**Next**  
Verify that online adapter learning is real.

---

## 4.10 rev1.9

**Purpose / idea**  
Prove that PatchAdapter parameters actually change during inference.

**Result / finding**

```text
Accepted samples         = 234
Adapter updated          = 234 / 234
Mean update loss         ≈ 0.000137
Mean adapter delta norm  ≈ 0.00259
```

**Discussion**  
The mechanism was confirmed to work, but this did not yet prove a performance benefit.

**Next**  
Reduce unsafe online updates.

---

## 4.11 rev1.10

**Purpose / idea**  
Add a stricter score gate and consistency gate before online learning.

**Result / finding**

| Metric | q0.95 | q0.90 |
|---|---:|---:|
| Accuracy | 88.33% | 88.00% |
| Anomaly Recall | 91.00% | **91.00%** |
| Bad Updates | 12 | **8** |
| Update Precision | 94.78% | **96.10%** |

Bad-update reduction:

\[
\frac{12-8}{12}\times100
\approx 33.3\%
\]

**Discussion**  
q0.90 was kept because it reduced contamination with almost no loss in anomaly sensitivity.

**Next**  
Check sensitivity to input order.

---

## 4.12 rev1.11

**Purpose / idea**  
Remove alphabetical sequence bias and test CTTA under different fixed image orders.

Seeds:

```text
42
130
2030
```

**Result / finding**

| Metric | Seed 42 | Seed 130 | Seed 2030 | Mean ± SD |
|---|---:|---:|---:|---:|
| Accuracy | 86.83% | 85.50% | 84.83% | **85.72% ± 1.02%** |
| Normal Recall | 87.00% | 81.00% | 79.67% | **82.56% ± 3.91%** |
| Anomaly Recall | 86.67% | 90.00% | 90.00% | **88.89% ± 1.92%** |
| Macro F1 | 86.83% | 85.47% | 84.79% | **85.70% ± 1.04%** |
| Pooled AUROC | 92.81% | 92.29% | 91.72% | **92.27% ± 0.54%** |

**Discussion**  
Detection performance stayed reasonably stable. Update counts varied more because CTTA is sequential and path-dependent.

**Next**  
Separate the contribution of calibration, adapter learning, and memory learning.

---

## 4.13 rev1.12

**Purpose / idea**  
Run a five-method ablation to identify what actually contributes.

| Experiment | Threshold | Adapter | Memory | Mode |
|---|---|---:|---:|---|
| Baseline | original | OFF | OFF | evaluate |
| Calibration | calibrated | OFF | OFF | evaluate |
| Memory CTTA | calibrated | OFF | ON | monitor |
| Adapter CTTA | calibrated | ON | OFF | monitor |
| Full CTTA | calibrated | ON | ON | monitor |

**Result / finding**

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% | 71.33% | 87.67% | 79.36% | 94.70% |
| Calibration | 84.17% | 77.33% | **91.00%** | 84.09% | 94.70% |
| **Memory CTTA** | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | 94.70% ± 0.02% |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

**Discussion**  
Calibration produced the largest deployment gain. Memory adaptation added the clearest CTTA improvement. Adapter-only learning remained close to calibration-only performance, while Full CTTA improved normal recall but did not clearly beat Memory CTTA overall.

**Next**  
Freeze this experiment cycle and move to broader validation: more sequences, longer deployment streams, cross-dataset testing, and stronger adapter objectives.

---

# 5. Final Configuration

| Parameter | Final value | Role |
|---|---:|---|
| `img_size` | 384 | Input resolution |
| `feature_choice` | `last2` | Local feature depth |
| `patch_grid` | 14 | Spatial patch grid |
| `patch_top_fraction` | 0.05 | Highest patch-distance fraction used for image score |
| `max_patch_memory` | 16000 | Maximum normal patch memory |
| `anomaly_quantile` | 0.995 | Deployment anomaly calibration |
| `update_quantile` | 0.90 | Online-update boundary |
| `online_lr` | 1e-4 | PatchAdapter learning rate |
| `online_steps` | 1 | Optimizer step per accepted image |
| `consistency_weight` | 1.0 | Consistency-loss weight |
| `anchor_weight` | 0.1 | Normal-anchor-loss weight |
| `consistency_threshold` | category-specific | Safety-gate threshold |

---

# 6. Reproducibility

For the final ablation:

1. use the same 600-image dataset;
2. use seeds `42`, `130`, and `2030`;
3. restart Flask before each CTTA run;
4. start from the same clean patch memory;
5. start from the same identity-initialized adapter;
6. reset the batch index;
7. clear the active prediction log;
8. save each run as a separate CSV.

Experiment selection:

```python
EXPERIMENT = "baseline"
EXPERIMENT = "calibration"
EXPERIMENT = "memory_ctta"
EXPERIMENT = "adapter_ctta"
EXPERIMENT = "full_ctta"
```

Recommended result layout:

```text
results/
├── baseline/
├── calibration/
├── memory_ctta/
├── adapter_ctta/
└── full_ctta/
```
