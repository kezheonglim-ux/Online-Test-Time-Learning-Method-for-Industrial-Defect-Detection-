# Code, Method, Experiment and Final Deployment Notes

This README is the technical entry point for the implementation under `code/`.

---

## Project Summary

This project develops an **online test-time learning method for industrial defect detection** using the MVTec AD dataset.

The final system uses:

- a frozen `YOLO26n-cls` backbone;
- local intermediate feature patches;
- category-specific normal patch memory;
- nearest-normal cosine-distance scoring;
- Top-5% patch aggregation;
- trusted-normal deployment calibration;
- a conservative score + consistency update gate;
- optional PatchAdapter parameter adaptation;
- optional online patch-memory adaptation;
- Flask for the model service;
- CiRA CORE for low-code batch operation.

The project started with a global image-level normal-reference detector and progressively moved to a local patch-memory design after the representation experiments showed that local features were much more effective for small industrial defects.

The final recommended deployment mode is:

```text
Memory CTTA
```

because it achieved the best overall Accuracy and Macro F1 in the final deployment ablation while keeping the YOLO26 backbone and PatchAdapter fixed.

### Project result at a glance

| Stage | Main result |
|---|---:|
| Global YOLO representation | 84.84% mean AUROC |
| Local patch representation | 93.02% mean AUROC |
| Final tuned patch configuration | 94.49% mean AUROC |
| Baseline deployment accuracy | 79.50% |
| Calibration-only accuracy | 84.17% |
| **Memory CTTA accuracy** | **85.22% ± 0.42%** |
| **Memory CTTA Macro F1** | **85.19% ± 0.43%** |

> **Important:** representation-stage AUROC and final deployment-ablation metrics come from different evaluation stages. They should not be interpreted as one continuous benchmark.

---

## Table of Contents

- [1. Code Map](#1-code-map)
- [2. End-to-End Execution Flow](#2-end-to-end-execution-flow)
- [3. Core Method](#3-core-method)
- [4. Experiment History](#4-experiment-history)
- [5. Final Deployment Result](#5-final-deployment-result)
- [6. Final Configuration](#6-final-configuration)
- [7. Deployment and CiRA CORE Flow](#7-deployment-and-cira-core-flow)
- [8. Reproducibility](#8-reproducibility)
- [9. Current Limitations and Next Work](#9-current-limitations-and-next-work)

---

# 1. Code Map

| File | Purpose | Usage | Main internal flow |
|---|---|---|---|
| [`offline_train.ipynb`](offline_train.ipynb) | Offline representation preparation and rev1.8 patch ablation. | Run before deployment export. | Prepare data → frozen YOLO features → patch ablation → all-category validation → export patch memory + threshold. |
| [`cira_ttl_anomaly.py`](cira_ttl_anomaly.py) | Core anomaly detector and CTTA logic. | Imported by Flask. | Preprocess → local patch features → nearest-normal score → prediction → safe gate → optional adapter/memory update. |
| [`app_ctta.py`](app_ctta.py) | Flask API and experiment controller. | Start before CiRA CORE batch testing. | Read request → load category state → select experiment mode → call detector → log result → return JSON. |
| [`auto_calibrate_threshold.py`](auto_calibrate_threshold.py) | Trusted-normal deployment calibration. | Run before deployment evaluation. | Score trusted normal images → q0.995 anomaly threshold → q0.90 update threshold → save threshold JSON. |
| [`calibrate_consistency.py`](calibrate_consistency.py) | Category consistency-threshold calibration. | Run after trusted-normal consistency logging. | Read trusted-normal consistency errors → q0.95 consistency threshold → save threshold. |
| [`python1.py`](python1.py) | CiRA CORE input/batch controller. | First Python stage in the Run flow. | Check stop state → load next image → determine category → prepare request information. |
| [`python2.py`](python2.py) | CiRA CORE response/result parser. | Runs after Flask REST response. | Validate JSON → parse prediction → update display state → return image/text/LED values. |
| [`Stop.py`](stop.py) | Controlled batch stop. | Used by CiRA CORE Stop flow. | Set stop condition → running flow checks before next image. |
| [`Reset.py`](reset.py) | Reset batch-control state. | Used by CiRA CORE Reset flow. | Clear stop/index state → prepare clean batch restart. |
| [`cira_test_flow.flow`](test_flow.flow) | Low-code workflow. | Import into CiRA CORE. | Run → Python1 → REST → Python2 → display → delay/next image; separate Stop and Reset paths. |

> File names should match the current repository copy. If the CiRA flow file is renamed, update the final row accordingly.

---

# 2. End-to-End Execution Flow

## 2.1 Offline representation preparation

```text
MVTec AD normal images
        ↓
offline_train.ipynb
        ↓
Frozen YOLO26n-cls
        ↓
Intermediate feature maps
        ↓
Local 14 × 14 patch representation
        ↓
Category normal patch memory
        ↓
Initial category thresholds
        ↓
Export deployment state
```

The backbone remains frozen. The offline notebook determines how normal images are represented and prepares the state later used by Flask.

## 2.2 Trusted-normal deployment calibration

```text
Trusted normal deployment images
        ↓
auto_calibrate_threshold.py
        ↓
anomaly_threshold
+
update_threshold

Trusted-normal consistency data
        ↓
calibrate_consistency.py
        ↓
consistency_threshold
```

Calibration adjusts the deployment operating point. It does **not** retrain YOLO26.

## 2.3 Online inference and optional CTTA

```text
CiRA CORE Run
        ↓
python1.py
        ↓
image path + category + experiment mode
        ↓
Flask /predict
        ↓
app_ctta.py
        ↓
cira_ttl_anomaly.py
        ↓
patch score + prediction
        ↓
safe update gate
        ↓
optional memory and/or adapter update
        ↓
JSON + CSV diagnostics
        ↓
python2.py
        ↓
CiRA CORE display
        ↓
next image
```

Every image receives a prediction first. Adaptation happens only when the sample passes the update gate.

---

# 3. Core Method

## 3.1 Local patch representation

The final detector uses intermediate YOLO26 features instead of only one whole-image embedding.

For the selected 14 × 14 grid:

```text
14 × 14 = 196 local patch positions
```

Each local feature is normalized before nearest-normal comparison.

## 3.2 Nearest-normal patch distance

For test patch `z_i` and category normal patch memory `M`:

```text
d_i = 1 - max cosine_similarity(z_i, m),  m ∈ M
```

Small `d_i` means the patch closely matches a normal reference. Large `d_i` means the local region is more unusual.

## 3.3 Top-fraction image score

The image score uses only the most abnormal local patches:

```text
K = max(1, ceil(N × r))
S(x) = average of Top-K patch distances
```

Final setting:

```text
N = 196 patches
r = 0.05
Top-5% ≈ 10 most abnormal patches
```

This prevents a small defect from being averaged away by the many normal regions in the image.

## 3.4 Deployment thresholds

```text
T_anom   = Q0.995(normal scores) + 0.10σ
T_update = Q0.90(normal scores)
T_update ≤ T_anom
```

The two thresholds have different roles:

```text
score < T_update
→ predicted normal
→ eligible for the next update-safety check

T_update ≤ score < T_anom
→ predicted normal
→ prediction only, no online update

score ≥ T_anom
→ predicted anomaly
→ no online update
```

## 3.5 Meaning of q0.95 and q0.90

`q0.95` means the 95th percentile of trusted-normal anomaly scores. `q0.90` means the 90th percentile.

Because lower scores are more normal-like:

```text
q0.90 = stricter update gate
q0.95 = more permissive update gate
```

rev1.10 compares these two update boundaries experimentally.

## 3.6 Safe update gate

```text
G_score  = [S(x) < T_update]
G_cons   = [E_cons < T_cons]
G_update = G_score AND G_cons
```

The score gate is the main protection; consistency is a secondary stability check.

## 3.7 PatchAdapter learning

PatchAdapter is a lightweight trainable transformation after the frozen YOLO26 patch features. YOLO26 itself does not update.

```text
L_online = αL_cons + βL_anchor
```

Final settings:

```text
consistency_weight = 1.0
anchor_weight      = 0.1
online_lr          = 1e-4
online_steps       = 1
```

The adapter is initialized close to identity so deployment begins from the offline feature representation.

## 3.8 Memory adaptation

When Memory CTTA is enabled and a sample passes the safe gate, accepted patch features are added to the category normal memory.

```text
memory adaptation
= change the stored normal reference

adapter adaptation
= change the lightweight feature transformation
```

The final patch memory is bounded at:

```text
max_patch_memory = 16000
```

---

# 4. Experiment History

The full project is treated as one continuous experiment path.

## 4.1 rev1.0 - First test-time workflow

**Purpose**  
Build the first end-to-end inference path.

**Finding**  
The basic image → feature → score → result flow worked.

---

## 4.2 rev1.1 - Normal-reference anomaly direction

**Purpose**  
Use normal appearance as the reference instead of requiring known defect classes.

**Finding**  
The anomaly direction was standardized so larger distance from normal means stronger anomaly evidence.

---

## 4.3 rev1.2 - Export and deployment loading

**Purpose**  
Allow the prepared category state to run outside the notebook.

**Finding**  
Model/state export and loading were stabilized.

---

## 4.4 rev1.3 - Automatic threshold calibration

**Purpose**  
Estimate category thresholds from normal data instead of manually fixing one threshold.

**Finding**  
The normal-only threshold process became repeatable.

---

## 4.5 rev1.4 - All-category expansion

**Purpose**  
Run the workflow across all 15 MVTec AD categories.

**Finding**

```text
Overall category accuracy:
62.53% → 79.81%
```

This stage mainly improved workflow consistency, category handling and threshold/state management.

---

## 4.6 rev1.5 - Experiment cleanup

**Purpose**  
Clean code, comments and experiment organization before representation comparison.

**Finding**  
No new performance claim.

---

## 4.7 rev1.6 - Global YOLO feature baseline

**Purpose**  
Measure the performance of one whole-image YOLO embedding.

**Result**

```text
Mean AUROC ≈ 84.84%
```

**Finding**  
Global features provide a useful baseline but can dilute small local defects.

---

## 4.8 rev1.7 - Local patch representation

**Purpose**  
Use local intermediate YOLO patches and compare them with normal patch memory.

**Result**

```text
Mean AUROC ≈ 93.02%
Improvement = +8.18 percentage points
```

**Finding**  
Local representation produced the largest feature-level gain in the project.

---

## 4.9 rev1.8 - Patch-representation ablation

> **Stored result:** [`code/result/rev1.8/`](result/rev1.8/)

**Purpose**  
Select one shared local-patch configuration.

Search space:

```text
image size       = 224, 384, 512
feature choice   = last1, last2, last3
patch grid       = 8, 14, 20
top fraction     = 1%, 3%, 5%, 10%
```

Total combinations:

```text
3 × 3 × 3 × 4 = 108
```

Ranking logic:

```text
Primary ranking      = Mean AUROC
Robustness ranking   = Minimum AUROC
Diagnostic check     = Mean Macro F1
Final validation     = all 15 categories
```

Final setting:

```text
img_size           = 384
feature_choice     = last2
patch_grid         = 14
patch_top_fraction = 0.05
max_patch_memory   = 16000
```

Final all-category result:

| Metric | Result |
|---|---:|
| Mean AUROC | **94.49%** |
| Accuracy | **82.36%** |
| Normal Recall | **84.59%** |
| Anomaly Recall | **80.86%** |
| Macro F1 | **78.42%** |

After this stage, the offline representation was frozen.

---

## 4.10 rev1.9 - PatchAdapter verification

> **Stored result:** [`code/result/rev1.9/`](result/rev1.9/)

**Purpose**  
Confirm that adapter learning is actually executed during inference.

**Result**

```text
Accepted samples        = 234
Adapter updated         = 234 / 234
Mean update loss        ≈ 0.000137
Mean adapter delta norm ≈ 0.00259
```

**Finding**  
The adapter genuinely changes its parameters. This verifies the mechanism, not its performance contribution.

---

## 4.11 rev1.10 - Safer update selection

> **Stored result:** [`code/result/rev1.10/`](result/rev1.10/)

**Purpose**  
Compare a more permissive q0.95 update threshold with the stricter q0.90 threshold.

| Metric | q0.95 | q0.90 |
|---|---:|---:|
| Accuracy | 88.33% | 88.00% |
| Normal Recall | 85.67% | 85.00% |
| Anomaly Recall | **91.00%** | **91.00%** |
| Macro F1 | 88.33% | 87.99% |
| Good updates | 218 | 197 |
| Bad updates | 12 | **8** |
| Update precision | 94.78% | **96.10%** |

```text
Bad-update reduction = 33.3%
```

**Finding**  
q0.90 was retained because it reduced defective updates while preserving anomaly recall.

---

## 4.12 rev1.11 - Shuffled-stream robustness

> **Stored result:** [`code/result/rev1.11/`](result/rev1.11/)

**Purpose**  
Check whether CTTA performance depends strongly on one fixed image order.

The same 600 images were shuffled using seeds `42`, `130`, and `2030`.

| Metric | Seed 42 | Seed 130 | Seed 2030 | Mean ± SD |
|---|---:|---:|---:|---:|
| Accuracy | 86.83% | 85.50% | 84.83% | **85.72% ± 1.02%** |
| Normal Recall | 87.00% | 81.00% | 79.67% | **82.56% ± 3.91%** |
| Anomaly Recall | 86.67% | 90.00% | 90.00% | **88.89% ± 1.92%** |
| Macro F1 | 86.83% | 85.47% | 84.79% | **85.70% ± 1.04%** |
| Pooled AUROC | 92.81% | 92.29% | 91.72% | **92.27% ± 0.54%** |

**Finding**  
Detection results are reasonably stable, but update behavior remains path-dependent because each accepted sample can change the state seen by later samples.

---

## 4.13 rev1.12 - Final component ablation

> **Stored result:** [`code/result/rev1.12/`](result/rev1.12/)

**Purpose**  
Separate the effect of calibration, memory adaptation and adapter adaptation.

| Experiment | Threshold | Adapter | Memory | Mode |
|---|---|---:|---:|---|
| Baseline | Original | OFF | OFF | evaluate |
| Calibration | Calibrated | OFF | OFF | evaluate |
| Memory CTTA | Calibrated | OFF | ON | monitor |
| Adapter CTTA | Calibrated | ON | OFF | monitor |
| Full CTTA | Calibrated | ON | ON | monitor |

Final result:

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% | 71.33% | 87.67% | 79.36% | 94.70% |
| Calibration | 84.17% | 77.33% | **91.00%** | 84.09% | 94.70% |
| **Memory CTTA** | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | 94.70% ± 0.02% |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

Contribution summary:

```text
Baseline → Calibration
79.50% → 84.17%
+4.67 percentage points
largest deployment gain

Calibration → Memory CTTA
84.17% → 85.22%
+1.05 percentage points
strongest additional CTTA gain

Calibration → Adapter CTTA
84.17% → 84.11%
no standalone accuracy gain

Memory CTTA → Full CTTA
85.22% → 85.17%
no additional overall gain
```

---

# 5. Final Deployment Result

The final deployment ablation is the rev1.12 experiment. Its saved outputs are kept under [`code/result/rev1.12/`](result/rev1.12/).

## 5.1 Detection result

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% | 71.33% | 87.67% | 79.36% | 94.70% |
| Calibration | 84.17% | 77.33% | **91.00%** | 84.09% | 94.70% |
| **Memory CTTA** | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | 94.70% ± 0.02% |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

### Final deployment interpretation

**Calibration** gave the largest deployment improvement:

```text
79.50% → 84.17% accuracy
```

**Memory CTTA** gave the strongest overall result:

```text
85.22% ± 0.42% accuracy
85.19% ± 0.43% Macro F1
```

**Adapter CTTA** confirmed that the adapter can learn online, but adapter-only performance remained close to calibration-only.

**Full CTTA** gave the best normal recall, but did not improve overall Accuracy or Macro F1 over Memory CTTA.

## 5.2 Online update quality

| Method | Good updates | Bad updates | Total updates | Update precision |
|---|---:|---:|---:|---:|
| Memory CTTA | 183.0 | 10.0 | 193.0 | **94.82% ± 0.14%** |
| Adapter CTTA | 174.3 | 8.0 | 182.3 | **95.61% ± 0.01%** |
| Full CTTA | 186.7 | 12.7 | 199.3 | **93.66% ± 0.55%** |

The update gate is effective but not contamination-free. Full CTTA accepts the most updates and also the most defective updates.

## 5.3 Final deployment decision

Recommended default:

```text
Memory CTTA
```

Reasons:

1. highest overall Accuracy;
2. highest Macro F1;
3. directly adapts the stored normal reference;
4. does not continuously change YOLO26 or PatchAdapter parameters;
5. simpler online state and easier reset;
6. avoids enabling an adaptive component that did not show a standalone performance gain.

The PatchAdapter remains in the repository as a valid experimental mechanism for future stronger domain-shift and online-objective studies.

---

# 6. Final Configuration

| Parameter | Final value | Role |
|---|---:|---|
| `img_size` | 384 | Input resolution |
| `feature_choice` | `last2` | Intermediate feature depth |
| `patch_grid` | 14 | 14 × 14 local grid |
| `patch_top_fraction` | 0.05 | Top abnormal patch fraction |
| `max_patch_memory` | 16000 | Bounded category patch memory |
| `anomaly_quantile` | 0.995 | Trusted-normal anomaly calibration |
| `update_quantile` | 0.90 | Strict online-update boundary |
| `online_lr` | 1e-4 | PatchAdapter learning rate |
| `online_steps` | 1 | Optimizer steps per accepted sample |
| `consistency_weight` | 1.0 | Online consistency-loss weight |
| `anchor_weight` | 0.1 | Feature-anchor loss weight |
| `consistency_threshold` | category-specific | Secondary safe-gate condition |

Final offline representation:

```text
Frozen YOLO26n
+ 384 input
+ last2 intermediate features
+ 14 × 14 patch grid
+ Top-5% patch scoring
+ max 16,000 normal patches
```

Final default deployment:

```text
calibrated thresholds
+ q0.90 score gate
+ consistency gate
+ Memory CTTA ON
+ PatchAdapter update OFF
+ YOLO26 frozen
```

---

# 7. Deployment and CiRA CORE Flow

## 7.1 Model/service layout

```text
C:\cira_ttl_model├── yolo26n-cls.pt
├── cira_ttl_anomaly.py
└── <category>    ├── patch_memory_bank.pt
    └── threshold.json

C:\cira_ttl_service└── app_ctta.py
```

The exact exported file names should follow the current code implementation.

## 7.2 CiRA CORE Run flow

```text
Run
 ↓
python1.py
 ↓
load/check next image
 ↓
Flask /predict
 ↓
cira_ttl_anomaly.py
 ↓
prediction + score + threshold + CTTA diagnostics
 ↓
python2.py
 ↓
Text + LED + image display
 ↓
next image
```

`python1.py` handles the input side of the batch sequence.

`python2.py` handles the returned Flask result and prepares the values displayed by CiRA CORE.

## 7.3 Stop flow

```text
Stop
 ↓
Stop.py
 ↓
set stop condition
 ↓
Run flow checks condition before next image
 ↓
batch ends safely
```

## 7.4 Reset flow

```text
Reset
 ↓
Reset.py
 ↓
clear batch-control state
 ↓
ready for a clean restart
```

CiRA CORE is the operator-facing workflow. Model scoring and CTTA state remain in Python.

---

# 8. Reproducibility

For the final deployment ablation:

1. use the same 600-image evaluation set;
2. use fixed shuffle seeds `42`, `130`, and `2030`;
3. restart Flask before each independent CTTA run;
4. restore the same clean category patch memory;
5. restore the same identity-initialized adapter;
6. load the same calibrated threshold state;
7. reset the CiRA CORE batch index;
8. clear or rotate the current CSV log;
9. run one experiment mode at a time;
10. save the rev1.12 runs under `code/result/rev1.12/`.

Experiment selection:

```python
EXPERIMENT = "baseline"
EXPERIMENT = "calibration"
EXPERIMENT = "memory_ctta"
EXPERIMENT = "adapter_ctta"
EXPERIMENT = "full_ctta"
```

| Mode | Calibration | Memory update | Adapter update |
|---|---:|---:|---:|
| `baseline` | original | OFF | OFF |
| `calibration` | ON | OFF | OFF |
| `memory_ctta` | ON | ON | OFF |
| `adapter_ctta` | ON | OFF | ON |
| `full_ctta` | ON | ON | ON |

---

# 9. Current Limitations and Next Work

Current limitations:

- some defective samples still pass the online update gate;
- the current PatchAdapter objective gives limited standalone performance gain;
- bounded recent-patch retention is simpler than a diversity-aware coreset;
- only three shuffled streams were used in the final robustness study;
- the deployment evaluation is limited to MVTec AD and the current 600-image stream;
- long-duration cumulative drift has not been tested;
- the current prototype does not include production camera or PLC integration.

High-value next experiments:

```text
Current memory policy
vs
k-center / coreset online memory

Current image-level safe gate
+
patch-level contamination filtering

Memory CTTA
under longer controlled domain shifts

YOLO26 frozen features
vs
DINOv2 / other stronger frozen representations

Current PatchAdapter objective
vs
stronger anomaly-aware online objective
```

---

## Final Project Verdict

```text
Global normal-reference baseline
        ↓
Local patch representation
        ↓
Trusted-normal calibration
        ↓
Safe CTTA gate
        ↓
Component ablation
        ↓
Memory CTTA selected for deployment
```

For the current project:

```text
Representation improvement  → local patches
Largest deployment gain     → calibration
Strongest online component  → memory adaptation
Recommended deployment      → Memory CTTA
```
