# Online Test-Time Learning Method for Industrial Defect Detection

This repository presents a practical industrial anomaly-detection system that can adjust to deployment changes without retraining the full visual backbone.

The final pipeline uses a **frozen YOLO26 classifier backbone**, **local patch features**, **category-specific normal patch memory**, **trusted-normal calibration**, and **safe online test-time adaptation (CTTA)**. Flask provides the inference API and CiRA CORE handles the low-code batch workflow.

> **Final recommendation:** use **Memory CTTA** as the default deployment mode. It gave the best overall accuracy and Macro F1, while avoiding unnecessary adapter parameter drift. Full CTTA remains available as a research mode.

---

## Table of Contents

- [1. Project Summary](#1-project-summary)
- [2. System Integration](#2-system-integration)
- [3. Final Method and Why It Was Selected](#3-final-method-and-why-it-was-selected)
  - [3.1 Local Patch Representation](#31-local-patch-representation)
  - [3.2 Patch Anomaly Score](#32-patch-anomaly-score)
  - [3.3 Deployment Calibration](#33-deployment-calibration)
  - [3.4 Safe Update Gate](#34-safe-update-gate)
  - [3.5 Online Adaptation](#35-online-adaptation)
- [4. Final Result](#4-final-result)
- [5. Main Findings and Final Verdict](#5-main-findings-and-final-verdict)
- [6. Deployment and CiRA CORE Operation](#6-deployment-and-cira-core-operation)
- [7. Limitations](#7-limitations)
- [8. Technical Detail](#8-technical-detail)

---

# 1. Project Summary

## Dataset

The project uses **MVTec AD**, an industrial anomaly-detection benchmark containing more than 5,000 high-resolution images across **15 object and texture categories**.

Official source:  
[MVTec AD — Industrial Anomaly Detection Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

Categories used:

```text
bottle, cable, capsule, carpet, grid,
hazelnut, leather, metal_nut, pill, screw,
tile, toothbrush, transistor, wood, zipper
```

Normal training images are used to build the reference representation. Defect labels are used for evaluation only.

## Final System Components

| Component | What it means in this project |
|---|---|
| **YOLO26n-cls** | Ultralytics nano classification model used only as the visual feature extractor. |
| **Frozen YOLO26** | YOLO weights are not changed during offline memory building or online deployment. This keeps the main representation stable. |
| **Local patch features** | Intermediate YOLO feature maps are divided into local spatial descriptors instead of reducing the whole image to one global vector. This keeps small defects visible. |
| **Patch memory** | A category-specific bank of normal patch embeddings. New images are scored by measuring how different their local patches are from this normal reference. |
| **Anomaly threshold** | Category-specific decision boundary used to classify an image as normal or anomalous. |
| **Score gate** | A stricter threshold used before online learning. A sample may be classified as normal but still be rejected for adaptation. |
| **Consistency gate** | Checks whether weakly and strongly perturbed versions of the same image produce similar patch features. Unstable samples are not used for online learning. |
| **PatchAdapter** | Small learnable scale-and-bias layer applied to patch features. YOLO stays frozen while this lightweight layer can adapt online. |
| **Flask** | Local REST API that loads the correct category model, runs prediction, logs CTTA diagnostics, and returns JSON to CiRA CORE. |
| **CiRA CORE** | Low-code runtime used to feed images to Flask, control the batch loop, and display the image, score, result, and LED status. |

## Trusted-Normal Calibration Set

The final deployment calibration uses **20 trusted-normal images per category**.

These images are stored separately from the mixed evaluation stream and are used only to estimate:

- anomaly threshold;
- update threshold;
- consistency threshold.

They are not used as defective training examples.

---

# 2. System Integration

```text
Offline preparation
      ↓
Frozen YOLO26 feature extractor
      ↓
Local patch representation
      ↓
Clean normal patch memory + offline threshold
      ↓
Trusted-normal deployment calibration
      ↓
Flask CTTA service
      ↓
Safe online update gate
      ↓
Memory and/or PatchAdapter update
      ↓
CiRA CORE batch workflow
      ↓
Prediction display + CSV logging
```

| Block | Usage |
|---|---|
| **Offline preparation** | Builds the initial normal reference from MVTec AD and selects one shared patch configuration. |
| **Frozen YOLO26** | Extracts visual features while keeping the backbone unchanged. |
| **Local patch representation** | Preserves spatial information so small local defects remain detectable. |
| **Normal patch memory** | Stores normal patch embeddings used as the nearest-normal reference during scoring. |
| **Deployment calibration** | Adjusts thresholds to trusted deployment-normal images before online evaluation. |
| **Flask CTTA service** | Provides category-aware inference, experiment control, logging, and checkpoint handling. |
| **Safe update gate** | Prevents uncertain samples from changing the online state. |
| **Online adaptation** | Updates memory, adapter, or both depending on the selected experiment mode. |
| **CiRA CORE** | Runs the image sequence and displays the returned prediction. |
| **CSV logging** | Keeps prediction scores, gate decisions, adapter changes, and memory updates for later analysis. |

> **Figure placeholder — Overall system integration**
>
> Add one figure showing offline preparation on the left and Flask + CiRA CORE deployment on the right.

---

# 3. Final Method and Why It Was Selected

The final method was selected from the experiment results, not from one fixed design chosen at the start.

Each step solved one issue found in the previous stage:

```text
Global feature
   ↓
Too coarse for small defects

Local patch feature
   ↓
Better local defect sensitivity

Offline patch tuning
   ↓
One stable shared configuration

Deployment calibration
   ↓
Adjust threshold to real deployment scores

Safe update gate
   ↓
Reduce bad samples entering online learning

Memory / adapter testing
   ↓
Measure what actually improves performance

Final choice
   ↓
Memory CTTA
```

The final deployment method is therefore:

```text
Frozen YOLO26
+ local patch features
+ calibrated thresholds
+ safe update gate
+ online patch-memory update
```

The PatchAdapter remains available for research, but it is not required in the default deployment because the final ablation did not show a clear standalone accuracy gain.

---

## 3.1 Why Local Patch Features Were Needed

The first approach used one global image feature. This was simple, but small defects could be averaged into the overall image representation.

The solution was to keep the image spatially separated into local patch features.

```math
F = f_{\theta}(x)
```

| Symbol | Meaning |
|---|---|
| `x` | input image |
| `fθ` | frozen YOLO26 feature extractor |
| `θ` | fixed YOLO parameters |
| `F` | intermediate feature maps |

The feature maps are converted into local patches:

```math
P = \{p_1, p_2, \ldots, p_N\}
```

where `pᵢ` is one local image region.

### Improvement

```text
Global feature AUROC ≈ 84.84%
Local patch AUROC   ≈ 93.02%
```

Gain:

```math
93.02 - 84.84 = 8.18
```

**Why it stayed in the final method:**  
This was the largest representation-level improvement. Local features were clearly better for defects that occupy only a small part of the image.

---

## 3.2 Why the Patch Score Uses Only the Most Abnormal Regions

Each test patch is compared with the closest normal patch stored in the category memory bank.

Nearest-normal distance:

```math
d_i =
1 -
\max_{m \in M}
\left(
\frac{z_i^\top m}
{\|z_i\|\,\|m\|}
\right)
```

| Symbol | Meaning |
|---|---|
| `zᵢ` | current test patch |
| `m` | one normal patch in memory |
| `M` | normal patch memory |
| `dᵢ` | distance from the nearest normal patch |

Interpretation:

```text
small dᵢ  → close to normal
large dᵢ  → locally unusual
```

Instead of averaging every patch, the detector keeps only the highest local distances:

```math
S(x) =
\frac{1}{K}
\sum_{i \in TopK(d)}
d_i
```

Final setting:

```text
patch_top_fraction = 0.05
```

So only the highest **5%** patch distances contribute to the image score.

### Why this was needed

If one defect occupies only a small area, averaging all patches can hide it.

Using the most abnormal patches keeps the score sensitive to local defects while still producing one image-level value.

### Final offline setting

The offline experiment compared:

```text
Image size     : 224 / 384 / 512
Feature depth  : last1 / last2 / last3
Patch grid     : 8 / 14 / 20
Top fraction   : 0.01 / 0.03 / 0.05 / 0.10
```

Selected:

```text
img_size           = 384
feature_choice     = last2
patch_grid         = 14
patch_top_fraction = 0.05
max_patch_memory   = 16000
```

Final offline validation:

| Metric | Result |
|---|---:|
| Mean AUROC | **94.49%** |
| Accuracy | **82.36%** |
| Normal Recall | **84.59%** |
| Anomaly Recall | **80.86%** |
| Macro F1 | **78.42%** |

**Why this setup was kept:**  
It gave a strong shared result across categories without needing a different model configuration for each product type.

---

## 3.3 Why Deployment Calibration Was Added

The offline threshold was built from the original dataset, but deployment images can shift because of lighting, camera position, image quality, or production variation.

The detector therefore recalibrates its operating threshold using trusted-normal deployment images.

Trusted-normal scores:

```math
S_{normal} = \{S_1, S_2, \ldots, S_n\}
```

with:

```text
n = 20 trusted-normal images per category
```

Anomaly threshold:

```math
T_{anom}
=
Q_{0.995}(S_{normal})
+
0.10\sigma
```

Update threshold:

```math
T_{update}
=
Q_{0.90}(S_{normal})
```

with:

```math
T_{update} \le T_{anom}
```

| Term | Meaning |
|---|---|
| `S_normal` | scores from trusted normal images |
| `Q0.995` | high quantile used for anomaly decision |
| `σ` | normal-score spread |
| `T_anom` | normal/anomaly decision threshold |
| `T_update` | stricter threshold used before online learning |

### Improvement

Final ablation:

```text
Baseline Accuracy     = 79.50%
Calibration Accuracy  = 84.17%
```

Gain:

```math
84.17 - 79.50 = 4.67
```

```text
+4.67 percentage points
```

**Why calibration stayed in the final method:**  
It produced the largest deployment-level improvement and confirmed that the original offline threshold should not be assumed to work unchanged in deployment.

---

## 3.4 Why a Safe Update Gate Was Needed

Online adaptation can help the model follow new normal variation, but it can also damage the model if a defective sample is accepted as normal.

A low anomaly score alone was therefore not considered enough.

The final update rule uses two checks:

```math
G_{update}
=
G_{score}
\land
G_{cons}
```

Score gate:

```math
G_{score}
=
[S(x) < T_{update}]
```

Consistency gate:

```math
G_{cons}
=
[E_{cons} < T_{cons}]
```

where:

| Term | Meaning |
|---|---|
| `S(x)` | current anomaly score |
| `T_update` | maximum score allowed for an update candidate |
| `E_cons` | difference between weak and strong image-view features |
| `T_cons` | category-specific consistency limit |
| `G_score` | score gate pass/fail |
| `G_cons` | consistency gate pass/fail |
| `G_update` | final permission to update |

The stricter score threshold was tuned from q0.95 to q0.90.

### Safety result

| Metric | q0.95 | q0.90 |
|---|---:|---:|
| Anomaly Recall | 91.00% | **91.00%** |
| Bad Updates | 12 | **8** |
| Update Precision | 94.78% | **96.10%** |

Bad-update reduction:

```math
\frac{12-8}{12}\times100 = 33.3\%
```

**Why the gate stayed in the final method:**  
It reduced contaminated updates by about one third without reducing anomaly recall.

The consistency gate remains a secondary safety check. It did not separate normal and defective images strongly enough to replace the anomaly score.

---

## 3.5 Why Memory CTTA Was Selected Over Full CTTA

Two online adaptation paths were tested:

```text
Memory CTTA
→ update normal patch memory only

Adapter CTTA
→ update PatchAdapter only

Full CTTA
→ update both
```

The original expectation was that Full CTTA would perform best because it combines both mechanisms.

The ablation showed otherwise:

| Method | Accuracy | Macro F1 | Normal Recall |
|---|---:|---:|---:|
| Calibration | 84.17% | 84.09% | 77.33% |
| **Memory CTTA** | **85.22% ± 0.42%** | **85.19% ± 0.43%** | 80.67% ± 1.20% |
| Adapter CTTA | 84.11% ± 0.10% | 84.04% ± 0.10% | 77.22% ± 0.19% |
| Full CTTA | 85.17% ± 0.44% | 85.14% ± 0.45% | **81.00% ± 1.33%** |

### What this means

- **Memory adaptation helped.**
  It improved both Accuracy and Macro F1 over calibration-only.

- **Adapter-only learning did not help enough.**
  The adapter was proven to update correctly, but its standalone result remained close to calibration-only.

- **Full CTTA was competitive but not clearly better.**
  It gave the best normal recall, but slightly lower overall Accuracy and Macro F1 than Memory CTTA.

### Final method decision

The recommended deployment mode is:

```text
Memory CTTA
```

because it gives the best balance of:

```text
accuracy
+ Macro F1
+ simpler online state
+ lower parameter-drift risk
```

The PatchAdapter remains in the project because it is technically valid and may become useful with a stronger online objective or a larger domain shift.

---

# 4. Final Result

Five methods were compared to isolate the effect of each component.

## Method Background

| Method | Threshold | Adapter | Memory | What this test isolates |
|---|---|---:|---:|---|
| **Baseline** | Original offline | OFF | OFF | Static offline detector with no deployment calibration or online learning. |
| **Calibration** | Calibrated | OFF | OFF | Measures the effect of trusted-normal threshold calibration only. |
| **Memory CTTA** | Calibrated | OFF | ON | Measures online normal-memory adaptation without parameter learning. |
| **Adapter CTTA** | Calibrated | ON | OFF | Measures PatchAdapter learning without changing normal memory. |
| **Full CTTA** | Calibrated | ON | ON | Tests both online mechanisms together. |

Each method was tested with the same 600-image evaluation stream under seeds:

```text
42
130
2030
```

## Detection Result

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% | 71.33% | 87.67% | 79.36% | 94.70% |
| Calibration | 84.17% | 77.33% | **91.00%** | 84.09% | 94.70% |
| **Memory CTTA** | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | 94.70% ± 0.02% |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

## Does the result match the original expectation?

**Partly.**

The expected outcome was that Full CTTA would be the strongest method because it combines both memory and adapter learning.

The experiment did **not** show a clear Full CTTA advantage.

Instead:

```text
Calibration     = 84.17% accuracy
Memory CTTA     = 85.22%
Adapter CTTA    = 84.11%
Full CTTA       = 85.17%
```

### Discussion

**Calibration behaved as expected.**  
It gave the largest deployment improvement, confirming that the offline threshold did not transfer perfectly to the deployment stream.

**Memory adaptation behaved positively.**  
It gave the best accuracy and Macro F1. This suggests that updating the normal reference is useful when normal appearance changes during deployment.

**Adapter-only learning did not improve the calibrated detector.**  
The adapter clearly changed its parameters, but those changes did not translate into better detection accuracy. This means a working online optimizer is not automatically a useful adaptation mechanism.

**Full CTTA did not clearly beat Memory CTTA.**  
It achieved slightly better normal recall, but also accepted more bad updates and gave slightly lower overall accuracy.

**AUROC and thresholded accuracy move differently.**  
Memory-based CTTA improved operating accuracy while category AUROC was slightly lower. This suggests that adaptation mainly improved the operating decision boundary and local normal reference rather than global score ranking.

---

# 5. Main Findings and Final Verdict

## Verdict

For the current project, the recommended default deployment mode is:

```text
Memory CTTA
```

### Why

1. It achieved the **highest overall accuracy**:
   ```text
   85.22% ± 0.42%
   ```

2. It achieved the **highest Macro F1**:
   ```text
   85.19% ± 0.43%
   ```

3. It improves the normal reference without changing YOLO or PatchAdapter parameters.

4. It is simpler to control and easier to reset than parameter adaptation.

5. It avoids adding a learning component that did not show a clear standalone benefit in the final ablation.

### Role of Full CTTA

`full_ctta` should remain in the repository as a research and comparison mode because:

- adapter learning is technically valid;
- it achieved the best normal recall;
- it may become useful with a stronger adapter loss or longer deployment shift.

It is not selected as the default operating mode because the present evidence does not show a clear overall advantage over Memory CTTA.

---

# 6. Deployment and CiRA CORE Operation

## 6.1 Folder Setup

```text
C:\cira_ttl_model\
├── yolo26n-cls.pt
├── cira_ttl_anomaly.py
└── <category>\
    ├── patch_memory_bank.pt
    └── threshold.json

C:\cira_ttl_service\
└── app_ctta.py

C:\cira_batch_test\
├── batch_index.txt
└── <category>\
    ├── good_*.png
    └── bad_*.png

C:\cira_ttl_logs\
C:\cira_ttl_checkpoints\
```

The recommended deployment setting in `app_ctta.py` is:

```python
EXPERIMENT = "memory_ctta"
```

## 6.2 Start the Flask Service

Open PowerShell or Command Prompt:

```bat
C:\cira_ttl_env\Scripts\activate
cd /d C:\cira_ttl_service
python app_ctta.py
```

Before starting the batch, confirm the terminal shows:

```text
Starting patch CTTA Flask service
EXPERIMENT: memory_ctta
MODEL_ROOT: C:\cira_ttl_model
```

Flask listens locally at:

```text
http://127.0.0.1:5000
```

Basic checks:

```text
GET /
GET /categories
GET /config?category=bottle
```

## 6.3 Launch CiRA CORE

1. Open **CiRA CORE**.
2. Load/import the project flow:
   ```text
   test_flow_rev1.3.flow
   ```
3. Confirm the batch folder and Flask endpoint used by the flow.
4. Reset the batch index before a new full run:
   ```text
   C:\cira_batch_test\batch_index.txt = 0
   ```
5. Make sure an old stop flag is not present:
   ```text
   C:\cira_batch_test\stop.txt
   ```
6. Start the Flask service first.
7. Press **Run** in CiRA CORE.

> **Figure placeholder — CiRA CORE complete flow**
>
> Add the Run / Stop / Reset workflow screenshot here.

## 6.4 What Happens During a Batch Run

```text
Run
 ↓
python1.py
 ↓
Find current image and category
 ↓
Send image_path + category to Flask
 ↓
Flask loads / reuses category detector
 ↓
CTTA prediction
 ↓
Write prediction_log.csv
 ↓
Return JSON
 ↓
python2.py
 ↓
Display result + score + image + LED
 ↓
Advance batch_index.txt
 ↓
Next image
```

### Python1

`python1.py`:

- checks the stop flag;
- collects images from valid category folders;
- applies the selected fixed shuffle seed;
- reads `batch_index.txt`;
- returns one image path and category;
- advances the index.

### Flask

`app_ctta.py`:

- receives the current image;
- loads the category memory and threshold;
- runs the selected experiment mode;
- saves prediction diagnostics;
- returns a stable JSON response.

### Python2

`python2.py`:

- validates Flask output;
- formats the result text;
- loads the current image for display;
- maps result to LED status:

```text
GREEN = normal
RED   = anomaly
GRAY  = error
```

## 6.5 Stop and Reset

### Stop

The CiRA Stop action creates:

```text
C:\cira_batch_test\stop.txt
```

Python1 detects the file, stops the loop, then removes the flag.

### Reset

Before another complete run:

```text
batch_index.txt → 0
```

For a clean CTTA experiment, also:

```text
restart Flask
clear prediction_log.csv
restore the clean starting model state
```

This prevents the previous online state from carrying into the next experiment.

---

# 7. Limitations

- Some defective samples are still accepted for online updates.
- The current PatchAdapter gives limited standalone performance gain.
- Memory adaptation is the dominant online component in the present implementation.
- Online adaptation improves normal recall but slightly reduces anomaly recall.
- Consistency error is useful as a safety signal but weak as a standalone defect discriminator.
- The trusted-normal calibration set is small compared with the full range of possible deployment variation.
- Only three shuffled streams were tested.
- Validation is limited to MVTec AD and the current 600-image evaluation setup.
- Multi-day or multi-week adaptation drift has not yet been studied.
- The project evaluates image-level anomaly detection; pixel-level defect segmentation is outside the current scope.

---

# 8. Technical Detail

Detailed code responsibilities, execution flow, revision-by-revision experiments, findings, discussions, and next-step reasoning are kept in:

[`code/README.md`](code/README.md)

This root README is intended to explain the final system, method selection, integration, and deployment result without repeating the full experiment history.
