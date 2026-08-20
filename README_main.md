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

The final method is the result of several experiments rather than one fixed design chosen at the beginning.

```text
Global image feature
        ↓
Local patch feature
        ↓
Offline patch tuning
        ↓
Verified online adapter learning
        ↓
Safe score + consistency gate
        ↓
Random-order testing
        ↓
Final component ablation
```

The main learning from these experiments was:

1. **Global features were too coarse for small defects.**
2. **Local patch features gave the largest representation gain.**
3. **Deployment calibration was necessary because offline and deployment score distributions differed.**
4. **Online updates needed a safety gate because bad samples could contaminate the normal memory.**
5. **Memory adaptation improved performance more reliably than adapter-only learning.**

For that reason, the recommended final deployment method is:

```text
Frozen YOLO26
+ local patch features
+ calibrated thresholds
+ score gate
+ consistency gate
+ online patch-memory update
```

The PatchAdapter remains supported, but is not required for the default deployment mode.

---

## 3.1 Local Patch Representation

The frozen YOLO26 backbone extracts intermediate feature maps:

$$
F = f_{\theta}(x)
$$

where:

- \(x\) = input image;
- \(f_{\theta}\) = YOLO26 feature extractor;
- \(\theta\) = frozen YOLO parameters;
- \(F\) = intermediate spatial feature maps.

Selected feature maps are resized to the same spatial grid and converted into patch descriptors:

$$
P = \{p_1,p_2,\ldots,p_N\}
$$

where \(p_i\) is one local patch feature.

### Why local patches were kept

The global-feature baseline reached about:

```text
Mean AUROC = 84.84%
```

After switching to local patch representation:

```text
Mean AUROC = 93.02%
```

This is an improvement of:

$$
93.02 - 84.84 = 8.18 \text{ percentage points}
$$

The result showed that local features were much better for defects that occupy only a small part of the image.

---

## 3.2 Patch Anomaly Score

Each test patch is compared with the closest normal patch in the category memory bank.

### Nearest-normal distance

$$
d_i =
\min_{m \in M}
\left(
1 -
\frac{z_i^\top m}
{\|z_i\|\|m\|}
\right)
$$

where:

- \(z_i\) = current test patch embedding;
- \(m\) = one patch stored in the normal memory;
- \(M\) = normal patch memory bank;
- \(d_i\) = nearest-normal cosine distance.

Interpretation:

```text
small d_i  → patch looks similar to normal memory
large d_i  → patch is locally unusual
```

### Image-level score

Using every patch would dilute small local defects, so only the most abnormal patches are aggregated:

$$
K =
\max
\left(
1,
\left\lceil
N \cdot r
\right\rceil
\right)
$$

$$
S(x) =
\frac{1}{K}
\sum_{i \in TopK(d)}
d_i
$$

where:

- \(N\) = total number of image patches;
- \(r\) = `patch_top_fraction`;
- \(K\) = number of highest-distance patches used;
- \(S(x)\) = image anomaly score.

Final setting:

```text
patch_top_fraction = 0.05
```

So the score uses the highest **5%** local patch distances.

### Tuning behind the final patch setup

The offline experiment compared:

- input size: `224`, `384`, `512`;
- feature depth: `last1`, `last2`, `last3`;
- patch grid: `8`, `14`, `20`;
- top fraction: `0.01`, `0.03`, `0.05`, `0.10`.

The selected shared configuration was:

```text
img_size           = 384
feature_choice     = last2
patch_grid         = 14
patch_top_fraction = 0.05
max_patch_memory   = 16000
```

It was chosen because it gave a strong mean result without using a different configuration for each category.

Final offline validation:

| Metric | Result |
|---|---:|
| Mean AUROC | **94.49%** |
| Accuracy | **82.36%** |
| Normal Recall | **84.59%** |
| Anomaly Recall | **80.86%** |
| Macro F1 | **78.42%** |

---

## 3.3 Deployment Calibration

Offline thresholds do not always transfer cleanly to deployment because image conditions can shift.

The trusted-normal deployment scores are:

$$
\mathcal{S}_{normal}
=
\{S_1,S_2,\ldots,S_n\}
$$

with:

```text
n = 20 trusted-normal images per category
```

### Anomaly threshold

$$
T_{anom}
=
Q_{0.995}(\mathcal{S}_{normal})
+
\lambda \sigma
$$

where:

- \(Q_{0.995}\) = 99.5th percentile of trusted-normal scores;
- \(\sigma\) = standard deviation of trusted-normal scores;
- \(\lambda\) = safety-margin factor;
- final `SAFETY_MARGIN_STD_FACTOR = 0.10`.

This threshold is used for the final normal/anomaly decision.

### Update threshold

Online learning uses a stricter boundary:

$$
T_{update}
=
Q_{0.90}(\mathcal{S}_{normal})
$$

subject to:

$$
T_{update} \le T_{anom}
$$

This separates two decisions:

```text
Is the image normal?
        vs
Is the image safe enough to learn from?
```

### Why q0.90 was selected

Two update quantiles were compared:

| Metric | q0.95 | q0.90 |
|---|---:|---:|
| Accuracy | 88.33% | 88.00% |
| Anomaly Recall | 91.00% | **91.00%** |
| Bad Updates | 12 | **8** |
| Update Precision | 94.78% | **96.10%** |

Bad-update reduction:

$$
\frac{12-8}{12}\times100
\approx 33.3\%
$$

q0.90 was kept because it reduced contaminated updates while preserving anomaly recall.

---

## 3.4 Safe Update Gate

A low anomaly score alone is not enough to trust a sample for online learning.

The final update rule requires both the score gate and consistency gate.

### Score gate

$$
G_{score}
=
[S(x) < T_{update}]
$$

### Consistency error

Two perturbed versions of the same image are processed:

$$
E_{cons}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
z_i^{weak}
-
z_i^{strong}
\right\|_2^2
$$

where:

- \(z_i^{weak}\) = patch feature after weak perturbation;
- \(z_i^{strong}\) = patch feature after stronger perturbation;
- \(E_{cons}\) = feature consistency error.

The category-specific consistency threshold is calibrated from trusted-normal data:

$$
T_{cons}
=
Q_{0.95}(E_{cons}^{normal})
$$

Consistency gate:

$$
G_{cons}
=
[E_{cons} < T_{cons}]
$$

Final gate:

$$
G_{update}
=
G_{score}
\land
G_{cons}
$$

### What the consistency experiment showed

The consistency gate was useful as a second safety condition, but consistency error alone did not clearly separate normal and defective images.

Therefore it is kept as a **supporting gate**, not the main anomaly detector.

---

## 3.5 Online Adaptation

Two online mechanisms were tested separately.

### Memory update

For an accepted normal-like image:

$$
M_{t+1}
=
M_t \cup Z_t
$$

where \(Z_t\) contains the accepted patch embeddings.

Memory size is capped:

$$
|M_t| \le M_{max}
$$

with:

```text
M_max = 16000 patches
```

When the limit is reached, the implementation keeps the newest accepted patches.

### PatchAdapter update

The PatchAdapter applies learnable scale and bias to patch features.

Its online loss combines consistency and normal-memory anchoring:

$$
L_{online}
=
\alpha L_{cons}
+
\beta L_{anchor}
$$

with:

```text
alpha = 1.0
beta  = 0.1
```

and:

$$
\phi_{t+1}
=
\phi_t
-
\eta
\nabla_{\phi}
L_{online}
$$

where:

- \(\phi\) = PatchAdapter parameters;
- \(\eta\) = online learning rate;
- `online_lr = 1e-4`;
- `online_steps = 1`.

The adapter verification test confirmed that learning was active:

```text
234 accepted samples
234 / 234 adapter updates
mean update loss        ≈ 0.000137
mean adapter delta norm ≈ 0.00259
```

This proved that the adapter changes online. The later ablation tested whether those changes improve detection performance.

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
