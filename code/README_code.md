# Code, Method and Experiment Notes

This file records how the final CTTA system is implemented and how the design changed across the experiments.

The root [`README.md`](../README.md) gives the project-level summary and final deployment result. This file stays closer to the code: what each file does, how data moves through the system, and what each revision proved.

---

## Table of Contents

- [1. Code Map](#1-code-map)
- [2. Execution Flow](#2-execution-flow)
- [3. Core Method](#3-core-method)
  - [3.1 Local Patch Representation](#31-local-patch-representation)
  - [3.2 Patch Anomaly Score](#32-patch-anomaly-score)
  - [3.3 Deployment Thresholds](#33-deployment-thresholds)
  - [3.4 Safe Update Gate](#34-safe-update-gate)
  - [3.5 PatchAdapter Update](#35-patchadapter-update)
  - [3.6 Memory Update](#36-memory-update)
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

| File | Purpose | When to use | Main flow |
|---|---|---|---|
| [`offline_train.ipynb`](offline_train.ipynb) | Build and validate the offline patch representation. | Run before deployment export or when changing patch settings. | Prepare data → frozen YOLO features → patch ablation → all-category validation → export `patch_memory_bank.pt` + `threshold.json`. |
| [`cira_ttl_anomaly.py`](cira_ttl_anomaly.py) | Core anomaly scoring and CTTA logic. | Imported by Flask for every prediction. | Preprocess → patch features → nearest-normal score → safe gate → optional adapter/memory update → final score. |
| [`app_ctta.py`](app_ctta.py) | Flask API and experiment controller. | Start before CiRA testing. | Read request → load category state → choose experiment → call detector → log → return fixed JSON. |
| [`auto_calibrate_threshold.py`](auto_calibrate_threshold.py) | Calibrate deployment anomaly and update thresholds. | Run with trusted `good_*` images before deployment testing. | Score trusted normals → q0.995 anomaly threshold → q0.90 update threshold → backup/update `threshold.json`. |
| [`calibrate_consistency.py`](calibrate_consistency.py) | Calibrate the consistency gate. | Run after trusted-normal consistency logging. | Read normal log → q0.95 consistency error → save category threshold. |
| [`python1.py`](python1.py) | CiRA batch loader and sequence controller. | First Python block in the CiRA loop. | Collect images → fixed-seed shuffle → read index → output image/category → advance index. |
| [`python2.py`](python2.py) | CiRA result parser and display formatter. | Run after the Flask REST response. | Validate JSON → parse result → choose LED → load image → output display values. |
| [`cira_test_flow.flow`](cira_test_flow.flow) | CiRA CORE workflow. | Import into CiRA CORE. | Run / Stop / Reset → Python1 → REST → Python2 → display → loop. |

---

# 2. Execution Flow

## 2.1 Offline model preparation

```text
MVTec AD
   ↓
offline_train.ipynb
   ↓
Frozen YOLO26
   ↓
Local patch embeddings
   ↓
Normal patch memory
   ↓
Offline threshold
   ↓
patch_memory_bank.pt + threshold.json
```

## 2.2 Deployment calibration

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

## 2.3 Online inference

```text
python1.py
   ↓
image path + category
   ↓
app_ctta.py
   ↓
cira_ttl_anomaly.py
   ↓
score_before
   ↓
safe update gate
   ↓
optional adapter / memory update
   ↓
score_after + prediction
   ↓
CSV log + Flask JSON
   ↓
python2.py
   ↓
CiRA result display
```

> **Figure placeholder — Code integration**
>
> Add one diagram showing how the offline export feeds the Flask/CiRA runtime.

---

# 3. Core Method

## 3.1 Local Patch Representation

YOLO26 is used as a **frozen feature extractor**. Its weights are not updated during CTTA.

For an input image \(x\):

```math
F = f_{\theta}(x)
```

where:

| Symbol | Meaning |
|---|---|
| \(x\) | input image |
| \(f_{\theta}\) | YOLO26 feature extractor |
| \(\theta\) | frozen YOLO parameters |
| \(F\) | intermediate spatial feature maps |

Selected feature maps are resized to the same grid and converted into local patch descriptors:

```math
P = \{p_1, p_2, \ldots, p_N\}
```

The final offline setting uses:

```text
img_size       = 384
feature_choice = last2
patch_grid     = 14
```

A `14 × 14` grid gives **196 local positions per selected representation** before flattening for memory comparison.

**Why this design:** the global-feature baseline was weaker for small defects. Local patches improved mean AUROC from about **84.84% to 93.02%**.

---

## 3.2 Patch Anomaly Score

Each test patch is compared with the most similar normal patch stored in the category memory bank.

### Step 1 — nearest-normal cosine distance

```math
d_i =
1 -
\max_{m \in M}
\left(
\frac{z_i^\top m}
{\|z_i\|\,\|m\|}
\right)
```

where:

| Symbol | Meaning |
|---|---|
| \(z_i\) | current test patch embedding |
| \(m\) | one normal patch in memory |
| \(M\) | category normal patch memory |
| \(d_i\) | distance from the nearest normal patch |

Interpretation:

```text
d_i close to 0  → patch looks normal
larger d_i      → patch is less similar to normal memory
```

### Step 2 — keep the most abnormal patches

```math
K =
\max
\left(
1,
\left\lceil
N \cdot r
\right\rceil
\right)
```

where:

- \(N\) = number of patch distances;
- \(r\) = `patch_top_fraction`;
- final \(r = 0.05\).

### Step 3 — image anomaly score

```math
S(x) =
\frac{1}{K}
\sum_{i \in TopK(d)}
d_i
```

Only the highest 5% patch distances contribute to the image score.

### Small example

If an image produces 196 patch distances:

```text
N = 196
r = 0.05
K = ceil(196 × 0.05) = 10
```

The detector averages the **10 most abnormal local distances**, not all 196 patches.

This avoids a small defect being hidden by many normal regions.

### Tuning performed

The offline ablation tested:

| Parameter | Values tested | Final |
|---|---|---:|
| Image size | 224, 384, 512 | **384** |
| Feature choice | last1, last2, last3 | **last2** |
| Patch grid | 8, 14, 20 | **14** |
| Top fraction | 0.01, 0.03, 0.05, 0.10 | **0.05** |

The selected shared setting reached:

```text
Mean AUROC      = 94.49%
Accuracy        = 82.36%
Normal Recall   = 84.59%
Anomaly Recall  = 80.86%
Macro F1        = 78.42%
```

---

## 3.3 Deployment Thresholds

Thresholds are recalculated from trusted-normal deployment images so the detector is not tied only to the offline score distribution.

### Anomaly threshold

```math
T_{anom}
=
Q_{0.995}(S_{normal})
+
0.10\,\sigma
```

where:

| Term | Meaning |
|---|---|
| \(S_{normal}\) | anomaly scores from trusted normal images |
| \(Q_{0.995}\) | 99.5th percentile |
| \(\sigma\) | standard deviation of trusted-normal scores |
| \(0.10\sigma\) | small safety margin |

This threshold answers:

> **Is this image normal or anomalous?**

### Update threshold

```math
T_{update}
=
Q_{0.90}(S_{normal})
```

with:

```math
T_{update} \le T_{anom}
```

This threshold answers a stricter question:

> **Is this sample safe enough to learn from?**

### Example

Assume:

```text
T_anom   = 0.060
T_update = 0.045
```

Then:

| Score | Prediction | Online learning |
|---:|---|---|
| `0.030` | Normal | Candidate for update |
| `0.052` | Normal | Rejected from update |
| `0.071` | Anomaly | Rejected from update |

So a normal prediction does **not** automatically enter CTTA.

### Why q0.90 was kept

| Metric | q0.95 | q0.90 |
|---|---:|---:|
| Accuracy | 88.33% | 88.00% |
| Anomaly Recall | 91.00% | **91.00%** |
| Bad Updates | 12 | **8** |
| Update Precision | 94.78% | **96.10%** |

Bad-update reduction:

```math
\frac{12-8}{12}\times100
=
33.3\%
```

**Decision:** keep q0.90. The small accuracy reduction was accepted because defective updates fell by about one third while anomaly recall stayed unchanged.

---

## 3.4 Safe Update Gate

The score gate is combined with a consistency check.

### Consistency error

Two noisy views of the same image are extracted:

```math
E_{cons}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
z_i^{weak}
-
z_i^{strong}
\right\|_2^2
```

The category-specific consistency threshold is calibrated from trusted normal samples:

```math
T_{cons}
=
Q_{0.95}
\left(
E_{cons}^{normal}
\right)
```

Final gate:

```math
G_{update}
=
\big[S(x) < T_{update}\big]
\land
\big[E_{cons} < T_{cons}\big]
```

### Example

```text
score_before        = 0.031
update_threshold    = 0.045     → PASS

consistency_error   = 0.00011
consistency_threshold = 0.00019 → PASS

update_allowed = True
```

If either gate fails, no online state is changed.

### Finding

The consistency error alone did not separate good and bad samples strongly. It is therefore used as a **secondary safety gate**, not as the main detector.

---

## 3.5 PatchAdapter Update

The PatchAdapter is a small learnable scale-and-bias layer applied after YOLO patch extraction.

YOLO26 stays frozen.

### Consistency loss

```math
L_{cons}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
z_i^{strong}
-
\operatorname{stopgrad}
\left(
z_i^{weak}
\right)
\right\|_2^2
```

`stopgrad()` keeps the weak view as a fixed target for that optimizer step.

### Normal-anchor loss

Each adapted patch is also pulled toward its nearest normal-memory anchor \(a_i\):

```math
L_{anchor}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
z_i^{weak}
-
a_i
\right\|_2^2
```

### Total online loss

```math
L_{online}
=
\alpha L_{cons}
+
\beta L_{anchor}
```

Final values:

```text
alpha / consistency_weight = 1.0
beta  / anchor_weight      = 0.1
online_lr                  = 1e-4
online_steps               = 1
```

Parameter update:

```math
\phi_{t+1}
=
\phi_t
-
\eta
\nabla_{\phi}
L_{online}
```

where \(\phi\) contains the PatchAdapter parameters.

### Verification result

```text
Accepted samples         = 234
Adapter updated          = 234 / 234
Mean update loss         ≈ 0.000137
Mean adapter delta norm  ≈ 0.00259
```

**Important:** this proved that parameter learning was active. It did **not** prove that adapter learning improves accuracy. That was tested later in the ablation study.

---

## 3.6 Memory Update

For an accepted sample, its current patch embeddings can be added to the normal memory:

```math
M_{t+1}
=
M_t
\cup
Z_t
```

where:

- \(M_t\) = current normal memory;
- \(Z_t\) = accepted patches from the current image.

Memory size is capped:

```math
|M_t| \le M_{max}
```

with:

```text
max_patch_memory = 16000
```

When the limit is exceeded, the implementation keeps the newest accepted patches.

**Purpose:** let the normal reference follow deployment variation without changing YOLO26.

---

# 4. Experiment History

Each revision answers one practical question. The important result is highlighted first, followed by the reason for the next experiment.

## 4.1 rev1.0

> **Main point:** first end-to-end test-time workflow was established.

**Purpose**  
Check that the basic inference path could run.

**Finding**  
The pipeline worked functionally.

**Discussion**  
The next gap was conceptual: industrial defects should not require every defect type to be known during training.

**Next**  
Move to normal-reference anomaly detection.

---

## 4.2 rev1.1

> **Main point:** changed the project direction to open-ended anomaly detection.

**Purpose**  
Use normal behaviour as the reference rather than known defect classes.

**Finding**  
The workflow became more suitable for unseen industrial defects.

**Discussion**  
A normal-reference detector can flag abnormal samples even when the exact defect class was not used in training.

**Next**  
Make the model output reusable outside the notebook.

---

## 4.3 rev1.2

> **Main point:** deployment model export became usable.

**Purpose**  
Prepare model state for Flask/CiRA use.

**Finding**  
Model-loading and export issues were resolved.

**Discussion**  
The detector could move from notebook-only testing into the deployment pipeline.

**Next**  
Handle the offline-to-deployment threshold shift.

---

## 4.4 rev1.3

> **Main point:** deployment threshold calibration was introduced.

**Purpose**  
Recalculate the operating threshold from trusted normal deployment images.

**Finding**  
The decision boundary could be adapted without retraining the backbone.

**Discussion**  
This became important later: the final ablation showed calibration alone gives the largest deployment accuracy gain.

**Next**  
Expand to all 15 categories.

---

## 4.5 rev1.4

> **Main point:** evaluation expanded to the full MVTec AD category set.

**Purpose**  
Check the workflow across all 15 categories.

**Result**

```text
Overall category accuracy:
62.53% → 79.81%
```

**Discussion**  
Full-category testing exposed larger variation between textures and objects. A shared configuration became more important than category-specific tuning.

**Next**  
Clean the implementation before representation experiments.

---

## 4.6 rev1.5

> **Main point:** maintenance revision; no new performance claim.

**Purpose**  
Simplify code and comments before changing the representation.

**Finding**  
The experiment structure became easier to compare and extend.

**Next**  
Establish a global-feature baseline.

---

## 4.7 rev1.6

> **Main point:** global YOLO features were not strong enough for small local defects.

**Purpose**  
Use one global YOLO embedding as the anomaly representation.

**Result**

```text
Mean AUROC ≈ 84.84%
```

**Discussion**  
Whole-image features dilute small defect regions.

**Next**  
Keep YOLO frozen but use local patches.

---

## 4.8 rev1.7

> **Main point:** local patches produced the largest representation improvement.

**Purpose**  
Compare local YOLO patches against a normal patch memory.

**Result**

```text
Mean AUROC ≈ 93.02%
```

Improvement over rev1.6:

```math
93.02 - 84.84 = 8.18
```

```text
+8.18 percentage points
```

**Discussion**  
Local defects no longer needed to dominate the whole-image embedding.

**Next**  
Tune the patch configuration and freeze one shared setting.

---

## 4.9 rev1.8

> **Main point:** offline representation was finalized.

**Purpose**  
Choose one configuration that performs well across categories.

**Selected setting**

```text
img_size           = 384
feature_choice     = last2
patch_grid         = 14
patch_top_fraction = 0.05
max_patch_memory   = 16000
```

**Result**

| Metric | Result |
|---|---:|
| Mean AUROC | **94.49%** |
| Accuracy | **82.36%** |
| Normal Recall | **84.59%** |
| Anomaly Recall | **80.86%** |
| Macro F1 | **78.42%** |

**Discussion**  
The representation was strong enough to freeze. Later work focused on deployment adaptation rather than changing YOLO features.

**Next**  
Verify that the online PatchAdapter really changes its parameters.

---

## 4.10 rev1.9

> **Main point:** online adapter learning was proven to be active.

**Purpose**  
Verify actual parameter change during inference.

**Result**

```text
Accepted samples         = 234
Adapter updated          = 234 / 234
Mean update loss         ≈ 0.000137
Mean adapter delta norm  ≈ 0.00259
```

**Discussion**  
This validated the mechanism, not its usefulness. A parameter can change without improving detection.

**Next**  
Make online updates safer before evaluating their contribution.

---

## 4.11 rev1.10

> **Main point:** q0.90 improved update safety with almost no detection penalty.

**Purpose**  
Reduce defective samples entering online learning.

**Result**

| Metric | q0.95 | q0.90 |
|---|---:|---:|
| Accuracy | 88.33% | 88.00% |
| Anomaly Recall | 91.00% | **91.00%** |
| Bad Updates | 12 | **8** |
| Update Precision | 94.78% | **96.10%** |

Bad-update reduction:

```math
\frac{12-8}{12}\times100 = 33.3\%
```

**Discussion**  
The stricter gate was accepted because unsafe updates matter more in a sequential system: contamination can affect later samples.

**Next**  
Check whether results depend heavily on image order.

---

## 4.12 rev1.11

> **Main point:** detection remained reasonably stable across different shuffled streams.

**Purpose**  
Remove alphabetical input-order bias.

Seeds:

```text
42
130
2030
```

**Result**

| Metric | Seed 42 | Seed 130 | Seed 2030 | Mean ± SD |
|---|---:|---:|---:|---:|
| Accuracy | 86.83% | 85.50% | 84.83% | **85.72% ± 1.02%** |
| Normal Recall | 87.00% | 81.00% | 79.67% | **82.56% ± 3.91%** |
| Anomaly Recall | 86.67% | 90.00% | 90.00% | **88.89% ± 1.92%** |
| Macro F1 | 86.83% | 85.47% | 84.79% | **85.70% ± 1.04%** |
| Pooled AUROC | 92.81% | 92.29% | 91.72% | **92.27% ± 0.54%** |

**Discussion**  
Detection metrics were relatively stable, while update counts moved more because CTTA is path-dependent.

**Next**  
Separate calibration, memory adaptation, and adapter adaptation in a final ablation.

---

## 4.13 rev1.12

> **Main point:** memory adaptation was the strongest online component.

**Purpose**  
Measure the contribution of each final component.

| Experiment | Threshold | Adapter | Memory | Mode |
|---|---|---:|---:|---|
| Baseline | original | OFF | OFF | evaluate |
| Calibration | calibrated | OFF | OFF | evaluate |
| Memory CTTA | calibrated | OFF | ON | monitor |
| Adapter CTTA | calibrated | ON | OFF | monitor |
| Full CTTA | calibrated | ON | ON | monitor |

**Result**

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% | 71.33% | 87.67% | 79.36% | 94.70% |
| Calibration | 84.17% | 77.33% | **91.00%** | 84.09% | 94.70% |
| **Memory CTTA** | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | 94.70% ± 0.02% |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

### What the ablation says

```text
Baseline → Calibration
79.50% → 84.17%
Largest deployment gain

Calibration → Memory CTTA
84.17% → 85.22%
Useful additional CTTA gain

Calibration → Adapter CTTA
84.17% → 84.11%
No standalone accuracy gain

Memory CTTA → Full CTTA
85.22% → 85.17%
No clear overall benefit from adding adapter learning
```

**Discussion**  
The result was only partly aligned with the original expectation. Full CTTA was expected to benefit from both online mechanisms, but memory-only CTTA gave slightly better overall accuracy and Macro F1.

**Conclusion**  
For the current implementation, **Memory CTTA is the strongest practical deployment mode**. Full CTTA remains useful for research because the adapter mechanism is valid and may benefit from a stronger objective or longer domain shift.

**Next**  
Move from small tuning changes to broader validation:

```text
more random sequences
long-duration adaptation
cross-dataset testing
factory-specific data
stronger adapter objective
safer update selection
```

---

# 5. Final Configuration

| Parameter | Final value | Role |
|---|---:|---|
| `img_size` | 384 | Input resolution |
| `feature_choice` | `last2` | Local YOLO feature depth |
| `patch_grid` | 14 | Spatial patch grid |
| `patch_top_fraction` | 0.05 | Fraction of highest local distances used in image score |
| `max_patch_memory` | 16000 | Maximum normal-memory size |
| `anomaly_quantile` | 0.995 | Deployment anomaly threshold |
| `update_quantile` | 0.90 | Stricter online-update threshold |
| `online_lr` | 1e-4 | PatchAdapter learning rate |
| `online_steps` | 1 | Optimizer steps per accepted image |
| `consistency_weight` | 1.0 | Consistency-loss weight |
| `anchor_weight` | 0.1 | Normal-anchor-loss weight |
| `consistency_threshold` | category-specific | Secondary safe-update gate |

---

# 6. Reproducibility

For the final ablation:

1. use the same 600-image evaluation set;
2. use seeds `42`, `130`, and `2030`;
3. restart Flask before each CTTA run;
4. start from the same clean patch memory;
5. start from the same identity-initialized PatchAdapter;
6. reset `batch_index.txt`;
7. clear the active prediction log;
8. save every run as a separate CSV.

Experiment selection in `app_ctta.py`:

```python
EXPERIMENT = "baseline"
EXPERIMENT = "calibration"
EXPERIMENT = "memory_ctta"
EXPERIMENT = "adapter_ctta"
EXPERIMENT = "full_ctta"
```

Recommended result structure:

```text
results/
├── baseline/
├── calibration/
├── memory_ctta/
├── adapter_ctta/
└── full_ctta/
```

Each folder should keep the three seed results.
