### Main Code Files

| File | Purpose |
|---|---|
| `train_rev<latest>.ipynb` | `Offline preparation notebook`. It includes `dataset preparation`, `feature extraction`, `memory bank construction`, `initial threshold calibration`, `evaluation`, and `export of model files`. |
| `app_ctta.py` | Category-aware Flask service file. It receives `image_path`, `category`, and `mode` from CiRA CORE, loads the corresponding category model files, runs prediction through the CTTA detector, and returns the JSON result to CiRA CORE. It supports `evaluate`, `monitor`, and `calibrate` modes. |
| `cira_ttl_anomaly.py` | Core `CTTA detector` file. It handles `YOLO26 feature extraction`, `adapter processing`, `normal memory bank comparison`, `anomaly score calculation`, `threshold decision`, and `optional online update`. |
| `auto_calibrate_threshold.py` | Deployment threshold calibration file. It calculates calibrated anomaly threshold and update threshold using trusted normal deployment images stored in `cira_ttl_calibration` under the target category folder. |
| 'test_flow_rev1.3.flow' | The flow shown in CiRA CORE platform. Consist of Run flow, Stop flow and Reset flow. A workable flow but need to open from CiRA CORE platfrom and with the direct setup correctly as [2.6.3.1 Deployment Folder Structure](../README.md#2631-deployment-folder-structure) |


## Code files Overview

| File | Purpose | Usage | Main Flow / Key Point |
|---|---|---|---|
| `offline_train.py` | Build the offline anomaly-detection model before deployment. | Run during model preparation and final offline validation. | Prepare MVTec data → extract frozen YOLO26 local features → test patch settings → build normal patch memory → calculate offline threshold → validate 15 categories → export `patch_memory_bank.pt` and `threshold.json`. |
| `cira_ttl_anomaly.py` | Core anomaly detector and CTTA logic. | Imported by the Flask service for every prediction. | Preprocess image → extract local patches → calculate nearest-normal anomaly score → apply safe update gate → optionally update PatchAdapter and/or memory → return prediction and CTTA diagnostics. |
| `app_ctta.py` | Flask API between CiRA CORE and the detector. | Start this service before running CiRA batch testing. | Receive image/category → load category model → select experiment mode → call detector → log result → save checkpoints when required → return stable JSON response to CiRA. |
| `auto_calibrate_threshold.py` | Calibrate deployment anomaly and update thresholds using trusted normal images. | Run before final deployment or after collecting new trusted-normal samples. | Read `good_*` images → calculate anomaly scores → use `q0.995` for anomaly threshold → use `q0.90` for update threshold → update each category `threshold.json` with backup. |
| `calibrate_consistency.py` | Calculate the consistency-gate threshold for each category. | Run after trusted-normal images have been logged with `consistency_error`. | Read prediction log → keep `good_*` samples → group by category → take 95th percentile consistency error → save `consistency_threshold` into `threshold.json`. |
| `python1.py` | CiRA batch image loader and test-sequence controller. | Used as the first Python block in the CiRA batch loop. | Check stop flag → collect valid category images → shuffle with fixed seed → read batch index → return current image/category → advance index until all images finish. |
| `python2.py` | Convert Flask prediction results into CiRA display outputs. | Used after the REST/API call in CiRA CORE. | Read Flask JSON → validate required fields → parse score/result → choose LED status → prepare display text → load current image → return clean CiRA output. |


The project is split into offline model preparation, deployment calibration, online CTTA inference, and CiRA CORE integration. Each script has one main responsibility so the training, adaptation, API, and UI workflow can be tested separately.

### Overall Code Flow

```text
offline_train.py
      ↓
patch_memory_bank.pt + threshold.json
      ↓
auto_calibrate_threshold.py
calibrate_consistency.py
      ↓
cira_ttl_anomaly.py
      ↓
app_ctta.py
      ↓
python1.py → CiRA REST call → python2.py


rev1.0
-----
- basic test-time phase flow


rev1.1
-------
- improved the test-time phase become open-ended anomaly detection

rev1.2
-------
- resolve the ultralytics load, and add few .pt save for CiRA CORE use

rev1.3
-------
- add auto calibration step

rev1.4
-------
- enable and run all 15 category to outputing the corresponding memory_bank.pt, threshold.json and ttl_adpater.pt
- improving overall category's accuracy from 62.53% to 79.81%

rev1.5
-------
- Update the comment

## Further Experiment 
### rev1.6 – Global Feature Baseline

rev1.6 evaluated whether the weak offline performance was caused mainly by the anomaly scoring formula.

The frozen YOLO26 feature extractor and normal memory bank were kept unchanged, while several scoring settings were tested:

- K=1 nearest neighbour
- K=3 / K=5 Top-K similarity
- Different reference/global weights
- Mean and median aggregation

The main evaluation metric was AUROC because it measures normal/anomaly score separation independently of the decision threshold.

**Result**

The best global formula was `k1_nearest` with a mean AUROC of approximately **0.8484**, compared with **0.8449** for the previous baseline.

The gain was small, and weak categories such as `grid`, `leather`, and `carpet` remained difficult.

**Conclusion**

Changing Top-K, weights, or aggregation alone was not enough. The main limitation was likely the feature representation rather than the scoring formula.

---

### rev1.7 – Local Patch Representation

rev1.7 introduced local intermediate YOLO26 features and a patch memory bank to preserve localized defect information.

The following representations were compared:

- Global nearest-neighbour score
- Global + patch hybrid score
- Patch-only anomaly score

**Result**

| Representation | Mean AUROC |
|---|---:|
| Global K1 | 0.8484 |
| Hybrid 75% Global / 25% Patch | 0.8977 |
| Hybrid 50% Global / 50% Patch | 0.9180 |
| Hybrid 25% Global / 75% Patch | 0.9260 |
| Patch Only | **0.9302** |

Patch features significantly improved several weak categories:

- Leather: 0.6997 → 0.9178
- Carpet: 0.7083 → 0.9209
- Pill: 0.7971 → 0.9056
- Metal Nut: 0.8993 → 0.9800
- Zipper: 0.8766 → 0.9740

Grid also improved from 0.4904 to 0.6759 but remains the weakest category.

**Conclusion**

rev1.7 shows that localized patch features are substantially better than the previous global-only embedding for industrial defect detection.

The current best offline candidate is therefore a **patch-based or patch-dominant representation**, with `patch_only` giving the highest mean AUROC.

The next experiment should focus on improving the remaining weak category, especially `grid`, by testing:
- different intermediate YOLO feature layers;
- patch grid size;
- patch memory-bank size;
- patch anomaly aggregation;
- image resolution.

After the final offline representation is fixed, the next stage is trusted-normal threshold calibration followed by CiRA CORE evaluate and monitor-mode CTTA testing.


### rev1.8 – Weak-Category Patch Optimization

**Objective**

rev1.8 focused on the remaining weak categories after rev1.7, especially `grid`, `carpet`, `leather`, `screw`, and `pill`.

The experiment tested:

- Image size: 224 / 384 / 512
- Patch grid: 8 / 14 / 20
- Different intermediate YOLO feature depths
- Patch anomaly aggregation using different top-fractions

The goal was to improve local defect representation before fixing the final offline model.

**Result**

The best fixed configurations on the weak-category experiment achieved mean AUROC above **0.92**.

The final all-category model achieved:

| Metric | Result |
|---|---:|
| Mean AUROC | **0.9449** |
| Mean Accuracy | 0.8236 |
| Mean Normal Recall | 0.8459 |
| Mean Anomaly Recall | 0.8086 |
| Mean Macro F1 | 0.7842 |

Strong categories include:

- Bottle: AUROC **1.0000**
- Hazelnut: **0.9993**
- Tile: **0.9928**
- Transistor: **0.9858**
- Cable: **0.9846**
- Metal Nut: **0.9805**
- Wood: **0.9798**
- Zipper: **0.9737**

The weakest category is now `grid` with AUROC **0.8246**, which is a clear improvement compared with the earlier global-only representation.

**Conclusion**

rev1.8 confirms that the patch-based representation substantially improves offline anomaly-score separation.

The mean AUROC increased to approximately **0.945**, indicating that the offline representation is sufficiently strong to proceed to the next stage.

Some categories still show weak thresholded accuracy or anomaly recall, especially:

- screw
- grid
- pill
- leather

However, their AUROC values indicate that the main remaining issue is threshold placement rather than feature separation.

Therefore, the offline model can now be frozen and the project should proceed to **trusted-normal deployment calibration**.

---

### rev1.9 – Restore Online Adapter Learning

**Purpose:**  
Activate and verify real PatchAdapter optimization during inference.

Previously, the system mainly updated the memory bank while the adapter optimization path was not actually executed.

**Added Verification:**

```text
update_loss
adapter_updated
adapter_delta_norm
updated_memory
```

**Result:**

```text
234 samples passed the update gate
234 / 234 adapter updates confirmed
update_loss was non-null
adapter_delta_norm > 0
```

**Conclusion:**  
Real online parameter adaptation was successfully restored and verified.

---

### rev1.10 – Safe Confidence and Consistency Update Gate

**Objective**

rev1.10 improves the safety of online test-time learning by preventing uncertain samples from directly updating the model.

The CTTA update rule was changed from a score-only condition to a dual safety gate:

```python
update_allowed = (
    score_before < update_threshold
    and consistency_error < consistency_threshold
)
```

Only samples that are:

1. confidently predicted as normal, and
2. stable under weak/strong augmentation

are allowed to update:

- the lightweight patch adapter
- the patch memory bank

The YOLO26 backbone remains frozen.


**Method**

A consistency error is measured between weakly and strongly augmented versions of the same image.

The consistency threshold was calibrated using trusted-normal images.

The first configuration used:

```text
Anomaly Quantile = 0.995
Update Quantile  = 0.95
```

The consistency gate alone reduced the total number of updates, but it did not sufficiently reduce defective-sample contamination.

Therefore, the update gate was made more conservative by changing:

```text
Update Quantile: q95 → q90
```

while keeping:

```text
Anomaly Quantile = 0.995
```

unchanged.


**q95 vs q90 Comparison**

| Metric | q95 | q90 | Change |
|---|---:|---:|---:|
| Accuracy | 88.33% | 88.00% | -0.33 pp |
| Normal Recall | 85.67% | 85.00% | -0.67 pp |
| Anomaly Recall | 91.00% | 91.00% | 0.00 pp |
| Macro F1 | 88.33% | 87.99% | -0.34 pp |
| AUROC | 93.99% | 93.06% | -0.93 pp |
| Good Samples Accepted | 218 | 197 | -21 |
| Bad Samples Accepted | 12 | 8 | -4 |
| Total Updates | 230 | 205 | -25 |
| Update Precision | 94.78% | 96.10% | +1.31 pp |

**Analysis**

The q90 configuration makes online adaptation more conservative.

The most important improvement is:

```text
Bad samples accepted for online learning:
12 → 8
```

This is a:

```text
33.3% reduction in contaminated updates
```

Update precision also improves from:

```text
94.78% → 96.10%
```

while anomaly recall remains unchanged at:

```text
91.00%
```

The trade-off is that fewer normal samples are accepted for adaptation:

```text
218 → 197
```

and overall accuracy decreases slightly:

```text
88.33% → 88.00%
```

The 0.33 percentage-point reduction in accuracy is small compared with the improvement in online-update safety.

**Final rev1.10 Configuration**

| Item | Final Setting |
|---|---|
| Anomaly Quantile | 0.995 |
| Update Quantile | 0.90 |
| Consistency Gate | Enabled |
| Patch Adapter Update | Enabled |
| Patch Memory Update | Enabled |
| YOLO26 Backbone | Frozen |

**Final Update Flow**

```text
Input Image
    ↓
Patch Feature Extraction
    ↓
Anomaly Score
    ↓
Score Confidence Gate
    ↓
Consistency Stability Gate
    ↓
Confident Normal Sample
    ↓
Online Patch-Adapter Update
    ↓
Patch Memory-Bank Update
```

**Conclusion**

Step 2 successfully improves the safety of the CTTA update mechanism.

The final q90 configuration reduces defective-sample contamination from **12 to 8 updates** and improves update precision from **94.78% to 96.10%**, while maintaining the same **91.00% anomaly recall**.

Although fewer normal samples are accepted and overall accuracy decreases slightly, the q90 configuration provides a better balance between adaptation capability and contamination control.

---

### rev1.11 Fixed-Seed Shuffled Evaluation

**Objective**

rev1.11 evaluates whether the online CTTA performance is affected by image order.

The previous evaluation used alphabetically sorted images, which could introduce sequence bias. A fixed-seed shuffle was therefore added so that good and defective samples are mixed while remaining reproducible.

```python
SHUFFLE_SEED = 42
rng = random.Random(SHUFFLE_SEED)
rng.shuffle(category_images)
```

Three independent shuffled streams were evaluated:

```text
Seed 42
Seed 130
Seed 2030
```

Each run used:

- 600 images
- 15 MVTec AD categories
- the same image set
- `MODE = monitor`
- final q90 safe-update configuration
- clean starting model state


**Results**

| Metric | Seed 42 | Seed 130 | Seed 2030 | Mean ± SD |
|---|---:|---:|---:|---:|
| Accuracy | 86.83% | 85.50% | 84.83% | **85.72% ± 1.02%** |
| Normal Recall | 87.00% | 81.00% | 79.67% | **82.56% ± 3.91%** |
| Anomaly Recall | 86.67% | 90.00% | 90.00% | **88.89% ± 1.92%** |
| Macro F1 | 86.83% | 85.47% | 84.79% | **85.70% ± 1.04%** |
| Pooled AUROC | 92.81% | 92.29% | 91.72% | **92.27% ± 0.54%** |
| Good Updates | 221 | 191 | 181 | **197.7 ± 20.8** |
| Bad Updates | 19 | 12 | 11 | **14.0 ± 4.4** |
| Total Updates | 240 | 203 | 192 | **211.7 ± 25.1** |
| Update Precision | 92.08% | 94.09% | 94.27% | **93.48% ± 1.21%** |

Category-averaged AUROC remained very stable:

```text
Seed 42   = 93.08%
Seed 130  = 93.41%
Seed 2030 = 93.14%
```

**Analysis**

The shuffled experiments show that CTTA performance remains reasonably consistent across different input orders.

The main detection metrics vary only slightly:

```text
Accuracy SD   ≈ 1.02%
Macro F1 SD   ≈ 1.04%
AUROC SD      ≈ 0.54%
```

This indicates that the model is not strongly dependent on one specific image sequence.

The number of online updates varies more between seeds because CTTA is sequential: earlier accepted samples change the adapter and memory bank and therefore influence later update decisions.

Despite this, anomaly recall remains stable at approximately:

```text
88.89% ± 1.92%
```

and update precision remains above 92% for all three streams.

**Conclusion**

rev1.11 confirms that the proposed online CTTA method is reasonably robust to different image orders.

Using fixed random seeds also provides a reproducible evaluation protocol and removes the bias caused by the previous alphabetical good/bad ordering.

---

### rev1.12 Final Ablation Study

**Objective**

rev1.12 evaluates the contribution of each component in the proposed online test-time learning framework.

Five configurations were compared using the same three fixed random seeds:

```text
42
130
2030
```

| Experiment | Calibration | Adapter Update | Memory Update |
|---|---:|---:|---:|
| Baseline | No | OFF | OFF |
| Calibration | Yes | OFF | OFF |
| Memory CTTA | Yes | OFF | ON |
| Adapter CTTA | Yes | ON | OFF |
| Full CTTA | Yes | ON | ON |

Each experiment used the same 600-image dataset and identical shuffled sequence for the corresponding seed.

**Final Ablation Results**

| Method | Accuracy | Normal Recall | Anomaly Recall | Macro F1 | Category AUROC |
|---|---:|---:|---:|---:|---:|
| Baseline | 79.50% ± 0.00% | 71.33% ± 0.00% | 87.67% ± 0.00% | 79.36% ± 0.00% | **94.70% ± 0.00%** |
| Calibration | 84.17% ± 0.00% | 77.33% ± 0.00% | **91.00% ± 0.00%** | 84.09% ± 0.00% | **94.70% ± 0.00%** |
| Memory CTTA | **85.22% ± 0.42%** | 80.67% ± 1.20% | 89.78% ± 0.38% | **85.19% ± 0.43%** | 93.11% ± 0.05% |
| Adapter CTTA | 84.11% ± 0.10% | 77.22% ± 0.19% | **91.00% ± 0.00%** | 84.04% ± 0.10% | **94.70% ± 0.02%** |
| Full CTTA | 85.17% ± 0.44% | **81.00% ± 1.33%** | 89.33% ± 0.88% | 85.14% ± 0.45% | 93.14% ± 0.18% |

**Online Update Behaviour**

| Method | Good Updates | Bad Updates | Total Updates | Update Precision |
|---|---:|---:|---:|---:|
| Memory CTTA | 183.0 | 10.0 | 193.0 | **94.82% ± 0.14%** |
| Adapter CTTA | 174.3 | **8.0** | 182.3 | **95.61% ± 0.01%** |
| Full CTTA | 186.7 | 12.7 | 199.3 | 93.66% ± 0.55% |

Adapter behaviour:

```text
Adapter CTTA:
~182 adapter updates/run
Memory updates = 0

Full CTTA:
~199 adapter updates/run
~199 memory updates/run
```

The average adapter parameter change remained non-zero, confirming that online parameter learning was active.

### Component Analysis

**Baseline → Calibration**

Calibration provides the largest single improvement:

```text
Accuracy:
79.50% → 84.17%   (+4.67 pp)

Normal Recall:
71.33% → 77.33%   (+6.00 pp)

Anomaly Recall:
87.67% → 91.00%   (+3.33 pp)
```

This confirms that trusted-normal deployment calibration is important for selecting an appropriate operating threshold.


**Calibration → Memory CTTA**

Online memory adaptation further improves:

```text
Accuracy:
84.17% → 85.22%

Macro F1:
84.09% → 85.19%

Normal Recall:
77.33% → 80.67%
```

Memory adaptation therefore contributes the clearest additional improvement after calibration.


**Calibration → Adapter CTTA**

Adapter-only CTTA produces almost the same detection performance as Calibration:

```text
Accuracy:
84.17% → 84.11%

Macro F1:
84.09% → 84.04%
```

Although the adapter parameters are actively updated, the current lightweight adapter provides limited direct performance gain when used alone.


**Memory CTTA → Full CTTA**

Full CTTA gives:

```text
Memory CTTA accuracy = 85.22%
Full CTTA accuracy   = 85.17%
```

The difference is very small.

Full CTTA achieves slightly better normal recall:

```text
80.67% → 81.00%
```

but also accepts more defective samples for adaptation.

Therefore, combining adapter and memory learning does not clearly outperform memory-only CTTA under the current configuration.


**Interpretation**

The ablation study shows that:

1. **Trusted-normal calibration is essential** and provides the largest performance improvement.
2. **Online memory adaptation provides the strongest CTTA contribution** after calibration.
3. The lightweight online adapter is genuinely learning, but its standalone detection benefit is limited.
4. Full CTTA remains competitive and stable across random image orders, but does not significantly outperform memory-only adaptation.
5. The safe update gate limits most online updates to normal samples, although some defective-sample contamination remains.


**Conclusion**

The final experiments confirm that online adaptation can improve deployment performance beyond the static calibrated model.

The strongest result is obtained using online memory adaptation:

```text
Baseline Accuracy      = 79.50%
Calibration Accuracy   = 84.17%
Memory CTTA Accuracy   = 85.22%
Adapter CTTA Accuracy  = 84.11%
Full CTTA Accuracy     = 85.17%
```

The results indicate that the main performance gain of the proposed CTTA framework comes from adaptive normal-memory updating, while the lightweight adapter provides a smaller contribution.

The ablation therefore provides clear evidence of the individual contribution of each component and identifies memory adaptation as the most effective online adaptation mechanism in the current implementation.

