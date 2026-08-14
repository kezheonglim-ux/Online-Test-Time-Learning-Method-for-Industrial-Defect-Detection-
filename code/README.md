### Main Code Files

| File | Purpose |
|---|---|
| `train_rev<latest>.ipynb` | `Offline preparation notebook`. It includes `dataset preparation`, `feature extraction`, `memory bank construction`, `initial threshold calibration`, `evaluation`, and `export of model files`. |
| `app_ctta.py` | Category-aware Flask service file. It receives `image_path`, `category`, and `mode` from CiRA CORE, loads the corresponding category model files, runs prediction through the CTTA detector, and returns the JSON result to CiRA CORE. It supports `evaluate`, `monitor`, and `calibrate` modes. |
| `cira_ttl_anomaly.py` | Core `CTTA detector` file. It handles `YOLO26 feature extraction`, `adapter processing`, `normal memory bank comparison`, `anomaly score calculation`, `threshold decision`, and `optional online update`. |
| `auto_calibrate_threshold.py` | Deployment threshold calibration file. It calculates calibrated anomaly threshold and update threshold using trusted normal deployment images stored in `cira_ttl_calibration` under the target category folder. |
| 'test_flow_rev1.3.flow' | The flow shown in CiRA CORE platform. Consist of Run flow, Stop flow and Reset flow. A workable flow but need to open from CiRA CORE platfrom and with the direct setup correctly as [2.6.3.1 Deployment Folder Structure](../README.md#2631-deployment-folder-structure) |


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

## More intense experiment to improve further each categories
### rev1.6 – Scoring Formula Ablation

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

### rev1.9 – Safe Confidence and Consistency Update Gate

**Objective**

rev1.9 improves the safety of online test-time learning by preventing uncertain samples from directly updating the model.

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

---

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

---

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

---

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

---

**Final rev1.9 Configuration**

| Item | Final Setting |
|---|---|
| Anomaly Quantile | 0.995 |
| Update Quantile | 0.90 |
| Consistency Gate | Enabled |
| Patch Adapter Update | Enabled |
| Patch Memory Update | Enabled |
| YOLO26 Backbone | Frozen |

---

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

---

**Conclusion**

Step 2 successfully improves the safety of the CTTA update mechanism.

The final q90 configuration reduces defective-sample contamination from **12 to 8 updates** and improves update precision from **94.78% to 96.10%**, while maintaining the same **91.00% anomaly recall**.

Although fewer normal samples are accepted and overall accuracy decreases slightly, the q90 configuration provides a better balance between adaptation capability and contamination control.


