# 1.0 Article Review

This research review records the main technical options considered during the project and why some ideas were selected while others were kept as alternatives.

The earlier review focused on industrial defect detection, model choices, learning strategies and low-code platforms. The later review extends the same structure to cover the patch-based anomaly method, safer online adaptation, memory management, streaming deployment and operator feedback.

The selection logic follows the implementation ideas used in the main project:

- frozen pre-trained feature extractor;
- normal patch memory;
- nearest-neighbour anomaly scoring;
- category-specific thresholds;
- trusted-normal deployment calibration;
- safe online test-time adaptation;
- lightweight deployment through Flask and CiRA CORE.

Main project reference: [2.2.1 Key Implementation Ideas and Supporting References](../README.md#221-key-implementation-ideas-and-supporting-references)

---

## Table of Contents

- [1.1 Overview of Industrial Defect Detection](#11-overview-of-industrial-defect-detection)
- [1.2 Summary of Literature Review Directions](#12-summary-of-literature-review-directions)
- [1.3 Modelling Option Analysis](#13-modelling-option-analysis)
- [1.4 Comparison of Alternative Learning Methods](#14-comparison-of-alternative-learning-methods)
- [1.5 Low-Code / Visual Platform Alternative Analysis](#15-low-code--visual-platform-alternative-analysis)
- [1.6 Patch-Based Anomaly Detection and Memory Options](#16-patch-based-anomaly-detection-and-memory-options)
- [1.7 Safe Test-Time Adaptation and Update Reliability](#17-safe-test-time-adaptation-and-update-reliability)
- [1.8 Continual and Streaming Memory Management](#18-continual-and-streaming-memory-management)
- [1.9 Human-in-the-Loop and Operator Feedback](#19-human-in-the-loop-and-operator-feedback)
- [1.10 Current Research Verdict and Future Options](#110-current-research-verdict-and-future-options)

---

# 1.1 Overview of Industrial Defect Detection

| Research Area | Modelling in Use | Methodology in Use | Platform / Setting | Strength | Challenge Faced | Link / Reference |
|---|---|---|---|---|---|---|
| Industrial anomaly segmentation with test-time learning | Anomaly segmentation model with lightweight test-time classifier | Uses test-time training to improve binary anomaly segmentation from anomaly score maps | Research implementation | Uses test information during inference and reduces dependence on fixed validation thresholds | Mainly segmentation refinement, not a complete online defect-detection and low-code workflow | Costanzino et al. (2024), [Test Time Training for Industrial Anomaly Segmentation](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Costanzino_Test_Time_Training_for_Industrial_Anomaly_Segmentation_CVPRW_2024_paper.html) |
| PCB class-incremental defect detection | PCB-YOLOX / YOLOX-S | Class-incremental learning with feature enhancement | PCB inspection | Improves small and weak defect recognition | Requires structured incremental training and labelled defect data | Ge et al. (2025), IEEE |
| PCB continuous defect detection under domain shift | Domain-incremental learning | Separates domain-invariant and domain-specific features | PCB inspection | Adapts to changing defect distributions while reducing forgetting | Requires domain-level training/update strategy | Yan et al. (2025), IEEE |
| Incremental meta defect detection for PCB | Incremental meta-learning detector | Meta-learning + incremental learning | PCB inspection | Useful for tiny defects and new categories with limited samples | Structured learning process; not direct test-time operation | Gung et al. (2022), IEEE |
| General continuous defect detection | Incremental knowledge-learning framework | Distillation, feature alignment and incremental updates | General defect detection | Retains previous knowledge while adding new categories | Usually staged rather than inference-time adaptation | Sun et al. (2024), IEEE TIM |
| Incremental defect detection with distillation | Elastic heterogeneous distillation | Knowledge transfer between old and new classes | Defect detection | Reduces catastrophic forgetting | Requires incremental sessions and labelled new-class samples | Shen et al. (2025), [DBLP](https://dblp.org/rec/journals/tase/ShenLZJZY25) |
| Flexible printed circuit defect detection | Adaptive dual-teacher incremental learning | Dual-teacher learning + feature distillation | Flexible PCB inspection | Balances old and new defect knowledge | Training-session based and relatively complex | Xiong et al. (2024), [DOI](https://doi.org/10.1016/j.compeleceng.2024.109337) |
| Photovoltaic class-incremental defect detection | Multi-scale feature decoupling | Similarity distillation | Photovoltaic inspection | Supports new defect classes while reducing forgetting | Domain-specific and not direct online adaptation | Wang et al. (2024), [DOI](https://doi.org/10.1016/j.measurement.2023.113997) |
| Weld surface online defect detection | GI2DPCA + FNN | Incremental feature extraction and recognition | Weld inspection | Supports online recognition with low delay | More traditional feature pipeline; less aligned with patch memory / CTTA | Wang et al. (2022), [DOI](https://doi.org/10.1016/j.eswa.2021.116407) |
| Unknown wafer defect detection | CNN incremental learning + reference buffer | Detects unknown wafer defects through incremental updates | Wafer inspection | Reduces full retraining need | Still relies on incremental learning setup | Zhao et al. (2024), [DOI](https://doi.org/10.1016/j.ress.2024.109966) |
| Wafer defect detection with limited samples | Incremental learning strategies | Limited-sample adaptation | Wafer maps | Relevant to rare / unknown defects | Specific to wafer data and structured updates | Ni et al. (2026), [DOI](https://doi.org/10.1016/j.compind.2025.104432) |
| Wafer defect detection with memory replay | Prompt-aware memory replay | Continual memory replay in a large industrial model | Wafer inspection | Shows value of memory/replay for adaptive inspection | Heavier and more complex than the current project target | Yue et al. (2026), [DOI](https://doi.org/10.1016/j.asoc.2026.114815) |
| Wafer surface defect pattern detection | ResNet incremental learning | Continuous defect-pattern learning | Wafer classification | Supports continuous learning | Focused on classification, not low-code online deployment | Yu et al. (2021) |
| Few-shot incremental surface defect detection | IF-DETR | Incremental few-shot transformer detector | Surface defect detection | Learns new defect categories from few examples | Requires labelled support data and fine-tuning | Yang et al. (2025), IJCNN |
| Few-shot packaging defect detection | Few-shot class-incremental model | Class feature interaction | Packaging inspection | Useful when only a few labelled defect examples exist | Still requires labels and training | Zhu et al. (2024), IEEE |
| Real-world industrial anomaly detection | ReinADNet | Normal-reference contrastive comparison | Real-world anomaly benchmark | Relevant to subtle defects, unaligned images and normal-reference comparison | Focuses on model/benchmark rather than online memory and low-code deployment | Wang et al. (2025), [OpenReview](https://openreview.net/forum?id=wEH5YGPSTx) |

### Verdict - 1.1

The review shows two broad directions:

1. **Supervised / incremental defect learning** works well when new defect labels and planned training sessions are available.
2. **Normal-reference anomaly detection and test-time adaptation** better match this project because the project should work with unknown defects and limited labelled anomaly data.

**Selection for this project:** keep the second direction. The main project already uses a frozen feature extractor, normal memory, deployment calibration and online adaptation, which matches the implementation ideas listed in the main README.

---

# 1.2 Summary of Literature Review Directions

| Main Direction | Summary | Fit to Current Project |
|---|---|---|
| Deep learning defect detection | CNN, YOLO, ResNet, DETR and transformers can achieve strong supervised detection. | Useful as feature-extractor background, but labelled boxes/classes are not assumed. |
| Incremental learning | Learns new categories or domains while controlling forgetting. | Relevant as a comparison, but usually requires staged updates and labelled data. |
| Few-shot / unknown defect detection | Helps when new defect examples are scarce. | Useful future option if a small labelled defect set becomes available. |
| Real-world anomaly detection | Focuses on lighting, alignment, subtle defects and normal-reference comparison. | **High relevance** to deployment conditions. |
| Test-time learning / adaptation | Uses test-time data to adjust the detector during inference. | **Core project direction.** |
| Low-code implementation | Connects the model to a visual workflow for operation. | **Core deployment requirement.** Existing literature is still limited compared with algorithm-only research. |

### Verdict - 1.2

The project should not be treated as only a model-selection problem. It needs to combine:

```text
anomaly representation
+ deployment calibration
+ safe online adaptation
+ practical low-code operation
```

This is why the implementation combines ideas from PatchCore-style memory, test-time adaptation, threshold calibration and CiRA CORE instead of copying one paper end-to-end.

---

# 1.3 Modelling Option Analysis

| Option | Type | Strength | Weakness | Fit to Project | Reference |
|---|---|---|---|---|---|
| ResNet / WideResNet | CNN feature extractor | Stable ImageNet features; common in PatchCore | Additional anomaly scoring is still needed | Strong baseline / alternative backbone | Roth et al. (2022), [PatchCore](https://arxiv.org/abs/2106.08265) |
| EfficientNet | CNN feature extractor | Good accuracy-efficiency balance | Feature-layer choice needs tuning | Possible efficiency alternative | Tan & Le (2019), [EfficientNet](https://arxiv.org/abs/1905.11946) |
| ViT | Transformer feature extractor | Strong global representation | Heavier and more tuning-intensive | Lower priority for current low-code deployment | Dosovitskiy et al. (2020), [ViT](https://arxiv.org/abs/2010.11929) |
| Faster R-CNN | Two-stage detector | Strong localization | Slow and requires labelled bounding boxes | Poor fit to current anomaly-only dataset setup | Ren et al. (2015), [Faster R-CNN](https://arxiv.org/abs/1506.01497) |
| SSD | One-stage detector | Fast and lightweight | Weaker on subtle/small defects; supervised labels required | Not preferred | Liu et al. (2016), [SSD](https://arxiv.org/abs/1512.02325) |
| DETR / RT-DETR | Transformer detector | End-to-end detection and strong localization | More compute and labelled detection data | Not preferred for this project scope | Carion et al. (2020), [DETR](https://arxiv.org/abs/2005.12872) |
| CLIP / DINOv2 | Foundation feature extractor | Strong transferable features; useful for zero-/few-shot settings | Larger and more complex for local deployment | Strong future comparison, especially if YOLO features become the bottleneck | Radford et al. (2021), [CLIP](https://arxiv.org/abs/2103.00020); Oquab et al. (2023), [DINOv2](https://arxiv.org/abs/2304.07193) |
| YOLO26 | Fast visual feature extractor | Lightweight and easy to integrate with Flask/CiRA | Not designed specifically for anomaly patches | **Selected backbone** for deployment practicality | Ultralytics documentation + YOLO-family literature |

### Verdict - 1.3

**YOLO26 remains selected as the practical backbone**, not because it is proven to be the best anomaly feature extractor, but because it fits the project constraints:

- fast local inference,
- already integrated with Flask and CiRA CORE,
- can remain frozen,
- no bounding-box retraining is required.

The review also identifies **DINOv2 / CLIP as the strongest future backbone comparison** if later work focuses on feature quality rather than deployment simplicity.

---

# 1.4 Comparison of Alternative Learning Methods

| Method | How It Works | Strength | Weakness / Risk | Effort with Current YOLO26 Setup | Verdict |
|---|---|---|---|---|---|
| Offline retraining | Collect, label and retrain | Controlled update | Slow and label-heavy | High | Not selected |
| Transfer learning / fine-tuning | Fine-tune pretrained model | Strong with enough labels | Risk of overfitting; separate training stage | Medium–High | Future option if labelled factory data exists |
| Incremental learning | Learn new stages/classes while reducing forgetting | Supports continuous known-class expansion | Needs staged training and forgetting control | High | Not selected for current inference-time goal |
| Few-shot learning | Learn from a few labelled examples | Useful for new defect types | Still needs support labels and training design | High | Future option |
| Unsupervised anomaly detection | Model normal data and detect deviation | Works without labelled defects | Static memory / threshold may drift | Low–Medium | **Selected base direction** |
| Self-supervised learning | Learn representations without labels | Can improve feature quality | Extra pretraining and validation | High | Future backbone option |
| Active learning | Ask operator to label uncertain cases | Uses human effort efficiently | Needs feedback/annotation workflow | Medium–High | Strong future extension |
| Rule-based inspection | Fixed vision rules | Simple and explainable | Weak under complex variation | Low | Only suitable as supplementary checks |
| Online test-time learning with memory | Update trusted normal reference during inference | No full retraining; adapts to deployment | Memory contamination risk | Medium | **Selected CTTA direction** |

### Verdict - 1.4

The selected route is:

```text
unsupervised normal-reference anomaly detection
+ deployment calibration
+ online memory adaptation
```

This directly supports the main README's implementation ideas: frozen feature extraction, nearest-neighbour comparison, memory updates and trusted-normal calibration.

The main unresolved problem from this section is **update reliability**, which motivated the later score gate and consistency gate.

---

# 1.5 Low-Code / Visual Platform Alternative Analysis

| Platform | Type | Strength | Weakness | Fit to Project |
|---|---|---|---|---|
| Node-RED | Flow-based IoT / API platform | Strong API, event, automation and device integration | Custom AI/image UI may need extra nodes/code | Strong alternative |
| KNIME | Visual data-science platform | Good experiment/data workflow | Less natural for live inspection control | Better for offline analytics |
| Orange | Visual ML / education tool | Easy interactive analysis | Limited production control workflow | Low |
| Microsoft Power Apps / Power Automate | Enterprise workflow platform | Business process / Microsoft integration | Cloud, connector and licensing complexity for local CV | Medium |
| Streamlit | Python web UI | Fast dashboard and image-result development | Not node-based low-code; requires Python | Good demo alternative |
| Gradio | Python ML interface | Very fast model demo UI | Not industrial flow control | Good demo/testing alternative |
| CiRA CORE | Visual low-code industrial / AI workflow | Node-based API connection, image/result display and control flow | Smaller ecosystem and public documentation | **Selected platform** |

References: [Node-RED](https://nodered.org/), [KNIME](https://www.knime.com/), [Orange](https://orangedatamining.com/), [Microsoft Power Platform](https://www.microsoft.com/en-us/power-platform), [Streamlit](https://streamlit.io/), [Gradio](https://www.gradio.app/).

### Verdict - 1.5

**CiRA CORE remains selected** because the project already requires:

```text
visual flow control
+ Flask API calls
+ image display
+ result / score / threshold display
+ Run / Stop / Reset operation
```

Node-RED remains the strongest alternative if the project later shifts toward larger IoT/device integration. Streamlit or Gradio are better if the goal changes from low-code industrial workflow to a polished web demo.

---

# 1.6 Patch-Based Anomaly Detection and Memory Options

The earlier review was completed before the project moved from global embeddings to local patch features. This section adds the memory-based approaches that are most relevant to the current architecture.

| Option | Main Idea | Strength | Risk / Cost | Relevance to Current Project | Reference |
|---|---|---|---|---|---|
| PatchCore | Store representative normal patch features and use nearest-neighbour distance | Strong industrial anomaly baseline; no defect labels required | Memory/search cost; static bank by default | **Directly supports current local patch + normal memory design** | Roth et al. (2022), [PatchCore](https://arxiv.org/abs/2106.08265) |
| FAPM | Patch-wise / layer-wise adaptive memory and sampling | Focuses on real-time efficiency | Different memory construction from current code | Useful if inference/memory speed becomes a bottleneck | Kim et al. (2022), [FAPM](https://arxiv.org/abs/2211.07381) |
| SoftPatch | Patch-level noise filtering before memory-bank construction | Designed for noisy normal training data | More memory filtering logic | **High-value future option** for preventing contaminated online memory | Jiang et al. (2024), [SoftPatch](https://arxiv.org/abs/2403.14233) |
| DMAD | Normal + abnormal dual memory | Can use available anomaly supervision | Needs reliable anomalous samples and extra memory logic | Future option if confirmed defect samples are available | Hu et al. (2024), [DMAD](https://arxiv.org/abs/2403.12362) |
| AnomalousPatchCore | Fine-tunes features using normal and anomaly examples before memory scoring | Can exploit available anomaly labels | Breaks the current fully frozen / label-light setup | Not selected now; useful if factory anomaly labels grow | Koshil et al. (2024), [AnomalousPatchCore](https://arxiv.org/abs/2408.15113) |
| High-resolution tiled anomaly detection | Split high-resolution images into tiles / ensembles | Preserves very small defects under GPU limits | More inference passes and aggregation logic | Useful if current 384 input misses tiny defects | Rolih et al. (2024), [CVPRW](https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Rolih_Divide_and_Conquer_High-Resolution_Industrial_Anomaly_Detection_via_Memory_Efficient_CVPRW_2024_paper.html) |
| Mahalanobis PatchCore | Covariance-aware patch distance + streaming-compatible memory construction | Models feature correlation and reduces memory pressure | More statistics and transformation logic; recent preprint | Worth testing if cosine distance becomes limiting | Ferrari et al. (2026), [arXiv](https://arxiv.org/abs/2605.27748) |
| RareCLIP | Online zero-shot anomaly detection with prototype patch memory and rarity estimation | Strong online / zero-shot direction; dynamic prototypes | Heavier CLIP-based pipeline and different scoring design | Strong research alternative, not immediate replacement | He et al. (2025), [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/He_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection_ICCV_2025_paper.html) |

### Project Evidence

The project experiment already showed why patch memory was selected:

```text
Global feature mean AUROC ≈ 84.84%
Local patch mean AUROC   ≈ 93.02%
Final tuned patch AUROC  ≈ 94.49%
```

### Verdict - 1.6

**Keep the current PatchCore-style local memory as the main design.**

Most useful next comparisons:

1. **SoftPatch-style memory filtering** - directly addresses the project's remaining bad-update contamination.
2. **Adaptive / coreset memory management** - useful for longer streams and memory limits.
3. **High-resolution tiled features** - only if tiny defects remain a known failure mode.
4. **DINOv2 / RareCLIP** - higher-cost backbone alternatives if feature quality becomes more important than deployment simplicity.

---

# 1.7 Safe Test-Time Adaptation and Update Reliability

The current project updates online state only when a sample appears safe. This section compares possible ways to make that decision more robust.

| Option | How It Works | Strength | Weakness | Fit to Current Project | Reference |
|---|---|---|---|---|---|
| Score threshold gate | Update only when anomaly score is comfortably below the anomaly boundary | Simple, fast, easy to explain | Can still accept low-scoring anomalies | **Implemented** |
| Consistency gate | Compare weak/strong augmented feature consistency | Adds a second independent safety check | Project results show weak good/bad separation by itself | **Implemented as secondary gate** |
| Entropy-based TTA | Minimize prediction entropy during test time | Simple adaptation principle | Designed mainly for classifiers; can fail under mixed anomalies | Conceptual support for lightweight adaptation | Wang et al. (2020), [TENT](https://arxiv.org/abs/2006.10726) |
| Continual TTA / restoration | Continual adaptation with mechanisms to reduce drift | Addresses long streams | More state and update complexity | Future option for long-duration deployment | Wang et al. (2022), [CoTTA](https://arxiv.org/abs/2203.13591) |
| Distribution alignment / optimal transport | Align target features with source memory during test time | Directly addresses source-target feature shift | Computationally more complex; research-stage option | Useful future comparison when domain shift is strong | TTAD, ICLR 2025 submission / OpenReview |
| Memory-noise filtering | Reject suspicious patches before they enter memory | Directly targets contamination | Requires a patch-level reliability score | **High-priority future option** | SoftPatch (2024) |
| Operator confirmation | Human validates uncertain updates | Strong protection against contamination | Requires operator interaction | Practical future extension with CiRA UI | Human-in-the-loop literature |

### Project Evidence

The project tested a stricter update threshold:

```text
q0.95 → q0.90
Bad updates:       12 → 8
Update precision:  94.78% → 96.10%
Anomaly recall:    91.00% → 91.00%
```

The consistency gate was retained, but its error distribution alone did not strongly separate good from bad samples.

### Verdict - 1.7

The current **score gate + consistency gate** is a reasonable low-complexity choice.

The next improvement should **not** simply add more thresholds. The strongest research direction is to improve sample reliability at patch level, for example:

```text
score gate
+ consistency gate
+ patch novelty / noise filter
```

This directly targets the contamination observed in the project's final experiments.

---

# 1.8 Continual and Streaming Memory Management

The final project keeps a bounded patch memory and appends accepted patches. This is simple, but a long-running industrial stream may need better memory selection.

| Option | Memory Strategy | Strength | Weakness | Suitability |
|---|---|---|---|---|
| FIFO / newest-patch retention | Keep latest accepted patches when memory is full | Very simple and follows recent deployment state | Can forget older valid normal modes | **Current implementation** |
| Random / reservoir sampling | Keep a representative sample over a long stream | Constant memory; easy streaming implementation | Does not explicitly preserve rare normal patterns | Easy future baseline |
| k-center / coreset selection | Keep diverse representative patches | Strong PatchCore connection; limits redundancy | More update computation | **Recommended future memory upgrade** |
| Per-task balanced coreset | Keep memory quota for previous/current domains | Reduces forgetting across shifts | Requires task/domain boundaries | Useful only if production stages are known |
| Incremental Gaussian / statistical memory | Update distribution parameters instead of raw patches | Compact memory | Distribution assumptions may be too simple | Alternative for constrained hardware |
| Covariance-aware streaming memory | Update reduced-space covariance + retrieval | Captures correlated features | More implementation complexity | Research option |
| Prototype memory | Keep representative centroids/prototypes | Efficient nearest-neighbour search | Prototype compression can lose fine detail | Useful for speed/scale |

Relevant work:

- PatchCore coreset memory: [Roth et al. (2022)](https://arxiv.org/abs/2106.08265)
- Continual anomaly-detection benchmark with PatchCore memory adaptation: [Bugarin et al. (2024), CVPRW](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Bugarin_Unveiling_the_Anomalies_in_an_Ever-Changing_World_A_Benchmark_for_CVPRW_2024_paper.pdf)
- On-device continual PatchCore with incremental coreset update: [Ren et al. (2025)](https://arxiv.org/abs/2512.13497)
- Streaming covariance-aware PatchCore: [Ferrari et al. (2026)](https://arxiv.org/abs/2605.27748)

### Verdict - 1.8

The current bounded FIFO-style memory is suitable for proving the CTTA idea, but it is **not yet the strongest long-term memory strategy**.

Recommended next experiment:

```text
Current FIFO memory
vs
k-center / coreset memory update
```

Measure:

```text
accuracy
anomaly recall
bad-update effect
memory diversity
memory size
runtime
```

This is more meaningful than only increasing `max_patch_memory`.

---

# 1.9 Human-in-the-Loop and Operator Feedback

CiRA CORE already provides Run / Stop / Reset and result display, so the platform can potentially support operator feedback without changing the full backend architecture.

| Option | Operator Action | Benefit | Risk / Cost | Project Fit |
|---|---|---|---|---|
| Confirm normal | Operator confirms a low-confidence normal before memory update | Strong protection against contamination | Adds manual workload | High for commissioning / small datasets |
| Reject update | Operator prevents suspicious sample entering memory | Simple safety override | Needs UI state and logging | High |
| Correct false positive | Add confirmed normal patches to memory after review | Can expand underrepresented normal modes | Wrong human label can contaminate memory | High-value future option |
| Save uncertain sample | Queue samples for later review | Builds useful factory dataset over time | Needs sample-management process | High |
| Active query | Ask operator only on uncertain / novel samples | Reduces review load | Requires uncertainty policy | Medium–High |
| Defect memory | Store confirmed anomaly patterns separately | May support known-defect reasoning | Can overfit and requires enough examples | Lower priority for current project |

Recent work on direct memory-bank correction shows that operator-confirmed updates can improve a PatchCore-style detector without retraining, especially when the initial normal bank is small. It also warns that evaluation must keep corrected images separate from held-out testing to avoid memorisation bias. See Abbas et al. (2026), [Training-Free Human-in-the-Loop Anomaly Detection via Memory Bank Correction](https://arxiv.org/abs/2608.17775).

### Verdict - 1.9

Human feedback is **not required for the current thesis result**, but it is one of the most practical future extensions because CiRA CORE already provides an operator-facing workflow.

A suitable future design is:

```text
high-confidence normal
→ automatic safe update

uncertain sample
→ no automatic update
→ operator Confirm / Reject

confirmed normal
→ controlled memory insertion
```

This would reduce contamination without forcing every image through manual review.

---

# 1.10 Current Research Verdict and Future Options

## A. Why the Current Method Was Selected

The current system combines several ideas from the review rather than following one paper exactly:

| Project Decision | Research Reason | Current Status |
|---|---|---|
| Frozen YOLO26 | Stable, lightweight deployment without full retraining | Selected |
| Local patch features | Better sensitivity to small defects than global embeddings | Selected and validated |
| Normal patch memory | Works without labelled defect classes | Selected |
| Nearest-normal scoring | Simple and compatible with PatchCore-style anomaly detection | Selected |
| Deployment calibration | Handles offline-to-deployment score shift | Selected and validated |
| q0.90 score gate | Reduced bad updates while preserving anomaly recall | Selected |
| Consistency gate | Adds a second update-safety check | Selected as secondary gate |
| Memory CTTA | Best final accuracy / Macro F1 among online variants | **Recommended deployment mode** |
| PatchAdapter CTTA | Online learning mechanism works but standalone gain is limited | Retained for research |
| Flask + CiRA CORE | Matches low-code integration objective | Selected |

This is consistent with the implementation ideas listed in the main README's [2.2.1 Key Implementation Ideas and Supporting References](../README.md#221-key-implementation-ideas-and-supporting-references).

## B. Research Options Worth Testing Next

| Priority | Option | Why It Is Worth Testing | Expected Project Impact |
|---:|---|---|---|
| **1** | Patch-level memory contamination filtering | Directly addresses remaining bad updates | Safer Memory CTTA |
| **2** | Coreset / k-center online memory update | Improves memory diversity under fixed size | Better long-stream stability |
| **3** | Human-confirmed memory correction | Fits CiRA operator workflow | Safer commissioning and real factory use |
| **4** | DINOv2 feature comparison | Tests whether YOLO features are still the representation bottleneck | Possible AUROC improvement |
| **5** | High-resolution tiled inference | Useful for tiny/local defects | Better small-defect sensitivity, higher compute |
| **6** | Distribution-alignment TTA | More direct handling of domain shift | Potential robustness improvement, higher complexity |
| **7** | Prototype / RareCLIP-style online memory | More scalable online zero-shot direction | Larger architecture change |
| **8** | Dual normal/anomaly memory | Useful only when reliable anomaly labels are available | Semi-supervised extension |

## C. Final Research Position

The evidence collected so far supports the following practical path:

```text
Frozen YOLO26
      ↓
Local patch representation
      ↓
Trusted-normal calibration
      ↓
Safe score + consistency gate
      ↓
Adaptive normal patch memory
      ↓
CiRA CORE low-code deployment
```

The project should **keep Memory CTTA as the main deployed method** because the final ablation showed the clearest practical gain from memory adaptation.

The next research work should focus less on adding another small adapter and more on the two remaining deployment risks:

```text
1. Which samples are safe to learn from?
2. Which accepted patches should remain in memory over a long stream?
```

These questions are directly supported by recent work on noisy memory, continual PatchCore, streaming memory and human-guided memory correction.

---

## Research Status

```text
General defect-detection review             COMPLETE
Model / backbone comparison                 COMPLETE
Learning-method comparison                  COMPLETE
Low-code platform comparison                COMPLETE
Local patch / memory-bank review            EXTENDED
Safe CTTA / contamination review            EXTENDED
Streaming memory options                    EXTENDED
Human-in-the-loop options                   EXTENDED
Cross-dataset / long-duration validation    FUTURE WORK
```
