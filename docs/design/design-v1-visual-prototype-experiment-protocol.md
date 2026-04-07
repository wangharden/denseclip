# DenseCLIP Few-Shot Anomaly Detection Experiment Protocol (v1)

> Purpose: turn the recent design discussion into a decision-complete experiment protocol.
> Goal: establish a clean visual-prototype baseline first, then compare it against a small learned head under the same split and metric setup.
> Scope: this document defines experiment logic, data protocol, control variables, metrics, and rollout order. It does not assume the legacy Mask R-CNN training path is the main route.

---

## 1. Core Decision

The first valid baseline is a visual-prototype method, not a prompt-learning method.

This means:

- Use a frozen visual encoder.
- Build prototypes from a tiny support set.
- Score query images by visual feature similarity.
- Do not train a text prompt module.
- Do not train a tiny anomaly head in the first baseline.

The purpose is to answer one question first:

Can visual features alone, under a strict few-shot protocol, separate normal from defect samples well enough on MVTec?

Only after this baseline is stable should later experiments add:

- defect/category text prompts
- prompt tuning
- adapter or LoRA
- a trainable tiny anomaly head

This is a control-variable decision. If the baseline already works, later gains can be attributed to the newly added module. If the baseline fails, the problem is more likely in the visual features, the data protocol, or the evaluation setup.

---

## 2. What The Current Few-Shot Code Is And Is Not

The current few-shot implementation is prototype-based in the visual sense, but it is not a prototype-prompt method.

More precisely:

- It extracts support features from a frozen visual backbone.
- It builds a `normal_prototype` and optionally a `defect_prototype`.
- It compares query patch features against those prototypes.
- It can then feed similarity maps into a tiny anomaly head.

Therefore:

- `prototype` currently means a visual feature prototype.
- It does not mean a text prompt prototype.
- It is not yet a category-specific or defect-specific prompt-learning system.

If a later prompt stage is added, the two natural prompt families are:

- category prompts: one prompt set per object category, such as `hazelnut`, `bottle`, `screw`
- defect prompts: one prompt set per defect type, such as `cut`, `crack`, `hole`

Those are explicitly out of scope for the v1 baseline.

---

## 3. Why The First Baseline Should Be Visual Prototype Only

### 3.1 Control Variables

The cleanest first comparison is:

- Stage A: visual prototype only
- Stage B: visual prototype + tiny anomaly head

This isolates the effect of learning a small head.

If Stage A and Stage B are mixed from the start, it becomes unclear whether gains or failures come from:

- the support/query split
- the visual backbone
- the prototype design
- the head optimization
- the thresholding strategy

### 3.2 Engineering Cost

The visual-prototype baseline is cheaper to build and easier to debug.

- No extra learned module is required.
- No optimizer behavior needs to be diagnosed.
- No training instability is introduced.
- The runtime path remains simple.

### 3.3 Compatibility Benefit

A pure visual-prototype path stays far away from the legacy MMDetection / MMCV detector stack.

That means it avoids dependence on:

- `RPN`
- `RoIAlign`
- `NMS`
- `mmcv.ops`

This is important for newer GPUs and for reducing framework-related failure modes.

---

## 4. Training Definition For The New Baseline

The new visual-prototype baseline should be treated as a no-training or near-no-training baseline.

The strict definition is:

- Extract support features from a frozen backbone.
- Aggregate those support features into prototypes.
- Directly evaluate query images against the prototypes.
- Do not update model weights.

In other words, the baseline does **not** mean:

"train on a small number of support samples and then validate."

That description belongs to the next-stage learned baseline, not the first one.

For v1 baseline, the correct wording is:

- support samples are used to construct prototypes
- query samples are used for evaluation
- no gradient-based optimization is required

If threshold calibration is needed, it should be treated as a small protocol choice, not as model training.

---

## 5. Recommended Experiment Stages

### Stage A: Visual Prototype Baseline

Definition:

- frozen visual backbone
- normal support set required
- defect support set optional
- no tiny head
- no prompt learning
- direct similarity-based anomaly scoring

Expected output:

- image-level anomaly score
- patch-level or pixel-level anomaly heatmap

Purpose:

- establish the clean baseline
- measure whether the visual representation is already useful for few-shot anomaly detection

### Stage B: Visual Prototype + Tiny Head

Definition:

- keep the same support/test protocol as Stage A
- keep the same backbone frozen
- add a tiny trainable head on top of similarity features
- train only that head

Purpose:

- measure the incremental value of a learned scorer
- compare directly against Stage A under the same data protocol

### Stage C: Prompt-Related Extensions

Definition:

- category prompts
- defect prompts
- prompt tuning or lightweight adapters

Purpose:

- test whether language priors improve anomaly discrimination after the visual baseline is already known

Stage C must not be merged into the initial baseline.

---

## 6. Data Protocol

The correct reference data source for a strict few-shot anomaly protocol is:

`C:\Nottingham\DenseCLIP-master\data\mvtec_anomaly_detection`

This is preferred over the flat `mvtec_coco` directory when the goal is a clean few-shot protocol, because the official MVTec structure already separates normal training images and test images.

### 6.1 Category Structure

For each category:

- `train/good`: normal training images
- `test/good`: normal test images
- `test/<defect_type>`: defect test images
- `ground_truth/<defect_type>`: pixel masks for defect regions

### 6.2 Stage A Split Rule

For Stage A, use:

- `support_normal`: a small subset sampled from `<category>/train/good`
- `support_defect`: optional, only if the chosen baseline is defect-aware
- `test_query`: all images from `<category>/test/good` plus all images from `<category>/test/<defect_type>`

Important constraints:

- Do not draw query images from `train/good`.
- Do not use the official test set for model training.
- Do not mix support construction and final evaluation within the same image pool unless this is explicitly an exploratory engineering shortcut.

### 6.3 Why The Previous `train_query` Split Is Not Ideal

The previous engineering implementation split a flat directory into:

- `support_normal`
- `support_defect`
- `train_query`
- `val_query`

That split was useful for bootstrapping a runnable pipeline, but it is not the cleanest few-shot protocol.

The main problem is that it allows a large remaining set of query images to participate in training logic, which weakens the "few-shot" constraint.

Therefore, the v1 experiment protocol should not treat that random flat split as the final evaluation protocol.

---

## 7. Should Defect Support Be Used In The First Baseline

There are two valid baseline variants:

### Variant A1: Normal-Only Prototype

- support uses only normal images
- anomaly score is defined by distance from normality

Pros:

- closer to classic anomaly detection
- simpler and cleaner

Cons:

- may be less informative when very small defect support is actually available

### Variant A2: Normal + Defect Prototype

- support uses a small normal set and a very small defect set
- anomaly score uses both normal and defect prototypes

Pros:

- closer to the target "few-shot defect-aware" setting
- may improve discrimination if defect support is representative

Cons:

- slightly less pure as an anomaly-only baseline

Recommended order:

1. run A1 first
2. run A2 second
3. compare both before introducing a trainable head

---

## 8. Metrics

The experiment should report at least:

- image-level AUROC
- pixel-level AUROC
- PRO or another region-level localization metric if available

Additionally record:

- per-category runtime
- support size
- whether defect support was used
- image resolution

These metadata are necessary because different support sizes and resolutions can change performance substantially.

---

## 9. Runtime And Scheduling Strategy

### 9.1 Is The Visual-Prototype Baseline Expensive

Not very expensive relative to the old detector-based route.

Reasons:

- the backbone is frozen
- no detector stack is active
- Stage A has no gradient-based optimization
- support features can be computed once and reused

Per-category runtime should be manageable on a single local GPU. The total runtime grows when all 15 MVTec categories are run, but the baseline itself is not a heavy training workload.

### 9.2 Should Multi-Process Parallel Training Be The Default

No, not on a single RTX 5060.

For a single GPU, running multiple category jobs in parallel usually:

- increases context switching
- increases VRAM pressure
- adds I/O contention
- makes failures harder to diagnose

The default policy should be:

- sequential category execution
- moderate `DataLoader` worker count
- local SSD reads when possible

Parallel multi-process scheduling only becomes a strong default if:

- there are multiple GPUs, or
- each category job is very long and the single-GPU throughput is clearly underutilized

For the current setup, that is not the primary optimization target.

---

## 10. Formal Comparison Matrix

The minimum comparison matrix should be:

| Stage | Backbone | Prototype | Tiny Head | Prompt | Weight Update | Goal |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Frozen | Normal only | No | No | No | strict anomaly baseline |
| A2 | Frozen | Normal + defect | No | No | No | defect-aware visual baseline |
| B1 | Frozen | Normal only | Yes | No | Tiny head only | measure learned scorer gain |
| B2 | Frozen | Normal + defect | Yes | No | Tiny head only | strongest non-prompt comparison |

Anything prompt-related should be compared only after these four rows are understood.

---

## 11. Implementation Priority

The recommended execution order is:

1. build the strict Stage A1 path on `mvtec_anomaly_detection`
2. validate image-level and pixel-level metrics on one category first
3. expand Stage A1 to all categories
4. add Stage A2
5. only then add Stage B with a tiny head

Recommended first category:

- `hazelnut`

Reason:

- moderate complexity
- visually intuitive defects
- convenient for debugging heatmaps

---

## 12. Decision Summary

The main decisions locked by this document are:

- v1 baseline is a visual-prototype method, not a prompt method
- v1 baseline should not train a tiny head
- the clean evaluation protocol should use `mvtec_anomaly_detection`, not only a flat random split
- the previous `train_query` design was an engineering bootstrap, not the final few-shot protocol
- single-GPU default should be sequential execution, not multi-process parallel category training

This gives a clean baseline first. After that, every later addition can be judged against something interpretable.
