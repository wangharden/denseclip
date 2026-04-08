# DenseCLIP Stage4 P0-P3 可执行实验清单

本文统一维护 Stage 4 新分支的脚本、参数、输出目录、gate 和当前状态。

## P0：reviewer / audit fix

### 目标

- 修复 prompt-text 的复现和阈值口径问题。

### 脚本

- `run_prompt_defect_text.py`
- `run_visual_layer_baseline.py`

### 当前状态

- 已完成
- 已修复：
  - `ftfy` 静默降级
  - prompt-text 全局 pooled F1 threshold
  - layer-baseline hard-coded seed
  - layer-baseline 误导性 selection metadata

## P1：weak5 修复验证

### 目标

- 确认阈值修复后，`prompt-text` 不再出现“全判异常”的退化口径。

### 脚本

- `run_prompt_defect_text.py`

### 命令

```powershell
python run_prompt_defect_text.py --cache-dir outputs/stage3/cache/weak5_bottle/seed42 --pretrained pretrained/RN50.pt --output-dir outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_p_v3_thresholdfix --scope weak5_bottle --seed 42 --num-resplits 3 --holdout-fraction 0.5 --feature-sources clip_global,denseclip_global,layer3_gap
```

### 输出目录

- `outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_p_v3_thresholdfix`

### gate

- 不允许 `prompt_plus_p__*` 再次出现全判异常
- 必须产出 `specificity` 和 `balanced_accuracy`
- 当前主线 winner 仍需保持 `clip_global`

### 当前状态

- 已完成
- 通过：口径修复有效

## P2：full-15 官方实验

### 目标

- 将修复后的 winner 推进到全品类，验证是否仍成立。

### cache

```powershell
python tools/cache_stage2_winner_maps.py --cache-root outputs/new-branch/cache --subset-name full15 --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper --seed 42 --support-normal-k 16 --support-defect-k 4 --image-size 320 --pretrained pretrained/RN50.pt --feature-layer layer4 --reference-topk 3 --subspace-dim 8 --topk-ratio 0.01 --aggregation-mode max --stage2-run-dir outputs/stage2/p4_full15_final/seed42/a1_sn016_sd004_layer4_neg_normal_subspace_refk003_max_patch_topk010_dim008 --batch-size 8
```

### 脚本

- `run_prompt_defect_text.py`

### 命令

```powershell
python run_prompt_defect_text.py --cache-dir outputs/new-branch/cache/full15/seed42 --pretrained pretrained/RN50.pt --output-dir outputs/new-branch/prompt-text/full15/seed42/prompt_p_v3_full15 --scope full15 --seed 42 --num-resplits 3 --holdout-fraction 0.5 --feature-sources clip_global,denseclip_global,layer3_gap
```

### 输出目录

- `outputs/new-branch/prompt-text/full15/seed42/prompt_p_v3_full15`

### gate

- 必须确认 `clip_global` 仍是主线 winner
- 必须确认 `denseclip_global` 和 `layer3_gap` 是否继续保留
- 若 `bottle` 仍明显回退，则阶段不得收口

### 当前状态

- 已完成
- 结论：`clip_global` 仍是 winner，但 Stage 4 未收口

## P3：Stage2-anchor text-hybrid 路线

### 目标

- 验证 `frozen Stage2 winner + conservative text residual` 是否能在 `weak5_bottle / seed42` 同时超过 `E0` 和当前 Stage 4 `prompt-text incumbent`
- 只有 weak5 通过，才允许推进到 `full15 / seed42` 和后续 `seed43 / seed44`

### 脚本

- `run_stage4_text_hybrid.py`

### 命令

```powershell
python run_stage4_text_hybrid.py --cache-dir outputs/stage3/cache/weak5_bottle/seed42 --output-dir outputs/new-branch/text-hybrid/weak5_bottle/seed42/anchor_hybrid_v3_supportnorm --scope weak5_bottle --seed 42 --baseline-experiments outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_p_v3_thresholdfix/experiments.csv
```

### 输出目录

- 官方 closeout：`outputs/new-branch/text-hybrid/weak5_bottle/seed42/anchor_hybrid_v3_supportnorm`
- 历史诊断：
  - `outputs/new-branch/text-hybrid/weak5_bottle/seed42/anchor_hybrid_v1`
  - `outputs/new-branch/text-hybrid/weak5_bottle/seed42/anchor_hybrid_v2_calib`

### gate

- 必须相对 `Stage2 E0` 和当前 `prompt-text incumbent` 至少在 `AUROC` 或 `AP` 上 +0.01
- 阈值指标不能再出现塌缩
- `bottle_image_auroc` 不允许比 `E0` 下降超过 0.02
- 只有 weak5 通过，才允许晋级 `full15 / multi-seed`

### 当前状态

- 已完成
- 结果：最佳 retained 行为 `hybrid_frozen__relu_z`
- 结论：超过了 `E0`，但没有超过当前 `prompt-text incumbent`
- 未晋级 `full15 / seed42`
- 当前 repo 约束下按 weak5 screening closeout

## P4：pure-visual PCA + coreset/farthest-point 路线

### 目标

- 验证 `SubspaceAD-style PCA normal-subspace` 和 `PatchCore-style greedy coreset / farthest-point memory reduction` 是否能在当前 repo 中形成 retained pure-visual weak5 winner
- 只有 weak5 明显通过 gate，才允许晋级 `full15 / seed42` 和后续 multi-seed

### 脚本

- `fewshot/coreset.py`
- `fewshot/patchcore_subspace.py`
- `tools/eval_structure_ablation.py`
- `tools/run_stage4_pure_visual.py`

### 命令

```powershell
python tools/run_stage4_pure_visual.py --mode screening --subset weak5_bottle --output-dir outputs/new-branch/pure-visual/weak5_bottle/seed42_screening_v3_coreset_subspace --methods baseline subspace coreset_subspace --subspace-dims 8 12 --coreset-ratios 0.1 0.25 0.5 --skip-existing
```

### 输出目录

- `outputs/new-branch/pure-visual/weak5_bottle/seed42_screening_v3_coreset_subspace`

### gate

- 相对 `Stage2 visual E0 = subspace(dim=8)`，`image AUROC` 或 `AP` 至少 `+0.01`
- `bottle_image_auroc` 不允许比 `E0` 回退超过 `0.03`
- 如果 weak5 出现明显 ranking / localization 双退化，则直接 closeout，不晋级 `full15 / multi-seed`

### 当前状态

- 已完成
- concrete route 已实跑：`PCA + coreset/farthest-point`
- 最好配置：`coreset_subspace(core=0.10, dim=12)`
- 结果：`image_auroc_mean 0.5125`，显著低于 `Stage2 subspace(dim=8) = 0.5947`
- `bottle_image_auroc 0.4280`，相对 `E0 = 0.9585` 灾难性回退
- 结论：按 weak5 screening 正式 closeout，不晋级 `full15 / seed42`，也不进入 `seed43 / 44`

## P5?reopened pure-visual DINOv2 PatchCore-like retained route

### ??

- ???? pure-visual???????? Stage2-like screening ??
- ???????? pure-visual SOTA ????`DINOv2 + patch-level NN retrieval`
- weak5 ?????????????? `full15 / seed42 -> seed43 -> seed44`

### ??

- `fewshot/dinov2_backbone.py`
- `tools/run_stage4_pure_visual_dinov2.py`

### weak5 ?? run

```powershell
python tools/run_stage4_pure_visual_dinov2.py --mode screening --subset weak5_bottle --seeds 42 --output-dir outputs/new-branch/pure-visual/weak5_bottle/seed42_dinov2_screening_v3_clean --feature-sources last last4_mean --methods patchcore_knn global_subspace --knn-topks 1 3 --coreset-ratios 1.0 --subspace-dims 64 128 --batch-size 4 --score-batch-size 2 --device auto
```

### retained weak5 winner

- `dinov2_vits14_last_patchcore_knn_k001_core1000_topk010`
- official dir?`outputs/new-branch/pure-visual/weak5_bottle/seed42_dinov2_screening_v3_clean`
- checker?`image_auroc_mean 0.9625`?`image_ap_mean 0.9792`?`balanced_accuracy 0.8741`?`bottle 1.0000`

### full15 official runs

```powershell
python tools/run_stage4_pure_visual_dinov2.py --mode screening --subset full15 --seeds 42 --output-dir outputs/new-branch/pure-visual/full15/seed42/dinov2_patchcore_k1_last_v1 --feature-sources last --methods patchcore_knn --knn-topks 1 --coreset-ratios 1.0 --batch-size 4 --score-batch-size 2 --device auto
python tools/run_stage4_pure_visual_dinov2.py --mode screening --subset full15 --seeds 43 --output-dir outputs/new-branch/pure-visual/full15/seed43/dinov2_patchcore_k1_last_v1 --feature-sources last --methods patchcore_knn --knn-topks 1 --coreset-ratios 1.0 --batch-size 4 --score-batch-size 2 --device auto
python tools/run_stage4_pure_visual_dinov2.py --mode screening --subset full15 --seeds 44 --output-dir outputs/new-branch/pure-visual/full15/seed44/dinov2_patchcore_k1_last_v1 --feature-sources last --methods patchcore_knn --knn-topks 1 --coreset-ratios 1.0 --batch-size 4 --score-batch-size 2 --device auto
```

### multi-seed summary

- `outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1`

### gate

- ???????
  - ranking ???? `Stage2 E0`
  - `balanced_accuracy / specificity` ??
  - `bottle` ?????????
  - ????????????? retained ??

### ????

- ???
- retained winner?`dinov2_vits14_last_patchcore_knn_k001_core1000_topk010`
- 3-seed checker?`AUROC 0.9744?0.0025`?`AP 0.9837?0.0007`?`balanced_accuracy 0.8704?0.0152`?`specificity 0.9650?0.0147`?`bottle 0.9994?0.0008`
- ???Stage 4 ?? winner ???? reopened pure-visual DINOv2 route
