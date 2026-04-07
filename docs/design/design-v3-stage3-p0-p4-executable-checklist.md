# DenseCLIP Stage3 P0-P4 可执行实验清单

本文用于承接 Stage 3 的可执行实验清单，统一维护：

- 脚本
- 参数
- 输出目录
- 判定门槛

Stage 3 主设计主线以 [二阶段结果分析以及三阶段实验设计](../总结/stage3/二阶段结果分析以及三阶段实验设计.md) 为准。  
已经执行的阶段需要在本文中保留“脚本 / 参数 / 输出目录 / 判定门槛 / 当前状态”；未执行阶段继续保留为计划条目。

当前自 `P2` 之后，Stage 3 的后续目标已收口为：**图像级异常识别优先**。  
其中 `P3` 已完成并写回，最新官方结果来源为 `seed42_monotonic`；`seed42_v3_multisplit` 与 `seed42_v2_rankproxy` 仅保留为历史 smoke。Stage 3 已收口，不再新增后续尝试，也不再围绕 `pixel / PRO` 展开主 gate。

## P0

### 目标

- winner map cache 与 baseline 回放校验

### 脚本

- `tools/cache_stage2_winner_maps.py`

### 参数

- subset：`weak5_bottle`
- categories：`leather / grid / carpet / screw / zipper / bottle`
- seed：`42`
- data_root：`data/mvtec_anomaly_detection`
- manifests_dir：`outputs/split_manifests/stage2`
- support：`sn16 / sd4`
- image_size：`320`
- pretrained：`pretrained/RN50.pt`
- feature_layer：`layer4`
- scorer：`neg-normal + max + patch`
- reference_topk：`3`
- subspace_dim：`8`
- topk_ratio：`0.01`
- batch_size：`8`
- stage2 winner source：`outputs/stage2/p4_full15_final/seed42/a1_sn016_sd004_layer4_neg_normal_subspace_refk003_max_patch_topk010_dim008`

### 输出目录

- `outputs/stage3/cache/weak5_bottle/seed42`
- 聚合入口：
  - `outputs/stage3/cache/weak5_bottle/seed42/replay_summary.json`
  - `outputs/stage3/cache/weak5_bottle/seed42/replay_summary.csv`

### 判定门槛

- cache 回放指标与 Stage 2 对应子集结果误差不超过 `±0.002`

### 当前状态

- 已完成
- replay check：`passed`
- max abs diff：`5.10e-7`
- 结论：可作为 Stage 3 后续训练的官方输入底座

## P1

### 目标

- `image-head-only` smoke，验证 image-level calibration 是否足以作为第一条训练主线

### 脚本

- `run_stage3_head.py`

### 参数

- cache_dir：`outputs/stage3/cache/weak5_bottle/seed42`
- output_dir：`outputs/stage3/p1_image_head/weak5_bottle/seed42`
- seed：`42`
- epochs：`100`
- batch_size：`32`
- lr：`1e-3`
- weight_decay：`1e-4`
- hidden_features：`64`
- dropout：`0.1`
- margin：`0.1`
- device：`auto`

### 输出目录

- `outputs/stage3/p1_image_head/weak5_bottle/seed42`
- 聚合入口：
  - `outputs/stage3/p1_image_head/weak5_bottle/seed42/experiments.csv`
  - `outputs/stage3/p1_image_head/weak5_bottle/seed42/per_category.csv`
  - `outputs/stage3/p1_image_head/weak5_bottle/seed42/summary.md`
  - `outputs/stage3/p1_image_head/weak5_bottle/seed42/train_history.json`

### 判定门槛

- weak-five `image AUROC mean >= E0 + 0.03`
- `bottle balanced` 不下降超过 `0.02`

### 当前状态

- 已完成
- 结果性质：`single-seed / subset / screening`
- weak-five `image AUROC mean`：`0.5220 -> 0.5122`
- `bottle balanced`：`0.9483 -> 0.9555`
- 判定：未通过，不升格、不扩 sweep，直接让位给 `P2`
- 说明：当前 `pixel AUROC / PRO` 来自冻结 `subspace cache`，这里只能做方向性判断

## P2

### 目标

- `map-space dual-head` 主实验，同时提升 weak classes 的 image-level ranking，并保住 pixel localization

### 脚本

- `run_stage3_head.py --mode p2`

### 参数

- cache_dir：`outputs/stage3/cache/weak5_bottle/seed42`
- output_dir：`outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2`
- 子集：`weak-five + bottle`
- seed：`42`
- epochs：`20`
- batch_size：`32`
- lr：`5e-4`
- weight_decay：`1e-4`
- patience：`8`
- device：`cpu`
- 输入 channel：`base_anomaly_map / subspace_residual_map / knn_top1_map / knn_top3_map / knn_gap_map`
- 当前实现：`MapFusionHead + raw-score residual image / raw-score residual pixel`
- 当前损失：`mask BCE + Dice + image BCE + smooth ranking + consistency`
- 训练口径：`BCE / ranking` 在 support-fitted affine score calibration 后的 logit 空间优化；评估保持 raw score space

### 输出目录

- 正式结果：`outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2`
- 辅助检查：
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_contract_check`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_rank_smoke`
- 历史诊断：
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_scorefix`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_smoke`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_smoke_residual`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_smoke_identity`
- 聚合入口：
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2/experiments.csv`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2/per_category.csv`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2/summary.md`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2/train_history.json`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2/identity_check.json`
  - `outputs/stage3/p2_dual_head/weak5_bottle/seed42_score_space_v2/rank_check.json`

### 判定门槛

- weak-five `image AUROC mean >= 0.56`
- weak-five `balanced mean >= 0.66`
- weak-five `pixel AUROC mean >= 0.72`
- `bottle balanced` 下降不超过 `0.02`

### 当前状态

- 已完成
- 结果性质：`single-seed / subset / screening`
- 官方来源：`seed42_score_space_v2`
- historical diagnostic：`seed42` 旧目录保留，但不再作为 gate 依据
- identity contract：`image drift = 0.0`，`pixel drift = 0.0`
- epoch 1 `rank = 0.1375`
- 总体结果：`image 0.5947 -> 0.5181`，`pixel 0.7778 -> 0.7094`，`balanced 0.6862 -> 0.6138`
- weak-five：`image 0.5220 -> 0.4865`，`pixel 0.7457 -> 0.6703`，`balanced 0.6338 -> 0.5784`
- `bottle balanced`：`0.9483 -> 0.7906`
- 判定：未通过，不晋级 `P3`
- 结论用途：保留为“为何放弃当前 dual-head 主线”的正式证据
- 下一步计划：转向 `image-only residual calibrator`，不再继续放大当前 `dual-head + pixel loss` 训练线

## P3

### 目标

- `image-only residual calibrator` 主线 smoke

### 脚本

- `run_stage3_image_only.py`

### 参数

- subset：`weak-five + bottle`
- smoke：`seed42`
- 复评：`42 / 43 / 44`
- 对比：`E0 / E1-rescal`
- 输入：`pooled map stats + winner_image_score (+ optional knn_gap uncertainty)`
- 模型：`z_final = z_frozen + α(x) · Δ(x)`
- 损失：`rank + image BCE + anchor + residual L2`
- 选模：`support-holdout image AUROC / ranking proxy`

### 输出目录

- 已完成官方结果：
  - `outputs/stage3/p3_image_only/weak5_bottle/seed42_v3_multisplit`
- 历史 smoke：
  - `outputs/stage3/p3_image_only/weak5_bottle/seed42_v2_rankproxy`
- 未来复评保留目录：
  - `outputs/stage3/p3_image_only/reval_seed{seed}`

### 判定门槛

- smoke：
  - weak-five `image AUROC mean >= 0.55`
  - 且相对 `E0` 至少提升 `+0.03`
  - `bottle image AUROC` 不下降超过 `0.03`
- 3-seed：
  - 至少 `2/3` 个 seed 的 weak-five `image AUROC` 高于 `E0`
  - 3-seed weak-five `image AUROC mean` 相对 `E0` 至少提升 `+0.03`
  - `bottle image AUROC mean` 不下降超过 `0.03`

### 当前状态

- 已完成
- 结果性质：`single-seed / subset / screening`
- 历史 smoke：`seed42_v3_multisplit`
- 历史 smoke：`seed42_v2_rankproxy`
- `py_compile`：passed
- monitor：`ok / no alerts`
- `latest_epoch`：`71`
- `gate_pass_smoke`：`False`
- `selection_epoch`：`11.7`
- query 侧表现：`E1-rescal (v3)` 与 `E0` 基本持平，未达到 smoke gate
- `E1-rescal (v3)`：`image 0.594707 -> 0.594240`，`balanced 0.686243 -> 0.686009`
- `weak-five image AUROC mean`：`0.521954 -> 0.521393`
- `bottle image AUROC`：`0.958475 -> 0.958475`
- 判定：未通过，不晋级 `P4`
- 结论用途：selector / gate 修复成功，但 query 增益不足，暂不扩到 full-15
- 历史记录：当时计划继续修 `P3 image-only`，先拿到一个 query 侧确实优于 `E0` 的 retained winner

## P3 final frozen text prior smoke

### 目标

- `frozen text prior` 最后一次 image-level smoke

### 脚本

- `run_stage3_text_prior.py`

### 参数

- subset：`weak-five + bottle`
- smoke：`seed42`
- 结果目录：`seed42_monotonic`
- 对比：`E0 / E1-text-prior`
- 输入：`P0` 产出的 cache map 与 `winner_image_score`
- 模型：`frozen text prior -> image gate`
- 策略：仅用 `text_gap` 做辅助特征，不做 prompt tuning / adapter / LoRA

### 输出目录

- 官方结果：
  - `outputs/stage3/p3_text_prior/weak5_bottle/seed42_monotonic`
- 历史 smoke：
  - `outputs/stage3/p3_image_only/weak5_bottle/seed42_v3_multisplit`
  - `outputs/stage3/p3_image_only/weak5_bottle/seed42_v2_rankproxy`

### 判定门槛

- smoke：弱五类 `image AUROC mean` 需高于 `E0`
- 且 `bottle image AUROC` 不能明显回退

### 当前状态

- 已完成
- 结果性质：`single-seed / subset / screening`
- `py_compile`：passed
- `gate_pass_smoke`：`False`
- `selection_beta_mean`：`0.0`
- `selection_tau_mean`：`-0.012923734`
- 六类 `per_category.csv` 与 `E0` 完全一致
- `holdout` 侧仍全是 `1.0`
- 判定：最优策略明确选择“不加 text prior”
- 结论：Stage 3 最后一次尝试失败，Stage 3 不再继续

## P4

### 目标

- full-15 image-level screening 与 finalist 正式收口

### 脚本

- 待 `P3` 产出 retained winner 后固化 full-15 runner 和聚合器

### 参数

- screening：先 `seed42`
- finalist：再补 `43 / 44`
- baseline：`frozen Stage 2 winner`
- candidate：`P3 retained image-only winner`
- 备选：只有当 `P3` 主线接近通过但 `leather / grid` 仍基本不动时，才额外加 `frozen text prior`

### 输出目录

- 预留：`outputs/stage3/p4_full15_final`

### 判定门槛

- `full-15 image AUROC mean >= 0.71`
- weak-five `image AUROC mean >= 0.56`
- strong control 不系统性塌陷；`bottle image AUROC` 相对 frozen baseline 不下降超过 `0.03`

### 当前状态

- 已收口
- 不再执行
- `P4` 仅保留为历史规划项，不再作为后续入口
