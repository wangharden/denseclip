# 阶段五执行说明：纯视觉 AnomalyDINO 诊断后重启 v1

本文档是 pure-visual 路线在本 repo 内重启 Stage 5 的唯一有效合同。

本轮不是回到旧的 `stage5_anomalydino_policy_align_v1/v2/v3`，而是按 `stage5fankui.txt` 的新诊断，把 Stage 5 改成一条更窄的 restart 路线。

## 重启原因

上一轮 Stage 5 的失败形态不是 image ranking 崩坏，而是：

- `image ranking` 基本仍强
- `PRO / localization` 持续退化
- 退化最像协议错配，而不是单一 backbone 失效

当前最强诊断固定为三条：

1. `mask` 被错误地施加到了 `support/reference bank` 和 `anomaly map` 上。
2. 原本应该 `category-conditional` 的 recipe，被写成了全局统一开关。
3. 在还没证明局部对齐有效前，就把 Stage 5 直接 closeout 到 Stage 6，信息增益不够。

因此本轮重启的目标不是回到“继续补丁式对齐官方所有细节”，而是先修最可能的合同错误。

## 单一目标

继续留在同一方法族：

- `DINOv2 PatchCore`

只回答一个问题：

- 如果把 Stage 5 的预处理策略改成 `query-only masking + category-gated policy + base reference bank`，能否恢复 weak5/full15 的 localization 方向，而不打坏 ranking 控制类。

## 单一假设

如果上一轮 Stage 5 失败的主因确实是：

- support/reference bank 被 mask 砍薄
- 全局 rotation/mask 误伤 texture 与方向敏感类
- score-map 后硬 mask 伤害边界与小缺陷

那么在不切换方法族的前提下，下面这条更窄的 restart policy 应该比旧 Stage 5 更合理：

- `reference_bank_view = base`
- `query_mask_mode = pre_score`
- `query_mask_policy = category_list`
- `rotation_policy = category_list`
- 不再把 hard mask 作为默认 anomaly-map 后处理

## 本轮固定策略

本轮 preflight 固定为：

- `method = patchcore_knn`
- `feature_source = last`
- `image_size = 448`
- `resize_mode = square`
- `reference_bank_view = base`
- `defect_bank_view = base`
- `query_mask_mode = pre_score`
- `query_mask_policy = category_list`
- `query_mask_categories = bottle capsule hazelnut metal_nut pill screw toothbrush`
- `rotation_policy = category_list`
- `rotation_categories = hazelnut metal_nut screw`

这意味着：

- query mask 只在 query 端参与 patch 选择
- support/reference bank 默认不再用 object-normalized mask 砍薄
- 旋转增强不再默认全类开启
- 旧的 global hard mask 路径不再是 restart 默认路径

## 三层级架构

本轮严格使用三层级架构。

### 第一层：主 Session

主 session 只负责：

- 固定本合同
- 读取 shared board / route board
- 在 checker milestone 做最终判断

主 session 不直接越级下发 `implementer / runner / checker / writeback`。

### 第二层：pure-visual Route Lead

route lead 由 `Fermat` 负责。

职责：

- 维护 `pure_visual_route_board.md`
- 维护 reopen milestone 的 `live_experiment_board.md`
- 分发 lower-level worker
- 在 `autorun_mode=on` 下，preflight 通过后自动推进

### 第三层：Role Worker

当前先走 `Smoke mode`：

- `implementer`
- `checker`
- `writeback`
- `doc-check`

只有当 weak5 preflight 通过，才允许自动升级到 `Training mode` 的 full15 official run。

## 官方范围与目录

weak5 preflight：

- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_restart_query_mask_category_gate_v1_preflight`

若 preflight 通过，full15 official：

- `outputs/new-branch/pure-visual/full15/stage5_restart_query_mask_category_gate_v1`

## 允许修改的文件

本轮只允许改：

- `tools/run_stage4_pure_visual_dinov2.py`
- `docs/总结/stage5/阶段五执行说明-纯视觉AnomalyDINO诊断后重启-v1.md`
- `outputs/new-branch/_ops/route_boards/pure_visual_route_board.md`
- `outputs/new-branch/_ops/live_experiment_board.md`

不允许：

- 切到 Stage 6 official repo
- 换 base family
- 引入新的 retrieval family
- 把任务改写成 prompt / hybrid

## weak5 Preflight Gate

只有全部满足，才允许自动推进到 full15：

1. `alerts = none`
2. `bottle image_auroc >= 0.9950`
3. `image_auroc_mean >= 0.9700`
4. `pro_mean >= 0.8900`
5. `screw image_auroc_mean >= 0.7975`
6. `zipper / leather / grid` 不接受明显 localization 崩坏或大面积 false positive

## Paper-Facing Gate

full15 retained gate 固定为：

1. `pro_mean >= 0.9000`
2. stretch target: `near 0.9200+`
3. `image_auroc_mean >= 0.9644`
4. `bottle_image_auroc >= 0.9950`
5. `screw image_auroc_mean >= 0.7975`

## Promotion Rule

如果：

- weak5 preflight 通过
- `autorun_mode = on`
- checker 没有记录 concrete blocker

那么 route lead 不得停在 preflight。

必须自动推进到：

- `outputs/new-branch/pure-visual/full15/stage5_restart_query_mask_category_gate_v1`

## Stop Rule

本轮 stop rule 不是“单次弱集失败就 closeout”。

执行规则固定为：

- `weak5 失败 -> 不停 -> 继续 bounded diagnosis + bounded code change`
- `full15 失败 -> 不停 -> 继续 bounded diagnosis + bounded code change`
- `full15 达到 retained gate -> 转入 attribution / writeback`

也就是说：

- `continue if gap remains`
- `close only on retained gate or hard blocker`

## 当前结论

pure-visual 当前正确状态不是 Stage 6。

当前正确状态是：

- `Stage 5 reopened`
- `query-only masking restart`
- `strict three-level execution`
- `smoke first, then auto-promote if clean`
