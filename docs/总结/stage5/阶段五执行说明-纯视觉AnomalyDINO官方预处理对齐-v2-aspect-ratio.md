# 阶段五执行说明：纯视觉 AnomalyDINO 官方预处理对齐 v2

本文档定义 pure-visual Stage 5 的下一轮 bounded reproduction 合同。

它沿用当前 Stage 5 的同一 base family：

- `DINOv2 PatchCore`

不引入新的 retrieval family，不引入 text / hybrid，不把任务改写成新的方法研究。

## 当前状态

已完成的 Stage 5 官方目录：

- `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v1`

checker verdict：

- `image_auroc_mean = 0.976766`
- `pro_mean = 0.824271`
- `bottle_image_auroc_mean = 0.999153`
- `screw_image_auroc_mean = 0.820007`

结论：

- `v1` 保住了 image ranking、`bottle` 和 `screw`
- 但主 reproduction 目标 `PRO` 明确失败
- 因此 route 不进入 attribution stop，而是继续下一轮 bounded reproduction

## 单一目标

本轮只补一个未闭环 parity step：

- `smaller-edge + aspect-ratio-preserving resize`

以及与它强耦合的两处实现口径：

- `score_map_outputs`
- `mask resize / output size`

一句话说：

- `v1` 先补齐了 masking / rotation / PRO implementation
- `v2` 只补齐 aspect-ratio-aware image geometry parity

## 单一假设

如果 `v1` 失败的主原因仍然来自当前 square-448 几何口径与官方 `smaller-edge` 几何口径不一致，那么：

- 把输入从强制 square resize 改成 smaller-edge resize
- 再把输出 score map、pixel mask 和 PRO 评估统一到真实非方形几何

应该能在不破坏 image ranking 的前提下，恢复一部分 `PRO`。

## 不允许做的事

- 不换 `base family`
- 不把任务改成新的 `support-aware retrieval` 研究
- 不加入 prompt / text / hybrid
- 不把 defect support 混进主 reproduction claim
- 不把本轮失败直接升级为 attribution closeout

## 允许修改的代码范围

本轮只允许改：

- `fewshot/dinov2_backbone.py`
- `fewshot/data.py`
- `fewshot/scoring.py`
- `tools/run_stage4_pure_visual_dinov2.py`

## 模式与角色

### 当前主模式

- 先进入 `Smoke mode`

用于：

- `weak5_bottle / seed42` 官方预检

角色分工：

- `implementer`
- `checker`
- `writeback`
- `doc-check`

只有 weak5 预检通过，才进入：

- `Training mode`

用于：

- `full15 / seed42,43,44` 官方 rerun

角色分工：

- `reviewer`
- `implementer`
- `runner`
- `monitor`
- `checker`

## 官方目录

Stage 5 v2 预检目录：

- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v2_preflight`

Stage 5 v2 官方 full15 目录：

- `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v2`

## 预检 gate

只有满足下面全部条件，才允许从 `weak5_bottle / seed42` 晋级到 `full15`：

1. `bottle_image_auroc >= 0.9950`
2. `image_auroc_mean >= 0.9625`
3. `pro_mean >= 0.8800`
4. 无 `non_finite_*`、`degenerate_*`、`threshold_metrics_collapsed`、`bottle_regression_severe` alerts

说明：

- weak5 不要求直接达到 full15 reproduction gate
- 它只负责确认 aspect-ratio parity 没把控制类和基本 ranking 打坏

## full15 reproduction gate

本轮 full15 官方 rerun 的完成条件固定为：

1. `pro_mean >= 0.9000`
2. `image_auroc_mean >= 0.9644`
3. `bottle_image_auroc_mean >= 0.9950`
4. `screw_image_auroc_mean >= 0.7975`
5. 无 collapse-style alerts

stretch target：

- `pro_mean` 尽量接近 `0.9200+`

## Route Stop Rule

当前 route 的自动推进规则改成下面这一条：

- 只要 reproduction gate 还没过，就继续下一个 bounded reproduction rerun
- 不因为单轮失败自动停在 diagnosis
- 只有在某一轮 clear reproduction gate 后，才停止继续实验，并把 route 切到 attribution / research handoff

也就是说：

- `复现失败 -> 继续实验`
- `复现完成 -> 开始归因并停止自动实验推进`

## Abort Rule

只有出现下面情况，才允许中断当前 reproduction 序列：

- 非有限指标或损失
- 结果目录缺失关键工件
- 明显阈值塌缩
- `bottle` 严重回退
- reviewer 确认本轮实现越界到新的方法族

不允许因为“这一轮没过 PRO gate”就直接停掉 route。

## 必须产出的工件

weak5 预检：

- `summary.md`
- `experiments.csv`
- `per_category.csv`
- `alerts.json`

full15 官方 rerun：

- `summary.md`
- `experiments.csv`
- `per_category.csv`
- `checker_verdict.md`

## 当前结论

Stage 5 现在不该停在 attribution，也不该重开大研究面。

正确动作是：

- 继续同一 reproduction family
- 只补 `aspect-ratio geometry parity`
- 先 weak5 预检
- 再 full15 官方 rerun
- 直到 reproduction gate 通过，才切到 attribution
