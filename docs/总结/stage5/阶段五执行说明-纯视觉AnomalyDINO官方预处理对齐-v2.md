# 阶段五执行说明：纯视觉 AnomalyDINO 官方预处理对齐 v2

本文档承接已完成的 Stage 5 v1 checker verdict。

它不是结果写回，而是当前 pure-visual 路线继续做 `复现优先` 推进时的唯一 v2 执行合同。

## 当前阶段判断

Stage 5 v1 官方 full15 多 seed run 已完成，结果是：

- `image_auroc_mean = 0.976766`
- `pro_mean = 0.824271`

结论不是路线崩溃，而是：

- image ranking 保持强
- `bottle` 与 `screw` 没塌
- 但主复现目标 `PRO >= 0.9000` 明确失败

因此当前不进入 attribution closeout，而是继续复现实验。

## 本轮唯一目标

继续沿 `DINOv2 PatchCore` 同一方法族推进 Stage 5 复现。

v2 只允许补当前已定位的剩余 parity step：

- `smaller-edge + aspect-ratio-preserving resize`
- 以及与之直接绑定的：
  - `score_map_outputs`
  - `mask resize`
  - `output size`

本轮不允许把失败重新解释成新的 retrieval 创新需求。

## 单一假设

如果 v1 仍然把所有图像强行压成 square `448x448`，而官方脚本采用的是：

- smaller-edge resize
- aspect-ratio preserving geometry
- 再对齐到 patch 倍数

那么当前 `PRO` 的主要剩余落差，仍可能有相当部分来自：

- anomaly map 几何被方形缩放扭曲
- foreground mask 与 score map 边界不再对应真实目标形状
- gt mask 与 upsampled score map 的对齐口径不一致

因此 v2 的单一假设固定为：

- `geometry-faithful smaller-edge aspect-ratio preprocessing inside the same DINOv2 PatchCore route can recover residual PRO without sacrificing ranking`

## 三层级 Agent 推进方式

本轮严格使用现有三层级架构。

### 第一层：主 Session

主 session 只负责：

- 固定 v2 合同
- 固定 stop rule
- 读取 route board / live board
- 在 milestone 处做最终判断

### 第二层：pure-visual Route Lead

route lead 继续由 `Fermat` 负责，按 `autorun_mode=on` 自推进。

本 tranche 的推进规则改为：

- 如果某轮复现实验失败：
  - 不停止
  - 回到 bounded diagnosis
  - 继续下一轮复现实验
- 如果某轮复现实验成功：
  - 不立即继续开新实验
  - 转入 attribution note
  - attribution 一开始就暂停 route，等待主 session 再决定

### 第三层：Role Worker

本轮采用两段式：

1. `Smoke mode`
   - `implementer`
   - `checker`
   - `writeback`
   - `doc-check`
2. `Training mode`
   - `reviewer`
   - `implementer`
   - `runner`
   - `monitor`
   - `checker`
   - `writeback`
   - `doc-check`

执行顺序：

1. implementer 落地 v2 parity 改动
2. weak5 preflight 作为唯一 smoke
3. checker 判定是否允许 full15 rerun
4. 若 smoke 通过，runner 启动 full15 官方 rerun
5. checker 给出复现 verdict
6. 若复现成功，转入 attribution note 并停止
7. 若复现失败，route 保持打开并继续下一 bounded rerun

## 允许改动的代码范围

本轮只允许改：

- `fewshot/dinov2_backbone.py`
- `fewshot/data.py`
- `fewshot/scoring.py`
- `tools/run_stage4_pure_visual_dinov2.py`

如无必要，不改其他文件。

## Reviewer 预跑约束

在任何 weak5 preflight 之前，必须先满足下面这些 reviewer 条件：

1. 合同必须继续冻结在这一条 parity step 上：
   - `smaller-edge + aspect-ratio-preserving resize`
   - `score_map_outputs / mask resize / output size`
2. 四个代码入口必须同时切到同一套几何口径：
   - `fewshot/dinov2_backbone.py`
   - `fewshot/data.py`
   - `fewshot/scoring.py`
   - `tools/run_stage4_pure_visual_dinov2.py`
3. cache 必须与 v1 隔离。
   - 只要 resize policy 变了，就不允许继续混用 v1 square cache
4. official output 目录必须与 v1 隔离。
5. run manifest / summary 必须能看出本轮确实使用的是 v2 aspect-ratio contract。

## 不允许做的事

- 不换 base family
- 不引入新 retrieval family
- 不把任务改写成 prompt / hybrid / support-aware 新方法筛选
- 不把 defect support 混进新的主 few-shot claim
- 不跳过 weak5 preflight 直接发 full15 rerun

## 官方目录

v2 weak5 preflight：

- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v2_preflight`

v2 full15 official rerun：

- `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v2`

## weak5 Preflight Gate

只有全部满足，才允许进入 full15 v2 official rerun：

1. `alerts = none`
2. `bottle image_auroc >= 0.9950`
3. `image_auroc_mean >= 0.9700`
4. `pro_mean >= 0.8900`
5. 不接受明显几何塌缩：
   - `zipper`、`tile`、`wood` 中任意一类若出现显著 localization 崩坏或大面积 false positive，直接判失败
6. `screw image_auroc_mean >= 0.7975`
7. `PRO` 不允许低于当前已记录的 v1 weak5 preflight 参考值 `0.897958`

这一步只负责判断：

- geometry parity 是否走对方向
- 是否没有打坏强控制类

## full15 Reproduction Gate

v2 official rerun 的复现目标仍然是 Stage 5 主 gate：

1. `pro_mean >= 0.9000`
2. stretch target：`near 0.9200+`
3. `image_auroc_mean >= 0.9644`
4. `bottle_image_auroc >= 0.9950`
5. `screw image_auroc_mean >= 0.7975`

## Stop Rule

这是本轮最重要的运行规则。

### 若复现实验失败

满足以下任一：

- weak5 preflight 未通过
- full15 official rerun 未通过 reproduction gate

则：

- `do not stop`
- route 保持 open
- 回到 bounded diagnosis
- 继续下一轮复现实验

尤其是：

- 若 `image ranking` 仍强，但 `PRO` 没有回到 weak5 参考带或 full15 gate，则判为 `复现未完成`，不能提前进入 attribution。

### 若复现实验成功

满足 full15 reproduction gate 后：

- 不再继续自动开新实验
- 立即转入 attribution
- attribution 只要求写出：
  - 哪些改动真正解决了复现 gap
  - 哪些 residual weakness 还在
  - 是否值得从 reproduction 转入 paper-facing innovation
- attribution 文档一开始写，就暂停 route

也就是说：

- `复现成功 -> 开始归因 -> 停`
- `复现失败 -> 不停 -> 继续实验`

## 必须产出的工件

weak5 preflight：

- `summary.md`
- `experiments.csv`
- `per_category.csv`
- `alerts.json`

full15 official rerun：

- `summary.md`
- `experiments.csv`
- `per_category.csv`
- `checker_verdict.md`

若 reproduction 成功并进入 attribution：

- 一份新的 attribution note，放在 `docs/总结/stage5/`

## 当前结论

对 pure-visual Stage 5 来说，现在不是 closeout，也不是自由原创。

当前正确状态是：

- `继续 bounded reproduction`
- `用三层级 agent 架构自推进`
- `直到某轮 rerun 完成复现`
- `一进入 attribution 就停止`
