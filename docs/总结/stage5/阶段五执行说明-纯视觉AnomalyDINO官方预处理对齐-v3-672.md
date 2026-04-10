# 阶段五执行说明：纯视觉 AnomalyDINO 官方预处理对齐 v3 672

本文档承接已完成的 Stage 5 v2 weak5 preflight verdict。

它不是结果写回，而是当前 pure-visual 路线继续做 `复现优先` 推进时的唯一 v3 执行合同。

## 当前阶段判断

Stage 5 v2 的 weak5 官方预检已完成，结果是：

- `image_auroc = 0.967186`
- `pixel_auroc = 0.855525`
- `pro = 0.785745`
- `bottle = 1.000000`
- `screw = 0.809968`

结论不是路线关闭，而是：

- `bottle` 与 `screw` 没塌
- 但 weak5 gate 明确失败
- 因此当前仍然处于 `继续复现`，不是 attribution

## v2 失败的关键归因

本轮弱集 `bottle / carpet / grid / leather / screw / zipper` 的原图本身就是 square：

- `bottle`: `900x900`
- 其余五类: `1024x1024`

因此 `image_size=448 + resize_mode=smaller_edge` 对这些 weak5 类并不会真正引入新的 aspect-ratio geometry。

这意味着：

- v2 weak5 的失败，不能再归因给 `448 aspect-ratio parity` 本身
- 对当前 weak5 而言，`smaller-edge@448` 基本等价于继续跑 `square@448`

## 本轮唯一目标

继续沿 `DINOv2 PatchCore` 同一方法族推进 Stage 5 复现。

v3 只允许测试一个新的 reproduction-only 变量：

- `image_size = 672`

其余保持不变：

- `method = patchcore_knn`
- `feature_source = last`
- `knn_topk = 1`
- `coreset_ratio = 1.0`
- `topk_ratio = 0.01`
- `resize_mode = smaller_edge`
- `resize_patch_multiple = 14`

## 单一假设

如果当前 weak5 失败的主要问题不是 `448 几何口径`，而是 localization 分辨率本身仍不足，那么：

- 在同一 `DINOv2 PatchCore` 路线下把输入分辨率提升到官方更强的 `672`
- 可能改善 `leather / zipper / grid` 这类 localization 敏感类的 `pixel_auroc / PRO`
- 且不需要提前跳到新的 retrieval 创新

因此 v3 的单一假设固定为：

- `official higher-resolution 672 inside the same DINOv2 PatchCore route can recover weak5 localization without breaking ranking`

## 三层级 Agent 推进方式

本轮继续严格使用现有三层级架构：

1. 主 session 固定 v3 合同与 gate。
2. pure-visual route lead 继续按 `autorun_mode=on` 推进。
3. role worker 只执行：
   - `runner`
   - `checker`
   - `writeback`
   - `doc-check`

规则保持不变：

- `复现失败 -> 不停 -> 继续实验`
- `复现成功 -> 开始归因 -> 停`

## 不允许做的事

- 不换 base family
- 不引入新 retrieval family
- 不把任务改写成 prompt / hybrid / support-aware 方法筛选
- 不引入新的结构性代码改动
- 不跳过 weak5 preflight 直接发 full15 rerun

## 允许改动的范围

本轮优先不改代码。

若运行中暴露出明确的工程阻塞，再单独冻结新的实现合同。

## 官方目录

v3 weak5 preflight：

- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v3_672_preflight`

v3 full15 official rerun：

- `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v3_672`

## weak5 Preflight Gate

只有全部满足，才允许进入 full15 v3 official rerun：

1. `alerts = none`
2. `bottle image_auroc >= 0.9950`
3. `image_auroc >= 0.9700`
4. `pro >= 0.8900`
5. `screw image_auroc >= 0.7975`
6. `zipper / leather / grid` 不接受明显 localization 崩坏或大面积 false positive

## full15 Reproduction Gate

v3 official rerun 的复现目标仍然是 Stage 5 主 gate：

1. `pro_mean >= 0.9000`
2. stretch target: `near 0.9200+`
3. `image_auroc_mean >= 0.9644`
4. `bottle_image_auroc >= 0.9950`
5. `screw image_auroc_mean >= 0.7975`

## Stop Rule

- `weak5 或 full15 失败 -> do not stop -> 继续 bounded reproduction`
- `full15 成功 -> 开始 attribution -> stop`

## 当前结论

对 pure-visual Stage 5 来说，v2 已经给出了一个关键负结论：

- `448 smaller-edge` 不是当前 weak5 residual gap 的主解释

因此 v3 的正确动作不是原创，而是继续做下一条官方 reproduction-only 检查：

- `672 resolution preflight`
