# 阶段五执行说明：纯视觉 AnomalyDINO 官方预处理对齐

本文档把阶段四的 pure-visual 研究结论转成阶段五的执行合同。

这不是结果写回，而是当前唯一有效的 Stage 5 合同。

## 单一目标

参考 `AnomalyDINO` 官方代码与论文描述，把官方 preprocessing policy 在当前 `DINOv2 PatchCore` 路线中尽量对齐，优先核对并落地：

- masking 的位置与启用条件
- rotation 的策略
- resolution
- patch extraction 口径
- `PRO` 实现

当前 Stage 5 的首要问题不是“再造一个 retrieval trick”，而是：

- 在尽量不换方法族的前提下，确认官方 preprocessing policy 对齐后，是否能把当前 full15 3-seed aggregate `PRO` 从 `0.8634` 拉向接近官方 `0.92+`

## 单一假设

如果当前 repo 的纯视觉 retained winner 还没有把 `AnomalyDINO` 的官方 preprocessing policy 对齐完整，那么：

- `masking`
- `rotation`
- `resolution`
- `patch extraction`
- `PRO implementation`

这些口径差异本身就可能是 `PRO` 落差的重要来源。

因此 Stage 5 的单一假设固定为：

- `official-policy-aligned AnomalyDINO preprocessing inside DINOv2 PatchCore can materially improve aggregate PRO without collapsing image ranking`

## Parity Audit

本轮对齐结论先固定为下面四点：

- `masking`：已从启发式边框裁剪/颜色裁剪，切到基于 patch feature 的 PCA foreground mask，并把 mask 下沉到特征缓存与参考库构建阶段。
- `rotation`：已把 support/reference 的 reference samples 扩展为官方 0/45/.../315 八角度族，按 feature map 级别做旋转增强。
- `resolution / patch extraction`：当前 runner 仍保持官方默认的 `448` 与 DINOv2 `patch size = 14` 的 square 路径，等价于官方默认分辨率口径；为了不越出当前 scorer/mask 约束，本轮不切到更激进的 aspect-ratio resize。
- `PRO`：已把近似阈值 sweep 改成官方式 connected-component PRO 计算，再用 `AU-PRO @ FPR<=0.3` 做归一化积分。

未完全对齐但已显式记录的差异：

- 当前输入仍是 square 448，而不是官方脚本里那种先按 smaller edge resize 再裁到 patch 倍数的完整图像变换。
- 这部分若要再进一步，需要同步改 `score_map_outputs` / mask resize / 官方输出尺寸口径，已超过本轮最小改动边界。

## 不允许做的事

- 不换 `base family`
- 不引入 prompt / text / hybrid 分支
- 不把 defect support 混进主 few-shot 对比
- 不把任务扩成新的 retrieval 拼盘

## 允许修改的代码范围

本轮只允许改：

- `tools/run_stage4_pure_visual_dinov2.py`
- `fewshot/dinov2_backbone.py`
- `fewshot/data.py`

如果为了对齐 `PRO` 口径必须抽取公共 helper，允许最小范围触碰：

- `tools/eval_structure_ablation.py`

除此之外，不扩散到无关模块。

## 官方输出目录

Stage 5 预检目录：

- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v1_preflight`

Stage 5 官方 full15 目录：

- `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v1`

只有 full15 多 seed 聚合结果才回答本阶段的主问题。

## 执行顺序

1. 做官方代码与当前实现的 parity audit：
   - masking 放在什么阶段
   - 哪些类别 / 条件启用 masking
   - rotation 怎么生成与使用
   - 输入分辨率与 patch 提取口径
   - `PRO` 的实现细节
2. 在当前 repo 里做最小实现对齐。
3. 跑 `weak5_bottle / 5-shot / seed42` 预检，确认：
   - 代码路径正确
   - 没有明显破坏 `bottle`
   - `PRO` 方向不是假提升
4. 只有预检通过，才进入：
   - `full15 / normal-only / shots=1,2,4,8,16 / seeds=42,43,44`
5. 用 multi-seed aggregate 决定 Stage 5 是否通过。

## Stage 5 gate

当前 baseline：

- full15 3-seed aggregate `PRO = 0.8634`
- full15 3-seed aggregate `image_auroc = 0.9744`
- `screw image_auroc_mean = 0.7975`
- `bottle_image_auroc ≈ 0.9994`

Stage 5 通过条件：

1. 主目标：
   - `pro_mean >= 0.9000`
2. stretch target：
   - `pro_mean` 尽量接近 `0.9200+`
3. 排名不能塌：
   - `image_auroc_mean >= 0.9644`
4. 强控制类不能坏：
   - `bottle_image_auroc >= 0.9950`
5. hard class 不允许明显倒退：
   - `screw image_auroc_mean >= 0.7975`

如果 `PRO` 上升但 `image_auroc`、`bottle` 或 `screw` 明显塌缩，则 Stage 5 视为未通过。

## 必须产出的工件

- 一份 parity audit 说明：
  - 当前实现 vs 官方实现差在哪
  - 哪些已对齐
  - 哪些无法完全对齐以及原因
- 预检目录中的：
  - `summary.md`
  - `experiments.csv`
  - `per_category.csv`
- full15 多 seed 聚合目录中的：
  - `summary.md`
  - `aggregate.csv`
  - `per_category_multiseed.csv`

## 当前结论

Stage 5 现在已经足够具体，可以进入实现与预检。

这一步的本质不是“发明新方法”，而是先回答一个更基础也更关键的问题：

- 当前 `PRO` 差距，到底有多少是方法口径差异造成的。
