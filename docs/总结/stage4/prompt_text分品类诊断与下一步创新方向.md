# prompt-text分品类诊断与下一步创新方向

## 1. 使用的结果源

- weak5 `v3 cscdefect`:
  [per_category.csv](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_ctx_v3_cscdefect_screen/per_category.csv)
- weak5 `v4 cscfull`:
  [per_category.csv](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/prompt-text/weak5_bottle/seed42/prompt_ctx_v4_cscfull_screen/per_category.csv)
- weak5 `PromptAD-like`:
  [per_category.csv](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/prompt-text/weak5_bottle/seed42/promptad_state_v1_screen/per_category.csv)
- full15 `PromptAD-like / seed42`:
  [per_category.csv](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/prompt-text/full15/seed42/promptad_state_v1_full15/per_category.csv)
- 纯视觉 retained winner 对照:
  [per_category_multiseed.csv](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/per_category_multiseed.csv)

## 2. 图示

- weak5 分品类 AUROC 对比:
  [prompt_text_weak5_category_auroc_compare.png](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/_ops/analysis/prompt_text_weak5_category_auroc_compare.png)
- PromptAD-like full15 对 E0 的分品类增益:
  [promptad_full15_delta_vs_e0.png](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/_ops/analysis/promptad_full15_delta_vs_e0.png)
- PromptAD-like weak5 相对 E0 的散点:
  [promptad_weak5_scatter_vs_e0.png](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/_ops/analysis/promptad_weak5_scatter_vs_e0.png)

## 3. 哪些类在拖后腿

先看最新完成的 `PromptAD-like weak5`。总体已经比旧 `v3/v4` 好，但拖后腿的类仍然很明确：

- `zipper`: `0.6989`
- `screw`: `0.5905`
- `grid`: `0.7197`

其中：

- `leather`: `0.9893`
- `carpet`: `0.9399`
- `bottle`: `0.9619`

已经很好，说明当前 prompt-text 路线不是“整体没学到”，而是明显偏向纹理型异常，对结构型、状态型、细粒度局部异常仍然弱。

再看 `PromptAD-like full15 / seed42`，拖后腿的类扩展为：

- `capsule`: `0.5553`
- `grid`: `0.5696`
- `screw`: `0.6653`
- `toothbrush`: `0.7147`
- `cable`: `0.7461`
- `transistor`: `0.7500`
- `zipper`: `0.7258`

这组类的共同点是：

- 异常往往是局部结构变化，而不是大面积纹理变化
- 正常和异常的全局语义词都很接近，靠“normal/anomaly”宽模板很难拉开
- 类内 defect state 多样，单一 anomaly prompt 容易把多个状态压成一个粗粒度文本原型

## 4. 图里显示的模式

图一显示：

- `v3 -> v4 -> PromptAD-like` 的演化里，`leather`、`carpet` 一直上升并接近饱和
- `zipper` 在 `v4` 甚至掉到 `0.4723`，到 `PromptAD-like` 才拉回 `0.6989`
- `screw` 和 `grid` 虽然从 `E0` 提升，但仍明显低于可保留水平

这说明现在的问题不是“PromptAD 框架本身无效”，而是 prompt bank 和损失都更偏向粗粒度 anomaly-vs-normal，对结构型 hard class 约束不够。

图二显示：

- 在 full15 上，PromptAD-like 对 `carpet`、`leather`、`hazelnut`、`pill` 这类类有大幅正增益
- 但 `capsule`、`grid`、`screw`、`toothbrush`、`zipper` 增益有限甚至仍偏低

这说明问题不是视觉底座已经到天花板。因为纯视觉 retained winner 对 `grid`、`zipper` 已经接近满分，所以当前 prompt-text 的瓶颈更像是文本建模与训练目标，而不是图像特征不够强。

图三显示：

- `bottle` 基本贴近对角线，说明控制类保护已经比前几轮好
- `zipper`、`screw` 明显落在“提升有限”的区域
- `carpet`、`leather` 明显高于对角线，说明 prompt-text 特别擅长纹理类

一句话总结：当前 PromptAD-like 已经学会“纹理异常提示”，但还没有学会“结构状态提示”。

## 5. 这说明下一步该改哪里

这轮不应再做多方法拼接，而应把 `PromptAD` 作为唯一底座，在它上面做创新。

### 5.1 prompt bank 该怎么改

优先方向是 `taxonomy-aware state prompt bank`。

做法：

- 保留 `PromptAD` 的 normal/anomaly 双分支框架
- 但 anomaly prompt 不再只有一个粗粒度 `defective {category}`
- 改成“共享异常干 + 类别状态叶子”的层级 prompt bank

建议先覆盖当前 hard class：

- `zipper`
  - `missing teeth`
  - `misaligned teeth`
  - `broken zipper edge`
  - `partial opening/closing defect`
- `screw`
  - `damaged slot`
  - `deformed head`
  - `thread defect`
  - `surface contamination`
- `grid`
  - `broken grid line`
  - `misaligned grid`
  - `missing segment`
  - `irregular pattern cell`

这里的关键不是把 prompt 变多，而是把异常词从“泛异常”换成“状态异常”。

### 5.2 loss 该怎么改

优先方向是 `hard-class-aware PromptAD loss`。

建议三项：

- `class-aware margin weighting`
  - 对 `zipper / screw / grid / capsule / toothbrush / transistor` 提更高 margin weight
  - 对已接近饱和的 `leather / carpet / bottle` 降低更新力度
- `control-preserving regularization`
  - 保留 `bottle`，并把 `zipper` 也纳入 control regularization
  - 目标不是把 `zipper` 冻死，而是防止强类先被 prompt 拉坏
- `hard negative defect-state mining`
  - 在同类内部做 normal vs hard defect-state 对比
  - 尤其对 `zipper` 和 `screw`，让最难 state 对形成主梯度，而不是被 easy texture 样本主导

### 5.3 不建议优先改什么

- 不建议先换新的视觉 backbone
- 不建议再做 `PromptAD + CoOp + AnomalyCLIP + adapter` 这种拼盘
- 不建议先加更复杂的阈值技巧来掩盖 hard class 失分

原因很简单：当前问题已经不是“总分不涨”，而是“某些结构型类一直没被 prompt bank 正确建模”。

## 6. 与纯视觉 retained winner 的对照

纯视觉 retained winner：

- [summary.md](/C:/Nottingham/DenseCLIP-master/outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/summary.md)

它说明：

- `grid`、`zipper` 在纯视觉路线里不是不可解
- 所以 prompt-text 的主问题不是“任务天然太难”
- 而是当前文本分支对结构状态的表达能力不足

这也意味着 prompt-text 继续推进时，应该把创新点定义为：

- `PromptAD on industrial state prompts`
- 而不是“再加一个视觉分支去兜底”

## 7. 下一条最值得跑的创新分支

推荐主线：

`PromptAD + taxonomy-aware state prompt bank + hard-class-aware control-preserving loss`

最小可行版本：

1. 只保留 `PromptAD` 主框架
2. 先给 `zipper / screw / grid` 加状态型 anomaly prompt bank
3. 对这三类加大 margin 和 defect-state mining 权重
4. 把 `zipper` 和 `bottle` 一起纳入 control-preserving regularization

如果这条线在 `weak5_bottle / seed42` 能把：

- `zipper` 拉回到 `>= 0.80`
- `screw` 拉到 `>= 0.72`
- `grid` 拉到 `>= 0.78`
- 同时 `bottle` 保持 `>= 0.95`

它就值得晋级 `full15`。

## 8. 结论

当前 prompt-text 路线已经不再是“完全不可用”，而是“强纹理、弱结构”。

真正拖后腿的类是：

- weak5: `zipper`, `screw`, `grid`
- full15: `capsule`, `grid`, `screw`, `toothbrush`, `cable`, `transistor`, `zipper`

因此下一步不该继续做 SOTA 拼盘，而应以 `PromptAD` 为唯一底座，围绕“状态型 prompt bank + hard-class-aware loss + control-preserving regularization”做创新。
