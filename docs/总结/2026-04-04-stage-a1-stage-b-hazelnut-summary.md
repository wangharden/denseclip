# DenseCLIP Few-Shot `hazelnut` 实验总结

## 1. 协议对齐情况

当前实验是按照 `C:\Nottingham\DenseCLIP-master\docs\design\design-v1-visual-prototype-experiment-protocol.md` 推进的。

对齐点如下：

- `Stage A1` 使用官方 `mvtec_anomaly_detection` 目录结构。
- `support_normal` 仅来自 `train/good`。
- `Stage A1` 冻结视觉 backbone，不训练 tiny head。
- `Stage A1` 直接在 `test/good + test/defect` 上评估。
- `Stage B` 保持同一官方协议，只新增 defect-aware prototype 和 tiny head 训练。

需要说明的一点：

- 协议文档里的 `A2` 还没有实现。
- 当前第二条对照线是 `Stage B learned-head`，不是 `A2 normal+defect prototype without training`。
- 由于官方 MVTec 没有 defect training split，`Stage B` 的 `support_defect` 是从一小部分 defect test 图中抽样后，再从最终评估集中剔除。这是当前实现的务实处理，不是最理想的长期协议。

## 2. 具体实验经过

### 2.1 Stage A1: normal-only visual prototype

入口脚本：

- `C:\Nottingham\DenseCLIP-master\run_stage_a1.py`

核心实现：

- `C:\Nottingham\DenseCLIP-master\fewshot\data.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\backbone.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\feature_bank.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\stage_a1.py`

本次保留结果对应配置：

- `category = hazelnut`
- `support_normal_k = 4`
- `image_size = 224`
- `batch_size = 4`
- `seed = 42`
- `save_visuals = false`

输出目录：

- `C:\Nottingham\DenseCLIP-master\outputs\stage_a1\hazelnut`

主要产物：

- `stage_a1_prototype.pt`
- `metrics.json`
- `predictions.csv`
- `support_paths.json`

### 2.2 Stage B: learned-head comparison

入口脚本：

- `C:\Nottingham\DenseCLIP-master\run_stage_b.py`

核心实现：

- `C:\Nottingham\DenseCLIP-master\fewshot\data.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\head.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\learned_head.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\feature_bank.py`
- `C:\Nottingham\DenseCLIP-master\fewshot\stage_a1.py`

本次短跑配置：

- `category = hazelnut`
- `support_normal_k = 8`
- `support_defect_k = 4`
- `image_size = 320`
- `batch_size = 8`
- `epochs = 3`
- `lr = 1e-3`
- `weight_decay = 1e-4`
- `seed = 42`

输出目录：

- `C:\Nottingham\DenseCLIP-master\outputs\stage_b_smoke\hazelnut`

主要产物：

- `stage_b_learned_head.pt`
- `metrics.json`
- `predictions.csv`
- `support_paths.json`
- `train_history.json`

## 3. 实验结果

### 3.1 Stage A1

来源：

- `C:\Nottingham\DenseCLIP-master\outputs\stage_a1\hazelnut\metrics.json`

结果：

- `num_support_normal = 4`
- `num_test_query = 110`
- `image_auroc = 0.46464285714285714`
- `pixel_auroc = 0.5363952752992882`

### 3.2 Stage B

来源：

- `C:\Nottingham\DenseCLIP-master\outputs\stage_b_smoke\hazelnut\metrics.json`

结果：

- `num_support_normal = 8`
- `num_support_defect = 4`
- `num_test_query = 106`
- `image_auroc = 0.4700757575757576`
- `pixel_auroc = 0.5346651964045523`

## 4. 结果分析

### 4.1 Stage A1 已经是有效的无训练基线

`Stage A1` 不是伪造 smoke，也不是玩具数据验证。

它已经满足以下条件：

- 使用官方 MVTec 目录协议
- 使用真实 `hazelnut` 数据
- support 和 test 分离
- 不训练权重
- 可以稳定输出 image-level 和 pixel-level 指标

因此它可以被视为“协议正确的无训练 baseline 已打通”。

### 4.2 当前分数水平接近随机

`image_auroc` 在 `0.46 ~ 0.47` 区间，按正常解释是差的。

- `0.5` 约等于随机排序
- `< 0.5` 说明排序方向可能不对，或者模型信号太弱

因此现在不能把这两组结果当作有效方法结论。

### 4.3 当前存在明显的 score direction 问题

对 `predictions.csv` 做检查后发现：

- `Stage A1` 原始 `image_auroc = 0.4646`
- 如果把分数反号，AUROC 约为 `0.5354`
- `Stage B` 原始 `image_auroc = 0.4701`
- 反号后约为 `0.5292`

这说明当前不是“完全没有信号”，而是：

- 有一部分信号
- 但当前 `image_score` 定义方向很可能反了，或者 anomaly score 的聚合方式不合适

### 4.4 Stage B 的 learned head 目前塌缩明显

从 `Stage B` 的 `predictions.csv` 和 `train_history.json` 看：

- image score 几乎是常数
- train loss 没有往好的方向收敛

当前最合理的解释是：

- tiny head 的训练数据太少
- supervision 只有 image-level BCE，约束太弱
- head 很容易学成常数输出

所以现在不能说 “learned-head 比 visual prototype 更强”，当前证据不支持这个结论。

### 4.5 A2 尚未实现

协议里最重要的中间对照其实是：

- `A1 = normal-only prototype`
- `A2 = normal + defect prototype`
- `B = normal + defect prototype + learned head`

目前少了 `A2`，导致我们无法单独判断：

- defect support 本身是否有帮助
- 还是只有 learned head 才可能有帮助

## 5. 当前结论

当前可以确认的结论只有这些：

- 协议文档已经落到代码和真实实验上。
- `Stage A1` 无训练基线已能稳定评估。
- `Stage B` learned-head 对照线已能训练和评估。
- 当前 `hazelnut` 结果接近随机，不能作为有效性能结论。
- 当前最大问题不是“代码没跑通”，而是“score 定义和对照矩阵还不够干净”。

## 6. 推荐下一步

推荐按下面顺序继续，而不是直接全类别扩展：

1. 实现 `A2`，即 `normal + defect prototype` 但仍然不训练 head。
2. 对 `A1 / A2 / B` 做 score direction 检查和统一定义。
3. 固定 `hazelnut` 做小规模 sweep：
   - `support_normal_k = 4 / 8 / 16 / 32`
   - `support_defect_k = 1 / 2 / 4 / 8`
4. 如果继续做 `B`，优先考虑更强监督而不是只加 epoch。

在这之前，不建议直接把当前版本扩到全 15 类然后汇总结果，因为这样会放大协议和 scoring 设计中的问题。
