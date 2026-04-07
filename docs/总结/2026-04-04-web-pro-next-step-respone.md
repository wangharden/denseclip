直接给决策。

第一，**先统一 split 与 scoring，再立刻补 A2；不要先继续堆 B，更不要提前做 prompt learning**。协议本身已经把 v1 基线定义为 frozen visual backbone 的 visual-prototype 路线，prompt 和更复杂扩展都应放在后面。

第二，**当前最先要修的不是“多跑几轮”，而是“把比较条件和 scoring 定义洗干净”**。现有 `hazelnut` 的 A1/B 结果并不能直接说明 learned head 有没有价值，因为它们的 `support_normal_k`、`image_size`、甚至 `num_test_query` 都不同；同时 A1/B 的 `image_auroc` 都低于 0.5，但 `pixel_auroc` 都略高于 0.53，这更像是 **image-level aggregation / scoring 定义先有问题**，不只是“完全没信号”。

第三，**B 路线若继续，优先改 supervision / loss，而不是先改 epoch，也不是先大改协议**。当前 B 只在 `support_normal + support_defect` 这些极少量图像上，用 image-level BCE 训练 tiny head；代码里 defect mask 已经能读到，但并没有进入 loss，head 还是一个很小的 1×1 conv 网络，塌缩成近常数输出并不意外。   

---

## 1. 先锁死“公平对照协议”

### 目标

把 A1 / A2 / B 放到**同一个可复现 split manifest**上，先消除比较污染。

### 现在必须承认的事实

当前 A1 和 B 的 query 集并不一样。`build_stage_b_split()` 会从 defect test 中抽 `support_defect`，再把这些图像从最终 `test_query` 里剔除；而 A1 是直接用全 test 集。现在把 A1 和 B 的数字并排看，本身就不够公平。 

### 要改的文件 / 模块

* `fewshot/data.py`

  * 新增统一 split builder，例如 `build_experiment_split(...)`
  * 支持：

    * `support_normal_k`
    * `support_defect_k`
    * `seed`
    * 保存 `support_normal / support_defect / query_eval` 全路径
* `run_stage_a1.py`

  * 增加 `--split-manifest`，允许 A1 在“与 A2/B 完全相同的 query 集”上重跑
* `run_stage_b.py`

  * 同样支持 `--split-manifest`
  * 输出 `split_manifest.json`
* 新增一个很小的工具脚本

  * 例如 `tools/make_split_manifest.py`

### 这里的决策

A1 以后保留两种结果：

1. **strict A1**：按协议，用全 test query，作为纯 anomaly baseline
2. **matched A1**：用与 A2/B 相同的 query manifest，只用于 apples-to-apples 对照

这样既不破坏协议纯度，也能保证 A1/A2/B 的比较干净。协议文档的核心就是“先有可解释的对照，再加模块”。

### 最小实验矩阵

先只做 `hazelnut`：

* `image_size = 320`
* `support_normal_k = 8`
* `support_defect_k = 4`
* `seed = 42, 43, 44`

### 验收标准

* A1/A2/B 都能加载同一个 `split_manifest.json`
* `support_*` 与 `query_eval` 零重叠
* matched A1 / A2 / B 的 `num_test_query` 完全一致
* 输出目录里必须有 `split_manifest.json`

---

## 2. 统一 scoring 模块，并先做 scoring-ablation

### 目标

先把“到底怎么从 similarity map 变成 anomaly score”定下来，再谈方法优劣。

### 为什么这一步优先级最高

A1 当前是：

* `normal_map = cos(local, normal_prototype)`
* `anomaly_map = 1 - normal_map`
* `image_score = topk_mean(upsampled_map)`  

这条链路里至少有三个可能同时出错的地方：

1. map 方向
2. image-level aggregation 方式
3. aggregation 是在 patch map 上做，还是在 upsample 后做

而你现在看到的是：`pixel_auroc` 略高于随机，`image_auroc` 却低于随机。我的判断是，**优先嫌疑不是 backbone，而是 image-score 聚合定义**。 

### 要改的文件 / 模块

* 新增 `fewshot/scoring.py`

  * `compute_similarity_maps(local_features, normal_prototype, defect_prototype=None)`
  * `build_score_map(mode=...)`
  * `aggregate_image_score(mode=..., on="patch|upsampled", topk_ratio=...)`
* `run_stage_a1.py`

  * 增加：

    * `--score-mode`
    * `--image-agg`
    * `--agg-input patch|upsampled`
* `fewshot/learned_head.py`

  * 改为复用同一个 similarity builder，避免 A1/A2/B 各自一套 scoring 逻辑
* 建议新增一个离线分析脚本

  * `tools/analyze_scoring.py`
  * 直接读保存下来的 raw maps，离线算 AUROC，避免每次都重跑 backbone

### 最小实验矩阵

先只做 `hazelnut`，并且只在 matched split 上做。

#### A1 map

* `1 - normal_map`

#### A2 map（实现后一起接入）

* `defect_map - normal_map`
* `normal_map - defect_map`（控制项，专门查方向）
* `0.5 * (1 - normal_map) + 0.5 * defect_map`

#### image aggregation

* `patch_max`
* `patch_topk_mean@1%`
* `patch_topk_mean@5%`
* 当前控制项：`upsampled_topk_mean@10%`

### 验收标准

* 选出一个默认 scorer，满足：

  * 3 个 seed 方向一致
  * 在 `hazelnut` 上 mean image AUROC 比当前默认设置至少高 **0.03**
  * 且 `mean(score_defect) > mean(score_good)` 在 3 个 seed 都成立
* 如果做不到，**先不要推进 B，也不要推进 prompt**
* 若 scoring 全部仍接近随机，就说明真正瓶颈更可能在 prototype 设计，而不是 sign

---

## 3. 立刻补 A2，但必须建立在统一 scorer 上

### 目标

补上协议中缺失的中间对照：`A2 = normal + defect prototype, no training`。协议里这一步本来就是必须的。

### 要改的文件 / 模块

* 新增 `run_stage_a2.py`
* `fewshot/data.py`

  * 复用统一 split builder
* `fewshot/scoring.py`

  * 接入 dual-prototype score map
* `fewshot/feature_bank.py`

  * 现有 `build_prototype_bank()` 已支持 `support_defect`，基本够用；只需把产物和路径记录补齐即可。

### 最小实验矩阵

先只做 `hazelnut`：

* A1 vs A2
* `support_normal_k = 8`
* `support_defect_k = 1, 4`
* 统一 `image_size = 320`
* `seed = 42, 43, 44`
* 使用上一步选出的默认 scorer

### 验收标准

* A2 相比 matched A1：

  * 平均 image AUROC 至少提升 **0.02**，或
  * 3 个 seed 里至少 2 个 seed 有稳定正提升
* 如果 A2 完全不比 A1 好，**不要急着救 B**

  * 那说明 defect support 本身都没有提供可用信号
  * 这时下一步应该改 `feature_bank.py`，做 **multi-prototype / memory bank / kNN prototype**，而不是上 prompt

这里我给一个非常明确的判断：**如果 A2 不成立，B 的优先级要下降；prompt learning 的优先级更要下降。**

---

## 4. B 路线继续时，优先改 supervision / loss，不先改协议

### 目标

先证明 tiny head 至少“学得动”，再讨论它“值不值得留”。

### 为什么不是先改协议

B 当前的 protocol 问题你已经知道了：`support_defect` 来自 test defect，再从最终评估里剔除。这个问题必须**明确标注**，但它不是当前 head 塌缩的最直接原因。最直接原因是：

* 训练图像极少
* 只有 image-level BCE
* defect mask 没参与监督
* 还用了带 ReLU / dropout 的小头，容易学成常数   

### 要改的文件 / 模块

* `fewshot/head.py`

  * 新增一个更简单的 `LinearAnomalyHead`（单层 1×1 conv）
  * 当前两层 head 保留，但 debug 默认先关 dropout
* `fewshot/learned_head.py`

  * 复用统一 similarity/scoring
  * 输出 patch logits，便于做 pixel loss
* 新增 `fewshot/losses.py`

  * `BCEWithLogitsLoss`（pixel）
  * `DiceLoss` 或 `Focal + Dice`
  * `image BCE` 作为辅损失，不再是唯一主损失
* `run_stage_b.py`

  * 增加：

    * `--head-type linear|mlp`
    * `--loss-mode image|pixel_image`
    * `--lambda-image`
    * `--lambda-pixel`
    * `--lambda-dice`
  * 训练日志里增加：

    * train image AUROC（仅作拟合诊断）
    * train patch Dice / IoU（仅 support defect 上）
    * image score 标准差（查常数塌缩）

### 我建议的默认 loss

`L = 1.0 * pixel_bce + 1.0 * dice + 0.25 * image_bce`

理由很直接：

* normal support 的 mask 全 0，可以约束正常图全图低响应
* defect support 的 mask 能直接告诉 head “哪里该亮”
* image BCE 只保留为辅助，不再承担全部监督

### 最小实验矩阵

先只做 `hazelnut`，且只用 matched split。

* Head:

  * `linear`
  * `current_mlp(no_dropout)`
* Loss:

  * `image_bce`（当前控制组）
  * `pixel_bce + dice + 0.25*image_bce`
* 固定：

  * `support_normal_k = 8`
  * `support_defect_k = 4`
  * `image_size = 320`
  * `seed = 42, 43, 44`

### 验收标准

先过“学得动”这一关，再看最终 query：

* train loss 明显下降
* support-train image AUROC > 0.8，或者 support-train patch Dice 明显高于 0
* image score 不再近常数
* 在 held-out `hazelnut` query 上，B 至少比 matched A2 提升 **0.02** 的 image AUROC 或 pixel AUROC

如果做不到，结论要直接：**tiny head 暂时不值得继续投入。**

---

## 5. 只在 `hazelnut` 上做 support sweep，别急着扩全类

### 目标

确认最优配置不是偶然撞出来的。

### 最小实验矩阵

仍然只做 `hazelnut`，只跑已经选出来的最佳 A1/A2/B 配置：

* `support_normal_k = 4, 8, 16, 32`
* `support_defect_k = 1, 4`（A2/B）
* `seed = 42, 43, 44`

### 验收标准

* 方法排序基本稳定，不要一换 support size 就反转
* 3 个 seed 的 image AUROC 标准差尽量 < `0.05`
* 热力图抽查 10 张 defect 图，至少大多数热点与 GT mask 有交集

---

## 6. 满足这些条件后，再扩到全 15 类

### 先只做 `hazelnut` 的内容

* scoring-ablation
* A2 实现与判断
* B 的 loss / supervision 修复
* support-size sweep

### 满足以下条件后，才能扩全 15 类

1. `split_manifest` 机制已经落地，A1/A2/B 的 matched 对照是公平的
2. 默认 scorer 已锁定，不再反复换 map / aggregation
3. `hazelnut` 上至少做完 3 个 seed
4. A2 的作用已经判清
5. 若保留 B，则 B 必须在 matched 协议下**稳定胜过** A2
6. 结果表里明确写清楚：A2/B 的 defect support 来自 held-out defect test，再从最终 query 中剔除

### 扩全 15 类的顺序

* 第一轮：**单 seed、顺序执行**
* 第二轮：只对保留下来的最终配置做多 seed

这也符合协议文档里“单 GPU 默认顺序执行”的思路。

---

## 对你四个问题的直接回答

### 1) 现在应该先实现 A2，还是先修 score direction？

**先修并统一 scoring；A2 紧接着上。**

更准确地说：

* **代码顺序**：先做 shared split + shared scoring
* **实验顺序**：先跑 scoring-ablation，再跑 A2

原因：

* 当前问题是跨 A1/A2/B 的公共问题
* 你现在最可疑的是 image-level aggregation，而不是单纯缺 A2
* A2 如果建立在一个还没定好的 scorer 上，结论会继续发虚

### 2) B 路线如果继续，优先改 loss、改 supervision，还是改 support/query 协议？

**优先改 supervision / loss。**

但要区分两件事：

* **比较公平性**：现在就要用 shared split manifest 修正
* **大改 protocol**：不是 B 修复的第一优先级

直接说，B 现在的头塌缩，首因是监督太弱，不是因为协议名字不够漂亮。

### 3) 是否应该先做统一 scoring-ablation？

**必须先做，而且不只是你列的 sign 版本，还要把 image-level aggregation 一起做。**

至少要统一检查：

* `1 - normal_map`
* `defect_map - normal_map`
* `normal_map - defect_map`
* `0.5 * (1 - normal_map) + 0.5 * defect_map`

以及：

* `patch_max`
* `patch_topk_mean@1%`
* `patch_topk_mean@5%`
* 当前 `upsampled_topk_mean@10%`

我会把这一步排在 A2 之前。

### 4) 当前是否值得把 prompt learning 提前？

**不值得。严格等 A1 / A2 / B 这三条视觉路线跑清楚以后再做。**

协议本身就已经明确：prompt 相关是后续 Stage C；在 visual baseline 和 tiny head 都还没判清之前引入 prompt，只会增加新的自由度，掩盖真正的问题。

---

## 最后补一个更硬的判断

如果你按上面顺序做完之后，`hazelnut` 上最好的 A1/A2 仍然基本贴着 0.5 走，那么**下一步也不该是 prompt**。
下一步应该是改 `feature_bank.py` 这一层，把“单一均值 prototype”升级为：

* multi-prototype
* support patch memory bank
* kNN / nearest-neighbor scoring

因为当前 prototype bank 只是把所有 support patch feature 直接求一个全局均值，这本身就可能过粗。

这套顺序的核心不是“多做几组实验”，而是先把 **scorer、split、公平对照** 三件事固定下来，再决定 learned head 和后续 prompt 是否值得存在。
