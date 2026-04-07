# 给网页版 Pro 模型的下一步规划输入稿

请基于下面这个项目状态，给出一个严格、可执行、按优先级排序的下一步研究/实现计划。从技术角度扎实推进。

## 项目背景

- 项目路径：`C:\Nottingham\DenseCLIP-master`
- 数据：`C:\Nottingham\DenseCLIP-master\data\mvtec_anomaly_detection`
- 协议文档：`C:\Nottingham\DenseCLIP-master\docs\design\design-v1-visual-prototype-experiment-protocol.md`
- 当前目标：基于 DenseCLIP/CLIP visual backbone 做 few-shot anomaly detection，不走旧的 Mask R-CNN 检测链

## 已实现内容

### Stage A1

- 入口：`C:\Nottingham\DenseCLIP-master\run_stage_a1.py`
- 逻辑：`normal-only visual prototype`
- support 仅来自 `<category>/train/good`
- backbone 冻结
- 不训练 tiny head
- 直接在 `<category>/test/good + test/<defect>` 上评估

### Stage B

- 入口：`C:\Nottingham\DenseCLIP-master\run_stage_b.py`
- 逻辑：`normal + defect prototype + learned head`
- support_normal 来自 `train/good`
- support_defect 目前从少量 defect test 图中抽样，然后从最终评估集中剔除
- 只训练 tiny head

## 已有结果（hazelnut）

### Stage A1

- `support_normal_k = 4`
- `image_size = 224`
- `image_auroc = 0.46464285714285714`
- `pixel_auroc = 0.5363952752992882`

### Stage B

- `support_normal_k = 8`
- `support_defect_k = 4`
- `image_size = 320`
- `epochs = 3`
- `image_auroc = 0.4700757575757576`
- `pixel_auroc = 0.5346651964045523`

## 已确认的问题

1. `image_auroc` 接近随机，不足以说明方法有效。
2. 把分数反号后，A1 和 B 都会略高于 `0.53`，说明当前 score direction 可能有问题。
3. B 的 tiny head 输出几乎是常数，train loss 也没有收敛，说明 learned-head 路线目前塌缩明显。
4. 协议里的 `A2 = normal + defect prototype without training` 还没有实现。

## 你需要输出的内容

请给出一个“下一步计划”，要求包括：

- 实现顺序
- 每一步的目标
- 每一步要修改哪些文件或模块
- 最小实验矩阵
- 验收标准
- 哪些实验先只做 `hazelnut`
- 哪些条件满足后再扩到全 15 类

请特别回答这些问题：

1. 现在应该先实现 `A2`，还是先修 score direction？
2. `B` 路线如果继续，应该优先改 loss、改 supervision，还是改 support/query 协议？
3. 是否应该先做一个统一的 scoring-ablation 实验：
   - `1 - normal_map`
   - `defect_map - normal_map`
   - `normal_map - defect_map`
   - 其他更合理的 image-level 聚合方式
4. 当前是否值得把 prompt learning 提前，还是应当严格等 A1/A2/B 这三条视觉路线跑清楚以后再做？

输出风格要求：

- 不要泛泛而谈
- 直接给决策
- 给出优先级
- 给出原因
- 给出执行顺序
