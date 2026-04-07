# DenseCLIP 工业质检本地环境配置与训练方案设计（v0，文档版）

> 范围：仅输出本地落地方案、环境约束、训练路径、命令建议与风险说明，不直接改代码。  
> 目标：在当前 `DenseCLIP-master` 目录下，为工业质检场景微调 DenseCLIP 制定一套可执行的本地训练方案。  
> 背景：你已在 Google Colab T4 上跑通一套老版本 DenseCLIP 训练链路，但训练速度一般，希望迁移到本地 GPU 训练。

---

## 1. 项目实体位置

- 仓库根目录：`C:\Nottingham\DenseCLIP-master`
- 检测训练入口：`detection/train.py`
- MVTec 检测配置：`detection/configs/mask_rcnn_denseclip_r50_fpn_1x_mvtec.py`
- 自定义检测模块：`detection/denseclip/`
- 分割训练入口：`segmentation/train_denseclip.py`
- 数据准备脚本：`prepare_mvtec_coco.py`
- Colab 环境与训练记录：`Untitled4.ipynb`
- 预训练 CLIP 权重：`pretrained/RN50.pt`
- 本地训练日志目录：`work_dirs/mask_rcnn_denseclip_r50_fpn_1x_mvtec/`

---

## 2. 本次问题的正确理解

本次不是单纯“把 Colab 命令搬到本地”。

真正需要解决的是三层问题：

### 2.1 环境层

- DenseCLIP 原仓库依赖老版 OpenMMLab 栈
- 你当前本地默认 Python 环境没有装齐 `mmcv/mmdet/mmseg/timm/opencv/ftfy/regex`
- 你当前本地 GPU 是较新的 `RTX 5060 Laptop GPU`

### 2.2 兼容层

Colab 中使用的是：

- `python=3.9`
- `pytorch=1.10.2`
- `torchvision=0.11.3`
- `cudatoolkit=11.3.1`
- `mmcv-full==1.3.17`
- `mmdet==2.17.0`
- `mmsegmentation==0.19.0`

这套组合对 Colab T4 可行，但不适合作为本地新卡的直接迁移方案。

### 2.3 训练层

即便代码入口已经做过一定本地化处理，真正决定能否训练的仍然是：

1. 本地 PyTorch 是否支持当前 GPU 架构
2. OpenMMLab 依赖是否完整
3. 自定义 DenseCLIP 模块是否能在当前 `mmdet/mmcv` 版本上正常注册和构图

---

## 3. 当前仓库现状评估

### 3.1 已具备的条件

- 本地仓库中已经存在 MVTec 检测配置
- `detection/train.py` 已加入本地 `sys.path` 修正
- 已引入 `custom_imports` 加载自定义数据集与模型
- `pretrained/RN50.pt` 已在仓库内
- `data/mvtec_coco/` 已存在 COCO 格式数据
- `work_dirs/` 已有历史本地训练日志

### 3.2 已确认的风险点

当前默认 Python 环境检测结果表明：

- `torch` 已安装，但 `mmcv/mmdet/mmseg/timm/cv2/ftfy/regex` 缺失
- 当前 `torch` 对本机 GPU 给出架构不兼容警告
- 因此当前默认环境不适合直接开训

### 3.3 结论

结论不是“代码完全不能用”，而是：

- 代码基础可复用
- 训练配置基础可复用
- 但必须先重建本地环境，再做最小训练改造

---

## 4. 落地目标

本地落地分成两个目标层级：

### 4.1 第一阶段目标：跑通

- 能在本地单卡启动 `detection/train.py`
- 能成功构建数据集、模型、优化器
- 能完成至少 1 个 epoch 或若干 iteration 的 smoke test

### 4.2 第二阶段目标：可持续训练

- 能稳定训练完整实验
- 能保存 checkpoint 和日志
- 能进行验证集评估
- 能根据显存与速度继续调参

---

## 5. 本地方案总路线

推荐路线不是直接复刻 Colab，而是采用“新卡优先”的兼容方案。

### 5.1 总策略

1. 先建立一套独立本地环境
2. 优先保证 PyTorch 对本地 GPU 架构支持正确
3. 再补齐 DenseCLIP 所需训练库
4. 再对训练代码做最小兼容修改
5. 最后执行 smoke test 和正式训练

### 5.2 不推荐路线

不推荐直接坚持以下组合：

- `torch 1.10 + cu113`
- `mmcv-full 1.3.17`
- 完整复刻 Colab 老环境

原因：

- 本地 GPU 较新，老版 PyTorch 很大概率无法正确支持
- 即使勉强装上，后续 CUDA 算子与驱动兼容仍有较大不确定性

---

## 6. 本地环境配置方案

## 6.1 方案选择

建议准备一套全新环境，例如：

- 环境名：`denseclip-local`
- Python：`3.10`

选择 `Python 3.10` 的原因：

- 兼容性比 3.12 更稳
- 对老中代的 OpenMMLab 生态更友好
- 对新版本 PyTorch 也没有明显阻碍

## 6.2 环境原则

本地环境要遵守以下原则：

1. 不污染你当前默认 Python
2. 不直接复用 Colab 的 Linux 包版本
3. 先装 PyTorch，再装 OpenMMLab 相关依赖
4. 每装完一层就做一次导入验证

## 6.3 建议安装层次

### 第一层：核心深度学习环境

- `python`
- `torch`
- `torchvision`

要求：

- PyTorch 版本必须明确支持本机 GPU 架构
- 这一层如果不成立，后面全部不成立

### 第二层：DenseCLIP 基础依赖

- `timm`
- `opencv-python`
- `ftfy`
- `regex`
- `tqdm`
- `pyyaml`
- `matplotlib`
- `pycocotools`

### 第三层：OpenMMLab 运行栈

- `mmcv`
- `mmdet`
- `mmsegmentation`

这里要注意：

- 不能机械照抄 Colab 的 `mmcv-full==1.3.17`
- 需要和本地 PyTorch 版本配套选择

---

## 7. 训练代码兼容策略

本地训练建议先按“最小改动”原则处理。

### 7.1 优先复用现有检测入口

优先使用：

- `detection/train.py`
- `detection/configs/mask_rcnn_denseclip_r50_fpn_1x_mvtec.py`

原因：

- 当前你的工业质检改造主要落在检测链路
- 该链路已有自定义数据集和配置基础

### 7.2 建议的最小代码改动方向

如果后续实测出现兼容问题，优先检查以下点：

1. `SyncBatchNorm`
- 单卡训练未必需要
- 某些环境下会带来额外兼容噪声

2. `custom_imports`
- 确认 `denseclip.datasets.mvtec_dataset`
- 确认 `denseclip.models`

3. `MVTecCocoDataset`
- 确认训练集不过滤无标注图片的逻辑仍然有效

4. `mmdet` 版本差异
- 如果使用的是比 2.17 更高的版本，需要关注配置字段和接口行为变化

### 7.3 改动原则

- 先不动模型结构
- 先不改 prompt 逻辑
- 先不改 backbone
- 只做启动所需兼容修补

---

## 8. 训练落地实施步骤

## 8.1 第一步：建立独立环境

输出目标：

- 拥有可激活的 `denseclip-local` 环境
- `python -c "import torch"` 成功
- `torch.cuda.is_available()` 为 `True`
- 不再出现“当前 GPU 架构不受支持”的警告

## 8.2 第二步：补齐项目依赖

输出目标：

- 可以成功导入：
  - `torch`
  - `torchvision`
  - `mmcv`
  - `mmdet`
  - `mmseg`
  - `timm`
  - `cv2`
  - `ftfy`
  - `regex`

## 8.3 第三步：做导入级验证

验证内容：

1. 运行最小导入脚本
2. 加载 `pretrained/RN50.pt`
3. 读取 `detection/configs/mask_rcnn_denseclip_r50_fpn_1x_mvtec.py`
4. 构建数据集
5. 构建模型但不训练

如果这一步失败，先修环境或兼容补丁，不直接开始训练。

## 8.4 第四步：做 smoke test

建议：

- 单卡
- 小 batch
- 1 epoch
- 或只跑少量 iteration

目标：

- 验证前向、反向、loss、checkpoint、log 全链路可工作

## 8.5 第五步：正式训练

正式训练时再逐步调整：

- `samples_per_gpu`
- `workers_per_gpu`
- 图像尺寸
- 学习率
- 验证频率

---

## 9. 建议的训练配置策略

### 9.1 初始策略

优先使用保守配置：

- `samples_per_gpu=1`
- `workers_per_gpu=2`
- 单卡训练
- 保持 `img_scale=(1333, 800)`

这样做的原因：

- 先保证稳定，不先追求极限吞吐
- 便于判断问题来自环境还是来自显存/数据

### 9.2 速度优化顺序

如果本地已经跑通，再按以下顺序优化速度：

1. 增大 `samples_per_gpu`
2. 增大 `workers_per_gpu`
3. 视情况降低输入尺寸
4. 视情况减少验证频率
5. 视情况增加日志间隔

### 9.3 不建议一开始就做的事

- 一开始就多卡
- 一开始就混合多个实验改动
- 一开始就同时改环境、模型、数据和 prompt

---

## 10. 风险分析

## 10.1 最大风险

最大风险不是代码本身，而是：

- 本地 GPU 架构支持问题
- PyTorch 与 OpenMMLab 版本配套问题

## 10.2 第二风险

`DenseCLIP` 原始项目建立在较老 `mmcv/mmdet/mmseg` 生态上。

如果本地最终采用较新的 OpenMMLab 版本，可能出现：

- 配置字段变化
- registry 行为变化
- head/backbone 构建接口差异

## 10.3 第三风险

工业质检数据与 COCO/ADE 的原始假设不同，可能出现：

- 正常样本占比高
- 异常样本少
- bbox/mask 极小
- 训练不稳定或收敛慢

这属于训练调参问题，不是第一阶段阻塞问题。

---

## 11. 回退方案

如果本地 GPU 训练短期仍不稳定，建议按以下优先级回退：

### 11.1 回退方案 A

保留本地代码与数据处理，训练继续放在 Colab 或云端。

本地负责：

- 数据准备
- 配置生成
- prompt 生成
- 日志分析

云端负责：

- 正式训练
- checkpoint 导出

### 11.2 回退方案 B

先只在本地跑 CPU/极小样本 smoke test，验证代码逻辑。

### 11.3 回退方案 C

如果检测链路兼容成本过高，可评估是否转为：

- 保留 DenseCLIP backbone 或文本提示思想
- 但换到更新、对新 GPU 更友好的训练框架

这个方案成本较高，当前不建议作为第一选择。

---

## 12. 阶段性交付标准

本方案落地完成的最低标准如下：

### 12.1 环境验收

- 本地独立环境可激活
- `torch` 可识别 GPU
- 不再出现 GPU 架构不支持警告
- `mmcv/mmdet/mmseg/timm/cv2/ftfy/regex` 全部可导入

### 12.2 代码验收

- `detection/train.py` 可启动
- `mask_rcnn_denseclip_r50_fpn_1x_mvtec.py` 可解析
- 自定义数据集与模型能成功注册

### 12.3 训练验收

- 能完成一次 smoke test
- 生成日志文件
- 生成 checkpoint
- 能读取验证结果

---

## 13. 本阶段建议执行顺序

建议严格按以下顺序推进：

1. 确认本地目标环境版本
2. 安装支持当前 GPU 的 PyTorch
3. 补齐 DenseCLIP 依赖
4. 做导入级验证
5. 做模型构建级验证
6. 做单卡 smoke test
7. 再决定是否修改训练代码
8. 再开始正式训练

---

## 14. 结论

本项目迁移到本地训练是可尝试的，但当前关键阻塞不在 API 接口，而在环境兼容。

从仓库状态看：

- 代码基础已经具备一定本地化改造
- 数据和配置已经基本到位
- 真正的第一优先级是重建本地训练环境

因此本阶段最合理的执行策略是：

- 先做“环境可用”
- 再做“训练可跑”
- 最后才做“性能优化”

如果后续继续推进，下一份文档应当进入：

- `v1 本地环境安装手册`
- 或 `v1 单卡 smoke test 执行清单`

