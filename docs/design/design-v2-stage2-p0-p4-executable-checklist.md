# DenseCLIP Stage2 P0-P4 可执行实验清单

> 基线来源：
> - `docs/design/design-v1-visual-prototype-experiment-protocol.md`
> - `docs/总结/阶段二文档索引.md`
> - `docs/总结/一阶段结果分析以及二阶段实验设计.md`
> - `docs/总结/阶段二实验思考过程以及相应调整.md`
> - `docs/总结/阶段二结果.md`
>
> 文档目标：
> - 把 P0-P4 改写成可直接执行的实验清单。
> - 每个实验包统一写清楚：`脚本 / 参数 / 输出目录 / 判定门槛`。
> - 默认面向单卡顺序执行，不设计并发调度。

---

## 1. 当前项目状态与约束

### 1.1 已有可运行入口

当前仓库已经具备以下入口脚本：

- `prepare_split_manifest.py`
- `run_stage_a1.py`
- `run_stage_a2.py`
- `run_stage_b.py`

当前 few-shot 路线的事实约束：

- 数据根目录默认使用 `data/mvtec_anomaly_detection`
- `support_normal` 从 `<category>/train/good` 采样
- `support_defect` 目前从 `<category>/test/<defect_type>` 采样，并从最终 `query_eval` 中剔除
- `run_stage_a1.py` 只使用 normal support
- `run_stage_a2.py` 使用 normal + defect prototype，但不训练 head
- `run_stage_b.py` 使用 normal + defect prototype，并训练 learned head
- 当前所有主脚本都会把结果写到 `--output-dir/<category>`

### 1.2 当前已知 `hazelnut` 共享 split 基线

以下结果已经存在，可作为 stage2 各阶段的对比起点。

| 实验 | 输出目录 | image AUROC | pixel AUROC | 备注 |
| --- | --- | ---: | ---: | --- |
| A1 shared scoring | `outputs/stage_a1_shared_scoring_smoke/hazelnut` | `0.4432` | `0.5388` | `one-minus-normal` |
| A2 defect-minus-normal | `outputs/stage_a2_smoke/hazelnut` | `0.4617` | `0.5728` | pixel 最强 |
| A2 normal-minus-defect | `outputs/stage_a2_smoke_normal_minus_defect/hazelnut` | `0.5462` | `0.4272` | image 最强，但 pixel 明显塌陷 |
| B shared scoring | `outputs/stage_b_shared_scoring_smoke/hazelnut` | `0.4992` | `0.5026` | learned head 未体现增益 |

由此锁定 stage2 的主问题：

- 先解决 scorer 和 prototype，不先切到 prompt
- 先把 A1/A2 跑稳，再重新判断 B 是否值得保留

---

## 2. 全局执行规范

### 2.1 统一数据与种子

除非某个实验卡片单独说明，默认参数如下：

- `category = hazelnut`
- `data_root = data/mvtec_anomaly_detection`
- `image_size = 320`
- `support_normal_k = 8`
- `support_defect_k = 4`
- `batch_size = 8`
- `workers = 0`
- `seed = 42, 43, 44`

### 2.2 统一 split manifest

所有 P0-P3 实验必须先冻结 shared manifest，再复用同一 manifest 做横向比较。

推荐命名：

- `outputs/split_manifests/stage2/{category}_sn{support_normal_k}_sd{support_defect_k}_seed{seed}.json`

生成命令模板：

```bash
python prepare_split_manifest.py \
  --category hazelnut \
  --support-normal-k 8 \
  --support-defect-k 4 \
  --seed 42 \
  --output outputs/split_manifests/stage2/hazelnut_sn8_sd4_seed42.json
```

说明：

- A1 也使用同一 manifest；A1 会自动忽略 manifest 内的 `support_defect`
- A2/B 必须复用同一 manifest，避免 support/query 池变化污染比较

### 2.3 统一输出目录

推荐 stage2 输出根目录：

```text
outputs/stage2/
  p0_scoring/
  p1_retrieval/
  p2_structure/
  p3_head/
  p4_full15/
```

命名原则：

- `--output-dir` 传实验根目录
- 最终结果落在 `--output-dir/<category>`
- 每次 sweep 必须把关键变量写入目录名

示例：

- `outputs/stage2/p0_scoring/a2_defect_minus_normal_topk010_seed42/hazelnut`
- `outputs/stage2/p1_retrieval/knn_bank_k8_seed43/hazelnut`

### 2.4 统一必备产物

A1/A2 至少保留：

- `metrics.json`
- `config.json`
- `predictions.csv`
- `support_paths.json`
- `split_manifest.json`

B 额外保留：

- `train_history.json`
- `checkpoint_summary.json`

新增脚本也应尽量沿用相同产物命名，避免后续汇总脚本分支过多。

---

## 3. P0：先锁定共享 scorer

### 3.1 目标

在固定 shared manifest 上，先确定一个后续 A1/A2/B 共用的默认 scorer。P0 不引入新表征，不引入新 head，只处理：

- score sign
- aggregation 方式
- aggregation 位置
- top-k 比例

### 3.2 实验清单

| 实验包 | 脚本 | 参数 | 输出目录 | 判定门槛 |
| --- | --- | --- | --- | --- |
| P0-1 基线重跑 | `prepare_split_manifest.py` + `run_stage_a1.py` + `run_stage_a2.py` | `seed=42,43,44`；A1 用默认 `one-minus-normal`；A2 依次跑 `defect-minus-normal`、`normal-minus-defect`、`blend`；`topk_ratio=0.1` | `outputs/stage2/p0_scoring/{exp_name}_seed{seed}/{category}` | 三个 seed 的结果可复现；目录和产物完整；作为后续门槛基线 |
| P0-2 scorer sweep | 新增 `tools/eval_scoring_ablation.py` | `score_mode in {one-minus-normal, defect-minus-normal, normal-minus-defect, blend}`；`aggregation in {topk_mean, max}`；`topk_ratio in {0.01,0.05,0.10}`；`aggregation_stage in {patch, upsampled}` | `outputs/stage2/p0_scoring/sweeps/{score_mode}_{aggregation}_{stage}_topk{ratio}_seed{seed}/{category}` | `mean(score_defect) > mean(score_good)` 在 3 个 seed 上方向一致；相对 A1 基线 `image_auroc` 平均提升至少 `+0.03`；`pixel_auroc` 相对 A2 pixel 最强版本不退化超过 `0.02` |
| P0-3 同编码器多层 scorer | 扩展 `fewshot/scoring.py`，并继续使用 `tools/eval_scoring_ablation.py` | 在 `layer3/layer4/local` 上做 `single` 与 `fuse(layer4+local)` 两种 scorer；其余沿用 P0-2 最优聚合策略 | `outputs/stage2/p0_scoring/multiscale_{variant}_seed{seed}/{category}` | 若多层融合进入候选，则必须同时优于 P0-2 最优单层 scorer 的 `image_auroc` 与 `pixel_auroc`，允许单项最多 `0.01` 波动 |

### 3.3 推荐命令模板

```bash
python run_stage_a1.py \
  --category hazelnut \
  --image-size 320 \
  --split-manifest outputs/split_manifests/stage2/hazelnut_sn8_sd4_seed42.json \
  --output-dir outputs/stage2/p0_scoring/a1_baseline_seed42
```

```bash
python run_stage_a2.py \
  --category hazelnut \
  --image-size 320 \
  --score-mode defect-minus-normal \
  --topk-ratio 0.1 \
  --split-manifest outputs/split_manifests/stage2/hazelnut_sn8_sd4_seed42.json \
  --output-dir outputs/stage2/p0_scoring/a2_defect_minus_normal_seed42
```

```bash
python tools/eval_scoring_ablation.py \
  --category hazelnut \
  --image-size 320 \
  --split-manifest outputs/split_manifests/stage2/hazelnut_sn8_sd4_seed42.json \
  --score-modes one-minus-normal defect-minus-normal normal-minus-defect blend \
  --aggregation-modes topk_mean max \
  --aggregation-stages patch upsampled \
  --topk-ratios 0.01 0.05 0.10 \
  --output-dir outputs/stage2/p0_scoring/sweeps
```

### 3.4 P0 完成条件

- 选出唯一默认 scorer
- 选出的 scorer 进入 `run_stage_a1.py`、`run_stage_a2.py`、`run_stage_b.py` 的默认配置
- 所有后续阶段不得再更换 scorer，除非重新开新阶段

---

## 4. P1：把单均值 prototype 升级为 retrieval / multi-prototype

### 4.1 目标

验证当前瓶颈是否主要来自 `fewshot/feature_bank.py` 的单均值 prototype 假设。

P1 仍然不引入 prompt，不引入新 backbone，优先做低成本结构升级。

### 4.2 实验清单

| 实验包 | 脚本 | 参数 | 输出目录 | 判定门槛 |
| --- | --- | --- | --- | --- |
| P1-1 mean prototype 对照组 | `run_stage_a1.py` + `run_stage_a2.py` | 固定 P0 默认 scorer；`support_normal_k in {4,8,16}`；`support_defect_k in {1,4}`；`seed=42,43,44` | `outputs/stage2/p1_retrieval/mean_proto_sn{n}_sd{d}_seed{seed}/{category}` | 作为 P1 内部对照，不单独晋级 |
| P1-2 PatchCore-like memory bank | 新增 `tools/run_retrieval_ablation.py`，内部调用扩展后的 `fewshot/feature_bank.py` | `bank_type=knn`；`knn_k in {1,3,5}`；`support_normal_k in {4,8,16}`；A1/A2 都跑 | `outputs/stage2/p1_retrieval/knn_bank_k{k}_sn{n}_sd{d}_seed{seed}/{category}` | 至少一条配置在 3 个 seed 中有 2 个 seed 同时优于 P0 默认 scorer；`image_auroc` 平均提升至少 `+0.03`；`pixel_auroc` 不低于 P0 最优 `-0.01` |
| P1-3 multi-prototype | 同上，新增 `fewshot/retrieval.py` 或 `fewshot/prototypes.py` | `bank_type=kmeans`；`prototype_k in {4,8}`；A1/A2 都跑 | `outputs/stage2/p1_retrieval/kmeans_proto_k{k}_sn{n}_sd{d}_seed{seed}/{category}` | 与 P1-2 使用同一门槛；若 image 与 pixel 再次严重对冲，则判定失败 |
| P1-4 FastRef-lite | 同上 | `bank_type=fastref_lite`；`refine_steps in {1,3}`；其余参数沿用 P1 最优 support 组合 | `outputs/stage2/p1_retrieval/fastref_step{s}_seed{seed}/{category}` | 只有当 refinement 后的结果稳定优于 P1-2/P1-3 最优方案，才保留到 P2/P3 |

### 4.3 实施依赖

P1 需要补的代码位点：

- `fewshot/feature_bank.py`
- `fewshot/scoring.py`
- 新增 `fewshot/retrieval.py` 或 `fewshot/prototypes.py`
- 新增 `tools/run_retrieval_ablation.py`

### 4.4 P1 完成条件

- 明确 single-mean prototype 是否被稳定击败
- 若被击败，锁定唯一 bank 方案进入 P2/P3
- 若未被击败，P2 直接转向 subspace / matching 路线

---

## 5. P2：补结构约束，而不是上 prompt

### 5.1 目标

解决 “patch 有信号，但 image 不稳定” 的第二层问题。P2 只允许两条路线：

- `SubspaceAD-style normal subspace residual`
- `lightweight correspondence / registration`

### 5.2 实验清单

| 实验包 | 脚本 | 参数 | 输出目录 | 判定门槛 |
| --- | --- | --- | --- | --- |
| P2-1 subspace 残差对照 | 新增 `tools/run_structure_ablation.py`，调用新增 `fewshot/subspace.py` | `method=subspace`；`subspace_dim in {16,32,64}`；`support_normal_k in {4,8,16}`；A1 路线优先 | `outputs/stage2/p2_structure/subspace_d{dim}_sn{n}_seed{seed}/{category}` | `image_auroc` 和 `pixel_auroc` 不得再次出现一升一降的大幅对冲；相对 P1 最优方案允许单项最大 `0.01` 波动，但另一项必须提升 |
| P2-2 lightweight matching | 同上，调用新增 `fewshot/matching.py` | `method=matching`；`match_k in {1,3,5}`；`spatial_window in {0,3,5}`；优先在 P1 最优 bank 上叠加 | `outputs/stage2/p2_structure/matching_k{k}_w{w}_seed{seed}/{category}` | 必须优于 P1 最优方案，且 3 个 seed 中至少 2 个 seed 的 `image_auroc` 和 `pixel_auroc` 同向改善 |
| P2-3 PRO 补齐 | 扩展评估脚本，统一输出 `pro` 指标 | 在 P2 最优结构方案上回算 `PRO`；不扫额外结构 | 复用对应实验目录，在 `metrics.json` 中新增 `pro` 字段 | `metrics.json` 必须包含 `pro`；P2 结束时所有 finalist 方案都要有 `image_auroc`、`pixel_auroc`、`pro` |

### 5.3 实施依赖

P2 需要补的代码位点：

- 新增 `fewshot/subspace.py`
- 新增 `fewshot/matching.py`
- 扩展 `fewshot/stage_a1.py` 的评估部分，补 `PRO`
- 新增 `tools/run_structure_ablation.py`

### 5.4 P2 完成条件

- 在不引入 prompt 的前提下，拿到一条结构稳定的视觉路线
- 指标报告从双指标补齐到三指标：`image_auroc` / `pixel_auroc` / `pro`
- 选出唯一 P2 finalist，供 P3 使用

---

## 6. P3：只有在视觉基线稳定后，才重评 learned head

### 6.1 目标

重新回答一个问题：在 P0-P2 已经稳定后，tiny head 是否还有净增益。

P3 禁止直接 “多加 epoch 试试”，必须先补 supervision 和 collapse 监控。

### 6.2 实验清单

| 实验包 | 脚本 | 参数 | 输出目录 | 判定门槛 |
| --- | --- | --- | --- | --- |
| P3-1 线性 head 控制组 | 扩展 `run_stage_b.py`；新增 `fewshot/losses.py` | `head_type=linear`；`dropout=0.0`；`epochs=20`；使用 P2 finalist similarity 输入 | `outputs/stage2/p3_head/linear_head_seed{seed}/{category}` | `train_history.json` 中必须记录 `train_image_auroc`、`train_pixel_dice`、`train_image_score_std`；`train_image_score_std` 不能近似 0 |
| P3-2 当前 MLP head 复评 | 扩展 `run_stage_b.py` | `head_type=current_mlp`；`dropout in {0.0,0.1}`；其余与 P3-1 相同 | `outputs/stage2/p3_head/mlp_head_drop{dropout}_seed{seed}/{category}` | 只要继续出现近常数输出，就直接判失败，不再继续加 epoch |
| P3-3 loss ablation | 同上 | `loss=image_bce` 对照 `loss=pixel_bce+dice+0.25image_bce`；固定最优 head 结构 | `outputs/stage2/p3_head/loss_{loss_name}_seed{seed}/{category}` | 只有当 query 集结果在 3 个 seed 中至少 2 个 seed 稳定超过 P2 finalist，B 才保留；否则降级为次线 |

### 6.3 实施依赖

P3 需要补的代码位点：

- `fewshot/head.py`
- `fewshot/learned_head.py`
- 新增 `fewshot/losses.py`
- 扩展 `run_stage_b.py`

建议新增字段到 `train_history.json`：

- `train_loss`
- `train_image_auroc`
- `train_pixel_dice`
- `train_image_score_std`

### 6.4 P3 完成条件

- 明确 learned head 是保留还是降级
- 若保留，锁定唯一 B-finalist
- 若降级，stage2 主线只保留 A 系列视觉路线

---

## 7. P4：满足门槛后再扩全 15 类

### 7.1 进入 P4 的前置条件

必须全部满足后，才允许扩全 15 类：

- P0 默认 scorer 已锁定
- P1/P2 已经拿到唯一视觉 finalist
- P3 已经明确 B 是否保留
- `metrics.json` 已统一包含 `image_auroc`、`pixel_auroc`、`pro`
- 输出目录和汇总格式已经稳定

### 7.2 实验清单

| 实验包 | 脚本 | 参数 | 输出目录 | 判定门槛 |
| --- | --- | --- | --- | --- |
| P4-1 15 类单 seed 全量跑 | 新增 `tools/run_full15_stage2.py` | `seed=42`；类别覆盖 MVTec 15 类；只跑视觉 finalist，若 B 保留则额外跑 B-finalist | `outputs/stage2/p4_full15/seed42/{exp_name}/{category}` | 全 15 类成功落盘；无类别缺失；每类都有完整 `metrics.json` |
| P4-2 finalist 三种子复评 | 同上 | 只对 P4-1 最强配置补 `seed=42,43,44` | `outputs/stage2/p4_full15/finalist_seed{seed}/{exp_name}/{category}` | 3 seed 均值和标准差可汇总；若跨 seed 波动过大，回滚到 P0-P3 重新定位 |
| P4-3 总表汇总 | 新增 `tools/aggregate_stage2_results.py` | 聚合所有类别与 seed 的 `metrics.json` | `outputs/stage2/p4_full15/summary/{exp_name}.csv` 和 `{exp_name}.md` | 产出均值、标准差、类别排名、失败类别名单；作为是否进入 Stage C 的唯一摘要 |

### 7.3 扩展顺序

P4 只允许如下顺序：

1. 单 seed 跑全 15 类
2. 只对 finalist 配置补 3 seeds
3. 最后做汇总表

不允许在单卡上同时并发多个类别进程。

---

## 8. 建议落地顺序

按工程优先级，建议实际执行顺序如下：

1. 先完成 P0，确定唯一 scorer
2. 再完成 P1，验证 single-mean prototype 是否是主瓶颈
3. 如果 P1 不够，再进入 P2
4. 只有视觉路线稳定后才进入 P3
5. 只有 P0-P3 全部判清后才进入 P4

一句话总结：

**stage2 不先追求“更会学”，而是先追求“同一份 anomaly signal 能被稳定表示、检索、聚合和评估”。**
