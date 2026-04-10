# [pure-visual] Research Note: Stage 5 AnomalyDINO 对齐后与 SOTA 仍有多大差距，以及差距主要来自哪一步

## Research Status

- Route: `pure-visual`
- Question: `写清当前 pure-visual 与 AnomalyDINO 论文级目标之间的差距，并检查改进路线中哪些步骤导致了这部分差距；同时说明 Stage 5 现在推进到哪里，以及为什么后台没有 python 在运行`
- Checked on: `2026-04-10`
- Latest route board: `outputs/new-branch/_ops/route_boards/pure_visual_route_board.md`
- New external search: `yes`
- Ready for training mode: `yes, but only for the bounded Stage 5 full15 launch`

## Repo Evidence

- 当前 pure-visual retained winner 的 full15 三种子聚合结果是 `image_auroc = 0.974412`、`pixel_auroc = 0.970188`、`pro = 0.863387`，说明主差距已经不是 image ranking，而是 localization / PRO。
- 对应本地聚合摘要在 `outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/summary.md` 和 `outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/aggregate.csv`。
- Stage 4 曾把下一条路收敛为 `support-aware preprocess-agnostic retrieval v5`，核心判断是“`screw / PRO` 的剩余短板来自单一预处理视角下的 patch 匹配不稳定”，见 `docs/总结/stage4/纯视觉执行说明-support-aware-preprocess-agnostic-v5.md`。
- 随后的研究说明把路线切回 `AnomalyDINO` 对齐，明确指出真正先要补的是 preprocessing parity，而不是再堆 retrieval trick，见 `docs/总结/stage4/纯视觉研究说明-AnomalyDINO对齐与验证阶梯.md` 和 `docs/总结/stage4/纯视觉研究说明-AnomalyDINO路线盲点与验证阶梯.md`。
- Stage 5 预检目录已经落盘，且只跑了 `weak5_bottle / seed42` 的 bounded preflight，不是 full15 正式 run：`outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v1_preflight`。
- 这次预检结果是 `image_auroc = 0.997619`、`pixel_auroc = 0.913123`、`pro = 0.897958`、`bottle = 0.997619`，无 alerts。对应文件：`summary.md`、`experiments.csv`、`run_manifest.json`。
- 预检 run manifest 只包含一个候选：`dinov2_vits14_last_patchcore_knn_k001_core1000_topk010`。这意味着这轮预检的改进来自 preprocessing parity，对象不是“新 retrieval trick”，而是“官方 policy 对齐后的同一 base route”。
- 本机当前没有正在运行的 `python.exe` / `python3.exe`，说明 Stage 5 现在停在“预检完成、等待 full15 发车”，不是“后台挂着长任务但你没看到”。

## Search Log

- Query: `AnomalyDINO WACV 2025 MVTec PRO 448 672 Table 2`
- Source opened: `AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2` https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf
- Why kept: `这是当前 pure-visual 路线直接对齐的主论文，且 Table 2 明确给出 MVTec few-shot 的 image/pixel/PRO 指标`

## Literature Evidence

- `AnomalyDINO` 论文把 few-shot MVTec 的主结果直接报告到 `1 / 2 / 4 / 8 / 16-shot`，并强调 preprocessing pipeline 是方法的一部分，不是可有可无的附属实现细节。
- 论文 Table 2 里，`AnomalyDINO-S (448)` 在 MVTec 上的 `PRO` 分别约为：
  - `1-shot: 91.7`
  - `2-shot: 92.0`
  - `4-shot: 92.4`
  - `8-shot: 92.7`
  - `16-shot: 92.9`
- 同一表中，`AnomalyDINO-S (672)` 的 `PRO` 更高，约为：
  - `1-shot: 92.7`
  - `2-shot: 93.1`
  - `4-shot: 93.4`
  - `8-shot: 93.8`
  - `16-shot: 94.0`
- 论文正文还明确指出：`448 vs 672` 的 detection performance 接近，但更高分辨率会改善 localization。

## Inference

- 当前 repo 与论文级目标的主差距，最稳妥的记法不是“纯视觉不行”，而是：`当前 pure-visual 的 image ranking 已经接近收口，但 localization / PRO 仍落后于 AnomalyDINO 论文级目标带`
- 如果用当前 repo 内部一直采用的 paper-facing target band `0.92+` 来记，full15 retained winner 的 `PRO = 0.863387`，与目标仍差约 `0.0566`。这就是 Stage 5 要优先解决的主差距。
- 这部分差距里，至少有一部分不是来自 backbone，也不是来自 retrieval family，而是来自此前没有把官方 preprocessing policy 对齐完整。证据是：Stage 5 预检没有引入新的 method family，只做 policy 对齐，`bottle` 预检 `PRO` 就已经到 `0.897958`。
- 改进路线里真正导致与 SOTA 仍有差距的关键步骤，按责任大小排序，更像是下面四件事：
  1. `Stage 4 retained winner` 直接以本地 `patchcore_knn` 结果作为主基线，但当时还没有完成官方 preprocessing parity audit。于是 `PRO = 0.8634` 这个差距里，混入了大量实现口径差异。
  2. `Stage 4 v5` 先把问题叙述成 `support-aware preprocess-agnostic retrieval`，本质上是过早把“口径差距”解释成“需要新的 retrieval innovation”。这一步没有错到不可用，但它混淆了 root cause 和 innovation point。
  3. `Stage 5` 已经修复了三项高影响差异：`masking` 放置位置、`rotation` 策略、`PRO` 实现口径。所以从现在起，若仍有剩余差距，就不能再笼统地归因给这三项。
  4. 当前仍未完全对齐的主差异是 `smaller-edge + aspect-ratio-preserving resize` 及其连带的 `score_map_outputs / mask resize / output size` 口径。这是目前最像“剩余 PRO 差距来源”的未闭环步骤。
- 还需要单独记一条：当前 Stage 5 预检只覆盖了 `weak5_bottle / seed42`，因此它证明的是“policy alignment 有效且没打坏强控制类”，还不能证明“full15 多 seed aggregate 也一定能到 `0.90` 或 `0.92+`”。这不是方法差距，而是验证尚未完成。
- 此外，当前实现里的 `masking` 仍是 repo 内的通用 PCA foreground mask 近似，并不是论文/官方脚本那种完全 object-dependent 的逐类 preprocess policy。这个近似足够支撑 Stage 5，但它仍可能留下 residual gap。
- 所以现在最准确的阶段判断是：`SOTA gap has been narrowed from a vague route-level criticism to one concrete unresolved parity step plus one missing full15 validation step.`

## Change Origin

- 本轮改动`不是`“完全照抄 AnomalyDINO 官方实现”。
- 本轮改动也`不是`“单纯拍脑袋乱试”。
- 更准确的定性是：`a bounded repo-specific approximation of the AnomalyDINO preprocessing idea, using a mix of paper-driven alignment targets and common object-localization heuristics under current repo constraints.`

可以分成三层：

1. 直接来自 `AnomalyDINO` 对齐目标的部分：
   - 把 preprocessing 当作主方法部件来对齐，而不是继续造新的 retrieval trick
   - 引入官方 8 角度 rotation family
   - 把 `PRO` 改成 official-style connected-component AU-PRO @ FPR<=0.3
   - 把 masking 从后处理思路，前移到 feature cache / reference construction 这类更靠前的位置
2. 属于 repo 内对“官方思路”的近似实现、也带有业界常见工程启发式色彩的部分：
   - 用 patch-feature PCA foreground mask 做通用前景估计
   - 对 anomaly map 做硬 mask
   - 用 object-normalized support bank 作为 reference bank 的主输入
   - 对所有类别较统一地使用这套 object-aware masking 逻辑
3. 仍未完成的官方对齐部分：
   - smaller-edge + aspect-ratio-preserving resize
   - 与之联动的 output-size / mask-resize / score-map 口径
   - 更 object-dependent 的逐类 preprocessing policy

因此，Stage 5 的问题不应记成：

- `the AnomalyDINO idea itself failed`

而应记成：

- `our repo-specific approximation to the AnomalyDINO preprocessing idea likely over-applied object-aware masking, which helped object-like categories but hurt texture-heavy or full-surface localization.`

## Decision

- Route state: `ready_to_resume`
- Why: `Stage 5 已完成 parity audit 和 weak5 预检，且预检无 alerts；当前不需要继续研究或继续改方法定义，下一步就是按合同发 full15 多 seed 正式 run`

## Bounded Next Branch

- Method family: `DINOv2 PatchCore`
- Target weakness: `aggregate PRO gap to the 0.92+ paper-facing target band, while preserving image ranking and bottle`
- Allowed code scope: `tools/run_stage4_pure_visual_dinov2.py`, `fewshot/dinov2_backbone.py`, `fewshot/data.py`, and if strictly needed for metric parity `tools/eval_structure_ablation.py`
- Output dir: `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v1`
- Gate:
  - `pro_mean >= 0.9000`
  - stretch target `near 0.9200+`
  - `image_auroc_mean >= 0.9644`
  - `bottle_image_auroc >= 0.9950`
  - `screw image_auroc_mean >= 0.7975`
- Next step: `launch the bounded Stage 5 full15 multi-seed run; do not reopen a new retrieval branch before this check is finished`

## Sources

- `outputs/new-branch/_ops/route_boards/pure_visual_route_board.md`
- `docs/总结/stage4/纯视觉执行说明-support-aware-preprocess-agnostic-v5.md`
- `docs/总结/stage4/纯视觉研究说明-AnomalyDINO对齐与验证阶梯.md`
- `docs/总结/stage4/纯视觉研究说明-AnomalyDINO路线盲点与验证阶梯.md`
- `docs/总结/stage5/阶段五执行说明-纯视觉AnomalyDINO官方预处理对齐.md`
- `outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/summary.md`
- `outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/aggregate.csv`
- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v1_preflight/summary.md`
- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v1_preflight/experiments.csv`
- `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v1_preflight/run_manifest.json`
- `AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2` https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf

## 2026-04-10 Stage 5 Closeout Note

Stage 5 到这里正式 closeout，但 closeout 的对象不是 `pure-visual` 整条科研路线，而是：

- `this repo-specific local approximation to the AnomalyDINO preprocessing idea`

新增官方结果证据如下：

- Stage 4 retained winner 仍是当前本 repo 内最强 pure-visual 结果：
  - `outputs/new-branch/pure-visual/full15/multiseed_summary/dinov2_patchcore_k1_last_v1/summary.md`
  - `image_auroc = 0.974412`
  - `pixel_auroc = 0.970188`
  - `pro = 0.863387`
- Stage 5 v1 full15 official:
  - `outputs/new-branch/pure-visual/full15/stage5_anomalydino_policy_align_v1/summary.md`
  - `pro = 0.824271`
- Stage 5 v2 weak5 aspect-ratio preflight:
  - `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v2_preflight/summary.md`
  - `pro = 0.785745`
- Stage 5 v3 weak5 672 preflight:
  - `outputs/new-branch/pure-visual/weak5_bottle/seed42/stage5_anomalydino_policy_align_v3_672_preflight/summary.md`
  - `pro = 0.595064`

因此被最终确认的判断是：

1. 当前 repo 内这条“局部对齐官方细节”的近似实现，随着对齐项增加并没有逼近论文带，反而持续偏离 Stage 4 retained winner。
2. 现在已经很难再把后续失败清楚地区分为：
   - `AnomalyDINO idea itself failed`
   - 还是 `our local port / approximation failed`
3. 继续在当前 repo 内沿这条近似实现做小修小补，信息增益已经很低，而且会进一步污染方法判断。

本轮 closeout 前考虑过、但明确拒绝的 next-best 方向是：

- 继续在本 repo 内追加新的局部 preprocessing / masking / resize / rotation 修正

拒绝原因：

- 它们都仍属于同一条 `local approximation`，不能回答“官方方法本身是否成立”
- 继续这样改，研究问题会被实现细节噪声淹没
- 当前最有信息增益的下一步，已经不是“再补本地细节”，而是“直接跑官方仓库”

所以 Stage 5 的正式结论是：

- `close out the local approximation line`
- `do not continue local patching in this repo`
- `reopen pure-visual as Stage 6: official-repo reproduction only`
