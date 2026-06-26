# Stage4 B7 Raw Graph 消融阶段报告

日期：2026-06-23

## 1. 实验目的

本消融实验用于验证：在最终 B7 gated dual-view 模型中，syntax view 使用实体折叠后的 coarse dependency graph 是否优于不做实体折叠的 raw dependency graph。

对照方式如下：

| 模型 | Syntax graph | 其他模块 |
|---|---|---|
| B7 coarse graph | `coarse_tokens / coarse_heads / coarse_labels` | PubMedBERT semantics view + GCN syntax view + gated fusion + BioBART |
| B7 raw graph | `tokens / dep_heads / dep_labels` | 与 B7 coarse 完全一致 |

因此，该实验主要隔离实体感知图粗化本身的影响。

## 2. 当前结果状态

目前 6 个 B7 raw graph full checkpoints 已完成训练：

- ChemProtSent seed 42 / 123 / 456
- CDRIntra seed 42 / 123 / 456

但 Stage4 test F1 当前只完成了两个单 seed 评估：

- ChemProtSent seed42
- CDRIntra seed42

因此，本报告中的 test split 结论只作为明日汇报的阶段性趋势，不作为最终三种子结论。

## 3. Test F1 单 seed 对比

### 3.1 总体指标

| Dataset | Model | Precision | Recall | Micro-F1 | Macro-F1 | mapped exact | eval loss |
|---|---|---:|---:|---:|---:|---:|---:|
| ChemProtSent | B7 coarse | 0.7811 | 0.6973 | **0.7368** | **0.7206** | **0.9012** | **0.0627** |
| ChemProtSent | B7 raw | 0.7703 | 0.6662 | 0.7145 | 0.6909 | 0.8939 | 0.0703 |
| CDRIntra | B7 coarse | **0.6892** | 0.6097 | 0.6470 | 0.6470 | 0.7540 | **0.2070** |
| CDRIntra | B7 raw | 0.6705 | **0.7067** | **0.6881** | **0.6881** | **0.7631** | 0.2216 |

### 3.2 Coarse - Raw 差值

| Dataset | Precision | Recall | Micro-F1 | mapped exact |
|---|---:|---:|---:|---:|
| ChemProtSent | +0.0108 | +0.0312 | +0.0224 | +0.0073 |
| CDRIntra | +0.0187 | -0.0971 | -0.0411 | -0.0091 |

## 4. 分数据集观察

### 4.1 ChemProtSent

ChemProtSent seed42 上，B7 coarse graph 明显优于 B7 raw graph，Micro-F1 提升约 2.24 个百分点。

该提升主要来自 recall 改善：coarse recall 为 0.6973，raw recall 为 0.6662。说明在 ChemProtSent 中，实体折叠后的依存图可能降低了多词实体内部噪声，使 syntax view 更容易捕捉与关系判断相关的外部结构。

按关系类型看：

| Label | Support | Coarse F1 | Raw F1 | Coarse - Raw |
|---|---:|---:|---:|---:|
| CPR:3 | 664 | 0.6962 | 0.6487 | +0.0475 |
| CPR:4 | 1654 | 0.7922 | 0.7711 | +0.0211 |
| CPR:5 | 181 | 0.7166 | 0.6631 | +0.0534 |
| CPR:6 | 291 | 0.8007 | 0.7280 | +0.0727 |
| CPR:9 | 643 | 0.5975 | 0.6434 | -0.0459 |

因此，ChemProtSent 的单 seed 结果支持实体折叠对大多数 CPR 类型有帮助，但并非所有关系类型都受益。`CPR:9` 上 raw graph 更好，说明部分关系可能仍依赖更细粒度的原始依存信息。

### 4.2 CDRIntra

CDRIntra seed42 上，B7 raw graph 反而优于 B7 coarse graph，Micro-F1 提升约 4.11 个百分点。

该差异主要来自 recall：raw recall 为 0.7067，coarse recall 为 0.6097。coarse graph 的 precision 更高，但召回下降较明显，最终 F1 不如 raw graph。

| Label | Model | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| CID | B7 coarse | 0.6892 | 0.6097 | 0.6470 |
| CID | B7 raw | 0.6705 | 0.7067 | 0.6881 |

这说明在 CDRIntra seed42 上，实体折叠可能提高了预测保守性或精度，但也压缩掉了一部分对 CID 正例召回有用的细粒度依存信息。

## 5. Checkpoint Dev 指标参考

训练阶段的 dev mapped exact 三种子结果如下：

| Dataset | B7 coarse dev mapped exact | B7 raw dev mapped exact | 观察 |
|---|---:|---:|---|
| ChemProtSent | 0.9014 ± 0.0046 | 0.9007 ± 0.0044 | 基本持平 |
| CDRIntra | 0.7712 ± 0.0020 | 0.7656 ± 0.0037 | coarse 略高且更稳 |

需要注意：dev mapped exact 包含大量 `NO_RELATION`，不能完全代表 positive-class F1。当前 CDRIntra 的 test F1 结果也说明，mapped exact 与正类召回之间可能存在差异。因此论文最终仍应以 Stage4 test F1 为主。

## 6. 当前阶段结论

当前结果不支持简单地说“实体折叠在所有数据集上都提升 B7”。更准确的阶段性结论是：

1. 实体折叠确实会影响最终 B7 gated dual-view 模型的性能，而不只是影响线性化 SDP baseline；
2. 在 ChemProtSent seed42 上，coarse graph 明显优于 raw graph，说明实体折叠可以提升多类别 ChemProt 关系识别；
3. 在 CDRIntra seed42 上，raw graph 反而优于 coarse graph，主要因为 raw graph 有更高正类召回；
4. 因此，实体折叠的收益具有数据集依赖性，本质上是在“去除实体内部噪声”和“保留细粒度依存信息”之间做 trade-off；
5. 当前只能作为单 seed 趋势，最终结论需要补齐 raw graph 的三种子 test F1。

## 7. 给老师汇报时的建议表述

可以这样汇报：

> 我们补充了 B7 raw graph 消融，目的是验证实体折叠在最终门控双视角模型中的贡献。当前已经完成两个数据集 seed42 的 test F1。ChemProtSent 上，B7 coarse graph 比 raw graph 高约 2.24 个 F1 点，说明实体折叠能改善多类别 ChemProt 关系识别；但 CDRIntra 上，raw graph 比 coarse graph 高约 4.11 个 F1 点，主要来自正类 recall 提升。因此目前不能简单说实体折叠在所有场景都提升，而应更谨慎地说：实体折叠会改变句法图的信息密度和 precision-recall trade-off，其收益具有数据集依赖性。完整结论会在补齐三种子 raw graph test F1 后确定。

## 8. 后续计划

1. 补齐 B7 raw graph 的剩余 4 个 test F1：
   - ChemProtSent seed123 / seed456
   - CDRIntra seed123 / seed456
2. 汇总 raw vs coarse 的三种子 Micro-F1 均值和标准差；
3. 对 CDRIntra 做错误分析，重点检查 raw graph 多召回的 CID 正例属于哪些句法结构；
4. 对 ChemProtSent 做 per-class 分析，重点解释为什么 `CPR:9` 上 raw graph 更好。
