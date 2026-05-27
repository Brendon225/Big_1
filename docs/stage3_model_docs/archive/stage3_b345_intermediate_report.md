# Stage 3 中间结果报告：B3/B4/B5 图增强基线

日期：2026-05-05

## 1. 文档定位

本文档记录阶段三前半段工作的中间结果。当前阶段的目标不是最终模型收敛，而是先在完整数据上验证三类图增强基线是否有效：

- B3：syntax-only，使用依存图结构编码后的句法表示作为 BioBART 的额外注入向量。
- B4：semantics-only，使用 PubMedBERT 的句子语义表示作为 BioBART 的额外注入向量。
- B5：dual-view concat，直接拼接 syntax view 与 semantics view，再投影为 BioBART 的额外注入向量。

这三组实验的作用是为后续 GM-IB/Ours 提供诊断性证据：如果简单语义、简单句法、简单拼接已经表现出不同数据集偏好，那么最终模型就不应只做 naive concat，而应显式学习“哪些图边、哪些结构信息值得保留”。

## 2. 已完成工作

### 2.1 图数据桥接

我们在不改动阶段二原始数据逻辑的前提下，新增了 Stage 3 图增强数据链路：

- `GraphBuilder`：从阶段一清洗后的样本中构建 coarse dependency graph。
- `BioREGraphDataset`：在 BioREDataset 基础上增加图字段，包括 `adj_matrix`、`dep_type_ids`、`node_mask`、`node_char_spans`、实体节点索引等。
- `BioREGraphCollator`：同时构造 BioBART 输入、PubMedBERT 输入和图张量 batch。
- `dep_type_vocab.json`：固定依存类型词表，保证本地 smoke 与 5090 full run 使用一致的图类型映射。

### 2.2 双视图模块

我们新增了两个互相解耦的表示模块：

- `SemanticsView`：基于 PubMedBERT，输出句子级语义向量 `H_sem` 与图节点初始表示 `X_init`。
- `SyntaxView`：基于 attentive GCN，对依存图节点做结构聚合，同时输出句法节点表示 `H_syn` 与边重要性初始分数 `a_ij`。

这里的设计重点是把 semantics view 与 syntax view 分开，使 B3/B4/B5 和后续 GM-IB/Ours 可以复用同一批底层模块。

### 2.3 B3/B4/B5 基线模型与训练器

我们实现了统一模型封装 `B345Model`：

- B3 使用 `SyntaxView` 的节点表示做 masked mean pooling，再注入 BioBART。
- B4 使用 `SemanticsView` 的 CLS 表示注入 BioBART。
- B5 拼接 B3/B4 两路向量后再投影注入 BioBART。

训练与评估部分实现了：

- `run_b345_smoke.py`：本地小规模链路验证。
- `run_b345_full.py`：5090 full run 入口。
- `trainer_b345.py`：支持 `label_rerank` 评估、`mapped_exact` 指标、best checkpoint、metrics history、early stopping。

## 3. 实验设置

当前 full run 使用如下共同设置：

- 数据集：ChemProtSent、CDRIntra。
- 随机种子：seed 42。
- 训练轮数：最多 20 epochs。
- early stopping：patience 5。
- 预测模式：`label_rerank`。
- 主指标：`mapped_exact`。
- 报告口径：使用 `best_metrics.json`，即验证集 `mapped_exact` 最优 epoch，而不是最后一个 epoch。

需要注意：阶段二 baseline 的现有 full result 主要是早期 1 epoch 训练结果，因此它可以作为“阶段二已完成基线”的参照，但不应被解释为完全等预算的最终对比。B3/B4/B5 三者之间的比较更公平，因为它们共享同一阶段三训练预算和评估流程。

## 4. 主要结果

### 4.1 ChemProtSent

| 模型 | best epoch | eval loss | mapped exact |
|---|---:|---:|---:|
| Stage 2 B2-coarse seed42 | 1 | 0.058921 | 0.866946 |
| B3 syntax-only | 15 | 0.122404 | 0.899948 |
| B4 semantics-only | 9 | 0.084065 | 0.901083 |
| B5 dual-view concat | 17 | 0.083922 | 0.896106 |

ChemProtSent 当前最优是 B4 semantics-only，`mapped_exact=0.901083`。相对于 Stage 2 B2-coarse seed42 的 `0.866946`，提升约 `+3.41` 个百分点。

### 4.2 CDRIntra

| 模型 | best epoch | eval loss | mapped exact |
|---|---:|---:|---:|
| Stage 2 B2-coarse seed42 | 1 | 0.107883 | 0.713408 |
| B3 syntax-only | 6 | 0.140148 | 0.764695 |
| B4 semantics-only | 16 | 0.221137 | 0.769305 |
| B5 dual-view concat | 14 | 0.251782 | 0.775451 |

CDRIntra 当前最优是 B5 dual-view concat，`mapped_exact=0.775451`。相对于 Stage 2 B2-coarse seed42 的 `0.713408`，提升约 `+6.20` 个百分点。

### 4.3 与阶段二三种子均值的参考对照

阶段二已完成三种子汇总中，B2-coarse 的结果为：

| 数据集 | Stage 2 B2-coarse 3-seed mean | Stage 3 当前 best seed42 | 差值 |
|---|---:|---:|---:|
| ChemProtSent | 0.870409 | 0.901083 | +0.030674 |
| CDRIntra | 0.697208 | 0.775451 | +0.078243 |

这个对照同样只作为趋势参考，因为 Stage 2 与 Stage 3 的训练预算并不完全一致。

## 5. 结果分析

第一，图增强方向是有效的。两个数据集上，B3/B4/B5 相比阶段二 B2-coarse seed42 都有明显提升，说明阶段三引入 semantics view 与 syntax view 后，模型确实获得了阶段二纯输入线性化以外的信息。

第二，不同数据集对信息源的偏好不同。ChemProtSent 上 B4 最高，说明该数据集更依赖局部语义与实体上下文表达，直接加入 PubMedBERT 语义表示已经非常强。CDRIntra 上 B5 最高，说明跨实体、句内结构关系对 CDRIntra 更关键，语义与句法互补更明显。

第三，naive dual-view concat 并不总是最优。B5 在 CDRIntra 上超过 B3/B4，但在 ChemProtSent 上反而低于 B4。这是一个重要信号：简单拼接可能带来冗余或噪声，特别是当 syntax view 对某些样本帮助有限时，模型没有机制主动压缩不必要结构。

第四，best epoch 的波动是正常现象。训练后期 `mapped_exact` 不一定单调上升，因此我们使用 `best_metrics.json` 和 early stopping，而不是把最后一个 epoch 当作最终结果。这对论文实验记录非常重要。

## 6. 对 GM-IB/Ours 的启示

当前结果直接支持下一步实现 GM-IB：

- Ours 不应只是把 syntax 与 semantics 拼接，而应学习图结构信息瓶颈。
- GM-IB 应该在边级别学习保留概率 `p_ij` 和采样/选择变量 `z_ij`。
- 压缩目标应鼓励模型只保留任务相关的依存边，降低无关结构对生成器的干扰。
- 训练时需要同时记录生成损失、压缩损失、平均保留边数和压缩比例，保证后续可以解释模型到底压缩了多少结构。

## 7. 下一步计划

下一步进入 Ours smoke 阶段：

1. 新增 `GMIB` 核心模块，实现边级门控、二值 Concrete 松弛、压缩损失与结构保留统计。
2. 新增 `EAGMIBModel`，复用 `SemanticsView`、`SyntaxView` 与 BioBART，把 GM-IB 输出作为生成器注入向量。
3. 新增 `trainer_ours.py` 与 `run_ours_smoke.py`，先在本地跑通 2 batch 级别 smoke。
4. smoke 通过后，再设计 5090 full run 配置，并与 B3/B4/B5 进行同预算对比。

当前 full run 目标线如下：

| 数据集 | 当前最强中间基线 | Ours 首要目标 |
|---|---:|---:|
| ChemProtSent | B4 0.901083 | 超过 B5 0.896106，并尽量超过 B4 0.901083 |
| CDRIntra | B5 0.775451 | 超过 B5 0.775451 |

## 8. Ours smoke 进展补充

在完成 B3/B4/B5 中间基线后，我们已经补齐了 EA-GMIB 的 smoke 链路：

- 新增 `GMIB` 核心模块，支持门控融合、边级保留概率、Binary Concrete 采样与压缩损失。
- 新增 `EAGMIBModel`，把语义视角、句法视角、GM-IB 与 BioBART 串联起来。
- 新增 `trainer_ours.py`、`run_ours_smoke.py` 与 smoke 配置文件。

本地 smoke 已在 ChemProtSent 与 CDRIntra 上跑通，说明：

- `forward` 和 `backward` 都正常。
- `tau` 退火接口正常。
- `avg_retained_arcs` 和 `compression_ratio` 可以被正确记录。
- eval 阶段的硬选择路径已经工作，且可以返回 label-rerank 结果。

这些 smoke 结果只用于验证工程链路，不作为论文主结果。
