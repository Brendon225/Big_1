# Stage 3 最终实验报告：双视角关系抽取模型

日期：2026-05-27

## 1. 文档定位

本文档是阶段三实验的最终归档报告，综合整理此前阶段三文档中的主要实验过程、模型设计、结果诊断与最终结论。重点参考文档包括：

- `docs/stage3_model_docs/stage3_b345_intermediate_report.md`
- `docs/stage3_model_docs/stage3_current_results_summary.md`
- `docs/stage3_model_docs/stage3_gmib_experiment_report.md`
- `docs/stage3_model_docs/stage3_multiseed_summary.md`

阶段三的核心目标是：在阶段二依存线性化 baseline 的基础上，进一步验证显式 syntax view 与 semantics view 是否能提升生物医学关系抽取性能，并最终形成一条稳定、可解释、可复现的双视角建模主线。

经过完整实验后，阶段三的最终主线已经明确：

1. Syntax view 与 semantics view 都能带来明显收益。
2. 不同数据集对两个视角的依赖不同。
3. GM-IB 式 hard compression 机制在当前生成式框架下不稳定，不适合作为最终主线。
4. B7 gated dual-view fusion 是当前最值得写入论文主线的模型：它在 ChemProtSent 上达到三种子最优，在 CDRIntra 上与 B5 非常接近且稳定性更好。

## 2. 实验脉络

阶段三不是一次性直接实现最终模型，而是按诊断路径逐步推进：

| 阶段 | 模型 | 目的 | 结论 |
|---|---|---|---|
| 图数据桥接 | GraphBuilder + BioREGraphDataset | 将阶段一清洗数据转为图增强输入 | 成功建立稳定图数据链路 |
| 单视角诊断 | B3 syntax-only | 验证依存图句法表示是否有效 | 两个数据集均显著优于阶段二 baseline |
| 单视角诊断 | B4 semantics-only | 验证 PubMedBERT 语义表示是否有效 | ChemProtSent 上尤其强 |
| 双视角诊断 | B5 dual-view concat | 验证 syntax + semantics 是否互补 | CDRIntra 上最强，说明结构互补有效 |
| 信息瓶颈探索 | GM-IB/Ours | 尝试学习可压缩依存子图 | 性能不稳定，hard selection 机制不可靠 |
| 最终双视角模型 | B7 gated dual-view fusion | 用连续门控替代硬压缩和 naive concat | ChemProtSent 三种子最优，CDRIntra 稳定且接近最优 |

这一路径的关键意义在于：我们不是盲目增加模型复杂度，而是先用 B3/B4/B5 证明两个视角确实有效，再用 GM-IB 的失败结果排除不稳定机制，最后收敛到更轻量、更稳健的 B7 gated fusion。

## 3. 数据与评估口径

阶段三主要在两个数据集上完成完整实验：

| 数据集 | 任务特点 | 阶段三关注点 |
|---|---|---|
| ChemProtSent | 化学物质-蛋白关系，句子级关系抽取 | 更依赖实体上下文与语义表达 |
| CDRIntra | 化学物质-疾病句内关系 | 更依赖句内结构和跨实体依存信息 |

统一评估口径如下：

| 项目 | 设置 |
|---|---|
| 主指标 | `mapped_exact` |
| 预测模式 | `label_rerank` |
| 模型选择 | 使用 `best_metrics.json` 中验证集 `mapped_exact` 最优 epoch |
| 训练上限 | 20 epochs |
| early stopping | patience = 5 |
| 多种子 | seed 42 / 123 / 456 |
| 数据视图 | coarse dependency graph |

说明：阶段二 baseline 与阶段三模型的训练预算并不完全相同，因此阶段二结果主要作为阶段性参考。阶段三内部 B4/B5/B7 的三种子比较更适合作为最终论文主表依据。

## 4. 模型设计

### 4.1 图增强数据链路

阶段三首先建立了不破坏阶段一、阶段二数据逻辑的图增强链路：

| 模块 | 作用 |
|---|---|
| `GraphBuilder` | 从清洗后样本构建 coarse dependency graph |
| `BioREGraphDataset` | 在文本样本上增加图字段，如邻接矩阵、依存类型、节点 mask、实体节点索引 |
| `BioREGraphCollator` | 同时构造 BioBART 输入、PubMedBERT 输入和图张量 batch |
| `dep_type_vocab.json` | 固定依存类型映射，保证本地 smoke 与 5090 full run 一致 |

### 4.2 双视角表示模块

阶段三将语义视角和句法视角解耦：

| 视角 | 模块 | 输出 | 作用 |
|---|---|---|---|
| Semantics view | `SemanticsView` | 句子级语义向量、节点初始表示 | 提供 PubMedBERT 生物医学语义信息 |
| Syntax view | `SyntaxView` | 句法节点表示、依存弧重要性初始分数 | 提供依存图结构聚合信息 |

这种解耦设计使 B3/B4/B5/B7 能复用同一底层表示模块，也使实验能够明确区分“语义收益”“句法收益”和“双视角融合收益”。

### 4.3 B3/B4/B5 诊断模型

| 模型 | 设计 | 实验意义 |
|---|---|---|
| B3 syntax-only | 只注入 syntax view pooling vector | 判断依存图结构本身是否有用 |
| B4 semantics-only | 只注入 PubMedBERT semantics vector | 判断语义增强是否有用 |
| B5 dual-view concat | 拼接 syntax vector 与 semantics vector 后投影 | 判断两个视角是否互补 |

B3/B4/B5 是阶段三的关键诊断组。它们表明：两个视角都有效，但简单拼接并不总是最优。

### 4.4 GM-IB/Ours 简要说明

GM-IB/Ours 的初衷是学习每条依存边的保留概率，通过 hard selection 得到可解释的依存子图，并用压缩损失控制保留边比例。

但正式实验和 threshold sweep 表明：

| 问题 | 现象 |
|---|---|
| hard selection 不稳定 | Ours-v1 best epoch 中 `eval_compression_ratio=0.0` |
| 边概率缺乏有效排序 | threshold sweep 对性能几乎无影响 |
| 压缩机制难解释 | 部分设置出现全空或近似全选 |
| 性能提升不稳 | CDRIntra 上 Ours-v1 只比 B5 多约 3 个 dev 样本 |

因此，GM-IB 不作为阶段三最终主线。它在报告中保留为探索性实验和负结果诊断，不建议在论文中作为主要贡献展开。

### 4.5 B7 gated dual-view fusion

B7 是阶段三最终主线模型。它不使用离散 hard selection，而是学习连续门控，在 semantic vector 和 syntax vector 之间进行自适应融合：

```text
g = sigmoid(W [h_sem; h_syn])
h_fused = g * h_sem + (1 - g) * h_syn
```

B7 的优势是：

| 优势 | 说明 |
|---|---|
| 连续可训练 | 避免 GM-IB hard selection 全空或全选 |
| 结构简单 | 相比 GM-IB 风险更低，训练更稳定 |
| 保留双视角 | 不像 B4/B3 只依赖单一信息源 |
| 比 naive concat 更自适应 | 相比 B5，模型可以学习视角权重 |
| 可记录 gate statistics | 可以分析语义/句法融合趋势 |

## 5. Seed42 完整阶段结果

本节汇总阶段三探索过程中的 seed42 结果，主要用于展示模型演进路径。最终论文主表建议优先采用第 6 节的三种子结果。

### 5.1 ChemProtSent seed42

| 模型 | 说明 | best epoch | mapped exact | eval loss | 备注 |
|---|---|---:|---:|---:|---|
| Stage2 B2-coarse | 阶段二依存线性化基线 | 1 | 0.866946 | 0.058921 | 阶段性参考 |
| B3 | syntax-only | 15 | 0.899948 | 0.122404 | 明显优于 Stage2 |
| B4 | semantics-only | 9 | 0.901083 | 0.084065 | 单视角中最强 |
| B5 | dual-view concat | 17 | 0.896106 | 0.083922 | 低于 B4 |
| Ours-v1 | GM-IB degree-pool | 14 | 0.895932 | 0.098628 | hard selection 全空 |
| Ours-v2 | selected-message | 15 | 0.903964 | 0.101205 | 数值强，但 compression 机制不理想 |
| B7 | gated dual-view | 4 | **0.906583** | 0.056853 | seed42 最优 |

ChemProtSent 上，语义视角非常强，B4 已经显著高于阶段二；B7 在 seed42 上进一步超过所有已有模型，说明连续门控融合可以在保持语义优势的同时吸收有用结构信息。

### 5.2 CDRIntra seed42

| 模型 | 说明 | best epoch | mapped exact | eval loss | 备注 |
|---|---|---:|---:|---:|---|
| Stage2 B2-coarse | 阶段二依存线性化基线 | 1 | 0.713408 | 0.107883 | 阶段性参考 |
| B3 | syntax-only | 6 | 0.764695 | 0.140148 | 明显优于 Stage2 |
| B4 | semantics-only | 16 | 0.769305 | 0.221137 | 优于 B3 |
| B5 | dual-view concat | 14 | 0.775451 | 0.251782 | 简单双视角最强 |
| Ours-v1 | GM-IB degree-pool | 10 | **0.776028** | 0.238254 | 仅比 B5 多约 3 个 dev 样本，hard selection 全空 |
| Ours-v2 | selected-message | 5 | 0.767576 | 0.119007 | 低于 B5 |
| B7 | gated dual-view | 9 | 0.769689 | 0.200311 | seed42 未超过 B5 |

CDRIntra 上，B5 的表现最有诊断意义：它说明语义和句法在该数据集上确实互补。Ours-v1 虽然 seed42 数值略高，但机制不可靠，因此不应作为最终主线。

## 6. 三种子最终结果

本节是阶段三最终报告的核心结果。B4/B5/B7 均已完成 seed 42/123/456 full run。

### 6.1 ChemProtSent 三种子结果

| 模型 | mapped exact mean | std | eval loss mean | 相对 Stage2 B2-coarse mean |
|---|---:|---:|---:|---:|
| Stage2 B2-coarse | 0.870409 | 0.002449 | 0.061042 | - |
| B4 semantics-only | 0.896776 | 0.004412 | 0.081550 | +0.026367 |
| B5 dual-view concat | 0.898493 | 0.002279 | 0.081942 | +0.028084 |
| B7 gated dual-view | **0.901374** | 0.004559 | **0.066236** | **+0.030965** |

ChemProtSent 上，B7 是三种子均值最高的阶段三模型，相比 Stage2 B2-coarse 三种子均值提升约 3.10 个百分点。虽然 B7 的标准差略高于 B5，但其均值和 eval loss mean 都是 B4/B5/B7 中最优。

### 6.2 CDRIntra 三种子结果

| 模型 | mapped exact mean | std | eval loss mean | 相对 Stage2 B2-coarse mean |
|---|---:|---:|---:|---:|
| Stage2 B2-coarse | 0.697208 | 0.019485 | 0.115773 | - |
| B4 semantics-only | 0.768472 | 0.005141 | 0.242170 | +0.071264 |
| B5 dual-view concat | **0.772314** | 0.007879 | 0.250330 | **+0.075106** |
| B7 gated dual-view | 0.771226 | **0.002033** | **0.162372** | +0.074018 |

CDRIntra 上，B5 的 `mapped_exact` 三种子均值略高于 B7，差值约 0.001088，即约 0.11 个百分点。但 B7 的标准差明显更小，eval loss mean 也明显更低，说明 B7 虽然不是主指标最高，但训练与评估更稳定。

### 6.3 B7 gate statistics

| 数据集 | gate_sem mean | gate_sem std | gate_syn mean | 解释 |
|---|---:|---:|---:|---|
| ChemProtSent | 0.497472 | 0.020517 | 0.502528 | 整体接近平衡，seed 间略有波动 |
| CDRIntra | 0.504926 | 0.006612 | 0.495074 | 整体接近平衡，跨 seed 更稳定 |

需要谨慎解释 gate：当前结果不支持“ChemProtSent 明显偏语义、CDRIntra 明显偏句法”这种强结论。更稳妥的表述是：B7 学到的是一种接近平衡的连续双视角融合机制，并通过端到端训练自动调节 syntax/semantics 的相对贡献。

## 7. 结果分析

### 7.1 阶段三整体有效

与阶段二 B2-coarse 相比，阶段三所有关键模型在两个数据集上都有明显提升：

| 数据集 | Stage2 B2-coarse mean | Stage3 最优三种子 mean | 提升 |
|---|---:|---:|---:|
| ChemProtSent | 0.870409 | B7 0.901374 | +0.030965 |
| CDRIntra | 0.697208 | B5 0.772314 | +0.075106 |

这说明阶段三引入的图增强语义/句法表示不是偶然收益，而是在完整数据和多种子设置下稳定优于阶段二依存线性化 baseline。

### 7.2 数据集偏好不同

ChemProtSent 中，B4/B5/B7 的排序为：

```text
B7 > B5 > B4
```

这说明仅使用语义视角已经很强，但经过门控融合后，模型仍能从 syntax view 中吸收额外有效信息。

CDRIntra 中，B4/B5/B7 的排序为：

```text
B5 > B7 > B4
```

这说明 CDRIntra 更依赖 syntax 与 semantics 的互补。B5 的 naive concat 在主指标上略强，但 B7 的低方差和低 eval loss 表明门控融合具有更好的稳定性。

### 7.3 GM-IB 是有价值的探索，但不是最终路线

GM-IB/Ours 的实验价值在于帮助我们确认：在当前 BioBART 生成式框架下，离散 hard edge selection 很难稳定地成为有效瓶颈。它暴露的问题不是工程链路不通，而是机制本身对性能贡献不稳、可解释性不可靠。

因此，论文主线不应强调“我们成功学习了可解释依存子图压缩”，而应强调经过诊断后，我们选择了更稳定的 continuous gated dual-view fusion。

### 7.4 B7 的论文定位

B7 适合作为阶段三最终模型，原因如下：

| 维度 | 判断 |
|---|---|
| 性能 | ChemProtSent 三种子最优；CDRIntra 接近最优 |
| 稳定性 | CDRIntra 标准差最低，eval loss 明显低于 B5 |
| 机制 | 连续门控比 hard compression 更可靠 |
| 叙事 | 自然承接 B3/B4/B5 的诊断结果 |
| 复杂度 | 比 GM-IB 更轻量，更容易复现 |

如果论文需要一个统一最终模型，建议将 B7 作为主模型；如果论文结果表允许按数据集分析，则可以诚实呈现 ChemProtSent 上 B7 最优、CDRIntra 上 B5 主指标略高但 B7 更稳定的 trade-off。

## 8. 推荐论文结果表

### 8.1 主表建议

论文主表建议采用三种子均值和标准差：

| 数据集 | 模型 | mapped exact mean ± std | 结论 |
|---|---|---:|---|
| ChemProtSent | Stage2 B2-coarse | 0.870409 ± 0.002449 | 阶段二结构线性化 baseline |
| ChemProtSent | B4 semantics-only | 0.896776 ± 0.004412 | 单语义视角强 |
| ChemProtSent | B5 dual-view concat | 0.898493 ± 0.002279 | 简单双视角有效 |
| ChemProtSent | B7 gated dual-view | **0.901374 ± 0.004559** | 最优 |
| CDRIntra | Stage2 B2-coarse | 0.697208 ± 0.019485 | 阶段二结构线性化 baseline |
| CDRIntra | B4 semantics-only | 0.768472 ± 0.005141 | 单语义视角有效 |
| CDRIntra | B5 dual-view concat | **0.772314 ± 0.007879** | 主指标最优 |
| CDRIntra | B7 gated dual-view | 0.771226 ± 0.002033 | 接近最优且更稳定 |

### 8.2 消融表建议

消融表可以只列 Stage3 内部模型：

| 数据集 | B4 semantics-only | B5 concat | B7 gated fusion | 主要观察 |
|---|---:|---:|---:|---|
| ChemProtSent | 0.896776 | 0.898493 | **0.901374** | 门控融合优于单语义和简单拼接 |
| CDRIntra | 0.768472 | **0.772314** | 0.771226 | 双视角互补明显；B7 更稳定 |

## 9. 最终结论

阶段三最终完成了从依存线性化 baseline 到显式双视角建模的完整实验闭环。

第一，syntax view 与 semantics view 的引入显著提升了关系抽取性能。相比阶段二 B2-coarse，阶段三最优模型在 ChemProtSent 上提升约 3.10 个百分点，在 CDRIntra 上提升约 7.51 个百分点。

第二，双视角建模比单纯依赖依存线性化更有效。B4/B5/B7 的结果共同表明，PubMedBERT 语义表示和依存图句法表示能够提供互补信息。

第三，GM-IB hard compression 在当前设置下不是理想最终方案。它的探索帮助我们识别了离散边选择机制的不稳定性，但其结果不应作为论文主要贡献。

第四，B7 gated dual-view fusion 是阶段三最适合写入论文主线的模型。它在 ChemProtSent 上取得三种子最优，在 CDRIntra 上与最优 B5 非常接近且稳定性更好，机制上也比 GM-IB 更清晰、更稳健。

最终建议的论文表述是：

```text
We find that explicitly modeling syntax and semantics as two complementary views substantially improves biomedical relation extraction over dependency linearization baselines. A lightweight gated dual-view fusion model achieves the best multi-seed performance on ChemProtSent and competitive, more stable performance on CDRIntra, while avoiding the instability of hard graph compression.
```

## 10. 后续工作建议

阶段三已经足以支撑论文实验主线。后续不建议继续大规模投入 GM-IB 压缩模块，除非论文需要专门讨论负结果或可解释性探索。

建议下一步工作集中在：

| 优先级 | 任务 | 目的 |
|---|---|---|
| 高 | 整理 Stage3 final table 与论文主表 | 固化最终结果 |
| 高 | 做 B7/B5 错误案例分析 | 解释 ChemProtSent 与 CDRIntra 差异 |
| 中 | 可视化 B7 gate statistics | 支撑双视角融合叙事 |
| 中 | 整理论文方法章节 | 将 SemanticsView、SyntaxView、Gated Fusion 写成统一框架 |
| 低 | GM-IB 负结果附录 | 如篇幅允许，可作为探索性分析 |
