# Stage 3 当前结果汇总与下一步模型目标

日期：2026-05-23

## 1. 文档定位

本文档用于归档 Stage 3 当前所有已完成 seed42 结果，并形成后续双视角模型优化的对比表雏形。此前我们已经完成：

- Stage 2：B2 coarse dependency linearization baseline。
- Stage 3：B3 syntax-only、B4 semantics-only、B5 dual-view concat。
- Stage 3：Ours-v1 GM-IB。
- Stage 3：Ours-v2 selected-message + target-ratio compression。

经过 GM-IB/Ours-v1/v2 的结果分析后，我们决定暂时不再把主力投入信息压缩模块，而是将论文主线转向更稳定、更有证据支撑的双视角 syntax-semantics 建模。

## 2. 结果口径

本表采用以下口径：

- 数据集：ChemProtSent、CDRIntra。
- 随机种子：seed 42。
- Stage 3 指标使用 `best_metrics.json`。
- 主指标：`mapped_exact`。
- `eval_loss` 仅作为辅助参考，不作为主排名标准。
- Stage 2 B2-coarse seed42 使用对应 checkpoint 的 `metrics.json`。

需要注意：Stage 2 与 Stage 3 的训练预算并不完全一致，因此 Stage 2 主要作为阶段性参照；Stage 3 内部模型之间的对比更关键。

## 3. ChemProtSent 当前结果

| 模型 | 说明 | best epoch | mapped exact | eval loss | hard compression | soft prob ratio |
|---|---|---:|---:|---:|---:|---:|
| Stage2 B2-coarse seed42 | 阶段二最佳依存线性化基线 | 1 | 0.866946 | 0.058921 | - | - |
| B3 syntax-only | 只注入 syntax view | 15 | 0.899948 | 0.122404 | - | - |
| B4 semantics-only | 只注入 PubMedBERT semantics view | 9 | 0.901083 | 0.084065 | - | - |
| B5 dual-view concat | syntax + semantics naive concat | 17 | 0.896106 | 0.083922 | - | - |
| Ours-v1 GM-IB | degree-pool GM-IB | 14 | 0.895932 | 0.098628 | 0.0000 | - |
| Ours-v2 selected-message | selected-message + target-ratio | 15 | **0.903964** | 0.101205 | 0.0000 | 0.499029 |

### 3.1 ChemProtSent 关键差值

| 对比 | mapped exact 差值 | 百分点 | 约等于 dev 样本数 |
|---|---:|---:|---:|
| B3 - Stage2 B2 | +0.033002 | +3.30 | +378 |
| B4 - Stage2 B2 | +0.034137 | +3.41 | +391 |
| B5 - Stage2 B2 | +0.029160 | +2.92 | +334 |
| Ours-v2 - B4 | +0.002881 | +0.29 | +33 |
| Ours-v2 - B5 | +0.007858 | +0.79 | +90 |

### 3.2 ChemProtSent 初步判断

ChemProtSent 当前数值最高的是 Ours-v2，`mapped_exact=0.903964`，超过 B4 和 B5。从纯性能角度看，这是 Stage 3 目前最好的 ChemProtSent seed42 结果。

但需要谨慎解释：Ours-v2 的 best epoch 中 `eval_compression_ratio=0.0`，说明 hard selection 仍然全空。因此，这个结果更适合被解释为 selected-message/entity-aware readout 的收益，而不是 GM-IB 成功学到了可解释的关键依存子图。

ChemProtSent 的稳定证据仍然指向 semantics view 很强：B4 已经达到 0.901083，高于 B3/B5。

## 4. CDRIntra 当前结果

| 模型 | 说明 | best epoch | mapped exact | eval loss | hard compression | soft prob ratio |
|---|---|---:|---:|---:|---:|---:|
| Stage2 B2-coarse seed42 | 阶段二最佳依存线性化基线 | 1 | 0.713408 | 0.107883 | - | - |
| B3 syntax-only | 只注入 syntax view | 6 | 0.764695 | 0.140148 | - | - |
| B4 semantics-only | 只注入 PubMedBERT semantics view | 16 | 0.769305 | 0.221137 | - | - |
| B5 dual-view concat | syntax + semantics naive concat | 14 | 0.775451 | 0.251782 | - | - |
| Ours-v1 GM-IB | degree-pool GM-IB | 10 | **0.776028** | 0.238254 | 0.0000 | - |
| Ours-v2 selected-message | selected-message + target-ratio | 5 | 0.767576 | 0.119007 | 1.0000 | 0.514173 |

### 4.1 CDRIntra 关键差值

| 对比 | mapped exact 差值 | 百分点 | 约等于 dev 样本数 |
|---|---:|---:|---:|
| B3 - Stage2 B2 | +0.051287 | +5.13 | +267 |
| B4 - Stage2 B2 | +0.055897 | +5.59 | +291 |
| B5 - Stage2 B2 | +0.062044 | +6.20 | +323 |
| Ours-v1 - B5 | +0.000576 | +0.06 | +3 |
| Ours-v2 - B5 | -0.007876 | -0.79 | -41 |

### 4.2 CDRIntra 初步判断

CDRIntra 当前数值最高的是 Ours-v1，`mapped_exact=0.776028`，但只比 B5 多 3 个 dev 样本。由于 Ours-v1 的 hard selection 也是全空，这个提升不宜被解释为 GM-IB 结构选择成功。

从稳定性和机制清晰度看，CDRIntra 更支持 B5 dual-view concat 这条主线。B5 显著高于 B3/B4，说明 CDRIntra 更依赖语义和句法的互补。

Ours-v2 在 CDRIntra 上退化到 0.767576，低于 B5 和 Ours-v1。其 `eval_compression_ratio=1.0`，说明 hard selection 近似全选，机制上也不理想。

## 5. Stage 2 三种子均值参考

阶段二三种子汇总中，与当前主线最相关的是 B2-coarse SDP：

| 数据集 | Stage2 B2-coarse 3-seed mean | std |
|---|---:|---:|
| ChemProtSent | 0.870409 | 0.002449 |
| CDRIntra | 0.697208 | 0.019485 |

Stage 3 seed42 相比 Stage 2 三种子均值的提升非常明显，但后续论文主表仍应优先补 Stage 3 关键模型的多种子结果。

## 6. 当前最强结果与推荐主线

| 数据集 | 当前 seed42 数值最高 | mapped exact | 机制可信度 | 推荐论文主线 |
|---|---|---:|---|---|
| ChemProtSent | Ours-v2 selected-message | 0.903964 | 中等，性能强但 hard selection 全空 | 语义主导 + entity-aware dual-view readout |
| CDRIntra | Ours-v1 GM-IB | 0.776028 | 较弱，只比 B5 多 3 样本且 hard selection 全空 | B5 dual-view concat / 稳定双视角融合 |

当前不建议继续深挖 GM-IB compression。更稳的方向是从 B5 出发，设计更简单、更稳定的双视角融合模型。

## 7. 对下一步 B7 的启示

现有结果给出三个重要启示：

1. ChemProtSent 上 B4 很强，说明语义视角贡献最大。新模型不应让句法噪声过度干扰语义表示。
2. CDRIntra 上 B5 最稳，说明 syntax 和 semantics 的互补确实有价值。新模型应保留双视角信息。
3. GM-IB hard selection 不稳定，说明下一步不应依赖离散边选择，而应使用连续、可训练、可解释的视角融合机制。

因此，推荐下一步实现：

```text
B7: gated dual-view fusion
```

B7 的核心思想：

- 输入 syntax vector 和 semantic vector。
- 学习一个 gate：`g = sigmoid(W[sem; syn])`。
- 融合：`h = g * sem + (1 - g) * syn`。
- 将融合向量注入 BioBART。
- 记录 gate 均值，用于解释不同数据集更偏语义还是句法。

相比 B5 naive concat，B7 的优势是：

- 参数少，训练风险低。
- 不依赖 hard threshold。
- 不会出现 compression ratio 全空或全满。
- 可以通过 gate statistics 做视角贡献分析。
- 更符合当前证据：ChemProtSent 需要偏语义，CDRIntra 需要双视角互补。

## 8. 下一步实验计划

建议按以下顺序推进：

1. 实现 B7 gated dual-view fusion。
2. 本地 smoke：ChemProtSent 和 CDRIntra 各 2 个 train/eval batch。
3. 5090 full run：先跑 seed42。
4. 如果 B7 seed42 在至少一个数据集上优于 B5，且另一个数据集不明显退化，则进入多种子验证。
5. 多种子优先补：
   - B4 ChemProtSent。
   - B5 CDRIntra。
   - B7 ChemProtSent/CDRIntra。
6. 最后整理 Stage 3 final results table，并更新论文叙事。

## 9. 当前阶段性结论

Stage 3 的稳定贡献不应再押注 GM-IB compression，而应转向 entity-aware syntax-semantics dual-view modeling。

当前证据支持如下叙事：

- Semantics view 对 ChemProtSent 最关键。
- Syntax + semantics dual-view 对 CDRIntra 最关键。
- Naive concat 有一定效果，但不够自适应。
- GM-IB compression 在当前生成式框架中机制不稳定。
- 下一步应采用更轻量、更稳健的 gated fusion 来学习数据集和样本级视角权重。
