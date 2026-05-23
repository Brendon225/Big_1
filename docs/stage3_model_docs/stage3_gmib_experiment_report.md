# Stage 3 GM-IB/Ours 实验报告

日期：2026-05-18

## 1. 文档定位

本文档记录阶段三后半段围绕 GM-IB/Ours 的实验过程、结果诊断和当前修复方案。前置文档 `stage3_b345_intermediate_report.md` 已经记录了 B3/B4/B5 三个中间基线的结果：B3 为 syntax-only，B4 为 semantics-only，B5 为 dual-view concat。由于 B3/B4/B5 结果显示语义视角和句法视角都有效，但 naive concat 并不稳定，我们继续实现 GM-IB/Ours，希望让模型学习“哪些依存边应该被保留”，从而替代简单拼接。

本文档重点记录：

- Ours-v1 的模型方法和正式实验结果。
- Ours-v1 暴露的问题：hard selection 失效、边概率塌缩、阈值调整无效。
- threshold sweep 的诊断结论。
- Ours-v2 的修复思路、代码改动和本地 smoke 验证结果。
- 后续 5090 full run 的判断标准。

## 2. 实验目标

GM-IB/Ours 的目标不是简单地再加一个图向量，而是实现一个可解释、可压缩的结构选择机制：

- 输入语义视角 `H_sem`、句法视角 `H_syn` 和初始弧重要性 `a_ij`。
- 学习每条依存弧的保留概率 `p_ij`。
- 训练时用 Binary Concrete 得到可微的软选择 `z_ij`。
- 推理/评估时用 hard selection 得到离散弧子图。
- 通过压缩损失控制保留边比例，避免模型把所有边都保留。
- 最终希望 Ours 在性能上超过 B5，同时在解释性上能给出合理的 selected dependency subgraph。

阶段目标线如下：

| 数据集 | 主要对照 | mapped exact |
|---|---:|---:|
| ChemProtSent | B5 dual-view concat | 0.896106 |
| ChemProtSent | B4 semantics-only | 0.901083 |
| CDRIntra | B5 dual-view concat | 0.775451 |

ChemProtSent 的最低目标是超过 B5，更理想目标是接近或超过 B4。CDRIntra 的目标是超过 B5。

## 3. Ours-v1 方法

Ours-v1 复用了前面已经验证过的三个核心模块：

- `SemanticsView`：基于 PubMedBERT 输出句子语义表示 `H_sem` 和图节点初始表示 `X_init`。
- `SyntaxView`：基于 attentive GCN 输出句法节点表示 `H_syn` 和初始弧分数 `a_ij`。
- `GMIB`：基于融合节点表示计算边保留概率 `p_ij` 和选择变量 `z_ij`。

Ours-v1 的训练路径如下：

1. PubMedBERT 编码实体标记后的句子，得到 `H_sem` 和 `X_init`。
2. SyntaxView 在依存图上编码 `X_init`，得到 `H_syn` 和 `a_ij`。
3. GM-IB 融合 `H_sem` 与 `H_syn`，并为每条依存边计算 `p_ij`。
4. 训练时通过 Binary Concrete 采样得到连续 `z_ij`。
5. 用 `z_ij` 的入边/出边度数为节点加权，池化得到一个 graph injection vector。
6. 将 injection vector 作为额外 token 拼接到 BioBART encoder 输入后训练生成目标。
7. 总损失为 `L_total = L_gen + beta * L_compress`。

Ours-v1 的关键配置：

| 参数 | 值 |
|---|---:|
| `beta` | `1e-6` |
| `compression_loss_type` | `l1` |
| `tau_init` | `1.0` |
| `tau_min` | `0.1` |
| `tau_anneal_rate` | `0.95` |
| `selection_threshold` | `0.475` |
| `readout_mode` | `degree_pool` |

## 4. Ours-v1 正式实验结果

Ours-v1 已在 ChemProtSent 和 CDRIntra 上完成 seed42 full run。

| 数据集 | 模型 | best epoch | mapped exact | eval loss | eval compression ratio |
|---|---:|---:|---:|---:|---:|
| ChemProtSent | B4 semantics-only | 9 | 0.901083 | 0.084065 | - |
| ChemProtSent | B5 dual-view concat | 17 | 0.896106 | 0.083922 | - |
| ChemProtSent | Ours-v1 | 14 | 0.895932 | 0.098628 | 0.0000 |
| CDRIntra | B5 dual-view concat | 14 | 0.775451 | 0.251782 | - |
| CDRIntra | Ours-v1 | 10 | 0.776028 | 0.238254 | 0.0000 |

从性能上看：

- ChemProtSent 上 Ours-v1 与 B5 非常接近，但低 2 个 dev 样本，没有严格超过 B5。
- CDRIntra 上 Ours-v1 比 B5 高 3 个 dev 样本，形式上达成了阶段三里程碑。

但从机制上看，两个 best epoch 的 `eval_compression_ratio` 都是 0。这意味着在评估时 hard selection 没有选中任何边，模型触发了 fallback pooling。因此，虽然 CDRIntra 的数值略高于 B5，但不能说当前 Ours-v1 已经稳定学到了可解释的关键依存子图。

## 5. Ours-v1 问题诊断

Ours-v1 的主要问题不是训练崩溃，也不是 label-rerank 无法工作，而是结构选择机制对最终预测影响太弱。

### 5.1 hard selection 全空

在 Ours-v1 best checkpoint 上：

| 数据集 | best epoch | train compression ratio | eval compression ratio |
|---|---:|---:|---:|
| ChemProtSent | 14 | 0.3041 | 0.0000 |
| CDRIntra | 10 | 0.1984 | 0.0000 |

训练时软选择仍有保留边，但评估时硬阈值路径没有保留任何边。这说明 `p_ij` 的绝对数值与固定阈值不匹配，且 hard path 缺少稳定约束。

### 5.2 threshold sweep 无法改善性能

为了确认问题是否只是阈值没有校准，我们加载 Ours-v1 的 `best_model`，不重新训练，只在 dev 集上做 threshold sweep。

ChemProtSent 结果：

| threshold | compression ratio | mapped exact |
|---:|---:|---:|
| 0.05 | 1.0000 | 0.895932 |
| 0.10 | 1.0000 | 0.895932 |
| 0.15 | 0.0000 | 0.895932 |
| 0.20 | 0.0000 | 0.895932 |
| 0.25 | 0.0000 | 0.895932 |
| 0.30 | 0.0000 | 0.895932 |
| 0.35 | 0.0000 | 0.895932 |
| 0.40 | 0.0000 | 0.895932 |
| 0.475 | 0.0000 | 0.895932 |

ChemProtSent 的 `p_ij` 几乎塌缩为常数：

| 统计量 | 数值 |
|---|---:|
| `p_mean` | 0.127671 |
| `p_min` | 0.127658 |
| `p_p50` | 0.127670 |
| `p_max` | 0.127682 |

因此，threshold 低于约 0.127 时几乎全选，高于约 0.127 时全不选，模型没有形成有意义的边排序。

CDRIntra 结果：

| threshold | compression ratio | mapped exact |
|---:|---:|---:|
| 0.05 | 1.0000 | 0.776028 |
| 0.10 | 0.7560 | 0.776028 |
| 0.15 | 0.6160 | 0.776028 |
| 0.20 | 0.5402 | 0.776028 |
| 0.25 | 0.0030 | 0.776028 |
| 0.30 | 0.0000 | 0.776028 |
| 0.35 | 0.0000 | 0.776028 |
| 0.40 | 0.0000 | 0.776028 |
| 0.475 | 0.0000 | 0.776028 |

CDRIntra 的 `p_ij` 分布比 ChemProtSent 稍好，但 mapped exact 对 threshold 完全不敏感。换句话说，边选择比例从全选到半选再到全不选，最终预测都不变。

### 5.3 诊断结论

threshold sweep 说明：问题不只是 `selection_threshold` 没调好，而是 Ours-v1 的结构选择对生成器决策影响不够强。

具体原因包括：

- `z_ij` 只影响最终 node pooling 的权重，没有真正进入图消息传递主干。
- BioBART 文本输入和 PubMedBERT 语义表示过强，模型可以绕开 graph selection。
- `l1` 压缩只鼓励概率变小，不鼓励概率保持在可控目标区间。
- ChemProtSent 的边概率塌缩为近似常数，缺少边级区分性。
- CDRIntra 虽然边概率有分布，但选择变化没有改变 label-rerank 决策。

因此，继续调 threshold 不会解决核心问题。我们需要改 Ours 的结构路径。

## 6. Ours-v2 修复方案

基于上述诊断，我们实现了 Ours-v2。Ours-v2 的目标是让 `z_ij` 更直接、更强地影响 BioBART 的注入向量，同时避免边概率塌缩。

### 6.1 selected-message readout

Ours-v1 的 readout 是 `degree_pool`：

- 根据 `z_ij` 的入边/出边度数给节点加权。
- 对节点表示做池化。
- 若 hard selection 全空，则 fallback 到所有节点均匀池化。

Ours-v2 新增 `readout_mode=selected_message`：

1. 使用 `z_ij` 构造 selected adjacency。
2. 对 selected adjacency 做归一化。
3. 用 selected edges 对融合节点表示做一次加权消息传递。
4. 得到 selected-message updated node representations。
5. 同时读取全局图表示、实体1节点表示、实体2节点表示和实体差异表示。
6. 拼接后投影为 BioBART injection vector。

这使得 `z_ij` 不再只是池化权重，而是直接决定哪些节点信息可以沿边传播。

### 6.2 target-ratio compression

Ours-v1 使用 `l1` 压缩，容易把概率整体压低。Ours-v2 新增：

- `compression_loss_type = target_ratio`
- `target_compression_ratio = 0.5`
- `target_ratio_loss_weight = 10.0`

目标是让平均边保留概率靠近 0.5，避免训练早期或中期迅速塌缩为全零/全一。

Ours-v2 的总损失仍为：

```text
L_total = L_gen + beta * L_compress
```

但此时：

```text
L_compress = target_ratio_loss_weight * (mean(p_ij) - target_ratio)^2
```

当前 v2 配置中 `beta = 0.1`，因此 target-ratio 项会参与训练，但不会压过生成损失。

### 6.3 新增诊断指标

Ours-v2 trainer 新增记录：

- `train_base_compress_loss`
- `train_target_ratio_loss`
- `train_prob_compression_ratio`
- `eval_base_compress_loss`
- `eval_target_ratio_loss`
- `eval_prob_compression_ratio`

其中 `prob_compression_ratio` 是 `mean(p_ij)`，它比 hard `compression_ratio` 更适合观察训练过程是否靠近目标比例。

## 7. Ours-v2 本地验证

Ours-v2 已完成本地 smoke 和 full-config override 验证。

ChemProtSent 本地 full-config smoke：

| 指标 | 数值 |
|---|---:|
| `train_prob_compression_ratio` | 0.5043 |
| `train_compression_ratio` | 0.4778 |
| `eval_prob_compression_ratio` | 0.5073 |
| `eval_compression_ratio` | 0.8478 |
| `eval_target_ratio_loss` | 0.000053 |
| `mapped_exact` | 1.0000 |

CDRIntra 本地 full-config smoke：

| 指标 | 数值 |
|---|---:|
| `train_prob_compression_ratio` | 0.5086 |
| `train_compression_ratio` | 0.4674 |
| `eval_prob_compression_ratio` | 0.5134 |
| `eval_compression_ratio` | 0.8125 |
| `eval_target_ratio_loss` | 0.000368 |
| `mapped_exact` | 0.5000 |

注意：上述 smoke 只跑 2 个 train batch 和 2 个 eval batch，因此 mapped exact 没有统计意义。这里真正重要的是：

- Ours-v2 能正常 forward/backward。
- `prob_compression_ratio` 接近目标 0.5。
- hard selection 不再全空。
- 新增指标能被正常写入 `metrics.json`。
- full v2 配置可以被 `run_ours_full.py` 正确读取并运行。

## 8. 当前代码与配置归档

本轮新增或修改的核心代码：

| 文件 | 作用 |
|---|---|
| `src/models/gmib.py` | 新增 target-ratio 压缩项和概率压缩比例诊断 |
| `src/models/ea_gmib.py` | 新增 `selected_message` 图读出路径 |
| `src/stage3/trainer_ours.py` | 新增 v2 配置读取和扩展 logging |
| `src/stage3/analyze_ours_threshold_sweep.py` | 新增 Ours-v1 threshold sweep 诊断脚本 |
| `src/tests/test_stage3_gmib.py` | 更新 GM-IB 单测，覆盖 target-ratio 路径 |

新增配置：

| 配置文件 | 用途 |
|---|---|
| `configs/stage3_smoke_ours_v2_chemprotsent.json` | ChemProtSent 本地 v2 smoke |
| `configs/stage3_smoke_ours_v2_cdrintra.json` | CDRIntra 本地 v2 smoke |
| `configs/stage3_full_ours_v2_chemprotsent_seed42.json` | ChemProtSent 5090 v2 full run |
| `configs/stage3_full_ours_v2_cdrintra_seed42.json` | CDRIntra 5090 v2 full run |

已通过验证：

- `compileall` 通过。
- `src/tests/test_stage3_gmib.py` 通过。
- `src/tests/test_stage3_modules.py` 通过。
- 旧 Ours smoke 兼容性通过。
- Ours-v2 ChemProtSent smoke 通过。
- Ours-v2 CDRIntra smoke 通过。
- Ours-v2 full 配置本地 override 验证通过。

## 9. 后续 5090 full run 计划

下一步在 5090 上运行 Ours-v2 full run：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/stage3/run_ours_full.py --config configs/stage3_full_ours_v2_chemprotsent_seed42.json
```

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/stage3/run_ours_full.py --config configs/stage3_full_ours_v2_cdrintra_seed42.json
```

跑完后回传：

- `best_metrics.json`
- `metrics.json`
- `metrics_history.json`
- `run_meta.json`
- 如需继续诊断，回传 `best_model/`

## 10. 判断标准

Ours-v2 full run 完成后，我们将从两个层面判断是否成功。

### 10.1 性能指标

| 数据集 | 最低目标 | 理想目标 |
|---|---:|---:|
| ChemProtSent | 超过 B5 0.896106 | 接近或超过 B4 0.901083 |
| CDRIntra | 超过 B5 0.775451 | 明显超过 Ours-v1 0.776028 |

### 10.2 机制指标

Ours-v2 不仅要看 mapped exact，还必须看结构选择是否健康：

- `eval_prob_compression_ratio` 应接近 0.5，至少不能塌到 0 或 1。
- `eval_compression_ratio` 不应长期为 0。
- `eval_target_ratio_loss` 应维持在较小范围。
- `metrics_history.json` 中不应出现长期 hard selection 全空。

如果 Ours-v2 性能提升，同时压缩比例健康，则可以将其作为正式 Ours 主结果。若性能没有提升，但机制指标健康，则说明结构路径修复有效，下一步应调目标比例或 beta。若性能和机制都不理想，则需要重新审视 GM-IB 与生成器之间的耦合方式。

## 11. 阶段性结论

本轮 GM-IB 实验给出了一个重要教训：性能小幅提升并不等于结构选择机制真正生效。Ours-v1 在 CDRIntra 上略超 B5，但 threshold sweep 表明它的 hard selection 对预测几乎没有影响，解释性不足。Ours-v2 针对这一问题进行了结构性修复，让 `z_ij` 直接参与 selected-edge message passing，并用 target-ratio loss 控制概率压缩比例。

因此，当前阶段的主线不是“继续微调 threshold”，而是等待 Ours-v2 full run，判断新的结构路径是否能同时带来性能收益和可解释的压缩行为。
