# Stage 4 实验指导指南：论文标准评估、F1 指标与最终分析

日期：2026-05-27

## 1. Stage4 定位

Stage4 的目标不是继续发明新模型，而是把 Stage3 已经确定的模型路线转化为论文可用的最终实验结果。

Stage3 已经完成的核心工作包括：

- 建立图增强数据链路。
- 完成 B3/B4/B5 单视角与双视角诊断实验。
- 完成 GM-IB/Ours 探索，并确认 hard compression 机制当前不适合作为最终主线。
- 实现 B7 gated dual-view fusion。
- 完成 B4/B5/B7 在 ChemProtSent 与 CDRIntra 上的三种子 full run。
- 得到阶段三最终结论：B7 在 ChemProtSent 上三种子最优；B5 在 CDRIntra 上主指标略优，B7 更稳定。

因此，Stage4 应该进入“最终评估与论文归档”阶段，重点解决以下问题：

1. 用关系抽取论文通用指标 F1 重新评估最终模型。
2. 在 test split 上生成最终结果，而不是只依赖 dev split 的 `mapped_exact`。
3. 建立最终论文主表、消融表、稳定性表和错误分析。
4. 将 Stage3 的模型发现转化为论文中的实验叙事。

一句话概括：

```text
Stage3 确定模型路线；Stage4 固化论文结果。
```

## 2. 为什么 Stage4 必须补 F1

Stage3 使用 `mapped_exact` 作为主指标，这是合理的阶段性选择，因为我们的模型是生成式 BioBART，需要先确认生成标签是否能被正确映射到合法关系标签。

但在关系抽取论文中，最终主表通常使用：

- Precision
- Recall
- Micro-F1
- Macro-F1
- Per-class F1

尤其是 biomedical relation extraction 场景，`NO_RELATION` 负类数量通常较多。如果把 `NO_RELATION` 纳入 micro-F1，指标可能被负类主导，无法真实反映正类关系抽取能力。

因此，Stage4 推荐采用以下最终指标口径：

| 指标 | 是否作为主指标 | 说明 |
|---|---|---|
| micro-F1 excluding `NO_RELATION` | 是 | 论文主指标，衡量正类关系整体抽取能力 |
| macro-F1 excluding `NO_RELATION` | 是 | 衡量各正类关系的平均表现，关注类别不平衡 |
| precision excluding `NO_RELATION` | 是 | 正类预测精度 |
| recall excluding `NO_RELATION` | 是 | 正类召回率 |
| per-class F1 | 是 | 用于细粒度错误分析 |
| mapped_exact | 辅助 | 保留 Stage3 可比性 |
| invalid_rate | 辅助 | 检查生成式输出合法性 |

建议论文主表以 `micro-F1` 为第一指标，`mapped_exact` 放在辅助列或附录表中。

## 3. Stage4 与原始实验计划的关系

原始规划中提出了五个研究问题：

| RQ | 原始问题 | 当前状态 | Stage4 任务 |
|---|---|---|---|
| RQ1 | 依存句法信息是否对生成式 BioRE 有帮助 | Stage2/Stage3 已初步回答 | 用 test F1 最终确认 |
| RQ2 | 实体感知图粗化是否有效 | Stage1/Stage2 已初步回答 | 在最终表中保留 Stage2 B2-coarse 作为参考 |
| RQ3 | 多视角是否优于单视角 | Stage3 已回答 | 用 B4/B5/B7 的 F1 消融表确认 |
| RQ4 | GM-IB 是否优于简单拼接 | Stage3 得到负结果 | 简写为探索性负结果，不作为主线 |
| RQ5 | 不同句长、实体复杂度下表现如何 | 尚未系统完成 | Stage4 重点补鲁棒性和错误分析 |

Stage4 的主要价值在于补齐 RQ1/RQ3 的论文级 F1 证据，并重点完成 RQ5。

## 4. Stage4 基本原则

Stage4 建议遵守以下原则：

1. 冻结模型结构，不再继续修改 B7 或 GM-IB。
2. 不重新训练已有 Stage3 full run，优先加载 best checkpoint 进行 test evaluation。
3. dev split 只用于选模和早停解释，test split 用于最终论文结果。
4. F1 指标必须基于 `mapped_pred` 与 gold label，而不是 raw generated string。
5. 所有最终表格都使用 seed 42/123/456 的 mean ± std。
6. GM-IB 只作为探索性结果简要记录，不再投入大规模调参。
7. 所有预测文件、指标文件和报告文件都要可复现、可回溯到 checkpoint。

## 5. 推荐目录结构

Stage4 推荐新增如下目录与文件：

```text
docs/stage4_experiments_docs/
  guide.md
  stage4_final_eval_report.md
  stage4_error_analysis_report.md
  stage4_paper_tables.md

outputs/stage4_predictions/
  chemprotsent/
  cdrintra/

checkpoints/
  stage4_f1_summary.csv
  stage4_f1_summary.md
```

其中：

| 文件 | 作用 |
|---|---|
| `guide.md` | 当前 Stage4 执行指南 |
| `stage4_final_eval_report.md` | 最终 F1 评估报告 |
| `stage4_error_analysis_report.md` | 错误分析、分桶分析、案例分析 |
| `stage4_paper_tables.md` | 可直接搬入论文的表格版本 |
| `stage4_f1_summary.csv` | 机器可读的最终指标汇总 |
| `outputs/stage4_predictions/` | 保存每个模型、数据集、seed 的预测结果 |

## 6. 推荐评估对象

Stage4 不需要把所有历史模型都重新评估一遍。建议分成主表模型、消融模型和附录模型。

### 6.1 主表模型

主表建议包括：

| 模型 | 原因 |
|---|---|
| Stage2 B2-coarse | 阶段二最佳结构线性化 baseline |
| B4 semantics-only | 单语义视角代表 |
| B5 dual-view concat | 简单双视角代表，CDRIntra 强 |
| B7 gated dual-view | Stage3 最终主模型 |

### 6.2 可选附录模型

附录可包括：

| 模型 | 是否必须 | 说明 |
|---|---|---|
| B3 syntax-only | 可选 | 如果需要完整单视角消融，则补充 |
| Ours-v1 | 可选 | GM-IB 探索性结果 |
| Ours-v2 | 可选 | GM-IB 修复尝试，不建议主表强调 |

如果时间有限，优先完成 B2-coarse、B4、B5、B7。

## 7. Stage4 任务拆解

### 7.1 Task 1：建立统一 F1 evaluation 脚本

目标：加载已有 best checkpoint，对 dev/test split 生成预测，并计算论文标准指标。

输入：

| 输入 | 示例 |
|---|---|
| checkpoint dir | `checkpoints/stage3_full_b7_chemprotsent_seed42` |
| config | `configs/stage3_full_b7_chemprotsent_seed42.json` |
| eval file | `data/experiment_views/coarsened/ChemProtSent_test.json` |
| label set | 数据集关系标签集合 |

输出：

| 输出 | 内容 |
|---|---|
| predictions jsonl/csv | sample id、gold label、raw pred、mapped pred、是否正确 |
| metrics json | precision、recall、micro-F1、macro-F1、mapped_exact、invalid_rate |
| confusion matrix | 每类关系混淆情况 |

评估逻辑要求：

1. 使用 `label_rerank` 保证输出尽量映射到合法标签。
2. 计算 F1 时排除 `NO_RELATION`。
3. 同时保留包含 `NO_RELATION` 的 accuracy/mapped_exact，方便与 Stage3 对齐。
4. 每个 checkpoint 单独输出预测文件，避免覆盖。

### 7.2 Task 2：完成 test split 三种子评估

推荐评估顺序：

```text
1. B7 ChemProtSent seed42/123/456
2. B7 CDRIntra seed42/123/456
3. B5 CDRIntra seed42/123/456
4. B5 ChemProtSent seed42/123/456
5. B4 ChemProtSent seed42/123/456
6. B4 CDRIntra seed42/123/456
7. Stage2 B2-coarse 两个数据集 seed42/123/456
```

这样排序的原因是：

- 先验证 Stage3 最终主模型 B7。
- 再验证 CDRIntra 上主指标最强的 B5。
- 最后补齐 B4 与 Stage2 baseline。

### 7.3 Task 3：生成最终 F1 汇总表

最终表建议格式：

| Dataset | Model | P | R | Micro-F1 | Macro-F1 | mapped exact | invalid rate |
|---|---|---:|---:|---:|---:|---:|---:|
| ChemProtSent | B2-coarse | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| ChemProtSent | B4 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| ChemProtSent | B5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| ChemProtSent | B7 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| CDRIntra | B2-coarse | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| CDRIntra | B4 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| CDRIntra | B5 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |
| CDRIntra | B7 | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std | mean |

表格排序建议：

1. Stage2 baseline
2. Single-view model
3. Naive dual-view model
4. Gated dual-view model

这条排序能自然呈现从线性化结构到图增强双视角的演进过程。

### 7.4 Task 4：做显著性与稳定性分析

Stage4 至少需要完成稳定性分析；如果时间允许，再做显著性分析。

稳定性分析：

| 分析 | 目的 |
|---|---|
| 三种子 mean ± std | 判断结果是否稳定 |
| min/max | 判断是否由单个 seed 支撑 |
| seed-level ranking | 判断模型排序是否一致 |

显著性分析可选方案：

| 方法 | 适用场景 |
|---|---|
| paired bootstrap | 比较两个模型在同一 test set 上的 F1 差异 |
| approximate randomization | 关系抽取常用显著性检验 |
| McNemar test | 比较两个模型正确/错误样本集合差异 |

最低要求：如果暂时不做严格显著性检验，报告中也要明确说明当前比较基于三种子均值与标准差。

### 7.5 Task 5：完成错误分析

错误分析建议围绕“为什么双视角有效”和“为什么 B7/B5 在两个数据集上表现不同”展开。

#### 7.5.1 Per-class analysis

输出每个正类关系的：

- support
- precision
- recall
- F1
- 主要混淆对象

重点观察：

1. B7 是否提升某些低频关系。
2. B5 是否在 CDRIntra 正类召回上更高。
3. B7 是否减少正类误判为 `NO_RELATION`。

#### 7.5.2 Confusion matrix

每个数据集至少输出 B5 与 B7 的 confusion matrix。

重点看：

| 错误类型 | 解释价值 |
|---|---|
| positive -> NO_RELATION | 召回不足 |
| NO_RELATION -> positive | 假阳性 |
| positive class A -> positive class B | 关系语义混淆 |

#### 7.5.3 Bucket analysis

建议按以下维度分桶：

| 分桶维度 | 目的 |
|---|---|
| sentence length | 回答长句中结构是否更有用 |
| entity token length | 回答多词实体是否受益于粗化 |
| entity distance | 回答远距离实体是否更需要 syntax |
| dependency path length | 回答长依存路径中双视角是否更强 |
| relation type | 找出具体收益关系类型 |

这部分直接对应原始计划中的 RQ5。

#### 7.5.4 Case study

建议每个数据集选 3-5 个典型样本：

| 类型 | 示例 |
|---|---|
| B7 correct, B5 wrong | 展示 gated fusion 的收益 |
| B5 correct, B7 wrong | 诚实分析 B7 局限 |
| Stage2 wrong, Stage3 correct | 展示图增强收益 |
| positive 被预测为 NO_RELATION | 分析召回瓶颈 |

case study 不宜太多，关键是要能支撑论文叙事。

### 7.6 Task 6：B7 gate statistics 可视化

Stage3 已经记录了 B7 的 gate 均值：

| 数据集 | gate_sem mean | gate_syn mean |
|---|---:|---:|
| ChemProtSent | 0.497472 | 0.502528 |
| CDRIntra | 0.504926 | 0.495074 |

Stage4 可进一步补充：

- 每个 seed 的 gate 分布。
- 正确样本 vs 错误样本的 gate 均值。
- 正类样本 vs `NO_RELATION` 样本的 gate 均值。
- 不同句长/实体距离 bucket 下的 gate 均值。

注意：当前 gate 均值接近 0.5，不应过度解释为“某数据集明显偏语义/句法”。更稳妥的论文表述是：

```text
The learned gates remain balanced on average, suggesting that the model does not collapse to a single view and can maintain both semantic and syntactic signals during fusion.
```

## 8. Stage4 推荐执行顺序

建议按以下顺序推进：

### Step 1：冻结 Stage3 结果

确认以下文件存在并归档：

- `docs/stage3_model_docs/stage3_final_experiment_report.md`
- `docs/stage3_model_docs/stage3_multiseed_summary.md`
- `checkpoints/stage3_multiseed_summary.csv`

### Step 2：实现 F1 evaluation

先只对一个 checkpoint 做 smoke：

```text
B7 ChemProtSent seed42 on test split
```

确认输出中包含：

- predictions file
- micro-F1
- macro-F1
- precision
- recall
- mapped_exact
- invalid_rate

### Step 3：批量评估 B7/B5/B4

完成两个数据集、三个 seed 的 test split 评估。

### Step 4：补 Stage2 baseline F1

优先补 B2-coarse，因为这是阶段二最相关 baseline。

### Step 5：生成最终表格

输出：

- `checkpoints/stage4_f1_summary.csv`
- `docs/stage4_experiments_docs/stage4_final_eval_report.md`
- `docs/stage4_experiments_docs/stage4_paper_tables.md`

### Step 6：错误分析与可视化

优先做：

1. per-class F1
2. confusion matrix
3. sentence length bucket
4. entity distance bucket
5. B5 vs B7 case study

## 9. 结果解释预案

Stage4 可能出现几种情况，需要提前设定解释策略。

### 情况 A：F1 与 mapped_exact 排序一致

如果 ChemProtSent 上 B7 仍然最优，CDRIntra 上 B5/B7 接近，则论文主线非常稳定：

```text
B7 is the final unified model, with strongest performance on ChemProtSent and competitive stable performance on CDRIntra.
```

### 情况 B：F1 中 B7 全面优于 B5

这是最理想情况，可以更坚定地把 B7 写成最终模型。

论文表述：

```text
Gated dual-view fusion consistently outperforms single-view and naive concatenation baselines under F1-based evaluation.
```

### 情况 C：F1 中 CDRIntra 上 B5 明显优于 B7

这并不推翻 Stage3，而是说明 CDRIntra 更偏好直接保留双视角信息。

论文表述：

```text
On CDRIntra, naive dual-view concatenation achieves the highest F1, while gated fusion provides a more stable but slightly conservative alternative.
```

### 情况 D：F1 与 mapped_exact 差异很大

需要检查：

1. `NO_RELATION` 是否正确排除。
2. label mapping 是否一致。
3. rare classes 是否影响 macro-F1。
4. 是否存在大量 positive -> NO_RELATION 错误。

如果 F1 结论不同，应以 F1 为论文主结果，mapped_exact 作为生成式评估辅助指标。

## 10. Stage4 交付物清单

Stage4 完成时，应至少交付以下内容：

| 交付物 | 是否必须 | 说明 |
|---|---|---|
| F1 evaluation 脚本 | 必须 | 加载 checkpoint，输出预测与 F1 |
| `stage4_f1_summary.csv` | 必须 | 所有模型三种子指标 |
| `stage4_final_eval_report.md` | 必须 | 最终实验报告 |
| `stage4_paper_tables.md` | 必须 | 可直接写入论文的表格 |
| predictions 文件 | 必须 | 支撑错误分析与可复现 |
| per-class F1 | 必须 | 关系抽取论文常用分析 |
| confusion matrix | 推荐 | 错误诊断 |
| bucket analysis | 推荐 | 回答 RQ5 |
| case study | 推荐 | 支撑论文定性分析 |
| gate visualization | 可选 | 支撑 B7 解释性 |

## 11. Stage4 成功标准

Stage4 成功的最低标准：

1. B2/B4/B5/B7 在 ChemProtSent 和 CDRIntra 上都有三种子 test F1。
2. 主表包含 micro-F1、macro-F1、precision、recall、mapped_exact。
3. F1 计算口径清楚说明是否排除 `NO_RELATION`。
4. 至少完成 per-class F1 和 confusion matrix。
5. 最终报告能清楚回答：为什么 Stage3 选择双视角，为什么不继续 GM-IB，B7 的优势和局限是什么。

Stage4 理想标准：

1. 增加显著性检验或 bootstrap confidence interval。
2. 增加句长、实体距离、依存路径长度分桶分析。
3. 增加 3-5 个典型样例分析。
4. 将最终结果表整理成论文可直接使用的中英文版本。

## 12. 建议的论文叙事

Stage4 完成后，论文实验部分可以形成如下叙事：

1. 阶段二证明直接依存线性化有一定帮助，但收益依赖数据集。
2. 阶段三证明显式 syntax/semantics 双视角表示比线性化更有效。
3. B4/B5 说明不同数据集对语义和句法互补的需求不同。
4. GM-IB hard compression 的负结果说明离散边选择在当前生成式框架下不稳定。
5. B7 使用连续门控融合两个视角，在保持机制简洁的同时获得稳定性能。
6. Stage4 的 F1/test 结果将作为最终论文证据。

推荐最终贡献表述：

```text
We build a generation-oriented biomedical relation extraction framework that explicitly models syntax and semantics as complementary views. Through systematic diagnostics, we show that continuous gated fusion is a more stable way to inject dependency-aware structural information into BioBART than hard graph compression.
```

## 13. 立即下一步

下一步建议从最小闭环开始：

1. 实现一个只评估 `stage3_full_b7_chemprotsent_seed42` 的 test F1 脚本。
2. 确认 `NO_RELATION` 排除逻辑正确。
3. 保存 prediction 文件。
4. 与当前 dev `mapped_exact=0.906583` 做 sanity check。
5. 确认无误后，再批量跑所有 seed 和模型。

不要一开始就批量跑全部模型。先把一个 checkpoint 的 F1 链路打穿，这会节省大量排错时间。
