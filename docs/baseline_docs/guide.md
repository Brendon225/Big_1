# 阶段二傻瓜式实验指南（Baseline 搭建与双层验证）

## 0. 文档定位

本指南只覆盖**阶段二**，不讨论阶段三以后的 GM-IB、双视角融合、正式消融等内容。阶段一已经完成数据准备、依存解析、实体感知图粗化、主实验视图冻结、DataLoader 打通与关键 bug 修复，因此现在可以直接进入 baseline 搭建。

本指南的目标不是讲概念，而是让实验室小白同学也能照着一步步把阶段二跑起来。

---

## 1. 阶段二到底要做什么

阶段二不再只是验证“依存信息是否有用”，而是做一个**双层验证**：

1. **结构有效性验证**：依存结构值不值得注入生成式模型？
2. **实体折叠早期验证**：同样是注入结构，使用粗化后的结构图是否会更有效？

因此，本阶段建议做 5 组 baseline：

* **B0**：Text-only
* **B1-raw**：原始依存树线性化拼接
* **B2-raw**：原始依存图上的最短依存路径（SDP）线性化拼接
* **B1-coarse**：粗化依存图线性化拼接
* **B2-coarse**：粗化图上的最短路径线性化拼接

原实验计划中，阶段二原本只安排了 B0 / B1 / B2，并要求先确认如果 B1/B2 比 B0 没提升，就要重新审视依存注入方式；同时，正式消融实验 A1-A5、分桶分析和 case study 仍然放在后续阶段。

**本指南的改动原则是：**

* 不推翻原计划；
* 保留阶段二“先看依存是否值得注入”的职责；
* 同时把阶段一已经做好的 `coarsened` 数据拉进来，做一个轻量的实体折叠早期验证。

---

## 2. 先记住这 6 条铁律

### 2.1 只用句级主实验视图，不要直接拿全文级数据做主结论

阶段一已经明确冻结主实验视图为：

* `same-sentence / local evidence`
* 负类统一为 `NO_RELATION`
* 主实验优先数据集为 `ChemProtSent` 和 `CDRIntra`

原因很简单：依存、SDP、图粗化、结构线性化本质上都是**句内结构**。如果你直接上全文级 `CDR / ChemProt`，就会混入跨句、长文截断、文档级证据聚合等问题，最后会把“结构有没有用”这个问题搅浑。

### 2.2 正式顺序：先 ChemProtSent，再 CDRIntra，最后再考虑 DDI

阶段一交接和归档文档都明确推荐：

1. `data/experiment_views/coarsened/ChemProtSent_*.json`
2. `data/experiment_views/coarsened/CDRIntra_*.json`
3. `data/coarsened/DDI_*.json`

### 2.3 阶段二不是正式消融

阶段二中的 `B1-raw vs B1-coarse`、`B2-raw vs B2-coarse` 只是**early probe（早期探针）**，用于先看趋势。
真正正式写进论文主结论的 RQ2 证据，后面仍然依赖：

* A1 `w/o Entity Coarsening`
* 按实体词数分桶
* case study 中的结构可视化对比

### 2.4 三种输入以外的东西尽量别动

做对比时，除了 `dep_view` 和 `dep_form`，其他条件尽量保持一致：

* 同一数据切分
* 同一模型
* 同一 target schema
* 同一训练轮数
* 同一随机种子列表

否则你后面没法判断性能差异来自“结构”，还是来自别的改动。

### 2.5 阶段二先回答“值不值得做”，不是“做到极致”

阶段二像打地基，不像装修。
先看方向对不对，再决定后面是否大规模投入。

### 2.6 出问题先看数据接口和输入文本，不要一上来怀疑模型

阶段一已经说明：

* `BioREDataset` 可直接读取新主实验视图
* `NO_RELATION` 已能识别
* `use_dep=False / True` 已能工作
* DataLoader smoke test 已通过

所以阶段二初期出问题，大概率不是“理论错了”，而是：

* 输入构造错了
* 结构串格式错了
* label 映射错了
* 数据文件路径读错了

---

## 3. 当前有哪些可直接使用的数据

## 3.1 主实验优先数据

### ChemProtSent

* 目录：`data/experiment_views/coarsened/ChemProtSent_*.json`
* 定义：同一句中的 `CHEMICAL × GENE-*`
* 正类：保留原标签 `CPR:3/4/5/6/9`
* 负类：同句共现但不在 gold 正例中的候选对，标为 `NO_RELATION`
* 当前状态：已完成 `processed -> parsed -> coarsened`，能被当前 DataLoader 直接读取

### CDRIntra

* 目录：`data/experiment_views/coarsened/CDRIntra_*.json`
* 定义：同一句中的 `Chemical × Disease`
* 正类：若 `(Chemical_MESH, Disease_MESH)` 出现在文档级 gold `CID` 中
* 负类：否则为 `NO_RELATION`
* 说明：它是句级子任务，不等于原始 document-level CDR 全任务，但更适合回答“依存知识是否有用”

## 3.2 补充数据

### DDI

* 目录：`data/coarsened/DDI_*.json`
* 说明：当前仍未额外构建 `DDISent`
* 训练注意：训练集必须做负例下采样；dev/test 保持原始分布

---

## 4. 阶段二推荐实验顺序

### 第 1 轮：只做最小冒烟

目标：确认数据能读、输入能构造、loss 能下降、模型能正常输出 label。

建议顺序：

1. `ChemProtSent`
2. `B0`
3. 单个随机种子
4. 少量 epoch

为什么先从 ChemProtSent 开始？
因为它的句级主实验视图规模最大，且原始 gold 正例中约 99% 具有同句支撑，更适合做阶段二第一块试金石。

### 第 2 轮：跑通原计划的原始三组

目标：保留原阶段二的 sanity check 价值。

顺序：

1. `ChemProtSent + B0`
2. `ChemProtSent + B1-raw`
3. `ChemProtSent + B2-raw`

### 第 3 轮：加入粗化结构早期验证

目标：让实体折叠尽早露面，看趋势。

顺序：

1. `ChemProtSent + B1-coarse`
2. `ChemProtSent + B2-coarse`

### 第 4 轮：迁移到第二个主数据集

顺序：

1. `CDRIntra + B0`
2. `CDRIntra + B1-raw`
3. `CDRIntra + B2-raw`
4. `CDRIntra + B1-coarse`
5. `CDRIntra + B2-coarse`

### 第 5 轮：正式跑多种子

所有正式结果建议跑 3 个随机种子，并报告均值和标准差。

---

## 5. 五组 baseline 的定义

## 5.1 B0：Text-only

### 输入

只包含原句 + 实体标记，不加任何 DEP 结构。

### 作用

这是纯文本 baseline，也是后面所有对比的零刻度线。

### 你要回答的问题

* 不加结构时，BioBART 能做到什么水平？
* 后面所有结构注入是否真的比纯文本更强？

---

## 5.2 B1-raw：原始依存树线性化拼接

### 输入

原句 + 实体标记 + 原始依存树线性化结果。

### 作用

这是最直接的结构注入法，用来回答“整棵原始依存树直接拼进去，有没有用”。

### 你要回答的问题

* 原始整树是否优于纯文本？

---

## 5.3 B2-raw：原始图上的最短依存路径（SDP）

### 输入

原句 + 实体标记 + 原始依存图上的最短路径线性化结果。

### 作用

这是更压缩、更聚焦的结构注入方式。

### 你要回答的问题

* 只保留关键路径，是否比整树更好？
* 原始图上，整树式注入和路径式注入谁更稳？

---

## 5.4 B1-coarse：粗化依存图线性化拼接

### 输入

原句 + 实体标记 + 粗化后的依存图线性化结果。

### 作用

这是实体折叠第一次正式进入阶段二。
阶段一已经完成实体感知图粗化，并生成了 `coarse_tokens / coarse_heads / coarse_labels / coarse_*` 等字段。

### 你要回答的问题

* 同样是整树线性化，去掉实体内部噪声弧之后，结构是否更适合生成模型？

---

## 5.5 B2-coarse：粗化图上的最短路径

### 输入

原句 + 实体标记 + 粗化图上的最短路径线性化结果。

### 作用

这是最建议重点观察的一组。
因为核心假设之一就是：多词实体内部噪声弧会污染关系信号；实体折叠后，路径会更短、更纯、更接近真正的关系支撑结构。

### 你要回答的问题

* 同样是走最短路径，粗化后是否比原始路径更有效？

---

## 6. 建议的输入模板

下面给的是**推荐模板**，不是死规定。核心原则是：
**五组 baseline 之间，除了 DEP 部分，其他内容尽量保持一致。**

## 6.1 B0

```text
[E1] aspirin [/E1] inhibits [E2] gene expression [/E2] .
```

## 6.2 B1-raw

```text
[E1] aspirin [/E1] inhibits [E2] gene expression [/E2] .
[DEP] aspirin -nsubj-> inhibits ; inhibits -obj-> expression ; expression -compound-> gene_expression [/DEP]
```

## 6.3 B2-raw

```text
[E1] aspirin [/E1] inhibits [E2] gene expression [/E2] .
[DEP] [E1] -nsubj-> inhibits -obj-> [E2] [/DEP]
```

## 6.4 B1-coarse

```text
[E1] aspirin [/E1] inhibits [E2] gene expression [/E2] .
[DEP] [E1] -nsubj-> inhibits ; inhibits -obj-> [E2] [/DEP]
```

## 6.5 B2-coarse

```text
[E1] aspirin [/E1] inhibits [E2] gene expression [/E2] .
[DEP] [E1] -nsubj-> inhibits -obj-> [E2] [/DEP]
```

### 注意

1. 粗化后的多词节点建议统一用下划线连接。
2. ROOT/self-loop 不要当有效弧写进结构串。
3. 如果后面发现 target schema 太复杂，阶段二建议先用**只生成 relation label**的最小任务形式，先保证 baseline 对比干净。

---

## 7. 代码层面的最小实现建议

阶段一已经把数据接口打通，但训练骨架还没开始，所以阶段二最重要的工程动作是：
**用最小代价搭一套统一可切换的 baseline 框架。**

## 7.1 推荐开关

```python
dep_view = "none"    # none | raw | coarse
dep_form = "none"    # none | tree | sdp
dataset_name = "ChemProtSent"   # ChemProtSent | CDRIntra | DDI
target_mode = "relation_only"
seed = 42
```

## 7.2 推荐最小模块

* `dataset.py`：读取样本
* `input_builder.py`：构造 encoder input / decoder target
* `trainer.py`：训练与验证
* `evaluator.py`：算指标
* `run_exp.py`：读配置并启动

## 7.3 为什么一定要这样拆

因为原计划里，阶段二后面还要接阶段三的 B3/B4/B5、GM-IB、Ours、B6、以及阶段四的正式消融。阶段二如果写成一堆复制粘贴脚本，后面会很痛苦。

---

## 8. 设备怎么分工

你们当前有两套环境：

* 笔记本：32G 内存、8G 4070、13代 i7
* 主机：5070

### 推荐分工

#### 笔记本

只做：

* 数据抽样检查
* DataLoader 联调
* 输入文本检查
* 小规模 smoke run
* 结果可视化和错误分析

#### 主机

优先做：

* 正式训练
* 多组 baseline
* 多随机种子实验
* best checkpoint 保存

### 原因

阶段二真正耗资源的不是“构造输入”，而是“多组实验 + 多随机种子”。
所以笔记本像厨房试菜，主机像正式开席。

---

## 9. 每一轮实验开始前必须检查的清单

## 9.1 数据层检查

* [ ] 读到的是否是 `ChemProtSent` / `CDRIntra` 的 sentence-level 主实验视图？
* [ ] 是否确认 `NO_RELATION` 已被当作负类？
* [ ] 若使用 DDI，是否只对 `train` 做了负采样，而没有污染 `dev/test`？

## 9.2 结构层检查

* [ ] `raw` 模式下，是否使用 `dep_heads / dep_labels`？
* [ ] `coarse` 模式下，是否使用 `coarse_heads / coarse_labels`？
* [ ] 是否排除了 ROOT/self-loop？
* [ ] 多词节点是否统一格式？

## 9.3 输入层检查

* [ ] B0 没有 DEP 段
* [ ] B1/B2 有 DEP 段
* [ ] raw 与 coarse 的差异只出现在结构串，不出现在其他地方
* [ ] 实体标记方式在五组里完全一致

## 9.4 训练层检查

* [ ] seed 是否记录
* [ ] checkpoint 保存路径是否正确
* [ ] dev 集上是否能输出合法 label
* [ ] loss 是否正常下降

---

## 10. 每组实验做完后必须记录的指标

阶段二至少记录下面这些：

### 主指标

* Micro-F1
* Macro-F1

### 辅助指标

* Precision
* Recall

### 生成质量指标

* Exact Match Rate
* Invalid Output Rate

### 结构效率指标

* Avg. Retained Arcs
* Compression Ratio

### 最低要求

至少要把下面三张表做出来：

#### 表 1：五组 baseline 总表

| Dataset | Model | Micro-F1 | Macro-F1 |  P |  R | EM | Invalid |
| ------- | ----: | -------: | -------: | -: | -: | -: | ------: |

#### 表 2：结构效率表

| Dataset | Model | Avg. Retained Arcs | Compression Ratio | Avg. Input Length |
| ------- | ----: | -----------------: | ----------------: | ----------------: |

#### 表 3：一句话结论表

| 对比                  | 结论   |
| ------------------- | ---- |
| B1-raw vs B0        | 是否更好 |
| B2-raw vs B1-raw    | 是否更好 |
| B1-coarse vs B1-raw | 是否更好 |
| B2-coarse vs B2-raw | 是否更好 |

---

## 11. 推荐的正式执行节奏

## Day 1：只做 B0 冒烟

### 任务

* 读 `ChemProtSent`
* 跑通 `B0`
* 看 loss、看输出、看标签是否合法

### 完成标准

* 模型能训练
* dev 不报错
* 输出不是乱码
* 能保存 checkpoint

---

## Day 2：把原计划三组跑通

### 任务

* `ChemProtSent + B1-raw`
* `ChemProtSent + B2-raw`

### 完成标准

* 能稳定训练
* 能正常验证
* 有第一版对比结果

---

## Day 3：加入实体折叠早期验证

### 任务

* `ChemProtSent + B1-coarse`
* `ChemProtSent + B2-coarse`

### 完成标准

* 五组结果齐
* 形成第一版趋势判断：

  * 结构有没有用？
  * SDP 是否比整树更好？
  * 粗化是否初步有益？

---

## Day 4：迁移到 CDRIntra

### 任务

* 按同样流程跑 `CDRIntra`

### 完成标准

* 第二个数据集的五组结果齐

---

## Day 5~6：补多种子正式结果

### 任务

* 5 组 × 2 数据集 × 3 seeds

### 完成标准

* 输出均值和标准差
* 形成阶段二正式结果表

---

## 12. 常见报错与排查顺序

## 12.1 loss 不下降

先查：

1. label 映射是否正确
2. target 是否过复杂
3. 输入是否被截断得太狠
4. B0 是否也不下降

### 判断逻辑

如果 B0 都不下降，先别怀疑结构；先怀疑训练和数据。

---

## 12.2 B1/B2 比 B0 还差很多

先查：

1. DEP 串是否太长
2. ROOT/self-loop 是否混进来了
3. 结构串格式是否混乱
4. raw / coarse 是否误用错字段

原计划明确说过：如果 B1/B2 比 B0 没提升，需要重新审视依存注入方式。

---

## 12.3 coarse 版本比 raw 还差

先查：

1. `coarse_heads / coarse_labels` 是否真的被正确读取
2. 多词实体是否被正确折叠
3. 线性化是否把粗化图写坏了
4. 是否因为样本截断触发了退化逻辑却没注意到

阶段一已经提示：当正文窗口截断且 `dep_view == coarse` 时，结构会自动退化到窗口内 raw dependency view。

---

## 12.4 DDI 跑不动或表现很怪

先查：

1. 有没有错误地把 dev/test 也做了负采样
2. 输入是否过长
3. 是否错误地把 DDI 当作阶段二主结论数据集

阶段一已经明确：DDI 当前更适合作为补充验证，不适合作为最先启动的主结论来源。

---

## 13. 阶段二完成标准

满足下面 6 条，就可以判定阶段二完成：

* [ ] `ChemProtSent` 上 B0、B1-raw、B2-raw、B1-coarse、B2-coarse 全部跑通
* [ ] `CDRIntra` 上同样五组全部跑通
* [ ] 每组至少有 3 个 seeds 的结果
* [ ] 五组 baseline 总表完成
* [ ] 结构效率表完成
* [ ] 能写出一页分析结论，回答下面三个问题：

  * 结构是否整体有用？
  * 路径是否优于整树？
  * 实体折叠是否初步有益？

---

## 14. 阶段二结束后，你应该得到什么

不是一句“模型训完了”，而是下面这些具体产物：

1. 一套统一可切换的 baseline 代码框架
2. 两个主数据集上的五组对比结果
3. 一份阶段二分析报告
4. 一个明确判断：

   * 后面是否值得继续做复杂结构建模？
   * 实体折叠是否值得进入正式消融主线？

如果阶段二做完，你已经能清楚回答这两个问题，那么这一步就不是“跑了几个 baseline”，而是真正把后面三周的方向照亮了。

---

## 15. 一句话总纲

**阶段二不是为了把所有花活都做出来，而是为了用最小代价确认两件事：结构值不值得做，实体折叠值不值得继续做。

