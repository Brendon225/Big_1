# 阶段一数据交接报告

**文档定位**：给下一位接手同学快速建立全局认知  
**适用场景**：开始做阶段二 baseline、主实验、误差分析之前  
**阅读目标**：读完后知道该用哪些数据、这些数据是怎么来的、哪些坑已经修掉、接下来该怎么接手

---

## 1. 阶段一已经完成到什么程度

阶段一“数据准备与预处理”已经完成。

当前我们不只是拿到了原始数据，而是已经构建出一套可直接支撑后续关系抽取实验的数据基础设施，包括：

1. 三个原始数据集的获取与统一格式化
2. 依存解析与 token 对齐
3. 实体感知图粗化
4. 抽样验证与关键 bug 修复
5. 数据统计与 DataLoader 接口
6. 主实验视图冻结
7. sentence-level 主实验视图构建与后处理

结论：

**阶段一已经可以判定完成，后续可进入 baseline 搭建。**

---

## 2. 当前有哪些数据

### 2.1 原始主线数据

目录：

- `data/processed/`
- `data/parsed/`
- `data/coarsened/`

数据集：

- `CDR`
- `ChemProt`
- `DDI`

说明：

- 这是项目最早的数据主线
- 保留了全文级/文档级实体对视图
- 适合做追溯、补充分析、与旧流程兼容

### 2.2 当前推荐的主实验数据

目录：

- `data/experiment_views/processed/`
- `data/experiment_views/parsed/`
- `data/experiment_views/coarsened/`

数据集：

- `ChemProtSent`
- `CDRIntra`

说明：

- 两者都使用 `same-sentence / local evidence` 视图
- 两者都显式包含 `NO_RELATION`
- 这是当前最推荐用于第一部分论文主实验的数据视图

---

## 3. 当前主实验视图为什么这样设计

本研究内容一的核心问题是：

**什么样的依存知识对下游模型真正有用。**

因此主实验视图不能再用全文级实体对直接做主结论，因为那会混入：

- 多句 forest 问题
- 跨句 SDP 不可定义问题
- 长文本截断问题
- 文档级证据聚合问题

所以主实验正式冻结为：

- **same-sentence / local evidence**
- **给定局部证据 + 实体对，生成关系标签**
- **负类统一为 `NO_RELATION`**

对应数据集：

- `ChemProtSent`
- `CDRIntra`

---

## 4. 三个数据集当前应该怎么理解

### ChemProt

当前最推荐使用：

- `ChemProtSent`

定义：

- 同一句中的 `CHEMICAL × GENE-*`
- gold 正例保留原标签 `CPR:3/4/5/6/9`
- 同句共现但不在 gold 正例中的候选对标为 `NO_RELATION`

现状：

- 已完成 `processed -> parsed -> coarsened`
- 已能被当前 DataLoader 直接读取

### CDR

当前最推荐使用：

- `CDRIntra`

定义：

- 同一句中的 `Chemical × Disease`
- 若对应 `(Chemical_MESH, Disease_MESH)` 出现在文档级 gold `CID` 中，则为正例
- 否则为 `NO_RELATION`

说明：

- `CDRIntra` 是句级子任务，不等于原始 document-level CDR 全任务
- 它更适合回答“依存知识是否有用”

### DDI

当前继续使用：

- `data/coarsened/DDI_*.json`

说明：

- 训练时必须做负例下采样
- 当前还没有额外派生 `DDISent`
- 因此 DDI 更适合作为辅助数据集，而不是当前最核心的主结论来源

---

## 5. 阶段一做了哪些关键处理

完整链路：

```text
原始数据
  -> processed
  -> parsed
  -> coarsened
```

### 5.1 processed

完成了：

- 多数据集统一 JSONL 格式
- 统一字段：`sentence / tokens / entity1 / entity2 / relation / is_positive`

### 5.2 parsed

完成了：

- SciSpaCy 依存解析
- 用解析后的 token 替换早期简单分词
- 重算实体 token 边界

### 5.3 coarsened

完成了：

- 多词实体折叠为超级节点
- 删除实体内部噪声弧
- 生成 `coarse_*` 系列字段

---

## 6. 已经修掉的关键问题

后续做实验时，可以默认下面这些坑已经被处理过：

1. `tokens` 与依存长度不一致  
已修复。

2. 实体重叠时图粗化出错  
已修复，使用安全退化策略。

3. DDI 默认负例采样污染 `dev/test`  
已修复，现在默认只影响 `train`。

4. 长样本正文截断后，DEP 仍引用全文  
已修复，正文与结构已对齐。

5. DDI 大文件整体读入内存  
已修复，改为字节偏移索引 + 按需读取。

6. 统计脚本把 ROOT/self-loop 当有效弧  
已修复。

7. DataLoader smoke test 兼容性问题  
已修复。

8. 修复后抽样验证已完成  
已完成 post-fix sample validation，当前 `data/coarsened/` 已无阻塞性结构错误，可直接供后续实验使用；剩余 warning 仅属于弱警告，不影响实验启动。

---

## 7. 当前应该优先使用哪些目录

如果你现在开始做阶段二 baseline，推荐顺序是：

1. `data/experiment_views/coarsened/ChemProtSent_*.json`
2. `data/experiment_views/coarsened/CDRIntra_*.json`
3. `data/coarsened/DDI_*.json`

原因：

- 前两个最符合本研究内容一的主问题
- DDI 当前仍保留全文级视图，更适合作为补充验证

---

## 8. 当前数据接口是否可用

已验证：

- `BioREDataset` 可直接读取新主实验视图
- `NO_RELATION` 已被识别为负类标签
- `use_dep=False` 与 `use_dep=True` 都能工作
- DataLoader smoke test 已通过

说明：

- 数据层和输入层已经可供后续 baseline 使用
- 当前阶段还没有开始训练骨架，这不属于阶段一问题

---

## 9. 接下来你该怎么接手

建议顺序：

1. 先读本文件
2. 再看实验计划书 `docs/0_experiment_plan_v1.md`
3. baseline 优先从 `ChemProtSent` 和 `CDRIntra` 开始
4. DDI 作为第三个数据集补入

如果只记一条：

**后续主实验优先使用 `data/experiment_views/coarsened/`，不要直接拿全文级 `CDR / ChemProt` 做第一部分论文的主结论。**

---

## 10. 最终结论

阶段一已经不只是“把数据下好”，而是已经形成了：

- 可追溯的原始主线数据
- 可直接用于主实验的句级视图
- 经过修复和验证的数据接口

一句话概括：

**我们已经拥有了一套可以支撑后续 baseline 和结构实验的数据基础设施。**
