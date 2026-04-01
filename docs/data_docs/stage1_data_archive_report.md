# 阶段一数据技术归档报告

**文档定位**：阶段一的技术归档与追溯文档  
**适用对象**：项目核心成员、后续排错者、论文撰写者  
**目的**：在不保留 6 份分散旧文档的前提下，保留阶段一的关键技术细节、设计决策与最终产物

---

## 1. 阶段一的总目标

阶段一服务于研究内容一：

**验证什么样的依存知识对下游关系抽取模型有用。**

因此，阶段一数据工作的目标不是“通用数据清洗”，而是建立一套能直接支撑后续结构实验的数据体系。

阶段一最终交付包括四层内容：

1. 原始数据获取与统一格式化
2. 依存解析与实体感知图粗化
3. 数据统计、DataLoader、接口验证
4. 主实验视图冻结与 sentence-level 视图构建

---

## 2. 原始数据来源与当前保留形态

### 2.1 CDR

- 原始来源：BioC XML
- 原始任务：document-level Chemical-Disease 关系
- 当前保留形态：
  - 原始全文级视图：`data/coarsened/CDR_*.json`
  - 句级主实验视图：`data/experiment_views/coarsened/CDRIntra_*.json`

### 2.2 ChemProt

- 原始来源：HuggingFace `bigbio/chemprot`
- 原始任务：化学物质-蛋白质关系抽取
- 当前保留形态：
  - 原始全文级视图：`data/coarsened/ChemProt_*.json`
  - 句级主实验视图：`data/experiment_views/coarsened/ChemProtSent_*.json`

### 2.3 DDI

- 原始来源：BRAT
- 原始任务：药物相互作用抽取
- 当前保留形态：
  - 原始全文级视图：`data/coarsened/DDI_*.json`

说明：

- DDI 目前仍未额外构建 `DDISent`
- 训练时必须做负例下采样

---

## 3. 原始主线处理流程

主线目录：

- `data/processed/`
- `data/parsed/`
- `data/coarsened/`

流程：

```text
raw -> processed -> parsed -> coarsened
```

### 3.1 processed

完成内容：

- 不同数据源统一成 JSONL
- 每条记录围绕“一个实体对实例”组织

核心字段：

- `sentence`
- `tokens`
- `entity1`
- `entity2`
- `relation`
- `is_positive`

### 3.2 parsed

完成内容：

- SciSpaCy `en_core_sci_lg` 依存解析
- 用解析 token 替换原始简单分词
- 用字符偏移重算实体 token 边界

新增字段：

- `dep_heads`
- `dep_labels`

### 3.3 coarsened

完成内容：

- 实体感知图粗化
- 多词实体折叠
- 实体内部噪声弧删除

新增字段：

- `coarse_tokens`
- `coarse_heads`
- `coarse_labels`
- `coarse_e1_idx`
- `coarse_e2_idx`
- `coarse_node_types`
- `coarse_status`

---

## 4. 阶段一关键 bug 与修复结论

### 4.1 token 与依存长度不一致

问题：

- 早期 `tokens` 仍来自简单分词
- `dep_heads/dep_labels` 来自 SciSpaCy
- 导致长度不一致

修复：

- 直接用 SciSpaCy token 覆盖 `tokens`
- 重新计算实体 token 边界

### 4.2 重叠实体导致图粗化错误

问题：

- 某些 biomedical tokenization 情况下，两个实体会映射到同一个 token
- 粗化时一个实体会覆盖另一个实体

修复：

- 引入 `overlap_passthrough`
- 共享节点类型记为 `entity1+entity2`
- 结构安全退化，不再损坏数据

### 4.3 DDI 负采样污染 dev/test

问题：

- 初版 DataLoader 默认对 `DDI train/dev/test` 都做负采样

修复：

- 现在只在 `DDI train` 默认启用 `1:3` 负采样
- `dev/test` 保持原始分布

### 4.4 长样本正文截断后结构仍来自全文

问题：

- 正文截断后，`DEP` 线性化仍然引用全文级结构

修复：

- 当正文窗口截断且 `dep_view == coarse` 时，自动退化为窗口内 `raw dependency view`

### 4.5 DDI 大文件整体读入内存

问题：

- 早期 Dataset 会 materialize 整个 DDI 训练集

修复：

- 改为字节偏移索引 + 按需读取
- 同时加入索引缓存

### 4.6 统计脚本口径错误

问题：

- ROOT/self-loop 被当成有效依存弧

修复：

- 统计时排除 self-loop
- 多词节点在线性化中下划线归一化

### 4.7 Collator 兼容性问题

问题：

- 旧式 `as_target_tokenizer()` 兼容性差

修复：

- 优先使用 `text_target`
- 回退到普通 tokenizer 编码

### 4.8 修复后抽样验证结论

在关键 bug 修复完成后，已对 `data/coarsened/` 执行 post-fix sample validation。

结论：

- 当前 `coarsened` 数据中已无阻塞性结构错误
- 抽样验证层面未再出现会阻止后续实验的 `fail`
- 剩余 warning 属于弱警告，主要用于误差分析与边界样本追踪，不影响后续 baseline 和主实验启动

---

## 5. 1.4 数据统计的最终结论

关键产物：

- `docs/data_docs/stats_1_4_report.md`
- `docs/data_docs/stats_1_4.json`

核心结论：

1. `CDR / ChemProt` 全文级样本偏长，但可控
2. `DDI` 全文级样本非常长，不能直接把全文正文 + 全文结构串同时送入模型
3. 原始全文级视图里，图粗化整体压缩率不高，原因是单词实体占多数
4. `DDI-false` 占比极高，训练时必须负采样

这部分统计支撑了后续 DataLoader 和主实验视图设计。

---

## 6. 1.5 DataLoader 与接口层最终状态

关键文件：

- `src/data/dataset.py`
- `src/data/dataloader.py`
- `src/data/collator.py`
- `src/tests/test_dataset.py`

当前能力：

- 可读取原始主线视图
- 可读取 sentence-level 主实验视图
- 支持 `use_dep=False / True`
- 能识别 `NO_RELATION`
- DDI 训练集支持默认负采样
- 大文件可按需读取
- smoke test 已通过

说明：

- 阶段一已经把数据接口打通
- 当前尚未开始训练骨架，这不属于阶段一问题

---

## 7. 主实验视图冻结

关键文档：

- `docs/data_docs/stage1_data_handoff_report.md`

冻结结论：

- 主实验视图：`same-sentence / local evidence`
- 负类标签：`NO_RELATION`
- 主实验数据集：
  - `ChemProtSent`
  - `CDRIntra`

原因：

- 依存、SDP、图粗化、结构线性化本质上是句内结构
- 若直接用全文级视图，会引入跨句与长文处理混杂因素

---

## 8. sentence-level 主实验视图构建结果

关键脚本：

- `src/preprocessing/build_experiment_views.py`
- `src/preprocessing/postprocess_experiment_views.py`

关键目录：

- `data/experiment_views/processed/`
- `data/experiment_views/parsed/`
- `data/experiment_views/coarsened/`

### 8.1 ChemProtSent

规模：

- train: `18608` 条
- dev: `11454` 条
- test: `16055` 条

正负比例：

- train: `4117 / 14491`

覆盖率：

- 原始 gold 正例中约 `99%` 具有同句支撑

### 8.2 CDRIntra

规模：

- train: `4844` 条
- dev: `5206` 条
- test: `5155` 条

正负比例：

- train: `1826 / 3018`

覆盖率：

- 文档级 gold relation 中约 `70%~75%` 具有同句支撑

### 8.3 粗化效果

当视图收缩到 sentence-level 后，图粗化更可观察：

- `ChemProtSent_*`：约 `-1.6% ~ -1.7%`
- `CDRIntra_*`：约 `-2.0% ~ -2.4%`

这比全文级视图更符合本研究内容一的实验直觉。

---

## 9. 当前推荐的后续实验使用顺序

### 9.1 第一优先级

- `data/experiment_views/coarsened/ChemProtSent_*.json`
- `data/experiment_views/coarsened/CDRIntra_*.json`

### 9.2 第二优先级

- `data/coarsened/DDI_*.json`

### 9.3 保留但不建议直接做主结论的数据

- `data/coarsened/ChemProt_*.json`
- `data/coarsened/CDR_*.json`

它们仍有价值，但更适合：

- 附录分析
- document-level 对照
- 追溯历史流程

---

## 10. 阶段一最终结论

阶段一现在已经形成了两条并行但互补的数据资产：

1. **原始主线数据**
   - 保留全文级/历史流程
   - 适合追溯与补充分析

2. **主实验 sentence-level 视图**
   - 更贴合“依存知识是否有用”的研究问题
   - 更适合作为后续 baseline 的首选数据

因此，从实验设计、数据质量、接口可用性三方面看：

**阶段一已经完成，并且已经为阶段二提供了可直接接手的高质量数据基础。**