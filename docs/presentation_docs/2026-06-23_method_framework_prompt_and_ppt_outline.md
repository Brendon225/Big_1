# 生成式生物医学关系抽取实验汇报材料

日期：2026-06-23  
用途：明天约 30 分钟组会/导师汇报  
目标听众：不熟悉 NLP、生物医学关系抽取、生成式模型的小白听众  
核心基调：像讲故事一样解释“为什么做、怎么做、做出了什么结论”

---

## 0. 汇报总策略

### I: Insight

这次汇报的核心矛盾不是“实验细节太多”，而是：

> 听众不一定懂 NLP，但他们必须听懂我们为什么需要实体级、句法级、语义级三层知识增强。

所以不要把汇报做成“模型结构堆叠说明书”，而要做成一条故事线：

1. 生物医学文献太多，人读不过来。
2. 我们希望模型自动抽取实体关系。
3. 生成式模型会生成答案，但不一定真正理解实体间结构。
4. 直接把依存树塞给模型不稳定。
5. 因此我们设计了实体级去噪、句法级建模、语义级增强和门控融合。
6. 实验证明：结构知识要经过表示学习和融合，而不是简单拼接。

### S: Strategy

30 分钟建议分成 7 个篇章：

| 篇章 | 页数 | 时间 | 目标 |
|---|---:|---:|---|
| 1. 问题引入 | 1-7 | 4 min | 让小白知道 BioRE 在干什么 |
| 2. 生成式模型与痛点 | 8-14 | 5 min | 说明为什么 text-only 不够 |
| 3. 总体方法框架 | 15-20 | 5 min | 建立“三副眼镜”的故事 |
| 4. Stage1 数据基础 | 21-25 | 4 min | 说明实验不是空中楼阁 |
| 5. Stage2 baseline | 26-30 | 4 min | 说明直接拼接结构不稳定 |
| 6. Stage3/4 核心模型与结果 | 31-39 | 7 min | 展示双视角和门控融合的价值 |
| 7. 总结与下一步 | 40-42 | 1 min | 收束创新点和后续工作 |

### O: Operation

PPT 制作建议：

- 每页只讲一个点。
- 少放大段文字，多用流程图、对比图、类比图。
- 结果表只放关键数字，完整表放附录或口头说明。
- “依存句法”不要作为唯一主角，要放在“多层结构化知识增强”里面讲。
- GM-IB 作为探索性诊断，不作为最终主贡献。

---

## 1. 图片生成提示词：版本 B，严肃学术风格，英文标签

### 1.1 推荐直接使用的 GPT 图片生成提示词

```text
Create a clean academic method framework diagram for a biomedical relation extraction study based on a generative model.

Style:
- Professional academic conference figure
- White or very light gray background
- Clean vector-like blocks and arrows
- Modern scientific color palette: deep blue, teal, soft orange, slate gray
- High readability, minimal decoration
- 16:9 landscape layout
- Use English labels only
- Avoid dense text and avoid tiny fonts

Diagram content:
Show a left-to-right pipeline with the following modules:

1. Input:
   A biomedical sentence containing two highlighted entities.
   Label: "Biomedical Sentence + Entity Pair"

2. Entity-level processing:
   Show multi-word biomedical entities being collapsed into compact nodes.
   Label: "Entity-aware Graph Coarsening"
   Small annotation: "reduce entity-internal noise"

3. Two-view representation learning:
   Split the pipeline into two parallel branches.

   Top branch:
   Label: "Syntax View"
   Show a dependency graph with nodes and directed edges.
   Small annotation: "coarse dependency graph + GCN"

   Bottom branch:
   Label: "Semantics View"
   Show contextual biomedical embeddings.
   Small annotation: "PubMedBERT semantic representation"

4. Fusion:
   Merge the two branches into a central module.
   Label: "Gated Dual-view Fusion"
   Visual metaphor: a small adaptive gate or slider balancing syntax and semantics.

5. Generative extraction:
   Connect the fused representation to a generator block.
   Label: "BioBART Generator"
   Small annotation: "relation label generation"

6. Output:
   Show a structured prediction card.
   Label: "Predicted Relation"
   Example text: "chemical - relation - gene/disease"

Add a subtle bottom ribbon showing the experimental logic:
"Text-only Baseline -> Dependency Linearization -> Graph-based Dual-view Modeling"

Important constraints:
- The figure should look like a serious research paper method diagram, not a cartoon.
- Keep all labels short and spelled correctly.
- Do not include Chinese characters.
- Do not include equations except one optional simple gate icon.
- Do not include excessive decorative molecules or DNA backgrounds.
- Make the main contribution visually obvious: entity-aware coarsening, syntax view, semantics view, gated fusion, BioBART generation.
```

### 1.2 如果图片模型容易写错英文，使用这个“少文字版本”

```text
Create a clean academic method framework diagram for a generative biomedical relation extraction model.

Use a 16:9 landscape layout, white background, professional conference-paper style, clean vector-like boxes and arrows.

Show the following visual pipeline with only very short English labels:

Input -> Entity Coarsening -> Syntax View
                          -> Semantics View
Syntax View + Semantics View -> Gated Fusion -> BioBART -> Relation

Visual requirements:
- Input: biomedical sentence with two highlighted entity spans.
- Entity Coarsening: multi-word entities collapsed into single graph nodes.
- Syntax View: dependency graph with directed edges.
- Semantics View: biomedical embedding blocks.
- Gated Fusion: adaptive gate combining two branches.
- BioBART: encoder-decoder generator block.
- Output: relation prediction card.

Use blue for text/generative model, teal for syntax graph, orange for semantic representation, gray for data preprocessing.
Keep the design clean, academic, and easy to relabel in PowerPoint.
No Chinese text.
No long paragraphs.
No decorative background.
```

### 1.3 英文标签与中文替换建议

| 图片英文标签 | PPT 中可替换为中文 |
|---|---|
| Biomedical Sentence + Entity Pair | 生物医学句子与实体对 |
| Entity-aware Graph Coarsening | 实体感知图粗化 |
| reduce entity-internal noise | 降低实体内部结构噪声 |
| Syntax View | 句法视角 |
| coarse dependency graph + GCN | 粗化依存图 + 图编码 |
| Semantics View | 语义视角 |
| PubMedBERT semantic representation | PubMedBERT 领域语义表示 |
| Gated Dual-view Fusion | 门控双视角融合 |
| BioBART Generator | BioBART 生成器 |
| Predicted Relation | 关系预测结果 |
| Text-only Baseline | 纯文本基线 |
| Dependency Linearization | 依存结构线性化 |
| Graph-based Dual-view Modeling | 图增强双视角建模 |

### 1.4 图片讲解时的一句话

> 这张图可以理解为：我们给生成式模型配了三层辅助信息，先把长实体压缩成更干净的节点，再分别从句法结构和领域语义两个角度理解句子，最后用门控机制决定两种信息如何融合，并交给 BioBART 生成关系标签。

---

## 2. 42 页 PPT 大纲

### Slide 1. 标题页

**标题**：基于生成式模型的生物医学关系抽取研究  
**副标题**：从实体感知图粗化到语义-句法双视角融合  
**内容**：姓名、导师、日期、课题方向。  
**视觉建议**：背景用淡色医学文献/知识图谱风格图，不要太花。  
**讲述提示**：今天不直接从模型公式讲起，而是先讲一个问题：模型怎么从复杂医学句子里读出实体关系。

### Slide 2. 今天要讲一个什么故事

**标题**：今天的故事线  
**内容**：
- 医学文献里有大量实体关系。
- 生成式模型能生成关系标签，但理解结构不稳定。
- 我们尝试用实体、句法、语义三层知识增强模型。
- 最终验证哪类知识更有效、怎样注入更稳。
**视觉建议**：四段式故事时间轴。  
**讲述提示**：把整场汇报比作侦探破案：实体是嫌疑人，句子是案发现场，关系是最终结论。

### Slide 3. 为什么要做生物医学关系抽取

**标题**：医学文献太多，人读不过来  
**内容**：
- 药物、疾病、基因、蛋白质之间存在大量关系。
- 这些关系散落在论文、摘要和数据库文本中。
- 自动抽取可以服务知识图谱、药物发现、临床辅助分析。
**视觉建议**：左侧文献堆积，右侧知识图谱。  
**讲述提示**：如果把医学文献看成一座图书馆，关系抽取就是自动给书中知识画索引。

### Slide 4. 什么是关系抽取

**标题**：关系抽取到底抽什么  
**内容**：
- 输入：一句话 + 两个实体。
- 输出：这两个实体之间是什么关系。
- 示例：chemical 与 gene/protein 之间可能是 inhibition、activation 等关系。
**视觉建议**：一句英文医学句子，高亮 entity1/entity2，箭头指向 relation label。  
**讲述提示**：它不是简单找关键词，而是判断“谁和谁发生了什么关系”。

### Slide 5. 用生活例子理解关系抽取

**标题**：先用生活句子热身  
**内容**：
- 句子：“小明把药交给医生。”
- 实体：小明、药、医生。
- 关系：小明-交给-医生；药-被交付给-医生。
**视觉建议**：简单人物/物品关系图。  
**讲述提示**：小白听众先理解任务形式，再切回医学文本。

### Slide 6. 生物医学场景为什么更难

**标题**：医学句子不像日常句子那么直接  
**内容**：
- 实体名很长。
- 句子结构复杂。
- 关系触发词可能离实体很远。
- 有大量 NO_RELATION 负例。
**视觉建议**：一条长句中多个彩色实体和触发词。  
**讲述提示**：医学句子像绕了几道弯的病例描述，模型要从里面找到真正有用的线索。

### Slide 7. 本论文的大问题

**标题**：大问题：让生成式模型更懂实体关系  
**内容**：
- 论文题目：基于生成式模型的生物医学关系抽取研究。
- 当前第一部分：研究怎样利用外部结构化知识增强生成式 BioRE。
- 重点不是重新发明依存分析，而是研究结构知识如何服务生成式抽取。
**视觉建议**：论文总问题树，当前工作高亮第一分支。  
**讲述提示**：提前化解“依存句法是不是老方法”的疑问。

### Slide 8. 为什么选择生成式模型

**标题**：从分类到生成：为什么用 BioBART  
**内容**：
- 传统方法多把关系抽取做成分类任务。
- 生成式模型可以把关系抽取转化为文本生成。
- BioBART 适合生物医学文本生成与理解。
**视觉建议**：分类器 vs 生成器对比。  
**讲述提示**：分类器像选择题，生成式模型像让学生写答案。

### Slide 9. 生成式模型的优势和风险

**标题**：会写答案，不代表真懂结构  
**内容**：
- 优势：输出灵活、适合结构化生成。
- 风险：可能依赖表层词序和语义相似性。
- 问题：遇到长距离关系、复杂句法时容易判断不稳。
**视觉建议**：学生写答案但旁边有复杂关系图。  
**讲述提示**：这是我们引入结构知识的根本原因。

### Slide 10. 先前生成式 BioRE 的核心问题

**标题**：先前方法的三个痛点  
**内容**：
1. Text-only 表示缺少显式结构。
2. 直接拼接依存结构容易引入噪声。
3. 单一知识源无法适应不同数据集。
**视觉建议**：三个警示卡片。  
**讲述提示**：我们的创新点是被这些痛点逼出来的，不是为了使用某个旧工具。

### Slide 11. 痛点一：多词实体像“长名字”

**标题**：多词实体会制造内部噪声  
**内容**：
- 生物医学实体常常由多个词组成。
- 实体内部依存边不一定帮助判断实体间关系。
- 如果不处理，结构模型可能被实体内部细节干扰。
**视觉建议**：长实体被折叠成一个超级节点。  
**讲述提示**：就像身份证号比完整户籍档案更适合快速识别一个人。

### Slide 12. 痛点二：整棵依存树像一张过密地图

**标题**：结构太多，也会变成噪声  
**内容**：
- 完整依存树包含很多与目标关系无关的边。
- 直接线性化拼接，模型要自己从噪声中找信号。
- 这在生成式模型中并不稳定。
**视觉建议**：复杂地图 vs 高亮最短路线。  
**讲述提示**：给司机整张城市管线图，不如给他一条清晰路线。

### Slide 13. 痛点三：语义和句法各有盲区

**标题**：只看语义或只看句法都不够  
**内容**：
- 语义视角能理解领域词汇和上下文。
- 句法视角能捕捉实体之间的结构路径。
- 不同数据集对两类信息依赖不同。
**视觉建议**：两副眼镜，一副看语义，一副看结构。  
**讲述提示**：语义像看意思，句法像看路线，两者合起来才更稳。

### Slide 14. 我们的研究问题

**标题**：我们真正想回答什么  
**内容**：
- 什么样的结构知识对生成式 BioRE 有用？
- 结构知识应该直接拼接，还是先表示学习？
- 实体级、句法级、语义级信息如何协同？
**视觉建议**：三个研究问题气泡。  
**讲述提示**：这一页是汇报的转折点：从背景进入我们的方案。

### Slide 15. 总体创新思路

**标题**：三层增强：实体、句法、语义  
**内容**：
- 实体级：实体感知图粗化，降低实体内部噪声。
- 句法级：粗化依存图与句法表示学习。
- 语义级：PubMedBERT 领域语义表示。
- 融合级：门控机制融合双视角。
**视觉建议**：三层金字塔或三段流水线。  
**讲述提示**：不要说“我们只做依存分析”，而要说“我们研究多层结构化知识增强”。

### Slide 16. 总体方法框架图

**标题**：我们的方法像一条加工流水线  
**内容**：
- 输入句子和实体对。
- 实体感知图粗化。
- 语义视角和句法视角并行编码。
- 门控双视角融合。
- BioBART 生成关系标签。
**视觉建议**：使用前面生成的英文框架图，再手动替换中文标签。  
**讲述提示**：这是全场最重要的总览图，可以多讲 1 分钟。

### Slide 17. 整个实验分成四个阶段

**标题**：实验不是一步到位，而是四阶段推进  
**内容**：
- Stage1：数据与结构视图构建。
- Stage2：依存线性化 baseline。
- Stage3：语义-句法双视角模型。
- Stage4：论文标准 F1 评估。
**视觉建议**：四阶段路线图。  
**讲述提示**：让听众知道后面不是散点实验，而是一条递进链。

### Slide 18. Stage1 要解决什么

**标题**：Stage1：先把地基打稳  
**内容**：
- 数据统一格式化。
- SciSpaCy 依存解析。
- 实体边界对齐。
- 实体感知图粗化。
- 构建 sentence-level 主实验视图。
**视觉建议**：建筑打地基类比。  
**讲述提示**：Stage1 不是普通清洗，而是为了结构实验服务的数据工程。

### Slide 19. Stage1 数据来源

**标题**：我们用了哪些数据集  
**内容**：
- ChemProt：化学物质-蛋白质关系。
- CDR：化学物质-疾病关系。
- DDI：药物-药物相互作用。
- 主实验聚焦 ChemProtSent 和 CDRIntra。
**视觉建议**：三张数据集卡片。  
**讲述提示**：DDI 可以说是补充验证，不作为主线重点展开。

### Slide 20. 为什么构建 sentence-level 视图

**标题**：为什么要缩到句内关系  
**内容**：
- 依存结构主要描述句内语法。
- 全文级会混入跨句证据、长文本截断等问题。
- 句级视图更适合检验“结构知识是否有用”。
**视觉建议**：全文长文档剪裁成单句样本。  
**讲述提示**：这是为了控制变量，不是偷懒。

### Slide 21. 实体感知图粗化

**标题**：把长实体折叠成一个节点  
**内容**：
- 多词实体折叠为超级节点。
- 删除实体内部噪声弧。
- 保留实体与外部谓词/修饰成分的关系。
**视觉建议**：before/after 依存图。  
**讲述提示**：这一步像把一个人的全名、头衔、单位先合成一个“实体身份证”。

### Slide 22. Stage1 最终产物

**标题**：Stage1 交付了什么  
**内容**：
- 原始主线数据：processed -> parsed -> coarsened。
- 主实验视图：ChemProtSent、CDRIntra。
- DataLoader/Collator 支持 text-only 与 dependency-enhanced 输入。
- 数据层已能支撑后续实验。
**视觉建议**：数据流目录图。  
**讲述提示**：强调“可复现、可接手、可支撑训练”。

### Slide 23. Stage2 要回答什么

**标题**：Stage2：依存结构直接加进去有用吗  
**内容**：
三个问题：
1. 结构信息是否值得注入？
2. tree 和 SDP 哪个更好？
3. raw 和 coarse 哪个更好？
**视觉建议**：三个问号卡片。  
**讲述提示**：Stage2 像试菜，先尝尝不同结构形式有没有价值。

### Slide 24. Stage2 baseline 设置

**标题**：五组 baseline 像五个参照物  
**内容**：
- B0：text-only。
- B1-raw：原始依存树线性化。
- B2-raw：原始 SDP 线性化。
- B1-coarse：粗化依存树线性化。
- B2-coarse：粗化 SDP 线性化。
**视觉建议**：五列模型对照表。  
**讲述提示**：这里不要讲太多实现，强调每组在回答什么问题。

### Slide 25. 什么是 SDP

**标题**：SDP：实体之间的最短路线  
**内容**：
- SDP = Shortest Dependency Path。
- 不看整棵树，只看两个实体之间的依存路径。
- 直觉上比完整树更聚焦关系证据。
**视觉建议**：整棵树中高亮一条路径。  
**讲述提示**：继续用地图类比：整棵树是城市地图，SDP 是导航路线。

### Slide 26. Stage2 的关键发现

**标题**：直接拼接结构不是总有效  
**内容**：
- ChemProtSent 上 B2-coarse 表现较好。
- CDRIntra 上 SDP 优于 tree，coarse SDP 略有优势。
- DDI 上 text-only 最强。
**视觉建议**：三个数据集的简化趋势图。  
**讲述提示**：这个结果非常重要，因为它证明“结构不是越多越好”。

### Slide 27. Stage2 给我们的启发

**标题**：第一条经验：结构要处理后再用  
**内容**：
- 依存信息有价值。
- 但直接线性化拼接不稳定。
- 实体粗化对路径型结构更有效。
- 后续需要表示学习和更精细融合。
**视觉建议**：从“拼接”走向“建模”的箭头。  
**讲述提示**：从这里自然过渡到 Stage3。

### Slide 28. Stage3 为什么升级为图增强模型

**标题**：从“把结构写进文本”到“让模型学习结构”  
**内容**：
- Stage2 是把依存结构转成文本拼接。
- Stage3 直接把粗化依存图变成图张量。
- 模型可以在图上学习节点和边的信息。
**视觉建议**：文本串 vs 图结构编码。  
**讲述提示**：这像从“把地图描述成一段话”升级为“直接给模型一张地图”。

### Slide 29. 句法视角：Syntax View

**标题**：句法视角看的是实体间路线  
**内容**：
- 输入：粗化依存图。
- 模块：Attentive GCN。
- 输出：句法表示向量。
- 作用：捕捉实体之间的结构路径和依存关系。
**视觉建议**：图节点消息传递示意。  
**讲述提示**：句法视角像看骨架，知道句子各部分怎么连起来。

### Slide 30. 语义视角：Semantics View

**标题**：语义视角看的是医学含义  
**内容**：
- 输入：生物医学句子。
- 模块：PubMedBERT。
- 输出：领域语义表示。
- 作用：理解医学术语、触发词和上下文语义。
**视觉建议**：词向量/embedding 方块。  
**讲述提示**：语义视角像读懂这句话在医学上是什么意思。

### Slide 31. B3/B4/B5 诊断模型

**标题**：先分别测试两副眼镜有没有用  
**内容**：
- B3 syntax-only：只看句法。
- B4 semantics-only：只看语义。
- B5 dual-view concat：语义和句法直接拼接。
**视觉建议**：三条并列模型路线。  
**讲述提示**：这是科学实验里的拆解：先分别测，再组合测。

### Slide 32. Stage3 诊断结论

**标题**：语义和句法都有效，但不能粗暴相加  
**内容**：
- 两个视角都优于 Stage2 线性化 baseline。
- ChemProtSent 中语义视角很强。
- CDRIntra 中双视角互补更明显。
- 简单 concat 不一定总是最稳。
**视觉建议**：语义/句法/双视角的趋势对比。  
**讲述提示**：这页讲“为什么还需要门控融合”。

### Slide 33. GM-IB 探索：我们学到了什么

**标题**：一次有价值的探索性负结果  
**内容**：
- 初衷：学习关键依存边，得到可解释子图。
- 问题：hard selection 容易全空、塌缩或对预测不敏感。
- 结论：不适合作为最终主线。
**视觉建议**：虚线分支或实验诊断卡片。  
**讲述提示**：不要回避失败，失败帮助我们排除了不稳定路线。

### Slide 34. B7：门控双视角融合

**标题**：最终选择：连续门控而不是硬筛边  
**内容**：
- 输入：semantic vector 和 syntax vector。
- 门控：自动学习两类信息的融合比例。
- 优点：连续可训练、稳定、保留双视角互补。
**视觉建议**：一个滑动门控/调音台，把两路信号合成一路。  
**讲述提示**：像调音台，不是把某一路完全关掉，而是学习合适比例。

### Slide 35. Stage4 为什么还要重新评估

**标题**：从开发指标走向论文标准指标  
**内容**：
- Stage3 主要用 mapped_exact 做模型选择。
- Stage4 统一在 test split 上计算 F1。
- Precision / Recall / Micro-F1 / Macro-F1 排除 NO_RELATION。
**视觉建议**：验证集选择 -> 测试集评估流程。  
**讲述提示**：这是把实验结果从“开发可用”转成“论文可报告”。

### Slide 36. Stage4 评估设置

**标题**：统一评估口径  
**内容**：
- 数据集：ChemProtSent、CDRIntra。
- 种子：42 / 123 / 456。
- 模型：B0、B1/B2、B4/B5/B7。
- 指标：P、R、Micro-F1、Macro-F1、mapped_exact、invalid_rate。
**视觉建议**：评估设置表。  
**讲述提示**：强调所有模型在同一规则下比较。

### Slide 37. 结果一：纯文本 BioBART 已经很强

**标题**：强 baseline 让提升更有含金量  
**内容**：
- CDRIntra：B0 Micro-F1 = 0.5457。
- ChemProtSent：B0 Micro-F1 = 0.6461。
- 最优 Stage3 模型仍明显超过 B0。
**视觉建议**：B0 vs 最优 Stage3 双柱图。  
**讲述提示**：如果 baseline 很弱，提升容易；baseline 已强还有提升，才说明方法有价值。

### Slide 38. 结果二：显式双视角建模优于线性化 baseline

**标题**：收益来自表示学习，不是简单拼接  
**内容**：
- CDRIntra：B2-coarse 0.5111 -> 最优 Stage3 0.6818。
- ChemProtSent：B2-coarse 0.6476 -> B7 0.7246。
- 说明结构知识需要经过模型化表示与融合。
**视觉建议**：两组提升箭头。  
**讲述提示**：这是本次汇报最关键的实验结论之一。

### Slide 39. 结果三：不同数据集偏好不同

**标题**：没有一种知识对所有场景都万能  
**内容**：
- ChemProtSent：B7 gated dual-view 最优，Micro-F1 = 0.7246。
- CDRIntra：B4 semantics-only 最优，Micro-F1 = 0.6818。
- B5/B7 接近，说明双视角仍有竞争力。
**视觉建议**：两个数据集左右对比。  
**讲述提示**：谨慎表达，不要过度声称“门控总是最优”。

### Slide 40. 我们的核心创新点

**标题**：创新点不只是“用了依存句法”  
**内容**：
1. 问题层面：面向生成式 BioRE 研究结构知识如何注入。
2. 数据层面：构建句级结构实验视图与实体感知粗化图。
3. 方法层面：语义-句法双视角表示学习。
4. 融合层面：用门控机制实现稳定双视角融合。
**视觉建议**：四个贡献卡片。  
**讲述提示**：这页要把老师可能关心的创新性讲清楚。

### Slide 41. 局限与谨慎表述

**标题**：哪些地方我们要诚实说明  
**内容**：
- 直接依存线性化不稳定，不应夸大。
- GM-IB hard selection 是探索性诊断，不作为最终主贡献。
- 不同数据集存在偏好差异。
- 后续需要错误分析、关系类别分析和更多可视化。
**视觉建议**：Limitations 与 Future Work 两栏。  
**讲述提示**：诚实不是削弱论文，而是让结论更可信。

### Slide 42. 总结页

**标题**：一句话总结  
**内容**：
> 生成式 BioRE 不应只依赖纯文本，也不能粗暴拼接依存结构；更有效的路线是先进行实体级结构去噪，再通过语义-句法双视角表示学习和融合机制，让结构知识以生成模型能够吸收的方式发挥作用。

**视觉建议**：回到总框架图，标出三层增强和最终输出。  
**讲述提示**：最后用一句温和但有力量的话收束：我们的工作是在回答“什么知识真正能帮助生成式模型理解关系”。

---

## 3. 30 分钟讲述节奏建议

| 时间 | 页码 | 讲法 |
|---:|---|---|
| 0:00-4:00 | 1-7 | 背景铺垫，少讲术语，多讲任务 |
| 4:00-9:00 | 8-14 | 引出生成式模型痛点 |
| 9:00-14:00 | 15-22 | 展示总体框架和 Stage1 |
| 14:00-18:00 | 23-27 | 讲 Stage2 baseline，强调“直接拼接不稳定” |
| 18:00-24:00 | 28-34 | 讲 Stage3 双视角和 B7 |
| 24:00-28:30 | 35-39 | 讲 Stage4 关键结果 |
| 28:30-30:00 | 40-42 | 创新点、局限、总结 |

---

## 4. 汇报时建议反复使用的三句话

1. **我们不是重新提出依存分析，而是在研究生成式 BioRE 如何有效吸收结构化知识。**
2. **实验说明，结构知识不是直接拼接就有用，而是需要实体级去噪、图表示学习和融合机制。**
3. **最终主线是多层结构化知识增强：实体级、句法级、语义级，再通过门控融合交给 BioBART 生成关系。**

---

## 5. 不建议在 PPT 中这样说

| 不建议说法 | 风险 | 建议替换 |
|---|---|---|
| 我们提出了一种新的依存分析方法 | 容易被质疑依存分析本身不新 | 我们提出面向生成式 BioRE 的结构知识增强方法 |
| 依存结构总是有效 | 与实验结果不一致 | 依存结构需要经过去噪、筛选或表示学习后才更稳定 |
| B7 在所有数据集上最好 | 与 CDRIntra test F1 不一致 | B7 在 ChemProtSent 最优，在 CDRIntra 接近最优且稳定 |
| GM-IB 是最终核心创新 | 与最终实验定位不符 | GM-IB 是探索性诊断，帮助我们识别 hard selection 不稳定 |
| PubMedBERT 是我们的主模型 | 会偏离生成式主线 | PubMedBERT 是语义视角编码器，主生成 backbone 是 BioBART |

---

## 6. 可直接放在 PPT 备注里的开场白

大家好，我今天汇报的题目是“基于生成式模型的生物医学关系抽取研究”。我不会一开始就讲模型公式，因为这个任务如果直接看模型结构会比较抽象。我们可以先把它理解成一个“医学文本侦探”问题：给模型一句复杂的医学句子，再指定两个实体，例如一个化学物质和一个蛋白质，模型需要判断它们之间到底有没有关系、是什么关系。

生成式模型像一个会写答案的学生，它可以生成关系标签，但问题是，它不一定真正理解句子内部的结构。尤其在生物医学文本里，实体很长、句子很复杂、真正有用的关系线索可能藏在依存路径和领域语义里。因此，我们的工作不是简单地把依存树加进去，而是想系统回答一个问题：什么样的结构化知识真正能帮助生成式模型做关系抽取，以及这些知识应该怎样注入才稳定。

---

## 7. 可直接放在总结页的结束语

总体来看，本阶段实验说明：纯文本 BioBART 已经是一个较强的生成式基线，但它仍然可以从结构化知识中受益。直接依存线性化并不稳定，实体感知图粗化可以缓解路径结构噪声，而显式语义/句法双视角建模相比线性化 baseline 有明显提升。因此，我们最终把创新点定位为面向生成式 BioRE 的多层结构化知识增强框架，而不是单独强调依存分析本身。

