# 实验计划书：面向生成式生物医学关系抽取的实体感知多视角信息蒸馏研究

## 一、实验摘要

本实验计划隶属于硕士大论文《基于生成式模型的生物医学关系抽取研究》的研究内容一。核心研究问题为：什么样的依存句法知识对下游生成式生物医学关系抽取（BioRE）最有用？

现有依存驱动的关系抽取（DDRE）方法存在两个关键问题：（1）生物医学文本中多词实体（如 non-small cell lung cancer）在依存树内部产生大量噪声弧，干扰实体间关系信号的传递；（2）原始依存树包含大量任务无关信息，简单拼接或线性化会淹没关键结构信息，无法为生成式模型提供高效的结构先验。

受 MS2IB（Multi-view Syntax-Semantics Information Bottleneck, ESWA 2025）的启发，本研究提出一种面向生成的实体感知多视角信息瓶颈方法（EA-GMIB）。该方法在依存树上执行实体感知的图粗化以消除实体内部噪声弧，结合语义视角补充被剪枝丢失的序列信息，并通过面向生成的信息瓶颈模块蒸馏出对 seq2seq 模型最有用的依存弧，最终以可控线性化序列的形式输出结构知识。

本研究的产出——经过蒸馏的高质量结构化依存序列——将直接作为大论文研究内容二（结构驱动的持续预训练与微调）的数据输入，形成完整的逻辑闭环。


---

## 二、立项依据

### 2.1 问题来源

生物医学关系抽取（BioRE）旨在从非结构化医学文本中自动提取实体间的语义关系三元组。生物医学文本具有句子长、从句多、被动结构复杂等特点，仅依赖线性上下文往往难以捕捉实体间准确的语义联系。依存句法树作为一种结构先验，能够刻画词与词之间的语法关系，为关系抽取提供重要的辅助信息。

然而，将依存句法知识引入生成式 BioRE 面临以下挑战：

**挑战一：多词实体内部的噪声弧问题。** 生物医学文本中大量存在多词实体（Multi-word Entity, MWE），如 "non-small cell lung cancer"、"vascular endothelial growth factor" 等。依存解析器会在这些实体内部生成复杂的依存弧（如 compound、amod 等），这些弧对关系抽取而言是纯噪声——它们干扰 GCN 的消息传递，污染最短依存路径（SDP），并增加线性化序列的冗余长度。

**挑战二：任务无关信息的干扰。** 原始依存树包含大量与关系抽取无关的结构信息。如果将整棵树直接线性化拼接到生成模型的输入中，不仅序列过长，而且关键结构信号会被冗余信息淹没。现有方法多采用注意力机制来缓解噪声，但缺乏理论保证。

**挑战三：与生成式模型的接口问题。** 现有 DDRE 方法几乎都面向判别式模型（分类任务），其信息筛选标准是分类准确率。但生成式模型的需求不同——它需要的是能够指导序列生成的结构知识，而非分类友好的特征向量。

### 2.2 相关工作与启发

**MS2IB（She et al., ESWA 2025）** 提出了多视角语法-语义信息瓶颈模型，通过信息瓶颈（IB）理论从语法视角和语义视角中蒸馏出对关系分类最有用的信息。该工作的核心启发在于：不去打造更好的解析器或更精巧的剪枝策略，而是用信息论的框架让模型自己学会"什么信息有用"。

**本研究与 MS2IB 的关键差异：**
- 任务层面：MS2IB 面向判别式 RE（分类），本研究面向生成式 RE（seq2seq）
- 视角设计：MS2IB 直接使用原始依存树，本研究引入实体感知的图粗化
- IB 目标：MS2IB 用分类损失驱动蒸馏，本研究用生成损失驱动蒸馏
- 应用领域：MS2IB 在通用/铁路数据集上验证，本研究聚焦生物医学领域
- 产出形式：MS2IB 产出特征向量用于分类，本研究产出结构化文本序列用于生成

### 2.3 与大论文整体框架的关系

本研究（研究内容一）在大论文中的定位是"学会筛选"——验证什么样的依存弧对生成式 BioRE 最有用，并产出一个可学习的弧筛选与线性化算法。其产出将直接服务于：

- **研究内容二（学会吸收）**：用本研究的算法大规模处理生物医学语料，生成结构化训练数据，用于生成式模型的持续预训练
- **研究内容三（学会迁移）**：本研究产出的结构序列可作为情景学习中结构检索的基础表示


---

## 三、研究目标与内容

### 3.1 研究目标

本研究的核心目标是：设计并验证一种面向生成式 BioRE 的依存知识蒸馏方法，使得经过蒸馏后的结构序列能够显著提升生成式模型的关系抽取性能，并为大论文后续阶段提供高质量的结构化数据输入。

具体而言，本研究需要回答以下五个研究问题（Research Questions）：

- **RQ1**：依存句法信息是否对生成式 BioRE 有帮助？（有用性验证）
- **RQ2**：实体感知的图粗化是否能有效消除多词实体内部的噪声弧？（实体折叠验证）
- **RQ3**：多视角（语法+语义）互补是否优于单视角？（多视角验证）
- **RQ4**：面向生成的信息瓶颈（GM-IB）是否优于简单拼接和面向分类的 IB？（蒸馏机制验证）
- **RQ5**：蒸馏后的结构序列在不同句长、不同实体复杂度下的表现如何？（鲁棒性分析）

### 3.2 研究内容

本研究包含以下四个核心研究内容：

**内容一：实体感知的依存图粗化（Entity-Aware Graph Coarsening）**

针对生物医学文本中多词实体在依存树内部产生噪声弧的问题，设计实体感知的图粗化算法。利用数据集提供的实体边界标注，将多词实体折叠为超级节点，消除实体内部的依存弧，超级节点继承其 head token 的外部依存弧。

**内容二：多视角表示学习（Multi-View Representation Learning）**

构建两个互补的视角来学习句子的结构与语义信息：
- 语法视角（Entity-Aware Syntax View）：在粗化后的依存图上，使用带注意力机制的 GCN 编码结构信息，同时为每条依存弧输出重要性评分
- 语义视角（Sequential Semantics View）：使用预训练语言模型（PubMedBERT）编码句子的上下文语义信息，补充因图粗化和剪枝丢失的序列语义

**内容三：面向生成的信息瓶颈蒸馏（Generation-oriented Information Bottleneck）**

设计面向生成任务的信息瓶颈模块（GM-IB），以生成式模型的序列生成损失（而非分类损失）作为信息保留的优化目标，以熵正则化作为信息压缩的约束，从两个视角中蒸馏出对生成最有用的依存弧子集。

**内容四：可控线性化与生成器融合（Controllable Linearization）**

将 GM-IB 筛选出的高重要性依存弧按照预定义模板线性化为结构文本序列，拼接到原始句子后作为生成式模型（BioBART）的输入，端到端训练并评估关系抽取性能。


---

## 四、实验 Pipeline 设计

### 4.1 总体架构

```text
输入: 生物医学句子 S + 实体对标注 (e1, e2)
                    |
    +===============+===============+
    |                               |
    v                               v
[View 1: Entity-Aware Syntax]   [View 2: Sequential Semantics]
    |                               |
    | 1a. 依存解析 (SciSpaCy)       | 2a. PubMedBERT 编码
    | 1b. 实体折叠 (MWE->超级节点)   |     得到上下文表示 H_sem
    | 1c. Attentive GCN 编码        |
    |     得到结构表示 H_syn         |
    |     + 每条弧的重要性分数 a_i   |
    |                               |
    +===============+===============+
                    |
                    v
        [GM-IB: 信息瓶颈蒸馏模块]
            |
            | - 信息保留: 由生成损失驱动
            | - 信息压缩: 熵正则化, 鼓励只保留少量关键弧
            | - 输出: 筛选后的依存弧子集 S*
            |
            v
        [可控线性化模块]
            |
            | - 将 S* 中的弧按模板转为文本序列
            | - 格式: [DEP] e1 -nsubj-> inhibits -obj-> e2 [/DEP]
            |
            v
        [BioBART 生成器]
            |
            | - 输入: 原始句子 + 实体标记 + 结构序列
            | - 输出: 关系标签 (或三元组文本)
            |
            v
        总损失 = $\mathcal{L}_{total} = \mathcal{L}_{gen} + \beta \cdot \mathcal{L}_{compress}$
```

### 4.2 数据预处理阶段（Stage 0）

#### 4.2.1 数据集选择与获取

| 数据集 | 关系类型 | 规模 | 特点 | 选择理由 |
|--------|---------|------|------|---------|
| CDR (BioCreative V) | 化学-疾病 (CID), 二分类 | ~1500 篇摘要 | 经典 BioRE 基准 | 几乎所有 BioRE 论文必用, 便于横向对比 |
| ChemProt | 化学-蛋白质, 5类+负例 | ~2400 篇摘要 | 关系类型多, 术语密集 | 难度更大, 区分度好 |
| DDI (Drug-Drug Interaction) | 药物相互作用, 4类 | ~1000 篇文档 | 句子较长, 实体重叠多 | 长句多, 能验证实体折叠价值 |

#### 4.2.2 依存解析

- 工具选择: SciSpaCy (en_core_sci_lg) 或 Stanza (biomedical package)
- 对训练集/验证集/测试集的所有句子执行依存解析
- 输出: 每个句子的依存树 (邻接矩阵 + 依存类型矩阵)
- 存储格式: JSON, 每条记录包含 tokens, deps, heads, entity_spans

#### 4.2.3 实体感知图粗化

对每个句子的依存树执行以下操作:

```text
算法: Entity-Aware Graph Coarsening
输入: 依存树 T = (V, E), 实体标注 entities = [(start, end, type), ...]
输出: 粗化后的依存图 T_coarse

步骤:
1. 对每个实体 ent = (start, end, type):
   a. 识别实体 span 内的所有 token: {t_start, t_start+1, ..., t_end}
   b. 确定 head token: 实体 span 中依存深度最浅的 token
      (即在依存树中最接近根节点的)
   c. 创建超级节点 super_node, 其文本为实体全称
   d. 删除实体内部所有 token 之间的依存弧
   e. 将 head token 的所有外部依存弧 (指向实体外部的弧) 转移给 super_node
   f. 删除实体内部的所有 token 节点

2. 重新编号节点, 更新邻接矩阵和依存类型矩阵
3. 返回 T_coarse
```

示例:

```text
原始依存树:
  inhibits -nsubj-> Aspirin
  inhibits -obj-> expression
  expression -compound-> gene
  cancer -compound-> lung
  cancer -compound-> cell
  cancer -amod-> non-small
  (实体1: Aspirin, 实体2: non-small cell lung cancer)

粗化后:
  inhibits -nsubj-> [Aspirin]
  inhibits -obj-> expression
  expression -compound-> gene
  expression -nmod-> [non-small cell lung cancer]
  (实体内部弧全部消除, 超级节点继承外部弧)
```

#### 4.2.4 生成式模型输入格式设计

```text
输入格式 (BioBART encoder input):
  <s> [E1] Aspirin [/E1] inhibits the [E2] gene expression [/E2] in tumor cells.
  [DEP] [E1] -nsubj-> inhibits -obj-> [E2] [/DEP] </s>

输出格式 (BioBART decoder target):
  Chemical-Disease | inhibits
```

说明:
- [E1]/[/E1] 和 [E2]/[/E2] 为实体标记符, 标识实体边界
- [DEP]/[/DEP] 为依存结构标记符, 包裹线性化后的依存弧序列
- 输出格式为 "关系类型 | 关键谓词" 的简洁形式

### 4.3 模型构建阶段（Stage 1）

#### 4.3.1 View 1: Entity-Aware Syntax View

**输入**: 粗化后的依存图 T_coarse = (邻接矩阵 X, 依存类型矩阵 T_dep)

**Attentive GCN 编码器** (参考 MS2IB 公式 1):

**Step 1：注意力机制捕获长距离依存**

$$H_{\text{att}} = \text{Attention}(Q=X_{\text{init}},\; K=X_{\text{init}},\; V=X_{\text{init}})$$

其中 $X_{\text{init}}$ 为节点的初始表示（可用 PubMedBERT 的 token embedding）。

**Step 2：GCN 聚合结构特征（2层）**

$$H_{\text{syn}}^{(l)} = \text{ReLU}\!\left(\tilde{A}_{\text{coarse}}\, H_{\text{syn}}^{(l-1)}\, W^{(l)}\right)$$

其中 $\tilde{A}_{\text{coarse}}$ 为归一化后的粗化邻接矩阵，最终取第2层输出 $H_{\text{syn}} = H_{\text{syn}}^{(2)}$。

**Step 3：弧级重要性评分**

对每条弧 $e_{ij}$，计算：

$$a_{ij} = \sigma\!\left(\text{MLP}\!\left([h_i \,;\, h_j \,;\, t_{ij}]\right)\right)$$

其中 $h_i,\, h_j$ 为节点表示，$t_{ij}$ 为依存类型嵌入。

**输出**:
- 结构表示 H_syn (节点级)
- 弧重要性分数 {a_ij} (弧级, 用于后续 GM-IB 筛选)

GCN 层数: 2 层 (参考 MS2IB 设置)

#### 4.3.2 View 2: Sequential Semantics View

**输入**: 原始句子 S (带实体标记)

**编码器**: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)

$$H_{\text{sem}} = \text{PubMedBERT}(S) \quad \in \mathbb{R}^{B \times L \times 768}$$

取 [CLS] token 表示作为句子级语义表示, 或取实体对应位置的表示作为实体级语义表示。

**输出**: 语义表示 H_sem

#### 4.3.3 GM-IB: 面向生成的信息瓶颈模块

这是本研究的核心创新模块。

**设计思路**:

MS2IB 的 IB 目标为：

$$\mathcal{L}_{\text{MS2IB}} = I(Z;Y) - \beta \left[ I(S_1;Z) + I(S_2;Z) \right]$$

其中 $Y$ 为离散关系类别标签。

本研究的 GM-IB 目标为：

$$\mathcal{L}_{\text{GM-IB}} = I(Z;Y_{\text{seq}}) - \beta \left[ I(S_1;Z) + I(S_2;Z) \right]$$

其中 $Y_{\text{seq}}$ 为生成目标序列（三元组文本）。

**关键区别**：$I(Z;Y_{\text{seq}})$ 的变分下界通过生成式解码器来近似：

$$I(Z;Y_{\text{seq}}) \geq \mathbb{E}\left[\log q(Y_{\text{seq}}|Z)\right]$$

其中 $q(Y_{\text{seq}}|Z)$ 即 BioBART 解码器的输出概率。

**具体实现**:

**Step 1：多视角信息融合**

将 $H_{\text{syn}}$ 和 $H_{\text{sem}}$ 通过门控机制融合：

$$g = \sigma\!\left(W_g \cdot [H_{\text{syn}} \,;\, H_{\text{sem}}] + b_g\right)$$

$$H_{\text{fused}} = g \odot H_{\text{syn}} + (1-g) \odot H_{\text{sem}}$$

**Step 2：弧筛选（信息压缩）**

对每条弧 $e_{ij}$，GM-IB 输出保留概率：

$$p_{ij} = \sigma\!\left(\text{MLP}_{\text{ib}}\!\left([h_i^{\text{fused}} \,;\, h_j^{\text{fused}} \,;\, a_{ij}]\right)\right)$$

- 训练时：使用 Gumbel-Softmax 进行可微的离散采样：$z_{ij} = \text{GumbelSoftmax}(p_{ij},\, \tau)$
- 推理时：直接取 top-$k$ 条弧（$k$ 为超参数，或按阈值截断）

**Step 3：信息保留（生成损失驱动）**

将筛选后的弧集合 $S^* = \{e_{ij} \mid z_{ij}=1\}$ 线性化为文本序列，拼接到原始句子后送入 BioBART 生成：

$$\mathcal{L}_{\text{gen}} = -\log P_{\text{BioBART}}\!\left(Y_{\text{seq}} \mid S,\, \mathcal{L}(S^*)\right)$$

**Step 4：信息压缩（熵正则化）**

$$\mathcal{L}_{\text{compress}} = \sum_{ij} p_{ij} \log p_{ij}$$

鼓励模型只保留少量最关键的弧。

**Step 5：总损失**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gen}} + \beta \cdot \mathcal{L}_{\text{compress}}$$

$\beta$ 控制压缩程度：$\beta$ 越大，保留的弧越少。

**Gumbel-Softmax 的作用**: 弧的保留/丢弃是离散决策, 无法直接反向传播。Gumbel-Softmax 提供了一种可微的近似, 使得整个模型可以端到端训练。温度参数 $\tau$ 在训练过程中逐渐退火（从 $1.0$ 降到 $0.1$），使采样逐渐趋近于硬选择。

#### 4.3.4 可控线性化模块

将 GM-IB 筛选出的弧集合 S* 转化为文本序列:

```text
线性化模板:
  [DEP] node_i -dep_type-> node_j ; node_k -dep_type-> node_l [/DEP]

排序规则:
  1. 优先保留实体对之间最短路径上的弧
  2. 其余弧按重要性分数 p_ij 降序排列
  3. 截断至最大长度 max_dep_len (超参数, 默认 30 tokens)

示例:
  筛选前 (原始依存树有 15 条弧)
  筛选后 (GM-IB 保留 5 条弧):
  [DEP] [E1] -nsubj-> inhibits -obj-> expression -nmod-> [E2] ;
        inhibits -advmod-> significantly [/DEP]
```

#### 4.3.5 BioBART 生成器

**模型**: BioBART-base (GanjinZero/biobart-base)

**输入拼接**:

```text
encoder_input:
  "[E1] aspirin [/E1] inhibits [E2] gene expression [/E2] .
   [DEP] [E1] -nsubj-> inhibits -obj-> [E2] [/DEP]"

decoder_target:
  "Chemical-Disease | inhibits"
```

**训练**: 标准 seq2seq teacher forcing, 损失函数为 token 级交叉熵

**关键**: BioBART 的参数在训练中更新 (微调), 但主要的学习信号通过 L_gen 反向传播到 GM-IB 模块, 驱动弧筛选的优化。

### 4.4 训练策略（Stage 2）

#### 4.4.1 端到端联合训练

整个模型 (Attentive GCN + PubMedBERT + GM-IB + BioBART) 端到端联合训练。

```text
训练流程:
  for each epoch:
    for each batch (sentence, entity_pair, gold_relation):
      # --- 前向传播 ---
      1. 加载预处理好的粗化依存图 (离线完成)
      2. H_syn, {a_ij} = AttentiveGCN(coarsened_dep_graph)
      3. H_sem = PubMedBERT(sentence + entity_markers)
      4. H_fused, {p_ij} = GMIB_fusion(H_syn, H_sem, {a_ij})
      5. S_star = GumbelSoftmax_sample({p_ij}, tau)   # 筛选弧子集
      6. dep_text = Linearize(S_star)                  # 线性化为文本
      7. input_seq = concat(sentence, entity_markers, dep_text)
      8. logits = BioBART(input_seq)

      # --- 损失计算 ---
      9.  $\mathcal{L}_{\text{gen}}$ = CrossEntropy(logits, gold_relation)
      10. $\mathcal{L}_{\text{compress}} = \sum_{ij} p_{ij} \log p_{ij}$   # 熵正则化
      11. $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gen}} + \beta \cdot \mathcal{L}_{\text{compress}}$

      # --- 反向传播 ---
      12. $\mathcal{L}_{\text{total}}$.backward()
      13. optimizer.step()

    # --- Gumbel 温度退火 ---
    $\tau$ = max(0.1, $\tau$ * anneal_rate)

    # --- 早停检查 ---
    if val_f1 not improved for 5 epochs: break
```

#### 4.4.2 超参数设置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| GCN 层数 | 2 | 参考 MS2IB |
| Attention heads | 8 | 参考 MS2IB |
| 学习率 | 2e-5 | BioBART 微调常用值 |
| Batch size | 8 | 受 GPU 显存限制 |
| Dropout | 0.1 | 标准设置 |
| $\beta$（IB 压缩系数）| grid search: {1e-4, 1e-5, ..., 1e-8} | 参考 MS2IB |
| $\tau$（Gumbel 温度）| $1.0 \to 0.1$（线性退火）| 标准 Gumbel-Softmax 设置 |
| max_dep_len | 30 tokens | 线性化序列最大长度 |
| Epochs | 30 | 早停 patience=5 |
| 优化器 | AdamW | 标准选择 |
| Warmup | 前 6% steps | 参考 MS2IB |


---

## 五、实验设计与评价方法

### 5.1 实验总览

本研究设计五组实验，分别回答五个研究问题：

| 实验组 | 对应 RQ | 实验内容 | 核心对比 |
|--------|---------|---------|---------|
| 主实验 | RQ1-RQ4 | 全模型对比 | Ours vs 所有 Baseline |
| 消融实验 | RQ2-RQ4 | 逐模块去除 | 验证每个组件的贡献 |
| 参数敏感性 | RQ4 | beta 参数搜索 | 压缩-保留的最优平衡点 |
| 鲁棒性分析 | RQ5 | 按句长/实体复杂度分桶 | 不同场景下的表现 |
| Case Study | RQ2,RQ4 | 可视化分析 | 定性展示筛选效果 |

### 5.2 主实验：全模型对比（回答 RQ1-RQ4）

#### Baseline 设计

| 编号 | 模型 | 说明 | 回答哪个 RQ |
|------|------|------|------------|
| B0 | BioBART (text-only) | 纯文本生成 baseline，不使用任何依存信息 | RQ1 对照组 |
| B1 | BioBART + Raw Dep Tree | 原始依存树直接线性化拼接到输入 | RQ1: 原始依存是否有用 |
| B2 | BioBART + SDP | 只使用实体对之间的最短依存路径 | RQ1: SDP 是否足够 |
| B3 | BioBART + GCN (syntax only) | 单视角：仅语法视角，GCN 编码后融合 | RQ3 对照组 |
| B4 | BioBART + PLM enhanced | 单视角：仅语义视角增强 | RQ3 对照组 |
| B5 | BioBART + Dual-view (concat) | 两视角简单拼接，不做信息蒸馏 | RQ4 对照组 |
| B6 | BioBART + Dual-view + Cls-IB | 两视角 + 分类式 IB（MS2IB 风格） | RQ4: 分类 IB vs 生成 IB |
| **Ours** | **BioBART + EA-Dual-view + GM-IB** | **实体感知两视角 + 生成导向 IB** | **完整模型** |

#### 主实验结果表格式

| Model | CDR F1 | CDR Prec. | CDR Rec. | ChemProt F1 | ChemProt Prec. | ChemProt Rec. | DDI F1 | DDI Prec. | DDI Rec. |
|-------|--------|-----------|----------|-------------|----------------|---------------|--------|-----------|----------|
| B0 | | | | | | | | | |
| B1 | | | | | | | | | |
| B2 | | | | | | | | | |
| B3 | | | | | | | | | |
| B4 | | | | | | | | | |
| B5 | | | | | | | | | |
| B6 | | | | | | | | | |
| **Ours** | | | | | | | | | |

#### 预期结论

- B0 < B1/B2: 依存信息对生成式 BioRE 有帮助（RQ1）
- B3/B4 < B5: 多视角优于单视角（RQ3）
- B5 < Ours: 信息蒸馏优于简单拼接（RQ4）
- B6 < Ours: 生成导向 IB 优于分类导向 IB（RQ4）

### 5.3 消融实验（回答 RQ2-RQ4）

| 实验 ID | 模型变体 | 去掉什么 | 验证什么 |
|---------|---------|---------|---------|
| A1 | Ours w/o Entity Coarsening | 不做实体折叠，使用原始依存树 | 实体感知图粗化的贡献（RQ2） |
| A2 | Ours w/o Syntax View | 去掉语法视角，仅用语义视角 | 依存结构信息的贡献（RQ3） |
| A3 | Ours w/o Semantics View | 去掉语义视角，仅用语法视角 | 序列语义补充的贡献（RQ3） |
| A4 | Ours w/o IB (simple concat) | 去掉 IB 模块，两视角直接拼接 | 信息蒸馏机制的贡献（RQ4） |
| A5 | Ours w/ Classification IB | 将生成损失替换为分类损失 | GM-IB vs 传统 IB 的差异（RQ4） |

#### 消融实验结果表格式

| ID | 模型变体 | CDR F1 | ChemProt F1 | DDI F1 |
|----|---------|--------|-------------|--------|
| Full | Ours (完整模型) | | | |
| A1 | w/o Entity Coarsening | | | |
| A2 | w/o Syntax View | | | |
| A3 | w/o Semantics View | | | |
| A4 | w/o IB | | | |
| A5 | w/ Classification IB | | | |

### 5.4 参数敏感性实验（回答 RQ4）

对 $\beta$ 参数进行 grid search：

```
beta_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
```

在每个数据集上分别跑完整模型，记录 Precision、Recall、F1，绘制折线图。

预期观察：F1 随 $\beta$ 先升后降，存在一个最优平衡点。$\beta$ 过小则压缩不足（保留太多噪声弧），$\beta$ 过大则压缩过度（丢失有用弧）。

### 5.5 鲁棒性分析（回答 RQ5）

#### 5.5.1 按句长分桶

| 句长区间 | B0 F1 | B1 F1 | Ours F1 | 提升幅度 |
|---------|-------|-------|---------|---------|
| 短句 (<=20 tokens) | | | | |
| 中句 (21-40 tokens) | | | | |
| 长句 (>40 tokens) | | | | |

预期：长句上 Ours 的优势最大，因为长句的依存树更复杂，噪声更多，信息蒸馏的价值更高。

#### 5.5.2 按实体词数分桶

| 实体类型 | B1 F1 | A1 F1 (w/o coarsening) | Ours F1 | 提升幅度 |
|---------|-------|----------------------|---------|---------|
| 单词实体 | | | | |
| 多词实体 (2-3 tokens) | | | | |
| 长实体 (>=4 tokens) | | | | |

预期：多词实体和长实体上，实体折叠的贡献最显著。

### 5.6 Case Study

选取 2-3 个典型句子，展示：

1. **实体折叠前后的依存图对比**：可视化原始依存树 vs 粗化后的依存图，直观展示噪声弧的消除效果
2. **GM-IB 弧筛选结果**：展示哪些弧被保留（高重要性）、哪些被压缩（低重要性），验证模型确实学会了筛选任务相关的弧
3. **线性化序列对比**：展示原始全树线性化 vs GM-IB 筛选后线性化的序列，对比长度和信息密度

### 5.7 评价指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 主指标 | Micro-F1 | 标准 RE 评价，所有关系类型的综合 F1 |
| 主指标 | Macro-F1 | 各关系类型 F1 的平均，关注少数类表现 |
| 辅助指标 | Precision | 精确率 |
| 辅助指标 | Recall | 召回率 |
| 生成质量 | Exact Match Rate | 生成的关系标签是否完全正确 |
| 生成质量 | Invalid Output Rate | 格式错误率（生成式模型特有问题） |
| 结构效率 | Avg. Retained Arcs | 平均保留弧数（衡量压缩程度） |
| 结构效率 | Compression Ratio | 保留弧数 / 原始弧数（衡量信息密度） |

### 5.8 统计显著性

- 所有实验跑 3 次（不同随机种子），报告均值和标准差
- 使用配对 t 检验（paired t-test）验证 Ours 与最强 baseline 之间的差异是否显著（p < 0.05）


---

## 六、实验进度安排

### 6.1 总体时间线

本研究计划总周期为 10-12 周，分为五个阶段：

```
Week 1-2:   [阶段一] 数据准备与预处理
Week 3-4:   [阶段二] Baseline 搭建与验证
Week 5-7:   [阶段三] 核心模型实现（EA-GMIB）
Week 8-9:   [阶段四] 全量实验与消融分析
Week 10-12: [阶段五] 补充实验与论文撰写
```

### 6.2 各阶段详细计划

#### 阶段一：数据准备与预处理（Week 1-2）

| 任务 | 具体内容 | 产出 | 预计耗时 |
|------|---------|------|---------|
| 1.1 数据集获取 | 下载 CDR、ChemProt、DDI 数据集，统一格式 | 标准化的 JSON 数据文件 | 2天 |
| 1.2 依存解析 | 用 SciSpaCy 对所有句子跑依存解析 | 每条句子的依存树（邻接矩阵+类型矩阵） | 2天 |
| 1.3 实体折叠 | 实现 Entity-Aware Graph Coarsening 算法 | 粗化后的依存图 | 3天 |
| 1.4 数据统计 | 统计句长分布、实体词数分布、依存弧数分布 | 数据分析报告，用于后续分桶实验 | 1天 |
| 1.5 输入格式构建 | 设计并实现 BioBART 的输入输出格式 | 数据加载器（DataLoader） | 2天 |

阶段一里程碑：能够加载数据并可视化原始依存树与粗化后依存图的对比。

#### 阶段二：Baseline 搭建与验证（Week 3-4）

| 任务 | 具体内容 | 产出 | 预计耗时 |
|------|---------|------|---------|
| 2.1 B0: Text-only | BioBART 纯文本关系抽取 baseline | B0 在三个数据集上的 F1 | 3天 |
| 2.2 B1: Raw Dep | 原始依存树线性化拼接 | B1 结果 | 2天 |
| 2.3 B2: SDP | 最短依存路径线性化拼接 | B2 结果 | 2天 |
| 2.4 结果分析 | 对比 B0/B1/B2，初步回答 RQ1 | 分析报告 | 1天 |
| 2.5 代码重构 | 整理代码结构，确保模块化可扩展 | 清晰的代码框架 | 2天 |

阶段二里程碑：B0-B2 跑通，初步验证依存信息对生成式 BioRE 的有效性。这一步非常关键——如果 B1/B2 相比 B0 没有提升，需要重新审视依存信息的注入方式。

#### 阶段三：核心模型实现（Week 5-7）

| 任务 | 具体内容 | 产出 | 预计耗时 |
|------|---------|------|---------|
| 3.1 Attentive GCN | 实现带注意力的 GCN 编码器 + 弧级重要性评分 | Syntax View 模块 | 3天 |
| 3.2 Semantics View | PubMedBERT 编码 + 表示提取 | Semantics View 模块 | 2天 |
| 3.3 B3/B4/B5 | 搭建单视角和双视角 baseline | B3-B5 结果 | 3天 |
| 3.4 GM-IB 模块 | 实现门控融合 + Gumbel-Softmax 弧筛选 + 熵正则化 | GM-IB 核心模块 | 5天 |
| 3.5 线性化模块 | 实现弧筛选后的可控线性化 | 线性化模块 | 2天 |
| 3.6 端到端联调 | 将所有模块串联，端到端训练调试 | 完整模型 Ours | 3天 |
| 3.7 B6: Cls-IB | 实现分类式 IB 作为对比 | B6 结果 | 2天 |

阶段三里程碑：完整模型 Ours 在至少一个数据集上跑通并优于 B5。

#### 阶段四：全量实验与消融分析（Week 8-9）

| 任务 | 具体内容 | 产出 | 预计耗时 |
|------|---------|------|---------|
| 4.1 全量主实验 | 所有模型在三个数据集上跑 3 次 | 主实验结果表 | 4天 |
| 4.2 消融实验 | A1-A5 五组消融 | 消融结果表 | 3天 |
| 4.3 beta 敏感性 | 8 个 beta 值 x 3 个数据集 | 敏感性曲线图 | 2天 |
| 4.4 统计检验 | 配对 t 检验 | 显著性结果 | 1天 |

阶段四里程碑：所有定量实验完成，主实验表和消融表填满。

#### 阶段五：补充实验与论文撰写（Week 10-12）

| 任务 | 具体内容 | 产出 | 预计耗时 |
|------|---------|------|---------|
| 5.1 鲁棒性分析 | 按句长/实体词数分桶分析 | 分桶结果表 | 2天 |
| 5.2 Case Study | 选取典型样例，可视化分析 | 可视化图表 | 2天 |
| 5.3 备选实验 | （如时间允许）加入 Lexical View | 三视角结果 | 3-5天 |
| 5.4 论文撰写 | 撰写研究内容一的完整论文章节 | 论文初稿 | 5-7天 |

阶段五里程碑：研究内容一的完整实验结果和论文章节初稿。

---

## 七、预期实验成果

### 7.1 核心产出

1. **一个可学习的依存弧筛选与线性化算法（EA-GMIB）**：该算法能够自动从依存树中筛选出对生成式 BioRE 最有用的弧子集，并线性化为结构文本序列。这是研究内容一的核心算法产出，将直接服务于大论文研究内容二的持续预训练数据生产。

2. **三个数据集上的完整实验结果**：在 CDR、ChemProt、DDI 三个生物医学关系抽取基准数据集上，验证 EA-GMIB 相比多种 baseline 的性能优势。

3. **系统的消融与分析**：通过消融实验和分桶分析，清晰地回答五个研究问题，为方法的每个组件提供实验证据。

### 7.2 预期性能提升

基于 MS2IB 在通用数据集上的实验结果（F1 提升约 0.3-2.0 个百分点），以及生物医学文本中多词实体更为普遍的特点，预期：

- 相比 text-only baseline（B0）：F1 提升 1.5-3.0 个百分点
- 相比原始依存树拼接（B1）：F1 提升 0.5-1.5 个百分点
- 相比无 IB 的双视角拼接（B5）：F1 提升 0.3-1.0 个百分点
- 实体折叠在多词实体子集上的贡献：F1 提升 1.0-2.0 个百分点

### 7.3 创新点总结

1. **问题层面**：首次系统研究面向生成式 BioRE 的依存知识蒸馏问题，明确提出"什么样的依存知识对生成式模型有用"这一研究问题
2. **方法层面**：
   - 提出实体感知的图粗化算法，针对性解决生物医学多词实体的依存噪声问题
   - 设计面向生成的信息瓶颈模块（GM-IB），用生成质量而非分类准确率驱动依存弧的筛选
   - 构建从蒸馏到线性化的完整 pipeline，产出可直接用于持续预训练的结构化文本
3. **实验层面**：在三个生物医学数据集上进行系统验证，并通过消融实验和分桶分析提供充分的实验证据

### 7.4 与大论文后续阶段的衔接

研究内容一的产出将以以下方式衔接到后续研究：

```
研究内容一产出                    研究内容二输入
EA-GMIB 算法  ──────────>  大规模处理 PubMed 摘要语料
                           生成结构化依存序列
                           作为持续预训练的训练数据

筛选后的结构序列  ──────────>  研究内容三输入
                           作为结构检索的基础表示
                           用于情景学习的示例选择
```

---

## 八、特别注意事项

### 8.1 技术风险与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|---------|
| GM-IB 训练不稳定 | 中 | 高 | 1) 先用简单的高斯假设（参考 MS2IB）；2) Gumbel 温度退火要缓慢；3) 如果 IB 不收敛，退化为注意力加权方案 |
| B1 相比 B0 无提升 | 低 | 高 | 说明原始依存树直接拼接不是好的注入方式，更加凸显信息蒸馏的必要性。调整线性化格式或注入方式 |
| 实体折叠效果不显著 | 中 | 中 | 可能是因为数据集中多词实体比例不高。在 DDI 数据集上重点验证（药物名通常较长） |
| GPU 显存不足 | 中 | 中 | 1) 减小 batch size；2) 冻结 PubMedBERT 部分层；3) 使用梯度累积 |
| BioBART 生成格式错误率高 | 低 | 中 | 1) 设计更严格的输出 schema；2) 后处理规则修正；3) 使用 constrained decoding |

### 8.2 实验规范

1. **可复现性**：固定随机种子（42, 123, 456），记录所有超参数，代码使用 Git 版本控制
2. **公平对比**：所有 baseline 使用相同的数据划分、相同的预训练模型、相同的训练 epoch 数
3. **计算资源**：预计需要单卡 GPU（RTX 3090 24GB 或同等），总训练时间约 100-150 GPU 小时
4. **代码结构**：模块化设计，每个组件（GCN、IB、线性化、生成器）独立封装，便于消融实验的开关控制

### 8.3 备选方案

如果主方案遇到重大障碍，以下备选方案可以启用：

**备选一：Lexical View 扩展**
如果两视角效果不够显著，加入词汇视角（实体词汇形态特征 + 轻量 embedding），构成三视角框架。

**备选二：简化 IB 为注意力加权**
如果 GM-IB 的变分近似训练不稳定，退化为可学习的注意力加权方案：用多头注意力替代 IB 模块，仍然保留"弧级筛选+线性化"的整体框架，只是筛选机制从信息论驱动变为注意力驱动。

**备选三：更换生成器**
如果 BioBART 效果不理想，可替换为 BioGPT 或 FLAN-T5-base，验证方法的生成器无关性。

### 8.4 关键检查点（Go/No-Go）

| 时间点 | 检查内容 | Go 条件 | No-Go 应对 |
|--------|---------|---------|-----------|
| Week 2 末 | 数据预处理完成 | 三个数据集全部处理完毕，实体折叠算法通过单元测试 | 缩减为两个数据集 |
| Week 4 末 | Baseline 结果 | B0 跑通且结果合理（与文献报告接近） | 检查数据处理和模型配置 |
| Week 7 末 | 核心模型 | Ours 在至少一个数据集上优于 B5 | 启用备选方案二（简化 IB） |
| Week 9 末 | 全量实验 | 主实验和消融实验完成 | 缩减分析实验范围 |

---

## 九、立即可以开始的工作

以下是你现在就可以着手的具体任务，按优先级排序：

### 第一步（今天就可以做）

1. **搭建项目代码框架**
   ```
   project/
   ├── data/                  # 数据存放
   │   ├── CDR/
   │   ├── ChemProt/
   │   └── DDI/
   ├── src/
   │   ├── data_processing/   # 数据预处理
   │   │   ├── dep_parser.py      # 依存解析封装
   │   │   ├── entity_coarsen.py  # 实体折叠算法
   │   │   └── linearizer.py      # 线性化模块
   │   ├── models/
   │   │   ├── syntax_view.py     # Attentive GCN
   │   │   ├── semantics_view.py  # PubMedBERT 编码
   │   │   ├── gmib.py            # GM-IB 模块
   │   │   └── generator.py       # BioBART 生成器
   │   ├── train.py               # 训练脚本
   │   ├── evaluate.py            # 评估脚本
   │   └── utils.py               # 工具函数
   ├── configs/                # 超参数配置
   ├── scripts/                # 实验运行脚本
   └── results/                # 实验结果存放
   ```

2. **下载数据集**
   - CDR: BioCreative V CDR corpus
   - ChemProt: BioCreative VI ChemProt corpus
   - DDI: DDIExtraction 2013 shared task

3. **安装依赖环境**
   ```bash
   pip install transformers datasets scispacy torch torch-geometric
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
   ```

### 第二步（本周内完成）

4. **实现实体折叠算法**（entity_coarsen.py）：这是最简单且最独立的模块，可以先写好并用几个例子验证正确性

5. **对一个数据集跑依存解析**：先拿 CDR 数据集试跑，观察依存树的质量和多词实体内部弧的情况

6. **搭建 B0 baseline**：BioBART text-only，确保生成式 RE 的基本 pipeline 跑通

### 第三步（下周完成）

7. **实现 B1 和 B2**：验证原始依存树和 SDP 的效果，初步回答 RQ1
8. **开始阅读 Gumbel-Softmax 相关论文**：为 GM-IB 的实现做理论准备

---

*文档版本：v1.0*
*创建日期：2026-03-15*
*状态：待执行*
*关联文档：start.pdf（大论文开题报告）、blueprint_by_GPT_V1.md（V1 蓝图，已迭代）、DP/1-s2.0-S0957417425018275-main.pdf（MS2IB 参考论文）*
