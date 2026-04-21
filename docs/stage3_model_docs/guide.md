# 阶段三实验行动指南：EA-GMIB 核心模型实现

**文档定位**：阶段三执行手册，面向实验室执行人员  
**对应计划**：`docs/0_experiment_plan_v1.md` 阶段三（Week 5-7）  
**阶段里程碑**：完整模型 Ours（EA-GMIB）在 ChemProtSent 或 CDRIntra 上端到端跑通，性能优于 B5  
**主力数据集**：ChemProtSent、CDRIntra（DDI 暂缓，时间充足时再处理）  
**核心原则**：先跑通流程，再扩大规模；先测单模块，再做联调；先本地 smoke，再服务器 full run

---

## 总体路线图

阶段三按以下顺序推进，每一步都必须验证通过再进行下一步：

```
[Day 0]   行前确认 + 环境准备 + 架构规划
[Day 1-2] 图数据构建桥接层（graph_builder + graph_collator）
[Day 3-4] 语义视角模块（PubMedBERT 编码）
[Day 5-7] 语法视角模块（Attentive GCN）
[Day 8-10] B3/B4/B5 中间基线：smoke → full run
[Day 11-15] GM-IB 核心模块（信息瓶颈蒸馏）
[Day 16]  线性化功能扩展
[Day 17-19] 完整模型组装与端到端联调
[Day 20-21] B6 分类式 IB 对照实验
```

---

## 第零步：行前确认（Day 0 上午）

在动任何代码之前，先完成以下确认，确保接手环境是健康的。

### 0.1 确认阶段一数据产物完整

检查以下目录和文件是否存在且非空：

```
data/experiment_views/coarsened/ChemProtSent_train.json
data/experiment_views/coarsened/ChemProtSent_dev.json
data/experiment_views/coarsened/ChemProtSent_test.json
data/experiment_views/coarsened/CDRIntra_train.json
data/experiment_views/coarsened/CDRIntra_dev.json
data/experiment_views/coarsened/CDRIntra_test.json
```

然后随机打开一条数据记录，确认它包含以下字段（这些是阶段三模型必需的）：

- `tokens`、`dep_heads`、`dep_labels`（原始依存信息）
- `coarse_tokens`、`coarse_heads`、`coarse_labels`（粗化后依存图）
- `coarse_e1_idx`、`coarse_e2_idx`（实体在粗化图中的节点索引）
- `coarse_node_types`（节点类型：普通节点 or 实体超级节点）
- `entity1`、`entity2`、`relation`（基本实体对信息）

如果上述字段缺失，说明阶段一数据没有完整传递，需要先追溯阶段一报告修复数据。

### 0.2 确认阶段二代码可运行

用一个 smoke 配置文件跑一次阶段二的 B0，确保现有代码框架没有损坏：

```bash
cd <项目根目录>
python src/baseline/run_smoke.py --config configs/baseline_local_smoke_chemprotsent_b0.json
```

看到 `eval_loss` 和 `mapped_exact` 有输出即可，不需要关注数值是否好看。

> **为什么要做这步？** 阶段三的代码会在阶段二的基础上扩展，如果起点就有问题，后续的 bug 会非常难定位。

### 0.3 记录当前环境信息

在终端执行并记录结果（粘贴到记事本备用）：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import transformers; print(transformers.__version__)"
```

---

## 第一步：环境准备（Day 0 下午）

### 1.1 下载 PubMedBERT 到本地

阶段三的语义视角模块需要 PubMedBERT。为了和已有的 BioBART 保持一致的本地化管理方式，将模型下载到项目的 `models/` 目录下。

**目标路径**：`models/pubmedbert-base/`

使用 Hugging Face 提供的下载方式：

```python
from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained("models/pubmedbert-base")
model.save_pretrained("models/pubmedbert-base")
```

> **注意**：文件较大（约 440MB），请确保网络畅通，或使用 `huggingface-cli` 工具下载。下载完成后，确认 `models/pubmedbert-base/` 目录下存在 `config.json`、`pytorch_model.bin`（或若干 `.safetensors` 分片）以及 `tokenizer_config.json`。

### 1.2 安装 PyTorch Geometric

阶段三的 GCN 实现依赖 PyTorch Geometric。安装时需要与当前 PyTorch 版本匹配：

```bash
# 先查看当前 PyTorch 和 CUDA 版本
python -c "import torch; print(torch.__version__)"

# 按照 https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# 选择对应版本安装，例如：
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-{YOUR_TORCH_VERSION}+{CUDA_VERSION}.html
```

将 `{YOUR_TORCH_VERSION}` 和 `{CUDA_VERSION}` 替换为你的实际版本（如 `2.1.0+cu121`）。

**验证安装**：

```python
import torch_geometric
print(torch_geometric.__version__)
```

### 1.3 验证 PubMedBERT 可正常加载

写一段简单的测试脚本（不需要保存为文件，直接在交互式 Python 中运行）：

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("models/pubmedbert-base")
model = AutoModel.from_pretrained("models/pubmedbert-base")
model.eval()

test_input = tokenizer("Aspirin inhibits gene expression.", return_tensors="pt")
with torch.no_grad():
    output = model(**test_input)

print("CLS embedding shape:", output.last_hidden_state[:, 0, :].shape)  # 应输出 torch.Size([1, 768])
```

看到 `torch.Size([1, 768])` 即表示 PubMedBERT 加载成功。

---

## 第二步：代码架构总览（Day 0，动手前必读）

> **这一步不写代码，只规划。** 架构不清楚就动手，后续会产生大量重构成本。

### 2.1 阶段三新增文件清单

阶段三在现有代码基础上，新增以下文件（不修改阶段二已有文件，保持阶段二可独立运行）：

```
src/
├── builders/
│   ├── input_builder.py          ← 已有，不动
│   └── graph_builder.py          ← 新增：将 coarsened JSON 转为图矩阵
│
├── data/
│   ├── dataset.py                ← 已有，不动（或最小扩展）
│   ├── dataloader.py             ← 已有，不动
│   ├── collator.py               ← 已有，不动
│   └── graph_collator.py         ← 新增：处理变长图数据的批处理对齐
│
├── models/                       ← 新增目录（阶段二没有此目录）
│   ├── __init__.py
│   ├── semantics_view.py         ← 新增：PubMedBERT 语义视角编码器
│   ├── syntax_view.py            ← 新增：Attentive GCN 语法视角编码器
│   ├── gmib.py                   ← 新增：GM-IB 信息瓶颈核心模块
│   ├── gmib_cls.py               ← 新增：分类式 IB（B6 用）
│   └── ea_gmib.py                ← 新增：完整 EA-GMIB 模型（组装以上所有模块）
│
├── stage3/                       ← 新增目录（与 baseline/ 并列）
│   ├── __init__.py
│   ├── run_b345_smoke.py         ← 新增：B3/B4/B5 本地 smoke runner
│   ├── run_b345_full.py          ← 新增：B3/B4/B5 服务器 full runner
│   ├── run_ours_smoke.py         ← 新增：完整模型本地 smoke runner
│   ├── run_ours_full.py          ← 新增：完整模型服务器 full runner
│   └── run_b6_smoke.py           ← 新增：B6 smoke runner
│
└── tests/
    └── test_stage3_modules.py    ← 新增：各模块独立单元测试

configs/
├── ...（已有阶段二配置，不动）
├── stage3_smoke_b3_chemprotsent.json     ← 新增
├── stage3_smoke_b4_chemprotsent.json     ← 新增
├── stage3_smoke_b5_chemprotsent.json     ← 新增
├── stage3_smoke_ours_chemprotsent.json   ← 新增
├── stage3_full_b3_chemprotsent_seed42.json  ← 新增（以此类推）
└── ...（full run 配置按数据集和 seed 命名）
```

### 2.2 模块依赖关系

理解各模块之间的数据流，是后续实现不出错的关键：

```
[coarsened JSON 数据]
        │
        ▼
[graph_builder.py]          ←── 把 coarse_tokens/heads/labels 转为矩阵
        │
        │  (adj_matrix, dep_type_matrix, node_count, e1_idx, e2_idx)
        ▼
[graph_collator.py]         ←── 对一个 batch 内变长矩阵做 padding 对齐
        │
        ▼
┌──────────────────────────────────────┐
│            EA-GMIB 模型              │
│                                      │
│  [semantics_view.py]                 │
│     PubMedBERT(sentence)             │
│     → H_sem (CLS)                    │
│     → X_init (token embeddings)      │
│            │                         │
│            ▼                         │
│  [syntax_view.py]                    │
│     AttentiveGCN(X_init, adj, dep)   │
│     → H_syn (node repr)              │
│     → a_ij (arc scores)              │
│            │                         │
│            ▼                         │
│  [gmib.py]                           │
│     GM-IB(H_syn, H_sem, a_ij)        │
│     → p_ij (arc keep probs)          │
│     → S* (selected arc subset)       │
│            │                         │
│  [linearize_selected_arcs()]         │
│     → dep_text (文本序列)             │
│            │                         │
│  BioBART(sentence + dep_text)        │
│     → L_gen                          │
│                                      │
│  L_total = L_gen + β * L_compress    │
└──────────────────────────────────────┘
```

### 2.3 编码规范（执行人员必须遵守）

1. **命名规范**：文件名小写加下划线（`graph_builder.py`），类名大写驼峰（`GraphBuilder`），函数名小写加下划线（`build_adj_matrix`），超参数统一通过配置文件传入，禁止在代码中硬编码数值

2. **解耦原则**：每个模块（`semantics_view.py`、`syntax_view.py`、`gmib.py`）必须可以独立实例化和测试，不得在构造函数中依赖其他模块的内部状态

3. **配置驱动**：阶段三所有超参数（GCN 层数、attention heads、β、τ 初始值、top-k 等）全部写进 JSON 配置文件，`configs/` 目录下按 `stage3_{scale}_{model}_{dataset}[_seed].json` 命名

4. **不破坏阶段二**：阶段三的所有新代码放在新文件或新目录中，不修改 `src/baseline/`、不修改已有 `collator.py`、不修改已有 `dataset.py`（如必须扩展 dataset，只允许子类继承，不修改原类）

5. **每个文件顶部写 docstring**：说明该模块的职责、输入、输出

---

## 第三步：图数据构建桥接层（Day 1-2）

这是阶段三第一个需要实现的模块，也是最基础的。它的作用是将阶段一产出的 coarsened JSON 数据，转换为 GCN 所需的矩阵格式。

### 3.1 新建：`src/builders/graph_builder.py`

**这个文件的职责**：给定一条 coarsened JSON 记录的图字段，输出：
- `adj_matrix`：N×N 的邻接矩阵（0/1，无向图，即如果 head→dep 存在，则 adj[head][dep] 和 adj[dep][head] 都置为1）
- `dep_type_ids`：N×N 的依存类型编码矩阵（把依存关系类型字符串映射为整数 ID，无弧的位置填0）
- `node_count`：N，该样本粗化图中的节点总数
- `e1_node_idx`：实体1在粗化图中的节点索引（直接来自 `coarse_e1_idx`）
- `e2_node_idx`：实体2在粗化图中的节点索引（直接来自 `coarse_e2_idx`）

**关键设计决策**：
- 邻接矩阵使用无向图（双向边），这是 GCN 的标准做法，有助于信息的双向传播
- ROOT 节点（head 指向自身或 head=-1）的弧跳过，不加入邻接矩阵
- 依存类型的整数映射需要维护一个全局词表（`dep_type_vocab`），这个词表在第一次遍历数据时构建，并保存为一个 JSON 文件（如 `data/dep_type_vocab.json`），后续所有实验共用同一个词表，**不得重复构建**

**构建词表的时机**：在开始正式训练之前，先跑一个数据预扫描脚本，遍历 ChemProtSent 和 CDRIntra 的所有 split，收集所有出现过的 `coarse_labels`，建立词表并保存。

**验证方法**：
1. 取一条 ChemProtSent 数据记录
2. 用 `graph_builder.py` 处理它
3. 打印 `node_count`、`adj_matrix.shape`、`dep_type_ids.shape`
4. 人工检查：`adj_matrix[i][j]` 为1的位置，应该对应 `coarse_heads[j] == i` 或 `coarse_heads[i] == j`

### 3.2 新建：`src/data/graph_collator.py`

**为什么需要单独的 collator？**

当一个 batch 里有多条样本时，每条样本的粗化图节点数 N 不同（有的句子短，图小；有的句子长，图大）。普通的 `torch.stack` 无法处理不同尺寸的矩阵。`graph_collator.py` 的职责就是对一个 batch 内的所有图矩阵做 padding，对齐到 `max_N × max_N`，并生成对应的 node mask。

**这个 Collator 需要输出什么**（每个字段都是 batch 维度的 Tensor）：

- 阶段二 Collator 已经输出的字段（文本侧）：`input_ids`、`attention_mask`、`labels`（直接复用阶段二的逻辑）
- **新增**（图侧）：
  - `adj_matrix`：形状 `[B, max_N, max_N]`，padding 区域填 0
  - `dep_type_ids`：形状 `[B, max_N, max_N]`，padding 区域填 0
  - `node_mask`：形状 `[B, max_N]`，有效节点位置为 True，padding 位置为 False
  - `e1_node_idx`：形状 `[B]`，每条样本实体1的节点索引
  - `e2_node_idx`：形状 `[B]`，每条样本实体2的节点索引
  - `pubmedbert_input_ids`：形状 `[B, seq_len]`，PubMedBERT 的文本输入（注意：PubMedBERT 和 BioBART 使用不同的 tokenizer，这里是 PubMedBERT 的 tokenizer 编码结果）
  - `pubmedbert_attention_mask`：形状 `[B, seq_len]`

**注意：PubMedBERT 有独立的 tokenizer**。在 collator 中，需要同时调用两个 tokenizer：BioBART 的 tokenizer（用于生成器输入）和 PubMedBERT 的 tokenizer（用于语义视角输入）。这两个 tokenizer 产出的 token 序列长度不同，是完全独立的两路输入。

**验证方法**：
1. 构造一个含2条样本的 mini batch
2. 跑一次 collator
3. 检查：`adj_matrix.shape[0]` == 2，`adj_matrix.shape[1]` == `adj_matrix.shape[2]` == `max_N`（等于这批样本中最大的节点数），`node_mask.sum(dim=1)` 等于每条样本的实际节点数

---

## 第四步：语义视角模块（Day 3-4）

### 4.1 新建：`src/models/semantics_view.py`

**这个模块的职责**：用 PubMedBERT 对含实体标记的句子进行编码，产出两种表示供后续使用：

1. **语义表示 H_sem**：取 `[CLS]` token 的最终隐层向量，形状 `[B, 768]`，代表整个句子的全局语义，用于后续 GM-IB 的融合
2. **节点初始化嵌入 X_init**：形状 `[B, max_N, 768]`，用于初始化 GCN 的节点特征

**关键问题：PubMedBERT 的 token 和粗化图节点的对齐**

这是阶段三最需要细心处理的工程问题之一。PubMedBERT 使用 WordPiece 分词，而粗化图使用 SpaCy 分词。两者不是一对一的关系。具体处理策略如下：

- 对于**普通节点**（对应一个 SpaCy token）：找到该 SpaCy token 在原始句子中的字符偏移，然后找到 PubMedBERT tokenization 中覆盖该偏移区间的所有 WordPiece token，取它们的最终隐层向量均值作为该节点的初始表示
- 对于**超级节点**（对应一个多词实体，如 "non-small cell lung cancer"）：找到整个实体 span 在原始句子中的字符偏移范围，取 PubMedBERT 中覆盖该范围的所有 WordPiece token 的均值

**实现建议**：在 `graph_builder.py` 阶段就计算好每个粗化节点对应的字符偏移区间，存入一个辅助字段（如 `coarse_node_char_spans`）。在 `semantics_view.py` 中，利用 PubMedBERT tokenizer 的 `return_offsets_mapping=True` 参数获得每个 WordPiece token 的字符偏移，然后做区间匹配。

**这个模块的类接口**（不写代码，只描述接口）：
- 输入：`pubmedbert_input_ids [B, L]`，`pubmedbert_attention_mask [B, L]`，`node_char_spans [B, max_N, 2]`（每个节点的起止字符偏移），`node_mask [B, max_N]`
- 输出：`H_sem [B, 768]`（CLS 表示），`X_init [B, max_N, 768]`（节点初始表示）

**验证方法**：
1. 取一条 ChemProtSent 样本
2. 用 `semantics_view.py` 处理
3. 检查输出形状：`H_sem.shape` == `[1, 768]`，`X_init.shape` == `[1, max_N, 768]`
4. 检查有效节点的表示范数是否合理（不应全为0或异常大），padding 节点的表示应为全0（通过 node_mask 屏蔽）

---

## 第五步：语法视角模块（Day 5-7）

### 5.1 新建：`src/models/syntax_view.py`

**这个模块的职责**：在粗化依存图上做结构编码，输出节点表示和弧级重要性分数。

**模块内部结构**（按照实验计划书的设计）：

**Step 1 - 多头注意力**：对 `X_init`（PubMedBERT 节点嵌入）做自注意力，得到 `H_att [B, max_N, 768]`。注意力只在有效节点之间计算（利用 `node_mask` 屏蔽 padding 节点）。注意力头数设为8（参考实验计划书）。

**Step 2 - 2层 GCN 聚合**：在粗化邻接矩阵上做图卷积。每一层 GCN 的操作是：`H^(l) = ReLU(A_norm * H^(l-1) * W^(l))`，其中 `A_norm` 是归一化的邻接矩阵（加上自环后做对称归一化，即 GCN 的标准做法）。第一层输入是 `H_att`，第二层输入是第一层输出，最终取第二层输出为 `H_syn [B, max_N, d_syn]`（`d_syn` 是 GCN 的隐层维度，推荐设为256）。

**Step 3 - 弧级重要性评分**：对图中存在的每条弧 `(i, j)`，计算重要性分数：`a_ij = sigmoid(MLP([h_i; h_j; t_ij]))`，其中 `h_i`、`h_j` 是两端节点的表示（来自 `H_syn`），`t_ij` 是依存类型的嵌入向量（从依存类型嵌入表查找）。MLP 是一个两层的全连接网络，输出维度为1（标量分数）。`a_ij` 存为一个稀疏的弧级分数矩阵，形状为 `[B, max_N, max_N]`（只有邻接矩阵中为1的位置有有效分数）。

**关键工程细节**：
- GCN 中的归一化邻接矩阵 `A_norm` 是在 `graph_collator.py` 中预计算好传入，还是在 `syntax_view.py` 的 forward 中计算？**推荐在 forward 中计算**，这样依赖关系更清晰，`graph_collator.py` 只负责传原始的 `adj_matrix`
- `d_syn` 和 BioBART 的 encoder hidden size（768）不同，在后续融合时需要一个线性投影层。推荐在 `ea_gmib.py` 的组装层处理这个投影，不在 `syntax_view.py` 内部处理（保持 `syntax_view.py` 单一职责）

**验证方法**：
1. 先用随机初始化的 `X_init` 测试 `syntax_view.py`，不依赖 PubMedBERT
2. 构造 batch_size=2，max_N=10 的假数据（随机邻接矩阵、随机 `X_init`）
3. 跑 forward pass，检查 `H_syn.shape` == `[2, 10, 256]`，`a_ij` 矩阵中有效弧的分数在 (0,1) 之间
4. 然后换用 `semantics_view.py` 产出的真实 `X_init`，再做一次验证

---

## 第六步：B3/B4/B5 中间基线实验（Day 8-10）

> **为什么在做 GM-IB 之前先做 B3/B4/B5？**
> 因为 GM-IB 模块构建在这三个中间基线之上。先把单视角和双视角拼接的版本跑通，一方面验证各模块接口没有问题，另一方面也获得后续消融实验所需的对照数据（B3/B4/B5 也是正式消融实验的一部分）。

### 6.1 B3/B4/B5 的设计

三个中间基线的核心区别在于：如何将结构/语义信息"注入"到 BioBART 的输入中。注入方式统一为：将结构/语义的**汇总向量**（压缩为一个固定维度的 token embedding）拼接在 BioBART 的 encoder 输入序列末尾（或前缀），让 BioBART 通过注意力机制自行决定如何利用这些信息。

**B3（语法单视角）**：只使用 GCN 输出的 `H_syn`。将 `H_syn` 做 mean pooling（对有效节点均值）得到一个 `[B, d_syn]` 的图级表示，再通过一个线性投影层映射到 BioBART 的 hidden size（768），作为一个特殊 `[GRAPH]` token 的 embedding，拼接到 BioBART encoder 输入序列的末尾。

**B4（语义单视角）**：只使用 PubMedBERT 的 `H_sem`（CLS 表示，`[B, 768]`）。同样通过投影层（此处维度不变，但加一个 LayerNorm）得到 `[SEMANT]` token 的 embedding，拼接到 BioBART encoder 输入。

**B5（双视角拼接）**：同时使用 B3 和 B4 的结果，将两个向量 concat 后通过一个投影层压缩回 768 维，作为 `[DUAL]` token 拼接到 BioBART 输入。不做任何信息蒸馏，直接拼接。

### 6.2 B3/B4/B5 的实验流程

**新建**：`src/stage3/run_b345_smoke.py` 和对应的配置文件

**配置文件关键字段**（在 configs/ 下新建，不修改阶段二配置）：

| 字段 | 说明 |
|------|------|
| `model_type` | "b3" / "b4" / "b5"，控制用哪个视角 |
| `dataset_name` | "ChemProtSent" 或 "CDRIntra" |
| `pubmedbert_path` | "models/pubmedbert-base" |
| `biobart_path` | "models/biobart-base" |
| `gcn_hidden_dim` | GCN 的隐层维度，推荐 256 |
| `gcn_layers` | 2 |
| `gcn_attn_heads` | 8 |
| `max_train_samples` | smoke 时设为 200，full run 时设为 null（用全量数据） |
| `batch_size` | smoke 时 1，full run 时 8 |
| `epochs` | smoke 时 2，full run 时 20 |
| `seed` | 42 |
| `dep_type_vocab_path` | "data/dep_type_vocab.json" |

**推进节奏**：

**Step 1（本地 smoke，Day 8）**：先跑 B3 的 smoke（200条样本，2 epoch）。目标是 forward pass 无报错，`eval_loss` 有值，`mapped_exact` 有值（数值是多少不重要）。

**Step 2（本地 smoke，Day 8）**：在 B3 通过后，依次跑 B4、B5 的 smoke。同样只看流程是否通，不看性能。

**Step 3（服务器 full run，Day 9-10）**：将配置文件的 `max_train_samples` 改为 null、`batch_size` 改为 8、`epochs` 改为 20，部署到 5070 服务器上做 full run。先跑 ChemProtSent，再跑 CDRIntra。每个 full run 跑完后记录 `mapped_exact`，用于后续对比。

> **显存注意事项**：B4 需要同时加载 PubMedBERT + BioBART，两个模型加起来约 1.6GB 参数。以 batch_size=8、序列长度128 估计，显存需求约 10-12GB。笔记本的 4070（8GB）可以跑 batch_size=1-2 的 smoke，full run 必须去服务器（5070，显存充足）。

---

## 第七步：GM-IB 核心模块（Day 11-15）

> **这是整个阶段三最核心、最复杂的模块，请分配最多的时间，不要赶。**

### 7.1 新建：`src/models/gmib.py`

**这个模块的职责**：接受 `H_syn`（GCN 输出）、`H_sem`（PubMedBERT CLS）和 `a_ij`（弧级重要性分数），输出：
- `p_ij`：每条弧的保留概率，形状 `[B, max_N, max_N]`（有效弧位置有值）
- `z_ij`：训练时的 Gumbel-Softmax 软采样，推理时的 top-k 硬选择
- `L_compress`：信息压缩损失（标量）

**GM-IB 的内部实现分为5个子步骤**：

**子步骤1 - 门控融合（GatingFusion）**

将 `H_syn`（节点级，`[B, max_N, d_syn]`）和 `H_sem`（句子级，`[B, 768]`）融合。

由于两者维度和语义层面不同，先对 `H_syn` 做 mean pooling 得到图级表示（形状 `[B, d_syn]`），然后通过线性投影对齐到同一维度 `d_fused`（推荐768），再做门控：

```
g = sigmoid(W_g · concat([H_syn_pooled; H_sem]) + b_g)
H_fused = g * H_syn_projected + (1-g) * H_sem
```

注意：这里的 `H_fused` 是句子级的融合表示（`[B, d_fused]`），用于后续弧级评分时为每个节点对提供全局上下文。

实际上，弧级评分时需要节点级表示。因此，门控融合的更精细做法是：对每个节点 i，将该节点的 `H_syn` 表示和全局的 `H_sem` 做融合，得到节点的融合表示 `H_fused_node [B, max_N, d_fused]`：

```
g_i = sigmoid(W_g · concat([h_syn_i; H_sem]) + b_g)
h_fused_i = g_i * W_syn(h_syn_i) + (1-g_i) * W_sem(H_sem)
```

**子步骤2 - 弧级保留概率（ArcScoringMLP）**

对每条存在的弧 `(i, j)`，利用两端节点的融合表示和弧的先验分数计算保留概率：

```
p_ij = sigmoid(MLP_ib(concat([h_fused_i; h_fused_j; a_ij_scalar])))
```

其中 `a_ij_scalar` 是来自 `syntax_view.py` 的弧重要性先验分数（标量）。MLP_ib 是一个两层全连接网络，输入维度 = `2 * d_fused + 1`，输出维度 = 1。

**子步骤3 - Gumbel-Softmax 弧选择（训练期间）**

由于弧的保留/丢弃是离散决策，不可直接反向传播，使用 Binary Concrete（二值 Gumbel-Softmax）近似：

```
z_ij = sigmoid((log(p_ij) - log(1-p_ij) + gumbel_noise_1 - gumbel_noise_2) / τ)
```

其中 `gumbel_noise_1, gumbel_noise_2 ~ Gumbel(0,1)`，`τ` 为温度参数（初始1.0，逐渐降至0.1）。`z_ij` 在 (0,1) 之间，当 `τ→0` 时趋近于0或1的硬选择。

Gumbel 分布的采样方法：`g = -log(-log(u + ε))` 其中 `u ~ Uniform(0,1)`，`ε` 是防止 log(0) 的小常数（如 1e-20）。

**推理期间**：不采样噪声，直接 `z_ij = 1 if p_ij > threshold else 0`（阈值法），或取 top-k 个分数最高的弧。具体使用哪种方式由配置文件控制（`inference_mode`: "threshold" 或 "topk"）。

**子步骤4 - 信息压缩损失（EntropyRegularization）**

信息压缩项用于鼓励模型只保留少量关键弧（稀疏化）。有两种合理实现，执行人员**在实际编码前必须和负责人确认使用哪种**：

**方案A（L1正则，最简单）**：`L_compress = mean(p_ij)`，对所有有效弧的保留概率求均值。β越大，模型被鼓励把 p_ij 推向0，保留越少弧。

**方案B（二值熵正则）**：`L_compress = mean(-p_ij * log(p_ij) - (1-p_ij) * log(1-p_ij))`，即所有有效弧的二值熵之和取均值。最小化此项会让每条弧的决策趋向确定性（p_ij 趋向0或1），结合生成损失的拉力，最终趋向只有少数弧的 p_ij 接近1。

**推荐先用方案A（简单，稳定），确认整个模型收敛后再尝试方案B。**

**子步骤5 - 总损失**

`L_total = L_gen + β * L_compress`

- `L_gen`：BioBART 的生成损失（标准 cross-entropy，来自 BioBART forward 的 `loss` 字段）
- `β`：超参数，从配置文件读取，初始实验设为 `1e-6`
- 注意：`L_compress` 和 `L_gen` 的数量级可能差异很大，β 的调整要参考两者各自的数量级

### 7.2 关键超参数 τ 的退火策略

温度 τ 的退火在 `trainer.py` 的训练循环中实现，**每个 epoch 结束时更新**：

```
τ_new = max(τ_min, τ_current * anneal_rate)
```

推荐配置：`τ_init=1.0`，`τ_min=0.1`，`anneal_rate=0.95`（30个epoch后 τ 约降至0.21）。

τ 应该通过 `model.update_temperature(τ_new)` 这样的接口更新，而不是通过全局变量传递。

### 7.3 GM-IB 的验证方法

**验证1 - 前向传播形状验证**：

构造 batch_size=2，max_N=8，有效弧数若干的假数据，跑 GM-IB 的 forward。检查：
- `p_ij.shape` == `[2, 8, 8]`（有效弧位置有值，无效位置为0）
- `z_ij` 的值在 (0,1) 之间（训练模式）
- `L_compress` 是一个标量

**验证2 - 梯度流通验证**：

跑一次 forward + backward，检查 GM-IB 内部参数（MLP_ib 的权重）是否有梯度（`.grad is not None`）。这验证了 Gumbel-Softmax 的可微性是否正常工作。

**验证3 - 温度影响验证**：

固定输入，用不同 τ（1.0，0.5，0.1）跑 forward，观察 `z_ij` 的分布：τ 越小，`z_ij` 越趋向 0 或 1（方差增大，均值不变）。

> **常见错误警告**：
> - `log(0)` 导致 NaN：在所有 log 计算中加 `ε = 1e-10` 保护
> - `p_ij` 梯度消失：检查 MLP_ib 是否有合适的初始化（推荐 Xavier 均匀初始化）
> - `L_compress` 数量级远大于 `L_gen`：先把 β 设成很小（1e-8 甚至更小）再观察

---

## 第八步：线性化功能扩展（Day 16）

### 8.1 扩展线性化逻辑

阶段二已有 `src/builders/input_builder.py`，其中的 `linearize_dependency_arcs` 和 `linearize_shortest_dependency_path` 已经实现了基本的线性化逻辑。

阶段三的 GM-IB 输出的是**动态选择**的弧子集 `S*`（由 `z_ij` 决定），需要新增一个函数来处理这种动态弧集合的线性化。

**新增方案**：在 `src/builders/input_builder.py` 末尾追加一个新函数 `linearize_selected_arcs()`，或者在 `src/builders/` 目录下新建 `gmib_linearizer.py`（推荐后者，保持解耦）。

**这个新函数的输入/输出**：
- 输入：`selected_arc_mask [max_N, max_N]`（布尔矩阵，True 表示该弧被选中）、`coarse_tokens`（节点文本列表）、`coarse_heads`（原始 head 数组，用于判断弧方向）、`coarse_labels`（依存类型列表）、`e1_node_idx`、`e2_node_idx`、`p_ij`（弧的保留概率，用于排序）、`max_dep_len`（最大线性化长度，来自配置文件）
- 输出：一个字符串，格式为 `"[DEP] node_i -dep_type-> node_j ; ... [/DEP]"`，与阶段二的格式保持一致

**排序规则**（按优先级）：
1. 优先包含实体1到实体2的最短路径上的弧
2. 其余弧按 `p_ij` 降序排列
3. 超过 `max_dep_len` 的部分截断

**推理阶段的使用方式**：在 `ea_gmib.py` 的推理 forward 中，先用 top-k 或阈值得到 `selected_arc_mask`，再调用 `linearize_selected_arcs` 得到 `dep_text`，拼接到句子后，送入 BioBART decode。

**训练阶段的处理方式**：训练时 `z_ij` 是连续值（Gumbel-Softmax 软采样），不直接用于线性化。训练时，GM-IB 的作用是通过 `z_ij` 加权 `H_fused`，将加权后的节点表示传递给 BioBART（而不是走文本线性化路径）。**文本线性化路径只在推理时启用**。

> 这是一个重要的设计决策：训练时走的是"软融合向量路径"（可微），推理时走的是"硬选择线性化路径"（可解释）。两条路径共用同一套 `p_ij` 参数。

---

## 第九步：完整模型组装与端到端训练（Day 17-19）

### 9.1 新建：`src/models/ea_gmib.py`

这个文件将所有模块组装为完整的 EA-GMIB 模型。它不包含任何新的模型逻辑，只负责：
1. 在 `__init__` 中初始化所有子模块（PubMedBERT、AttentiveGCN、GM-IB、BioBART，以及各维度投影层）
2. 在 `forward` 中按照实验计划书的顺序串联各步骤
3. 计算并返回 `L_total`、`L_gen`、`L_compress` 以及各中间量（用于 logging 和 debug）
4. 提供一个 `update_temperature(tau)` 接口供 trainer 在 epoch 结束时调用

**训练时的 forward 流程**（严格按照以下顺序）：
1. `semantics_view` forward → `H_sem`, `X_init`
2. `syntax_view` forward（接收 `X_init`, `adj_matrix`, `dep_type_ids`, `node_mask`）→ `H_syn`, `a_ij`
3. `gmib` forward（接收 `H_syn`, `H_sem`, `a_ij`, `node_mask`，训练模式下启用 Gumbel 采样）→ `p_ij`, `z_ij`, `L_compress`
4. 生成 BioBART encoder 的增强表示（将 `z_ij` 加权的节点表示池化后投影，拼接到 BioBART encoder 输入）
5. BioBART forward → `L_gen`
6. `L_total = L_gen + β * L_compress`，返回

**推理时的 forward 流程**：
1. 前3步同训练时，但 GM-IB 使用硬选择（`p_ij > threshold`）而非 Gumbel 采样
2. 调用 `linearize_selected_arcs` 生成 `dep_text`
3. 将 `dep_text` 拼接到句子后，送入 BioBART.generate() 解码

### 9.2 新建：`src/stage3/trainer.py`

这个文件实现阶段三专用的训练循环，区别于阶段二的训练方式：

**关键新增逻辑**：
1. 每个 epoch 结束时，调用 `model.update_temperature(tau)` 做 τ 退火
2. 训练 log 中除 `train_loss`、`eval_loss`、`mapped_exact` 外，还要额外记录：
   - `avg_retained_arcs`：每个 batch 平均保留的弧数（等于 `z_ij.sum() / batch_size`）
   - `compression_ratio`：保留弧数 / 总有效弧数
   - `current_tau`：当前温度值
   - `L_gen_value`、`L_compress_value`：两项损失的独立值（用于诊断 β 是否合适）

**早停机制**：以 `eval mapped_exact` 为标准，patience=5，连续5个epoch不提升则停止。

### 9.3 端到端训练的推进节奏

**Step 1 - 本地 smoke（Day 17，使用笔记本）**：

配置文件 `configs/stage3_smoke_ours_chemprotsent.json` 设置：
- `max_train_samples: 200`
- `max_eval_samples: 50`
- `batch_size: 1`
- `epochs: 2`
- `β: 1e-6`
- `inference_mode: "threshold"`，`threshold: 0.5`

目标：forward + backward 无报错，`L_gen` 数值合理（参考阶段二 B0 的 `eval_loss` 范围），`avg_retained_arcs` 大于0（说明有弧被选中），`L_compress` 是合理标量。

**如果 smoke 失败的排查顺序**：
1. 先隔离测试 `semantics_view.py`（独立跑，看输出形状）
2. 再隔离测试 `syntax_view.py`（用 semantics_view 的输出作为输入）
3. 再隔离测试 `gmib.py`（用前两步的输出）
4. 最后联调 `ea_gmib.py`

**Step 2 - 服务器 full run（Day 18-19，使用 5070 主机）**：

smoke 通过后，修改配置为 full run 设置并部署到服务器：
- `max_train_samples: null`（全量）
- `batch_size: 8`
- `epochs: 30`
- 初始 β 先用 `1e-6`
- 随机种子先只跑 `seed=42`

**判断 β 是否合适**：观察训练过程中的 `avg_retained_arcs`。如果始终接近总弧数（几乎不压缩），说明 β 太小，需要增大；如果迅速降至0附近（过度压缩），说明 β 太大，需要减小。理想情况是 `compression_ratio` 在 0.3-0.7 之间收敛。

**里程碑验证**：`Ours` 在 ChemProtSent 或 CDRIntra 上的 `mapped_exact`（seed=42）优于同数据集的 B5 结果（参考阶段二结果表），则阶段三核心里程碑达成。

---

## 第十步：B6 分类式 IB 对照实验（Day 20-21）

### 10.1 新建：`src/models/gmib_cls.py`

B6 是 `Ours` 的一个变体，将信息瓶颈的驱动信号从**生成损失**替换为**分类损失**。这个实验的目的是验证 GM-IB（生成导向）是否优于传统的分类导向 IB（RQ4）。

`gmib_cls.py` 的实现应尽量复用 `gmib.py` 的结构，只修改以下两处：
1. 在 GM-IB 模块顶部增加一个**分类头**：一个线性层，输入 `H_fused`（融合后的句子表示），输出维度等于关系类别数（ChemProtSent 有5+1类，CDRIntra 有2类）
2. 将 `L_gen` 替换为 `L_cls = CrossEntropy(分类头输出, gold_relation_id)`

注意：B6 不需要 BioBART 的生成解码。在推理时，分类头的 argmax 输出即为预测的关系类别。

### 10.2 B6 的实验流程

同样遵循"先本地 smoke，再服务器 full"的节奏。配置文件新建 `configs/stage3_smoke_b6_chemprotsent.json` 和 `configs/stage3_full_b6_chemprotsent_seed42.json`。

评价指标需要特别注意：B6 是分类模型，需要计算标准的 Micro-F1 和 Precision/Recall（而不是 `mapped_exact`）。需要确认评估脚本是否支持分类模式的 F1 计算（可能需要扩展 `src/stage3/trainer.py` 的评估部分）。

---

## 附录A：常见错误与排查清单

| 错误现象 | 最可能的原因 | 排查步骤 |
|---------|------------|---------|
| `adj_matrix` 形状不一致报错 | graph_collator 的 padding 逻辑有 bug | 打印 batch 内每个样本的 `node_count`，检查 padding 是否对齐到 `max_N` |
| PubMedBERT tokenizer 报找不到词表 | `models/pubmedbert-base/` 目录结构不完整 | 检查目录下是否有 `tokenizer_config.json` 和 `vocab.txt` |
| GCN 输出全为 0 | 归一化邻接矩阵计算有误（除数为0） | 检查是否有节点度为0的孤立节点，确保归一化时加了自环 |
| `L_total` 出现 NaN | log 计算遇到0或负数 | 在所有 log 和 sigmoid 后加 ε 保护；检查 `p_ij` 是否真的在 (0,1) 内 |
| Gumbel 采样梯度为 None | `z_ij` 的计算路径被截断（如误用了 `.detach()`） | 跑 `torch.autograd.gradcheck` 或手动 backward 后打印各参数的 `.grad` |
| `avg_retained_arcs` 始终为 0 | β 太大或 threshold 太高，所有弧都被压缩掉 | 先将 β 减小10倍；推理时降低 threshold 到 0.3 |
| `avg_retained_arcs` 等于总弧数 | β 太小，信息压缩完全不起作用 | 将 β 增大10倍，观察变化 |
| B6 的 F1 无法计算 | 评估代码没有处理分类模式 | 检查 `trainer.py` 的评估分支是否按 `model_type` 区分了生成模式和分类模式 |
| 显存 OOM（本地 smoke） | batch_size 仍然过大 | 降至 batch_size=1；同时冻结 PubMedBERT 的前6层（只 fine-tune 后6层） |

---

## 附录B：阶段三里程碑检查清单

在阶段三结束时，逐项确认以下内容已完成：

**环境与数据**
- [ ] PubMedBERT 已下载至 `models/pubmedbert-base/` 且可正常加载
- [ ] PyTorch Geometric 已安装且版本匹配
- [ ] `data/dep_type_vocab.json` 已生成，覆盖 ChemProtSent 和 CDRIntra 所有 split

**代码模块**
- [ ] `src/builders/graph_builder.py` 完成，单元测试通过
- [ ] `src/data/graph_collator.py` 完成，batch padding 逻辑验证通过
- [ ] `src/models/semantics_view.py` 完成，输出形状验证通过
- [ ] `src/models/syntax_view.py` 完成，输出形状和梯度验证通过
- [ ] `src/models/gmib.py` 完成，三项验证（形状/梯度/温度影响）均通过
- [ ] `src/builders/gmib_linearizer.py` 完成，线性化格式与阶段二一致
- [ ] `src/models/ea_gmib.py` 完成，训练和推理两条路径均测试通过
- [ ] `src/models/gmib_cls.py` 完成
- [ ] `src/stage3/trainer.py` 完成，含 τ 退火和扩展 logging

**实验结果**
- [ ] B3 在 ChemProtSent 和 CDRIntra 上 full run 完成（seed=42），结果记录在案
- [ ] B4 同上
- [ ] B5 同上
- [ ] Ours（EA-GMIB）在 ChemProtSent 或 CDRIntra 上 full run 完成（seed=42），且优于 B5
- [ ] B6 在至少一个数据集上 full run 完成（seed=42）

**文档**
- [ ] 每个新增文件顶部有 docstring（说明职责、输入、输出）
- [ ] configs/ 下所有阶段三配置文件命名规范，且包含完整超参数记录
- [ ] 本阶段实验过程关键决策（如 β 选择、τ 退火方案、L_compress 实现方式）记录在 `docs/record/` 下的个人日志中

---

*文档版本：v1.0*  
*创建日期：2026-04-21*  
*状态：待执行*  
*关联文档：`docs/0_experiment_plan_v1.md`、`docs/stage2_baseline_docs/stage2_baseline_report.md`、`DP_paper/1_multi_view.pdf`*
