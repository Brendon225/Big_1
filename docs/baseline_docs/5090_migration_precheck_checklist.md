# GPU 迁移与预检清单（5090 / 4090 通用）

## 0. 先给结论

当前 `.gitignore` 的设计是合理的，适合把**代码与配置**上传到 GitHub，但**不适合把完整运行环境一起上传**。

也就是说：

- GitHub 上的内容，已经足够你在 5090 上**拉下代码骨架**
- 但还不足够让 5090 **直接开跑**

因为下面这些东西被故意忽略了，不会随 GitHub 一起过去：

- `data/`
- `models/`
- `venv/`
- `checkpoints/`

这不是问题，反而是正确做法。  
正确的迁移思路是：

1. **代码走 GitHub**
2. **数据和模型单独拷贝**
3. **venv 在 5090 上重新创建**

一句话理解：

> GitHub 负责搬“菜谱”，手动传输负责搬“食材”，5090 本地重新搭“厨房”。

补充：

- 这份清单同样适用于实验室 `4090`，只是训练速度和可承受 batch 上限可能略有差异。

---

## 1. 当前 GitHub 上应该包含什么

如果你准备去 5090 上拉代码，那么 GitHub 上至少应该包含下面这些内容：

- `src/`
- `configs/`
- `docs/baseline_docs/`
- `requirements.txt`
- `.gitignore`

当前从仓库状态来看，阶段二核心文件已经在 Git 跟踪里，包括：

- `src/baseline/run_smoke.py`
- `src/baseline/run_seed_sweep.py`
- `src/builders/input_builder.py`
- `configs/*.json`
- `requirements.txt`

这说明：

**只要你把当前改动 commit 并 push，上 5090 后通过 GitHub 拉代码这一部分是没问题的。**

### 1.1 必加检查（非常重要）

请务必确认 `src/data` 目录也在 Git 跟踪中，否则服务器上会出现“本地能跑、服务器找不到模块”的问题。

建议命令：

```powershell
git ls-files src/data
```

成功标志：

- 输出中包含 `src/data/dataset.py`
- 输出中包含 `src/data/dataloader.py`
- 输出中包含 `src/data/collator.py`

如果为空：

```powershell
git add src/data
git commit -m "track src/data for server runs"
git push origin <你的分支名>
```

---

## 2. 推荐迁移方案

推荐采用下面这个方案：

### 方案：GitHub 拉代码 + 手动拷数据与模型 + 5090 重建 venv

#### 为什么推荐这个方案

- 代码通过 GitHub 拉取，版本最清晰，后续更新最方便
- 数据和模型单独拷，避免 GitHub 被大文件拖垮
- venv 重新创建，避免“本机能跑，换机就炸”的环境污染问题

#### 不推荐什么

不建议直接把本机整个 40 多 GB 项目文件夹原封不动搬过去，尤其不要指望把本机 `venv/` 整个搬到 5090 后继续用。

原因很简单：

- 路径可能不同
- CUDA / 驱动环境不同
- 已编译包可能不兼容
- 一旦报错，很难判断是代码问题还是旧环境问题

---

## 3. 正式开始前的总流程

你可以把整个迁移流程理解成 4 个阶段：

1. **本地冻结版本**
2. **5090 迁移代码、数据、模型**
3. **5090 预检**
4. **5090 正式跑数**

下面按照真正执行顺序，一步一步写。

---

## 4. 第一步：本地冻结版本

### 你要做什么

在本地电脑上，把当前已经验证通过的阶段二代码固定下来。

### 操作

1. 先检查工作区状态
2. 确认只提交阶段二相关代码和配置
3. 提交并 push 到 GitHub

### 建议命令

```powershell
git status
git add src configs requirements.txt .gitignore docs/baseline_docs
git commit -m "stage2 baseline local smoke pipeline ready"
git push origin <你的分支名>
```

### 成功标志

- `git status` 显示工作区干净，或者只剩你明确不准备提交的文件
- GitHub 仓库页面上能看到最新 commit
- 5090 上后续 `git clone` 或 `git pull` 能拿到这些文件

### 特别提醒

如果 `docs/record/` 是你自己本地记录用的草稿，而不是必须同步给 5090 的内容，可以先不提交。

---

## 5. 第二步：在 5090 上拉代码

### 你要做什么

在 5090 机器上创建一个新的工作目录，然后从 GitHub 拉取项目代码。

### 操作

1. 新建一个项目目录
2. `git clone`
3. 进入项目目录

### 建议命令

```powershell
mkdir D:\big_1
cd D:\big_1
git clone https://github.com/Brendon225/big_1.git
cd big_1
```

### 成功标志

在 5090 上，你能看到至少这些目录或文件：

- `src/`
- `configs/`
- `docs/`
- `requirements.txt`
- `.gitignore`

### 如果这里失败

优先排查：

- GitHub 网络
- 仓库权限
- 你是否 push 了最新代码

---

## 6. 第三步：单独拷贝数据和模型

### 你要做什么

把**阶段二真正运行需要的食材**搬到 5090。

### 当前阶段二最少需要拷贝的内容

#### 数据

至少拷这两个目录下的文件：

- `data/experiment_views/coarsened/ChemProtSent_*.json`
- `data/experiment_views/coarsened/CDRIntra_*.json`

如果你后面要补 DDI，再额外拷：

- `data/coarsened/DDI_*.json`
- `data/coarsened/.index_cache/`（可选，用于减少首次索引等待时间）

#### 模型

把整个本地模型目录拷过去：

- `models/biobart-base/`

这个目录下至少应包括：

- `config.json`
- `model.safetensors` 或 `pytorch_model.bin`
- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`
- `special_tokens_map.json`

### 成功标志

在 5090 上，这两个路径必须存在：

- `data/experiment_views/coarsened/ChemProtSent_train.json`
- `models/biobart-base/config.json`

如果要跑 DDI，额外必须存在：

- `data/coarsened/DDI_train.json`
- `data/coarsened/DDI_dev.json`
- `data/coarsened/DDI_test.json`

### 通俗理解

代码是菜谱，数据和模型是食材。  
你把菜谱拉下来后，如果厨房里没有米和菜，还是做不出饭。

---

## 7. 第四步：在 5090 上重新创建 venv

### 你要做什么

在 5090 上新建一个干净的 Python 虚拟环境，不搬本地的 `venv/`。

### 建议命令

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 `requirements.txt` 里的 `torch==...+cu128` 安装失败，建议替代顺序：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 成功标志

下面命令能正常输出版本：

```powershell
python -V
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
```

### 可能遇到的坑

如果 `pip install -r requirements.txt` 在 `torch==...+cu128` 这里报错，不要慌。  
这通常不是代码错，而是 5090 机器上的 PyTorch/CUDA 轮子源或 CUDA 环境要单独对齐。

处理思路：

1. 先确认 5090 机器的 CUDA 驱动正常
2. 必要时先单独装 PyTorch
3. 再继续安装剩余依赖

---

## 8. 第五步：检查 5090 的 CUDA 是否真的可用

### 你要做什么

确认 5090 不只是“有显卡”，而是真的能被 PyTorch 看到。

### 建议命令

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

### 成功标志

你应该看到：

- `True`
- 正确的 GPU 名称

如果这里不对，**不要继续跑实验**。  
先把 CUDA 问题解决掉。

---

## 9. 第六步：检查数据与模型路径

### 你要做什么

在正式跑脚本前，先确认路径真的没写错。

### 建议命令

```powershell
Test-Path data\experiment_views\coarsened\ChemProtSent_train.json
Test-Path data\experiment_views\coarsened\CDRIntra_train.json
Test-Path models\biobart-base\config.json
```

### 成功标志

三个命令都返回：

```text
True
```

---

## 10. 第七步：做 5090 最小预检

### 你要做什么

先别一口气跑 10 个配置 × 3 seeds。  
先跑一组最小预检，确认 5090 这台机器上的训练链路没问题。

### 推荐预检配置

- 数据集：`ChemProtSent`
- 模型：`B0`
- seed：`42`
- epoch：`1`
- 小 batch、小 eval

### 建议命令

```powershell
python src\baseline\run_smoke.py --config configs\baseline_local_smoke.json --output_dir checkpoints\server_precheck_b0_chemprotsent_seed42
```

### 成功标志

至少满足下面 4 条：

- 能正常开始训练
- `train_loss` / `eval_loss` 是数字，不是 `nan`
- 能输出合法 label 评估结果
- `checkpoints\server_precheck_b0_chemprotsent_seed42` 目录成功生成

### 如果这里失败

优先排查：

1. 模型路径
2. 数据路径
3. venv 是否装对
4. CUDA 是否可用

不要一上来怀疑研究思路。

---

## 10.1 DDI 上机前专项预检（新增）

由于 DDI 文件体量大，建议在服务器上先做 DDI 的最小预检，再进入正式跑数。

### 推荐预检设置

- 数据集：`DDI`
- baseline：先 `B0`
- seed：`42`
- `epochs=1`
- 开启小规模截断（例如 `max_train_batches=100`）只做链路检查

### 预检关注点

1. DataLoader 是否只对 `train` 做负采样（`dev/test` 保持原始分布）
2. 首轮索引构建是否完成且无阻塞报错
3. 显存与速度是否在可接受范围
4. `metrics.json` 与 `run_meta.json` 是否正常落盘

### 成功标志

- 无 OOM / 无 NaN
- 训练正常推进
- 输出目录结构完整

---

## 10.2 4090 与 5090 的参数起步建议（新增）

建议起步（保守值）：

| GPU | 建议 batch_size | 建议 eval_batch_size | dtype |
|---|---:|---:|---|
| 4090 24GB | 4~8 | 4~8 | float32 |
| 5090（大显存） | 8~16 | 8~16 | float32 |

说明：

- 请先从保守值起步，预检通过再逐步加大 batch。
- 不建议在未验证前直接切回 fp16。

---

## 11. 第八步：做 5090 小规模正式前验证

### 你要做什么

当 `B0` 预检通过后，再跑 `ChemProtSent` 五组单 seed，确认所有 baseline 都能在 5090 上稳定跑。

### 推荐顺序

1. `B0`
2. `B1-raw`
3. `B2-raw`
4. `B1-coarse`
5. `B2-coarse`

### 成功标志

- 五组都能正常结束
- 每组都有 `metrics.json`
- 每组都有 `run_meta.json`
- 每组都有 checkpoint 目录

### 为什么这一步不能省

这一步像“正式宴会前的彩排”。  
如果彩排都没走完，就直接开正式跑数，后面出错会很难收拾。

---

## 12. 第九步：再开始正式跑大流程

### 重要提醒

当前我们本地和脚本里已经打通的是 **smoke 流程**。  
它非常适合：

- 验证链路
- 验证环境
- 验证 baseline 组合是否能跑

但它还不是“最终正式论文配置”。

### 正式跑大流程前，你还需要确认

1. `epochs` 不再是 smoke 的小值
2. `max_train_batches` / `max_eval_batches` 不再人为截断
3. 输出目录命名规范清晰
4. 三个 seeds 固定为：`42, 123, 456`
5. 汇总表路径固定

### 正式阶段建议顺序

1. 先跑 `ChemProtSent` 五组 × 3 seeds
2. 再跑 `CDRIntra` 五组 × 3 seeds
3. 最后跑 `DDI`（建议先单 seed，再扩到 3 seeds）
4. 最后汇总结果

### 断点恢复建议（新增）

长任务中断后，避免重复开销的建议：

1. 优先复用已存在的 `checkpoints/<run_name>/metrics.json`
2. 批量脚本按“未完成配置 + 未完成 seed”继续跑
3. 每天跑完后先备份当日 `checkpoints/` 与汇总 CSV

---

## 13. 每一步的最低成功标准总表

| 阶段 | 你在做什么 | 最低成功标志 |
|------|------------|--------------|
| 本地冻结 | 固定代码版本 | GitHub 上能看到最新 commit |
| 5090 拉代码 | 下载菜谱 | `src/`、`configs/`、`requirements.txt` 存在 |
| 拷数据模型 | 搬食材 | `data/...json` 与 `models/biobart-base/config.json` 存在 |
| 重建 venv | 搭厨房 | `import torch, transformers` 成功 |
| CUDA 检查 | 看灶台能不能点火 | `torch.cuda.is_available() == True` |
| 最小预检 | 先试一道菜 | `loss` 正常，checkpoint 生成 |
| 小规模彩排 | 五组 baseline 单 seed | 五组都能跑通 |
| 正式跑数（前两数据集） | 真正开席第一轮 | 10 组 × 3 seeds 全完成 |
| 正式跑数（DDI） | 真正开席第二轮 | DDI 5 组 × 3 seeds 全完成（或按时间缩减并记录） |

---

## 14. 最后一句建议

如果你是实验小白，就记住这一条：

**先让 5090 跑通一个最小样例，再让它跑完整套实验。**

不要跳步。  
跳步通常不是节省时间，而是把错误留到最难排查的时候再爆炸。
