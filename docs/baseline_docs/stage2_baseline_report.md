# 阶段二实验报告

## 1. 目的

阶段二的目标是完成生成式关系抽取 baseline 的系统搭建与验证，重点回答以下三个问题：

1. 结构信息是否值得注入生成式模型？
2. 在结构注入中，`tree` 与 `sdp` 哪种形式更有效？
3. 在结构注入中，`raw` 与 `coarse` 哪种视图更有效？

本阶段覆盖三个数据集：

- `ChemProtSent`
- `CDRIntra`
- `DDI`

其中：

- `ChemProtSent` 与 `CDRIntra` 是句级主实验视图
- `DDI` 作为阶段二补充验证数据集，在服务器上完成 full-run

---

## 2. 方法

### 2.1 baseline 设计

本阶段统一采用 5 组 baseline：

- `B0`: text-only
- `B1-raw`: `raw + tree`
- `B2-raw`: `raw + sdp`
- `B1-coarse`: `coarse + tree`
- `B2-coarse`: `coarse + sdp`

所有 baseline 在以下条件下保持一致：

- backbone: `models/biobart-base`
- `target_mode = relation_only`
- `prediction_mode = label_rerank`
- `model_dtype = float32`

### 2.2 训练设置

统一 full-run 设置：

- `epochs = 1`
- `ChemProtSent` / `CDRIntra`：本地运行
- `DDI`：实验室服务器运行
- 随机种子：`42 / 123 / 456`

### 2.3 代码与模块结构

本阶段完成了阶段二运行骨架与解耦重构，关键文件如下：

#### 数据与输入层

- `src/data/dataset.py`
- `src/data/dataloader.py`
- `src/data/collator.py`
- `src/builders/input_builder.py`

#### 运行脚本

- `src/baseline/run_smoke.py`
- `src/baseline/run_seed_sweep.py`

#### 测试

- `src/tests/test_dataset.py`

#### 配置文件

本阶段共使用两类配置：

1. smoke 配置
   - `configs/baseline_local_smoke*.json`
   - `configs/baseline_local_smoke_cdrintra*.json`
   - `configs/baseline_server_precheck_ddi_b0.json`

2. full-run 配置
   - `configs/baseline_local_full_chemprotsent_*.json`
   - `configs/baseline_local_full_cdrintra_*.json`
   - `configs/baseline_server_full_ddi_*.json`

### 2.4 关键工程结果

本阶段除了跑数本身，还完成了以下关键工程工作：

1. 冻结了阶段二统一实验接口：
   - `dataset_name`
   - `dep_view`
   - `dep_form`
   - `target_mode`
   - `model_name`
   - `seed`
2. 将输入构造逻辑从数据层解耦到 `src/builders/input_builder.py`
3. 建立了 smoke -> full-run -> seed sweep 的逐层推进流程
4. 统一了运行输出：
   - `run_meta.json`
   - `metrics.json`
   - `checkpoints/`

---

## 3. 过程记录

### 3.1 ChemProtSent 与 CDRIntra

前两个数据集的阶段性工作已在早期报告中完成归档，核心内容已在本报告中吸收整合，不再保留独立阶段性报告作为正式版本。

完成过程包括：

1. 本地 smoke 打通五组 baseline
2. 本地 full-run 跑通 `seed=42`
3. 通过 `run_seed_sweep.py` 补齐 `seed=123/456`
4. 生成前两个数据集的三种子汇总结果

### 3.2 DDI

DDI 由于文件更大，在实验室 GPU 服务器上完成：

1. 先做 `DDI + B0` 服务器预检
2. 再做 `DDI + B0 full-run`
3. 逐个完成：
   - `B1-raw`
   - `B2-raw`
   - `B1-coarse`
   - `B2-coarse`
4. 最后补齐 `seed=123/456`

日志文件：

- `logs_full_ddi_seeds123_456.txt`

### 3.3 结果文件产物

每个 full-run 目录包含：

- `model/`
- `metrics.json`
- `run_meta.json`

命名形式：

- `checkpoints/full_b0_chemprotsent_seed42`
- `checkpoints/full_b1_raw_cdrintra_seed123`
- `checkpoints/full_b2_coarse_ddi_seed456`

汇总文件：

- `checkpoints/full_summary_stage2_local_3seeds_merged.csv`
- `checkpoints/full_summary_stage2_local_seeds123_456.csv`
- `checkpoints/full_summary_ddi_seeds123_456.csv`

---

## 4. 结果

### 4.1 三数据集统一结果表

最终总表见：

- [stage2_baseline_final_results_table.md](/d:/Desktop_D/postgraduate/研二/big/big_1/docs/baseline_docs/stage2_baseline_final_results_table.md)

### 4.2 关键结果摘要

#### ChemProtSent

- `B2-coarse` 最优
- `eval_loss_mean = 0.061042`
- `mapped_exact_mean = 0.870409`

#### CDRIntra

- `B2-coarse` 略优于 `B2-raw`
- `eval_loss_mean = 0.115773`
- `mapped_exact_mean = 0.697208`

#### DDI

- 当前设置下 `B0` 最优
- `eval_loss_mean = 0.013009`
- `mapped_exact_mean = 0.978713`

### 4.3 输出合法性

三个数据集全部 15 组 baseline：

- `invalid_rate_mean = 0.000000`

说明阶段二的标签输出链路稳定。

---

## 5. 结论

阶段二已经完成以下工作：

1. 三个数据集全部完成 5 组 baseline
2. 三个数据集全部补齐 3 个随机种子
3. 本地与服务器两套环境都验证了阶段二流程可运行
4. 统一结果表、结果文件和日志文件均已生成

从实验执行角度，可以判定：

**阶段二实验已经正式收尾。**

从结果趋势角度，可得到如下结论：

1. `ChemProtSent` 上，结构注入整体有效，其中 `coarse + sdp` 最优。
2. `CDRIntra` 上，`sdp` 路线优于 `tree` 路线，`coarse + sdp` 略优。
3. `DDI` 上，当前阶段二设置下 text-only baseline 已经很强，结构注入没有超过 `B0`。

---

## 6. 注意事项

1. 本阶段统一使用的是 `prediction_mode=label_rerank`，适合阶段二结构注入对比与工程闭环验证。
2. 论文终稿若需严格的 `Micro-F1 / Macro-F1 / Precision / Recall` 指标，应在后续评估阶段补充统一计算脚本。
3. `docs/record/2baseline_record.md` 是个人日志，本报告未对其写入。

