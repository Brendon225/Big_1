# 阶段二实验报告（ChemProtSent + CDRIntra）

## 1. 文档定位

本文档记录阶段二中前两个主数据集（`ChemProtSent`、`CDRIntra`）的完整 baseline 实验工作与产物，用于：

- 阶段二阶段性归档
- 后续 DDI 实验交接
- 论文实验过程可追溯

说明：

- 本文档不写入 `docs/record/2baseline_record.md`（该文件为个人日志）
- 本文档聚焦“已完成且可复现”的工作

---

## 2. 阶段目标与范围

本阶段目标：

1. 在句级主实验视图上完成 5 组 baseline：
   - `B0`: text-only
   - `B1-raw`: raw + tree
   - `B2-raw`: raw + sdp
   - `B1-coarse`: coarse + tree
   - `B2-coarse`: coarse + sdp
2. 覆盖两个主数据集：
   - `ChemProtSent`
   - `CDRIntra`
3. 完成 3 个 seeds（`42/123/456`）的本地 full-run（`epoch=1`）
4. 输出可汇总的结构化结果文件

---

## 3. 环境与运行设置

硬件环境（本地）：

- CPU: 12th Gen i7
- RAM: 32GB
- GPU: RTX 4070 Laptop (8GB)

软件环境：

- Python: 3.10.11
- PyTorch: 2.11.0+cu128
- Transformers: 5.4.0
- venv: `venv`

关键训练设置（full-run）：

- `model_name`: `models/biobart-base`
- `model_dtype`: `float32`
- `target_mode`: `relation_only`
- `prediction_mode`: `label_rerank`
- `batch_size`: `4`
- `eval_batch_size`: `4`
- `epochs`: `1`
- `max_src_len`: `256`
- `max_input_len`: `512`

---

## 4. 代码与配置产物

### 4.1 核心代码文件

- `src/baseline/run_smoke.py`
- `src/baseline/run_seed_sweep.py`
- `src/builders/input_builder.py`
- `src/builders/__init__.py`
- `src/tests/test_dataset.py`

### 4.2 关键能力落地

1. 统一 baseline 开关：
   - `use_dep`
   - `dep_view` (`raw/coarse`)
   - `dep_form` (`none/tree/sdp`)
   - `target_mode` (`relation_only`)
2. 输入构造与数据读取解耦：
   - 输入构造沉淀到 `src/builders/input_builder.py`
   - 数据层保留索引、读取、截断等职责
3. 本地自动化跑数：
   - 支持单配置运行
   - 支持多配置多 seed 批量 sweep
   - 自动落盘 `run_meta.json` + `metrics.json`

### 4.3 配置文件产物

Smoke 配置（阶段打通）：

- `configs/baseline_local_smoke*.json`
- `configs/baseline_local_smoke_cdrintra*.json`

Full-run 配置（正式本地结果）：

- `configs/baseline_local_full_chemprotsent_b0.json`
- `configs/baseline_local_full_chemprotsent_b1_raw.json`
- `configs/baseline_local_full_chemprotsent_b2_raw.json`
- `configs/baseline_local_full_chemprotsent_b1_coarse.json`
- `configs/baseline_local_full_chemprotsent_b2_coarse.json`
- `configs/baseline_local_full_cdrintra_b0.json`
- `configs/baseline_local_full_cdrintra_b1_raw.json`
- `configs/baseline_local_full_cdrintra_b2_raw.json`
- `configs/baseline_local_full_cdrintra_b1_coarse.json`
- `configs/baseline_local_full_cdrintra_b2_coarse.json`

---

## 5. 结果与文件产物

### 5.1 单次运行产物

每个实验目录（`checkpoints/full_*_seed*`）包含：

- `model/`
- `run_meta.json`
- `metrics.json`

### 5.2 汇总产物

- `checkpoints/full_summary_stage2_local_seeds123_456.csv`
- `checkpoints/full_summary_stage2_local_3seeds_merged.csv`

其中 `full_summary_stage2_local_3seeds_merged.csv` 为当前阶段二前两数据集的主要汇总表。

---

## 6. 三种子结果快照（3-seed mean/std）

数据来源：

- `checkpoints/full_summary_stage2_local_3seeds_merged.csv`

### 6.1 ChemProtSent

| dep_view | dep_form | eval_loss_mean | eval_loss_std | mapped_exact_mean | mapped_exact_std |
|---|---|---:|---:|---:|---:|
| raw | none | 0.063350 | 0.007824 | 0.859612 | 0.021647 |
| raw | tree | 0.065115 | 0.001365 | 0.862261 | 0.006101 |
| raw | sdp | 0.069411 | 0.005242 | 0.855625 | 0.001465 |
| coarse | tree | 0.063092 | 0.002866 | 0.866888 | 0.011505 |
| coarse | sdp | **0.061042** | 0.002018 | **0.870409** | 0.002449 |

观察：

- 在本阶段设置下，`coarse + sdp`（B2-coarse）表现最佳。

### 6.2 CDRIntra

| dep_view | dep_form | eval_loss_mean | eval_loss_std | mapped_exact_mean | mapped_exact_std |
|---|---|---:|---:|---:|---:|
| raw | none | 0.126219 | 0.020741 | 0.660712 | 0.053767 |
| raw | tree | 0.131078 | 0.009447 | 0.634460 | 0.025128 |
| raw | sdp | 0.116575 | 0.005288 | 0.696184 | 0.009670 |
| coarse | tree | 0.125597 | 0.005888 | 0.642208 | 0.019994 |
| coarse | sdp | **0.115773** | 0.008337 | **0.697208** | 0.019485 |

观察：

- 在本阶段设置下，`coarse + sdp`（B2-coarse）与 `raw + sdp`（B2-raw）表现最接近，前者略优。

### 6.3 输出合法性

所有组别：

- `invalid_rate_mean = 0.000000`

说明标签输出链路稳定。

---

## 7. 关键实验记录（执行过程）

1. 完成 smoke 打通（两数据集五组）
2. 完成 `ChemProtSent` full-run seed=42
3. 完成 `CDRIntra` full-run seed=42
4. 通过 `run_seed_sweep.py` 补齐 `123/456`
5. 生成 3-seed 合并汇总表

---

## 8. 可复现实验命令（本阶段）

单配置 full-run 示例：

```powershell
.\venv\Scripts\python.exe src\baseline\run_smoke.py --config configs\baseline_local_full_chemprotsent_b0.json
```

批量补 seed 示例：

```powershell
.\venv\Scripts\python.exe src\baseline\run_seed_sweep.py --configs configs\baseline_local_full_chemprotsent_b0.json ... --seeds 123 456 --summary_csv checkpoints\full_summary_stage2_local_seeds123_456.csv
```

---

## 9. 当前阶段结论与后续建议

结论：

1. 阶段二前两个主数据集（ChemProtSent、CDRIntra）已完成 5 组 baseline × 3 seeds 的本地 full-run。
2. 关键产物（配置、模型输出、汇总表）已齐备，可支撑论文阶段二主体结果描述。
3. 当前阶段二尚未完成部分主要为 DDI（建议在实验室服务器进行）。

建议：

1. 进入 DDI 前，先在服务器做最小预检（B0, 1 epoch, 小 batch）。
2. DDI 结果完成后，统一生成阶段二总表（含三数据集）。

---

## 10. 注意事项（交接）

1. 本报告引用的是 `prediction_mode=label_rerank` 下的结果，适合阶段二结构对比与流程验证。
2. 论文终稿若需要严格标准指标（Micro-F1 / Macro-F1 / P / R）与统一解码策略，应在后续评估脚本中进一步固定和补充。
3. `docs/record/2baseline_record.md` 为个人日志，本报告未对其写入。

