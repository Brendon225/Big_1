# Stage 3 Ours 正式实验运行指南

日期：2026-05-06

## 1. 正式配置

本轮正式实验先跑 Ours/EA-GMIB 的两个主力数据集 seed42：

| 数据集 | 配置文件 | 输出目录 |
|---|---|---|
| ChemProtSent | `configs/stage3_full_ours_chemprotsent_seed42.json` | `checkpoints/stage3_full_ours_chemprotsent_seed42` |
| CDRIntra | `configs/stage3_full_ours_cdrintra_seed42.json` | `checkpoints/stage3_full_ours_cdrintra_seed42` |

核心设置：

- `model_dtype: float32`
- `batch_size: 8`
- `eval_batch_size: 8`
- `epochs: 20`
- `early_stopping_patience: 5`
- `metric_for_best: mapped_exact`
- `beta: 1e-6`
- `tau_init: 1.0`
- `tau_min: 0.1`
- `tau_anneal_rate: 0.95`
- `selection_threshold: 0.475`
- `save_best_model: true`
- `save_final_model: false`

`save_final_model=false` 只影响磁盘保存，不影响训练与精度。正式报告使用 `best_metrics.json`，不是最后一个 epoch 的 `metrics.json`。

## 2. 5090 正式运行命令

在实验室机器项目根目录执行：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/stage3/run_ours_full.py --config configs/stage3_full_ours_chemprotsent_seed42.json
```

ChemProtSent 完成后，再运行：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python src/stage3/run_ours_full.py --config configs/stage3_full_ours_cdrintra_seed42.json
```

如果 GPU 上还有其他同学的大进程，优先等待机器空闲。不要为了强行跑通而改 `model_dtype` 或随意降低训练 batch size。

## 3. 本地 smoke 验证命令

本地已经使用正式配置做过 smoke 验证。验证方式是读取 full JSON，但临时覆盖为 1 epoch、2 个 train batch、2 个 eval batch：

```bash
venv/Scripts/python.exe src/stage3/run_ours_full.py --config configs/stage3_full_ours_chemprotsent_seed42.json --epochs 1 --max_train_batches 2 --max_eval_batches 2 --batch_size 1 --eval_batch_size 1 --save_best_model false --save_final_model false --output_dir checkpoints/stage3_smoke_from_full_ours_chemprotsent_seed42
```

```bash
venv/Scripts/python.exe src/stage3/run_ours_full.py --config configs/stage3_full_ours_cdrintra_seed42.json --epochs 1 --max_train_batches 2 --max_eval_batches 2 --batch_size 1 --eval_batch_size 1 --save_best_model false --save_final_model false --output_dir checkpoints/stage3_smoke_from_full_ours_cdrintra_seed42
```

两个 smoke 均已通过，说明正式配置的路径、模型组装、GM-IB logging、label-rerank 评估和 checkpoint 写入链路均正常。

## 4. 跑完后需要回传的文件

每个正式输出目录至少回传：

- `best_metrics.json`
- `metrics.json`
- `metrics_history.json`
- `run_meta.json`

如果需要复现实验或继续分析模型，也回传：

- `best_model/`

其中 `best_model/ea_gmib_model_state.pt` 保存了 EA-GMIB 的完整模型状态，不能只回传 HuggingFace 子目录。

## 5. 结果判断

当前中间基线目标线：

| 数据集 | 需要优先超过的基线 | mapped exact |
|---|---|---:|
| ChemProtSent | B5 dual-view concat | 0.896106 |
| ChemProtSent | B4 semantics-only | 0.901083 |
| CDRIntra | B5 dual-view concat | 0.775451 |

ChemProtSent 的最低里程碑是超过 B5，更理想目标是超过 B4。CDRIntra 的核心目标是超过 B5。

同时观察：

- `eval_compress_loss`
- `eval_avg_retained_arcs`
- `eval_compression_ratio`
- `tau`

如果 `eval_compression_ratio` 长期接近 0 或 1，说明压缩强度或阈值可能需要后续调参；第一次正式 run 先不要中途改配置，完整跑完后再分析。
