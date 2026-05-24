# Stage3 Multi-Seed Run Plan

## Goal

当前目标是把 Stage3 的核心双视角模型从单种子结果推进到三种子证据。已有 seed42 结果；本轮新增 seed123/456 配置，优先验证 B7 在 CDRIntra 上的稳定性，同时补齐 B4/B5/B7 的主要对比表。

## Configs Added

本轮新增 12 个 full-run 配置：

| model | dataset | seeds |
| --- | --- | --- |
| B4 semantics-only | ChemProtSent | 123, 456 |
| B4 semantics-only | CDRIntra | 123, 456 |
| B5 dual-view concat | ChemProtSent | 123, 456 |
| B5 dual-view concat | CDRIntra | 123, 456 |
| B7 gated dual-view | ChemProtSent | 123, 456 |
| B7 gated dual-view | CDRIntra | 123, 456 |

说明：新配置均保持 `float32`、`batch_size=8`、`eval_batch_size=8`、`epochs=20`、`early_stopping_patience=5`、`prediction_mode=label_rerank`，并设置 `save_best_model=true` 与 `save_final_model=false`。这只影响磁盘保存策略，不改变训练精度或评估逻辑。

## Recommended 5090 Order

第一优先级：用户本轮明确新增的 B7 CDRIntra 多种子。

```bash
python src/stage3/run_b345_full.py --config configs/stage3_full_b7_cdrintra_seed123.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b7_cdrintra_seed456.json
```

第二优先级：支撑最终主结论的强模型多种子。

```bash
python src/stage3/run_b345_full.py --config configs/stage3_full_b7_chemprotsent_seed123.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b7_chemprotsent_seed456.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b5_cdrintra_seed123.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b5_cdrintra_seed456.json
```

第三优先级：补齐 B4/B5/B7 对比表。

```bash
python src/stage3/run_b345_full.py --config configs/stage3_full_b4_chemprotsent_seed123.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b4_chemprotsent_seed456.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b4_cdrintra_seed123.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b4_cdrintra_seed456.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b5_chemprotsent_seed123.json
python src/stage3/run_b345_full.py --config configs/stage3_full_b5_chemprotsent_seed456.json
```

## Copy-Back Checklist

每个实验完成后，把对应 checkpoint 目录传回本地 `checkpoints/`，至少需要包含：

| file | purpose |
| --- | --- |
| `best_metrics.json` | 最佳 epoch 的主结果，汇总脚本优先读取 |
| `metrics_history.json` | 训练轨迹与早停诊断 |
| `run_meta.json` | 配置和环境元信息 |
| `best_model/` | 需要后续错误分析或复现实验时使用 |

## Local Summary Command

结果传回本地后运行：

```bash
venv/Scripts/python.exe src/stage3/summarize_stage3_multiseed.py
```

脚本会生成：

| output | content |
| --- | --- |
| `checkpoints/stage3_multiseed_summary.csv` | 每个 seed 的机器可读结果 |
| `docs/stage3_model_docs/stage3_multiseed_summary.md` | 聚合 mean/std 与 per-seed Markdown 表 |

当前 seed123/456 尚未回传时，表中对应行会显示 `missing`，这是预期行为。
