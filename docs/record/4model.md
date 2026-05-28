## 2026-05-27

1. 直接开始阶段四的实验，有24个

## 2026-05-28

1. 做完6个了，其实很快，继续做，然后要不停上传之前没有的checkpoints过去host4

```python
1. python src/stage4/evaluate_f1.py --config configs/stage3_full_b5_chemprotsent_seed42.json --checkpoint_dir checkpoints/stage3_full_b5_chemprotsent_seed42 --model_family stage3 --model_alias b5 --split test --output_dir outputs/stage4_predictions/full

2. python src/stage4/evaluate_f1.py --config configs/stage3_full_b5_chemprotsent_seed123.json --checkpoint_dir checkpoints/stage3_full_b5_chemprotsent_seed123 --model_family stage3 --model_alias b5 --split test --output_dir outputs/stage4_predictions/full

3. python src/stage4/evaluate_f1.py --config configs/stage3_full_b5_chemprotsent_seed456.json --checkpoint_dir checkpoints/stage3_full_b5_chemprotsent_seed456 --model_family stage3 --model_alias b5 --split test --output_dir outputs/stage4_predictions/full

4. python src/stage4/evaluate_f1.py --config configs/stage3_full_b5_cdrintra_seed42.json --checkpoint_dir checkpoints/stage3_full_b5_cdrintra_seed42 --model_family stage3 --model_alias b5 --split test --output_dir outputs/stage4_predictions/full

5. python src/stage4/evaluate_f1.py --config configs/stage3_full_b5_cdrintra_seed123.json --checkpoint_dir checkpoints/stage3_full_b5_cdrintra_seed123 --model_family stage3 --model_alias b5 --split test --output_dir outputs/stage4_predictions/full

6. python src/stage4/evaluate_f1.py --config configs/stage3_full_b5_cdrintra_seed456.json --checkpoint_dir checkpoints/stage3_full_b5_cdrintra_seed456 --model_family stage3 --model_alias b5 --split test --output_dir outputs/stage4_predictions/full

7. python src/stage4/evaluate_f1.py --config configs/stage3_full_b4_chemprotsent_seed42.json --checkpoint_dir checkpoints/stage3_full_b4_chemprotsent_seed42 --model_family stage3 --model_alias b4 --split test --output_dir outputs/stage4_predictions/full

8. python src/stage4/evaluate_f1.py --config configs/stage3_full_b4_chemprotsent_seed123.json --checkpoint_dir checkpoints/stage3_full_b4_chemprotsent_seed123 --model_family stage3 --model_alias b4 --split test --output_dir outputs/stage4_predictions/full

9. python src/stage4/evaluate_f1.py --config configs/stage3_full_b4_chemprotsent_seed456.json --checkpoint_dir checkpoints/stage3_full_b4_chemprotsent_seed456 --model_family stage3 --model_alias b4 --split test --output_dir outputs/stage4_predictions/full

10. python src/stage4/evaluate_f1.py --config configs/stage3_full_b4_cdrintra_seed42.json --checkpoint_dir checkpoints/stage3_full_b4_cdrintra_seed42 --model_family stage3 --model_alias b4 --split test --output_dir outputs/stage4_predictions/full

11. python src/stage4/evaluate_f1.py --config configs/stage3_full_b4_cdrintra_seed123.json --checkpoint_dir checkpoints/stage3_full_b4_cdrintra_seed123 --model_family stage3 --model_alias b4 --split test --output_dir outputs/stage4_predictions/full

12. python src/stage4/evaluate_f1.py --config configs/stage3_full_b4_cdrintra_seed456.json --checkpoint_dir checkpoints/stage3_full_b4_cdrintra_seed456 --model_family stage3 --model_alias b4 --split test --output_dir outputs/stage4_predictions/full

13. python src/stage4/evaluate_f1.py --config configs/baseline_local_full_chemprotsent_b2_coarse.json --checkpoint_dir checkpoints/full_b2_coarse_chemprotsent_seed42 --model_family baseline --model_alias b2_coarse --seed 42 --split test --output_dir outputs/stage4_predictions/full

14. python src/stage4/evaluate_f1.py --config configs/baseline_local_full_chemprotsent_b2_coarse.json --checkpoint_dir checkpoints/full_b2_coarse_chemprotsent_seed123 --model_family baseline --model_alias b2_coarse --seed 123 --split test --output_dir outputs/stage4_predictions/full

15. python src/stage4/evaluate_f1.py --config configs/baseline_local_full_chemprotsent_b2_coarse.json --checkpoint_dir checkpoints/full_b2_coarse_chemprotsent_seed456 --model_family baseline --model_alias b2_coarse --seed 456 --split test --output_dir outputs/stage4_predictions/full

16. python src/stage4/evaluate_f1.py --config configs/baseline_local_full_cdrintra_b2_coarse.json --checkpoint_dir checkpoints/full_b2_coarse_cdrintra_seed42 --model_family baseline --model_alias b2_coarse --seed 42 --split test --output_dir outputs/stage4_predictions/full

17. python src/stage4/evaluate_f1.py --config configs/baseline_local_full_cdrintra_b2_coarse.json --checkpoint_dir checkpoints/full_b2_coarse_cdrintra_seed123 --model_family baseline --model_alias b2_coarse --seed 123 --split test --output_dir outputs/stage4_predictions/full

18. python src/stage4/evaluate_f1.py --config configs/baseline_local_full_cdrintra_b2_coarse.json --checkpoint_dir checkpoints/full_b2_coarse_cdrintra_seed456 --model_family baseline --model_alias b2_coarse --seed 456 --split test --output_dir outputs/stage4_predictions/full
```