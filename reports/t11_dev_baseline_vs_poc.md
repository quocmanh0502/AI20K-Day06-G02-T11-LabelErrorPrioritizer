## Bảng 1 · Baseline vs PoC — split DEV, top-10, nhãn tự gán

> Sinh bởi `python tools/t11_dev_compare.py` (không gõ tay). Nhãn: `reports/t11_dev_selfcheck.csv`.
> Không phải metric chính thức: Error Recall@B trên heldout cần `gt/` của giảng viên.

| Run | Lỗi trong top-10 (dev) | Precision@10 | Wilson 95% CI | Loại lỗi phủ | Runtime |
|---|---|---|---|---|---|
| baseline (random, seed 42) | 1/10 | 0.10 | [0.02, 0.40] | 1/4 (box_shift) | 0.0749 s / 260 items |
| PoC (4 detectors) | 9/10 | 0.90 | [0.60, 0.98] | 3/4 (class_swap, duplicate, missing) | 0.3349 s / 260 items |

Δ Precision@10 = +0.80 (9 vs 1 lỗi). Cùng seed 42, cùng budget, cùng pool dev.
