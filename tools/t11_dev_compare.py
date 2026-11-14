"""Baseline vs PoC trên split DEV, dùng nhãn tự gán (reports/t11_dev_selfcheck.csv) — không cần gt/.

    python tools/t11_dev_compare.py
-> reports/t11_dev_baseline_vs_poc.md + reports/t11_dev_metrics.json

Heldout chính thức (Error Recall@B) cần gt/ của giảng viên: python -m src.evaluate --track t11 --pred out/poc/ranking.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

FX = Path("fixtures/t11_label_errors")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def dev_top(ranking: Path, k: int) -> pd.DataFrame:
    ann = pd.read_csv(FX / "annotations.csv").set_index("ann_id").image_id
    images = pd.read_csv(FX / "images.csv")
    dev = set(images.loc[images.split == "dev", "image_id"])
    r = pd.read_csv(ranking)
    r["image_id"] = [i[4:] if i.startswith("img:") else ann[i] for i in r.item_id]
    r = r[r.image_id.isin(dev)].sort_values("score", ascending=False, kind="mergesort")
    return r.head(k)


def evaluate(name: str, run: Path, labels: pd.DataFrame, k: int) -> dict:
    top = dev_top(run / "ranking.csv", k).merge(labels, on="item_id", how="left")
    unlabeled = int(top.my_label.isna().sum())
    if unlabeled:
        raise SystemExit(f"{name}: {unlabeled} mục trong top-{k} dev chưa có nhãn trong selfcheck: "
                         f"{top.loc[top.my_label.isna(), 'item_id'].tolist()}")
    hits = int((top.my_label == "error").sum())
    lo, hi = wilson(hits, k)
    types = sorted(set(top.loc[top.my_label == "error", "my_error_type"].dropna()))
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    return {"run": name, "k": k, "errors_in_top_k": hits, "precision_at_k": hits / k,
            "wilson95": [round(lo, 3), round(hi, 3)], "error_types_found": types,
            "items": top.item_id.tolist(), "runtime_seconds": info.get("runtime_seconds"),
            "n_items": info.get("stats", {}).get("n_items"), "seed": info.get("seed")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="out/baseline")
    ap.add_argument("--poc", default="out/poc")
    ap.add_argument("--labels", default="reports/t11_dev_selfcheck.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="reports/t11_dev_baseline_vs_poc.md")
    a = ap.parse_args()

    labels = pd.read_csv(a.labels)[["item_id", "my_label", "my_error_type"]].drop_duplicates("item_id")
    b = evaluate("baseline (random, seed 42)", Path(a.baseline), labels, a.k)
    p = evaluate("PoC (4 detectors)", Path(a.poc), labels, a.k)

    def row(m):
        return (f"| {m['run']} | {m['errors_in_top_k']}/{m['k']} | {m['precision_at_k']:.2f} "
                f"| [{m['wilson95'][0]:.2f}, {m['wilson95'][1]:.2f}] | {len(m['error_types_found'])}/4 "
                f"({', '.join(m['error_types_found']) or '—'}) | {m['runtime_seconds']} s / {m['n_items']} items |")

    md = "\n".join([
        f"## Bảng 1 · Baseline vs PoC — split DEV, top-{a.k}, nhãn tự gán",
        "",
        "> Sinh bởi `python tools/t11_dev_compare.py` (không gõ tay). Nhãn: `reports/t11_dev_selfcheck.csv`.",
        "> Không phải metric chính thức: Error Recall@B trên heldout cần `gt/` của giảng viên.",
        "",
        f"| Run | Lỗi trong top-{a.k} (dev) | Precision@{a.k} | Wilson 95% CI | Loại lỗi phủ | Runtime |",
        "|---|---|---|---|---|---|",
        row(b), row(p),
        "",
        f"Δ Precision@{a.k} = {p['precision_at_k'] - b['precision_at_k']:+.2f} "
        f"({p['errors_in_top_k']} vs {b['errors_in_top_k']} lỗi). Cùng seed 42, cùng budget, cùng pool dev.",
        "",
    ])
    Path(a.out).write_text(md, encoding="utf-8")
    Path(a.out).with_name("t11_dev_metrics.json").write_text(
        json.dumps({"baseline": b, "poc": p}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
