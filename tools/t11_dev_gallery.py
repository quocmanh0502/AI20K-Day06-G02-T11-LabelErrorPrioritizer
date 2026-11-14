"""Xem top-N mục nghi ngờ trên DEV (không đọc gt/) + tạo file để tự gán nhãn.

    python tools/t11_dev_gallery.py --ranking out/poc/ranking.csv --top 20
-> out/dev_check/gallery.png       (đỏ = annotation đang bị nghi, vàng = annotation khác, xanh = model)
-> out/dev_check/self_labels.csv   (bạn điền cột my_label: error / ok / unsure)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

FX = Path("fixtures/t11_label_errors")


def draw_item(item, ann, pred, scale=2):
    img = Image.open(FX / item.path).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale))
    dr = ImageDraw.Draw(img)
    for p in pred[pred.image_id == item.image_id].itertuples():
        x, y, w, h = (v * scale for v in (p.x, p.y, p.w, p.h))
        dr.rectangle([x, y, x + w, y + h], outline="lime", width=1)
        dr.text((x, y + h + 1), f"{p[2]} {p.conf:.2f}", fill="lime")
    for a in ann[ann.image_id == item.image_id].itertuples():
        x, y, w, h = (v * scale for v in (a.x, a.y, a.w, a.h))
        hit = a.ann_id == item.item_id
        color = "red" if hit else "yellow"
        dr.rectangle([x, y, x + w, y + h], outline=color, width=3 if hit else 1)
        dr.text((x, y - 11), a[3], fill=color)
    dr.rectangle([0, 0, img.width, 14], fill="black")
    dr.text((3, 2), f"#{item.rank} {item.item_id}  {item.reason}  score={item.score:.2f}", fill="white")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default="out/poc/ranking.csv")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="out/dev_check")
    args = ap.parse_args()

    ann = pd.read_csv(FX / "annotations.csv")
    pred = pd.read_csv(FX / "model_predictions.csv")
    images = pd.read_csv(FX / "images.csv")
    dev = set(images.loc[images.split == "dev", "image_id"])          # CHỈ dev — không nhìn heldout

    r = pd.read_csv(args.ranking)
    r = r[r.image_id.isin(dev)].head(args.top).reset_index(drop=True)
    r["rank"] = range(1, len(r) + 1)

    tiles = [draw_item(it, ann, pred) for it in r.itertuples()]
    cols = 3
    W, H = tiles[0].size
    grid = Image.new("RGB", (W * cols, H * ((len(tiles) + cols - 1) // cols)), "black")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * W, (i // cols) * H))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.save(out / "gallery.png")

    labels = out / "self_labels.csv"
    if not labels.exists():                                           # không ghi đè nhãn bạn đã điền
        r[["rank", "item_id", "reason", "score"]].assign(my_label="", my_error_type="", note="").to_csv(labels, index=False)
    print(f"gallery -> {out / 'gallery.png'} ({len(r)} items) | labels -> {labels}")


if __name__ == "__main__":
    main()
