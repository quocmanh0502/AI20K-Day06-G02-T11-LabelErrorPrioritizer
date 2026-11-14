"""Ảnh trước/sau cho các nhãn đã sửa ở bước review.

    python tools/t11_before_after.py --run out/poc --out demo/t11_before_after.png
Trái = annotations gốc (fixtures), phải = corrected_annotations.csv; box bị sửa tô đỏ, box mới tô xanh dương.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

FX = Path("fixtures/t11_label_errors")
SCALE = 2


def _draw(img_path: Path, boxes: pd.DataFrame, highlight: set[str], title: str) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    for b in boxes.itertuples(index=False):
        hit = b.ann_id in highlight or str(b.ann_id).startswith("new_")
        color = ("deepskyblue" if str(b.ann_id).startswith("new_") else "red") if hit else "yellow"
        xy = [b.x * SCALE, b.y * SCALE, (b.x + b.w) * SCALE, (b.y + b.h) * SCALE]
        dr.rectangle(xy, outline=color, width=3 if hit else 1)
        dr.text((xy[0], max(0, xy[1] - 11)), f"{b[2]}", fill=color)
    dr.rectangle([0, 0, img.width, 14], fill="black")
    dr.text((3, 2), title, fill="white")
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/poc")
    ap.add_argument("--out", default="demo/t11_before_after.png")
    a = ap.parse_args()
    run = Path(a.run)
    before = pd.read_csv(FX / "annotations.csv")
    after = pd.read_csv(run / "corrected_annotations.csv")
    changes = pd.read_csv(run / "label_changes.csv")
    images = pd.read_csv(FX / "images.csv").set_index("image_id")

    rows = []
    for c in changes.itertuples(index=False):
        image_id = c.item_id[4:] if c.item_id.startswith("img:") else before.loc[before.ann_id == c.item_id, "image_id"].iloc[0]
        path = FX / images.loc[image_id, "path"]
        hl = {c.item_id}
        left = _draw(path, before[before.image_id == image_id], hl, f"TRUOC  {c.item_id}")
        right = _draw(path, after[after.image_id == image_id], hl, f"SAU  {c.kind}: {c.after}")
        rows.append((left, right))

    W, H = rows[0][0].size
    grid = Image.new("RGB", (W * 2 + 6, (H + 6) * len(rows)), "black")
    for i, (l, r) in enumerate(rows):
        grid.paste(l, (0, i * (H + 6)))
        grid.paste(r, (W + 6, i * (H + 6)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(a.out)
    print(f"{len(rows)} fixes -> {a.out}")


if __name__ == "__main__":
    main()
