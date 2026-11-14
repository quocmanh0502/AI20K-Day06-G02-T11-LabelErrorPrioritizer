from __future__ import annotations

import pandas as pd


def iou(a, b) -> float:
    """a, b = (x, y, w, h). Trả IoU trong [0, 1]."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_to_predictions(ann: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Với mỗi annotation, tìm prediction cùng ảnh có IoU lớn nhất (bất kể class).

    Thêm cột: best_iou, best_pred_class, best_pred_conf.
    """
    rows = []
    for _, a in ann.iterrows():
        cands = pred[pred.image_id == a.image_id]
        best = (0.0, None, 0.0)
        for _, p in cands.iterrows():
            v = iou((a.x, a.y, a.w, a.h), (p.x, p.y, p.w, p.h))
            if v > best[0]:
                best = (v, p["class"], p.conf)
        rows.append({"ann_id": a.ann_id, "best_iou": best[0],
                     "best_pred_class": best[1], "best_pred_conf": best[2]})
    return ann.merge(pd.DataFrame(rows), on="ann_id")

def score_class_swap(m: pd.DataFrame, min_iou: float = 0.5) -> pd.Series:
    """Khung của model trùng khá khớp (IoU ≥ 0.5) nhưng class khác → nghi gán nhầm class.
    Điểm = độ tự tin của model: model càng chắc thì càng đáng nghi."""
    hit = (m.best_iou >= min_iou) & (m["class"] != m.best_pred_class)
    return m.best_pred_conf.where(hit, 0.0)


def score_box_shift(m: pd.DataFrame, low: float = 0.05, high: float = 0.6) -> pd.Series:
    """IoU nằm giữa (0.05, 0.6): có chạm vật thể nhưng lệch. IoU càng thấp → lệch càng nhiều → điểm càng cao."""
    hit = (m.best_iou > low) & (m.best_iou < high)
    return ((high - m.best_iou) / high * m.best_pred_conf).where(hit, 0.0)


def score_duplicate(m: pd.DataFrame, min_iou: float = 0.5) -> pd.Series:
    """Hai annotation CÙNG ảnh, CÙNG class, đè lên nhau (IoU ≥ 0.5) → một cái là bản sao.
    Chỉ nghi mạnh cái khớp model KÉM hơn; cái còn lại điểm thấp (không lãng phí 2 chỗ trong top-B)."""
    s = pd.Series(0.0, index=m.index)
    for _, g in m.groupby("image_id"):
        idx = list(g.index)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = m.loc[idx[i]], m.loc[idx[j]]
                if a["class"] != b["class"]:
                    continue
                v = iou((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h))
                if v >= min_iou:
                    worse, better = (idx[i], idx[j]) if a.best_iou <= b.best_iou else (idx[j], idx[i])
                    s[worse] = max(s[worse], v)
                    s[better] = max(s[better], 0.3 * v)
    return s


def score_missing(ann: pd.DataFrame, pred: pd.DataFrame, images: pd.DataFrame,
                  min_conf: float = 0.5, max_iou: float = 0.1) -> dict:
    """Prediction tự tin (conf ≥ 0.5) mà KHÔNG chạm annotation nào (IoU < 0.1) → ảnh có thể thiếu box.
    Trả {image_id: điểm}; điểm = conf cao nhất của các prediction 'mồ côi' trong ảnh."""
    out = {}
    for img in images.image_id:
        a = ann[ann.image_id == img]
        best = 0.0
        for p in pred[pred.image_id == img].itertuples():
            overlap = max([iou((p.x, p.y, p.w, p.h), (q.x, q.y, q.w, q.h)) for q in a.itertuples()] + [0.0])
            if p.conf >= min_conf and overlap < max_iou:
                best = max(best, p.conf)
        out[img] = best
    return out


# Ngưỡng chọn trên DEV (xem reports/t11_dev_selfcheck.csv) và ĐÓNG BĂNG trước khi chấm heldout.
# Có thể ghi đè trong configs/lab.yaml -> params.
DEFAULT_PARAMS = {
    "swap_min_iou": 0.5,        # class_swap: khung model phải trùng ít nhất chừng này
    "shift_low_iou": 0.05,      # box_shift: dưới mức này coi như "model không thấy", không phải lệch
    "shift_high_iou": 0.6,      # box_shift: trên mức này coi như khớp bình thường
    "dup_min_iou": 0.5,         # duplicate: 2 annotation cùng class đè nhau ít nhất chừng này
    "miss_min_conf": 0.5,       # missing: model phải tự tin ít nhất chừng này
    "miss_max_iou": 0.1,        # missing: prediction không chạm annotation nào quá mức này
    "no_match_score": 0.2,      # điểm nghi nhẹ cho khung model không thấy (vật thể tí hon)
}


def build_scores(ann: pd.DataFrame, pred: pd.DataFrame, images: pd.DataFrame,
                 params: dict | None = None) -> pd.DataFrame:
    """Trả bảng: item_id (cột đầu), score, reason, image_id, path — cho MỌI annotation + MỌI ảnh."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    m = match_to_predictions(ann, pred)
    det = pd.DataFrame({
        "class_swap": score_class_swap(m, p["swap_min_iou"]),
        "box_shift": score_box_shift(m, p["shift_low_iou"], p["shift_high_iou"]),
        "duplicate": score_duplicate(m, p["dup_min_iou"]),
    })
    det["no_model_match"] = (m.best_iou <= p["shift_low_iou"]).astype(float) * p["no_match_score"]
    best = det.max(axis=1)
    ann_items = pd.DataFrame({"item_id": m.ann_id, "score": best,
                              "reason": det.idxmax(axis=1).where(best > 0, "looks_ok"),
                              "image_id": m.image_id})
    miss = score_missing(ann, pred, images, p["miss_min_conf"], p["miss_max_iou"])
    img_items = pd.DataFrame({"item_id": ["img:" + i for i in miss], "score": list(miss.values()),
                              "reason": ["missing" if v > 0 else "looks_ok" for v in miss.values()],
                              "image_id": list(miss)})
    items = pd.concat([ann_items, img_items], ignore_index=True)
    items = items.merge(images[["image_id", "path"]], on="image_id", how="left")
    return items.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
