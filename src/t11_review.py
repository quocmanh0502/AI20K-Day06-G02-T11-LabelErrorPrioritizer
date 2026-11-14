"""T11 human-in-the-loop: review workbench, label fixes, decision policy.

Override syntax (reviewer_decision = edit):
    class=<cls>                 đổi class của annotation
    box=<x>,<y>,<w>,<h>         đổi toạ độ annotation
    replace=<cls>,<x>,<y>,<w>,<h>  đổi cả class lẫn toạ độ
    delete                      xoá annotation (vd bản duplicate)
    add=<cls>,<x>,<y>,<w>,<h>   thêm box mới cho mục img:<image_id> (missing)
"""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import pandas as pd

CONFIRMED = {"edit", "approve"}
VALID_CLASSES = ("car", "motorbike", "pedestrian", "sign")
CHANGE_COLUMNS = ["item_id", "kind", "before", "after", "reviewer", "note"]
TEMPLATE = Path(__file__).parent / "templates" / "t11_review_workbench.html"
REVIEW_FIELDS = ["reviewer", "reviewer_decision", "override_value", "review_note", "reviewed_at"]


# ------------------------------------------------------------------ parsing
def _floats(parts: list[str], item_id: str, value: str) -> list[float]:
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{item_id}: toạ độ không phải số trong override_value={value!r}") from None
    if len(nums) != 4 or nums[2] <= 0 or nums[3] <= 0:
        raise ValueError(f"{item_id}: cần x,y,w,h với w>0, h>0 trong override_value={value!r}")
    return nums


def _class(cls: str, item_id: str, value: str) -> str:
    cls = cls.strip()
    if cls not in VALID_CLASSES:
        raise ValueError(f"{item_id}: class {cls!r} không hợp lệ (override_value={value!r}); dùng {VALID_CLASSES}")
    return cls


def parse_override(item_id: str, value: str) -> tuple[str, dict]:
    """Trả (kind, payload). kind ∈ delete/class/box/replace/add. Sai cú pháp -> ValueError."""
    v = str(value).strip()
    is_img = item_id.startswith("img:")
    if v == "delete":
        if is_img:
            raise ValueError(f"{item_id}: không thể delete một mục ảnh (img:)")
        return "delete", {}
    op, _, arg = v.partition("=")
    args = arg.split(",") if arg else []
    if op == "add":
        if not is_img:
            raise ValueError(f"{item_id}: add= chỉ dùng cho mục img:<image_id>")
        if len(args) != 5:
            raise ValueError(f"{item_id}: add cần class,x,y,w,h (override_value={v!r})")
        return "add", {"class": _class(args[0], item_id, v), "xywh": _floats(args[1:], item_id, v)}
    if is_img:
        raise ValueError(f"{item_id}: mục img: chỉ hỗ trợ add=... (override_value={v!r})")
    if op == "class" and len(args) == 1:
        return "class", {"class": _class(args[0], item_id, v)}
    if op == "box":
        return "box", {"xywh": _floats(args, item_id, v)}
    if op == "replace" and len(args) == 5:
        return "replace", {"class": _class(args[0], item_id, v), "xywh": _floats(args[1:], item_id, v)}
    raise ValueError(f"{item_id}: override_value không hợp lệ: {v!r}")


# ------------------------------------------------------------------ apply
def apply_label_fixes(ann: pd.DataFrame, reviewed: pd.DataFrame, audit=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Áp các quyết định `edit` lên BẢN SAO annotations. Trả (fixed_ann, changes).

    File/bảng gốc không bị đổi. Mỗi quyết định (kể cả không sửa) được ghi audit log.
    """
    fixed = ann.copy()
    changes: list[dict] = []
    n_added = 0
    for r in reviewed.itertuples(index=False):
        item_id, decision = str(r.item_id), str(r.reviewer_decision)
        value = str(r.override_value).strip()
        if decision != "edit" or not value:
            if audit:
                audit.log(actor=r.reviewer, action=f"review:{decision}", item_id=item_id, reason=r.review_note)
            continue
        kind, p = parse_override(item_id, value)
        if kind == "add":
            image_id = item_id[4:]
            new_id = f"new_{image_id}_{n_added}"
            while new_id in set(fixed.ann_id):
                n_added += 1
                new_id = f"new_{image_id}_{n_added}"
            n_added += 1
            x, y, w, h = p["xywh"]
            row = {"ann_id": new_id, "image_id": image_id, "class": p["class"], "x": x, "y": y, "w": w, "h": h}
            fixed = pd.concat([fixed, pd.DataFrame([row])], ignore_index=True)
            before = ""
        else:
            mask = fixed.ann_id == item_id
            if not mask.any():
                raise ValueError(f"{item_id}: không tìm thấy annotation để sửa ({value!r})")
            before = json.dumps(fixed.loc[mask].iloc[0].to_dict(), ensure_ascii=False, default=str)
            if kind == "delete":
                fixed = fixed[~mask].reset_index(drop=True)
            else:
                if "class" in p:
                    fixed.loc[mask, "class"] = p["class"]
                if "xywh" in p:
                    fixed.loc[mask, ["x", "y", "w", "h"]] = p["xywh"]
        changes.append({"item_id": item_id, "kind": kind, "before": before, "after": value,
                        "reviewer": r.reviewer, "note": r.review_note})
        if audit:
            audit.log(actor=r.reviewer, action=f"label_fix:{kind}", item_id=item_id,
                      before=before, after=value, reason=r.review_note)
    return fixed, pd.DataFrame(changes, columns=CHANGE_COLUMNS)


def review_stats(reviewed: pd.DataFrame | None, changes: pd.DataFrame | None) -> dict:
    if reviewed is None or len(reviewed) == 0:
        return {"n_reviewed": 0, "n_confirmed": 0, "n_false_positive": 0, "n_escalated": 0, "n_fixed": 0}
    d = reviewed.reviewer_decision
    return {"n_reviewed": int(len(reviewed)), "n_confirmed": int(d.isin(CONFIRMED).sum()),
            "n_false_positive": int((d == "reject").sum()), "n_escalated": int((d == "escalate").sum()),
            "n_fixed": int(len(changes)) if changes is not None else 0}


# ------------------------------------------------------------------ decision
def decide_policy(stats: dict, rework_rate: float = 0.30) -> tuple[str, str, str]:
    """Chính sách chốt trước khi chạy eval. Trả (status, reason_code, reason)."""
    n = int(stats.get("n_reviewed", 0))
    conf, esc, fixed = int(stats.get("n_confirmed", 0)), int(stats.get("n_escalated", 0)), int(stats.get("n_fixed", 0))
    if n == 0:
        return "HOLD", "REVIEW_PENDING", "Chưa có kết quả review của người kiểm tra cho queue top-B."
    if esc > 0:
        return "HOLD", "ESCALATIONS_OPEN", f"Còn {esc} mục escalate chưa được xác minh."
    rate = conf / n
    if rate >= rework_rate:
        return ("REWORK", "HIGH_CONFIRMED_ERROR_RATE",
                f"{conf}/{n} mục trong queue là lỗi thật ({rate:.0%}) ≥ {rework_rate:.0%} → lô nhãn có lỗi hệ thống, trả vendor gán lại.")
    if conf > fixed:
        return "HOLD", "CONFIRMED_NOT_FIXED", f"{conf - fixed} lỗi đã xác nhận nhưng chưa có bản sửa."
    return "GO", "ERRORS_FIXED", f"Tỷ lệ lỗi {rate:.0%} < {rework_rate:.0%} và mọi lỗi xác nhận đã được sửa."


# ------------------------------------------------------------------ workbench (contact sheet)
def build_workbench_items(queue: pd.DataFrame, ann: pd.DataFrame, pred: pd.DataFrame, images: pd.DataFrame,
                          image_root: Path, reviewed: pd.DataFrame | None = None) -> list[dict]:
    """Dữ liệu cho trang review: ảnh (base64), box người (cam/đỏ), box model (xanh), trạng thái review."""
    session = uuid.uuid4().hex
    img_meta = images.set_index("image_id")
    rev = reviewed.set_index("item_id") if reviewed is not None and len(reviewed) else None
    items = []
    for q in queue.to_dict("records"):
        item_id = str(q["item_id"])
        image_id = item_id[4:] if item_id.startswith("img:") else str(ann.loc[ann.ann_id == item_id, "image_id"].iloc[0])
        meta = img_meta.loc[image_id]
        boxes, original = [], None
        for a in ann[ann.image_id == image_id].itertuples(index=False):
            hit = a.ann_id == item_id
            b = {"x": float(a.x), "y": float(a.y), "w": float(a.w), "h": float(a.h), "ann_id": a.ann_id,
                 "class": a[2], "color": "red" if hit else "orange", "label": f"{a.ann_id} {a[2]}"}
            boxes.append(b)
            if hit:
                original = b
        for p in pred[pred.image_id == image_id].itertuples(index=False):
            boxes.append({"x": float(p.x), "y": float(p.y), "w": float(p.w), "h": float(p.h), "class": p[1],
                          "conf": float(p.conf), "color": "green", "label": f"prediction {p[1]} {p.conf:.2f}"})
        src = "data:image/jpeg;base64," + base64.b64encode((image_root / meta["path"]).read_bytes()).decode()
        applied = None
        if rev is not None and item_id in rev.index:
            applied = {k: str(rev.loc[item_id, k]) for k in REVIEW_FIELDS if k in rev.columns}
            q.update(applied)
        q.update({"image_id": image_id, "src": src, "width": int(meta["width"]), "height": int(meta["height"]),
                  "classes": list(VALID_CLASSES), "applied_review": applied, "review_session_id": session,
                  "original_box": original, "boxes": boxes})
        items.append({k: ("" if isinstance(v, float) and pd.isna(v) else v) for k, v in q.items()})
    return items


def render_workbench(items: list[dict], out_html: Path, template: Path = TEMPLATE) -> Path:
    page = template.read_text(encoding="utf-8")
    if "__ITEMS__" not in page:
        raise ValueError(f"template {template} thiếu placeholder __ITEMS__")
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page.replace("__ITEMS__", payload, 1), encoding="utf-8")
    return out_html
