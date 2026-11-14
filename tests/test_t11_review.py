"""Tests cho src/t11_review.py — sửa nhãn, audit log, chính sách quyết định, trang review."""
import json
import re

import pandas as pd
import pytest

from src.common.review import REVIEW_COLUMNS, AuditLog, load_review_decisions
from src.t11_review import (CHANGE_COLUMNS, apply_label_fixes, build_workbench_items, decide_policy,
                            render_workbench, review_stats)

ANN = pd.DataFrame([
    ("a1", "im1", "motorbike", 10.0, 10.0, 20.0, 40.0),
    ("a2", "im1", "car", 100.0, 50.0, 40.0, 20.0),
    ("a3", "im1", "car", 101.0, 50.0, 40.0, 20.0),
], columns=["ann_id", "image_id", "class", "x", "y", "w", "h"])


def _reviewed(rows):
    """rows = [(item_id, decision, override, reviewer)] -> DataFrame giống output load_review_decisions."""
    return pd.DataFrame([{"item_id": i, "reviewer_decision": d, "override_value": o, "reviewer": who,
                          "review_note": "note"} for i, d, o, who in rows])


# ---------------------------------------------------------------- success
def test_class_box_delete_add_replace_are_applied_on_a_copy():
    original = ANN.copy()
    reviewed = _reviewed([
        ("a1", "edit", "class=pedestrian", "Manh"),
        ("a2", "edit", "box=1,2,3,4", "Manh"),
        ("a3", "edit", "delete", "Manh"),
        ("img:im1", "edit", "add=sign,5,5,10,10", "Manh"),
    ])
    fixed, changes = apply_label_fixes(ANN, reviewed)
    pd.testing.assert_frame_equal(ANN, original)                      # bảng gốc không đổi
    f = fixed.set_index("ann_id")
    assert f.loc["a1", "class"] == "pedestrian"
    assert f.loc["a2", ["x", "y", "w", "h"]].tolist() == [1, 2, 3, 4]
    assert "a3" not in f.index
    new = fixed[fixed.ann_id.str.startswith("new_")]
    assert len(new) == 1 and new.iloc[0]["image_id"] == "im1" and new.iloc[0]["class"] == "sign"
    assert list(changes.kind) == ["class", "box", "delete", "add"]


def test_replace_changes_class_and_box_together():
    fixed, changes = apply_label_fixes(ANN, _reviewed([("a1", "edit", "replace=pedestrian,11,9,18,42", "Manh")]))
    row = fixed.set_index("ann_id").loc["a1"]
    assert row["class"] == "pedestrian" and row[["x", "y", "w", "h"]].tolist() == [11, 9, 18, 42]
    assert changes.kind.tolist() == ["replace"]


def test_reject_and_approve_change_nothing_but_keep_columns():
    fixed, changes = apply_label_fixes(ANN, _reviewed([("a1", "reject", "", "Manh"), ("a2", "approve", "", "Manh")]))
    pd.testing.assert_frame_equal(fixed, ANN)
    assert changes.empty and list(changes.columns) == CHANGE_COLUMNS


def test_audit_log_records_who_changed_what(tmp_path):
    audit = AuditLog(tmp_path / "audit_log.jsonl")
    apply_label_fixes(ANN, _reviewed([("a1", "edit", "class=pedestrian", "Manh"),
                                      ("a3", "edit", "delete", "Manh"),
                                      ("a2", "reject", "", "Manh")]), audit)
    events = audit.read()
    assert [e["action"] for e in events] == ["label_fix:class", "label_fix:delete", "review:reject"]
    assert all(e["actor"] == "Manh" for e in events)
    assert '"motorbike"' in events[0]["before"] and events[0]["after"] == "class=pedestrian"


# ---------------------------------------------------------------- failure / edge
@pytest.mark.parametrize("item_id,value", [
    ("a1", "class=truck"),            # class lạ
    ("a1", "box=1,2,3"),              # thiếu toạ độ
    ("a1", "box=1,2,0,4"),            # w = 0
    ("a1", "colour=red"),             # lệnh lạ
    ("a1", "add=car,1,2,3,4"),        # add trên annotation
    ("img:im1", "delete"),            # delete trên ảnh
    ("img:im1", "class=car"),         # class trên ảnh
    ("zzz", "delete"),                # ann_id không tồn tại
])
def test_invalid_overrides_fail_loudly(item_id, value):
    with pytest.raises(ValueError):
        apply_label_fixes(ANN, _reviewed([(item_id, "edit", value, "Manh")]))


def test_reviewed_row_without_reviewer_is_rejected(tmp_path):
    q = pd.DataFrame([{c: "" for c in REVIEW_COLUMNS}])
    q.loc[0, ["rank", "item_id", "reviewer_decision", "override_value"]] = ["1", "a1", "edit", "class=car"]
    p = tmp_path / "review_queue_reviewed.csv"
    q.to_csv(p, index=False)
    with pytest.raises(ValueError, match="reviewer"):
        load_review_decisions(p)


# ---------------------------------------------------------------- decision policy
@pytest.mark.parametrize("stats,expected", [
    ({}, ("HOLD", "REVIEW_PENDING")),
    ({"n_reviewed": 10, "n_confirmed": 2, "n_escalated": 1, "n_fixed": 2}, ("HOLD", "ESCALATIONS_OPEN")),
    ({"n_reviewed": 10, "n_confirmed": 9, "n_fixed": 9}, ("REWORK", "HIGH_CONFIRMED_ERROR_RATE")),
    ({"n_reviewed": 10, "n_confirmed": 2, "n_fixed": 1}, ("HOLD", "CONFIRMED_NOT_FIXED")),
    ({"n_reviewed": 10, "n_confirmed": 2, "n_fixed": 2}, ("GO", "ERRORS_FIXED")),
])
def test_decide_policy_branches(stats, expected):
    status, code, reason = decide_policy(stats)
    assert (status, code) == expected and reason


def test_review_stats_counts_decisions():
    reviewed = _reviewed([("a1", "edit", "class=car", "M"), ("a2", "approve", "", "M"),
                          ("a3", "reject", "", "M"), ("img:im1", "escalate", "", "M")])
    s = review_stats(reviewed, pd.DataFrame([{}]))
    assert (s["n_reviewed"], s["n_confirmed"], s["n_false_positive"], s["n_escalated"], s["n_fixed"]) == (4, 2, 1, 1, 1)


# ---------------------------------------------------------------- workbench page
def test_workbench_page_embeds_queue_items(tmp_path):
    from PIL import Image
    (tmp_path / "images").mkdir()
    Image.new("RGB", (320, 180)).save(tmp_path / "images" / "im1.jpg")
    images = pd.DataFrame({"image_id": ["im1"], "path": ["images/im1.jpg"], "width": [320], "height": [180]})
    pred = pd.DataFrame([("im1", "pedestrian", 10.0, 10.0, 20.0, 40.0, 0.9)],
                        columns=["image_id", "class", "x", "y", "w", "h", "conf"])
    queue = pd.DataFrame({"rank": [1, 2], "item_id": ["a1", "img:im1"], "score": [0.9, 0.5]})
    items = build_workbench_items(queue, ANN, pred, images, tmp_path)
    out = render_workbench(items, tmp_path / "contact_sheet.html")
    page = out.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const ITEMS=(\[.*?\]);\n", page, re.S).group(1))
    assert [d["item_id"] for d in data] == ["a1", "img:im1"]
    red = [b for b in data[0]["boxes"] if b["color"] == "red"]
    assert len(red) == 1 and red[0]["ann_id"] == "a1"
    assert any(b["color"] == "green" for b in data[0]["boxes"])
    assert data[1]["original_box"] is None and "__ITEMS__" not in page
