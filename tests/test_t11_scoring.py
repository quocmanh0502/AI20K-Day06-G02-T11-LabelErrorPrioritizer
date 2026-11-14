"""Tests cho src/t11_scoring.py — dữ liệu tự dựng, không cần gt/."""
import pandas as pd
import pytest

from src.t11_scoring import build_scores, iou

IMAGES = pd.DataFrame({"image_id": ["im1", "im2"], "path": ["images/im1.jpg", "images/im2.jpg"]})


def _ann(rows):
    return pd.DataFrame(rows, columns=["ann_id", "image_id", "class", "x", "y", "w", "h"])


def _pred(rows):
    return pd.DataFrame(rows, columns=["image_id", "class", "x", "y", "w", "h", "conf"])


# ---------------------------------------------------------------- IoU
def test_iou_basic_cases():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)       # trùng khít
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0                     # không chạm
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)  # lệch nửa khung


def test_iou_zero_area_box_does_not_crash():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------------------------------------------------------------- success: mỗi loại lỗi lên đầu bảng
def test_each_error_type_is_ranked_above_clean_boxes():
    ann = _ann([
        ("clean", "im1", "car", 10, 10, 40, 20),            # khớp model, đúng class -> sạch
        ("swap", "im1", "motorbike", 100, 10, 40, 20),      # model nói car, cùng chỗ -> class_swap
        ("shift", "im1", "car", 215, 10, 40, 20),           # lệch so với model -> box_shift
        ("dupA", "im2", "sign", 10, 10, 20, 20),            # 2 khung sign đè nhau -> duplicate
        ("dupB", "im2", "sign", 11, 10, 20, 20),
    ])
    pred = _pred([
        ("im1", "car", 10, 10, 40, 20, 0.9),
        ("im1", "car", 100, 10, 40, 20, 0.9),
        ("im1", "car", 200, 10, 40, 20, 0.9),
        ("im2", "sign", 10, 10, 20, 20, 0.9),
        ("im2", "car", 200, 100, 40, 20, 0.95),             # model thấy xe, không ai vẽ -> missing ở im2
    ])
    out = build_scores(ann, pred, IMAGES).set_index("item_id")

    assert out.columns[0] == "score" and out.index.name == "item_id"
    assert out.loc["swap", "reason"] == "class_swap"
    assert out.loc["shift", "reason"] == "box_shift"
    assert out.loc["img:im2", "reason"] == "missing"
    assert out.loc["clean", "reason"] == "looks_ok"
    for bad in ["swap", "shift", "img:im2"]:
        assert out.loc[bad, "score"] > out.loc["clean", "score"]
    # duplicate: chỉ MỘT khung trong cặp bị nghi mạnh (không phí 2 chỗ review)
    dup_scores = sorted([out.loc["dupA", "score"], out.loc["dupB", "score"]])
    assert dup_scores[1] >= 0.5 and dup_scores[0] < 0.5


# ---------------------------------------------------------------- edge / failure
def test_image_without_predictions_is_not_flagged_missing():
    ann = _ann([("a1", "im1", "car", 10, 10, 40, 20)])
    pred = _pred([])
    out = build_scores(ann, pred, IMAGES).set_index("item_id")
    assert out.loc["img:im1", "score"] == 0.0
    assert out.loc["img:im2", "reason"] == "looks_ok"


def test_low_confidence_orphan_prediction_is_ignored():
    """Model không chắc (conf < miss_min_conf) thì không được báo ảnh thiếu khung."""
    ann = _ann([("a1", "im1", "car", 10, 10, 40, 20)])
    pred = _pred([("im1", "car", 10, 10, 40, 20, 0.9), ("im1", "sign", 250, 5, 10, 10, 0.3)])
    out = build_scores(ann, pred, IMAGES).set_index("item_id")
    assert out.loc["img:im1", "score"] == 0.0


def test_params_override_changes_behaviour():
    ann = _ann([("a1", "im1", "car", 10, 10, 40, 20)])
    pred = _pred([("im1", "sign", 250, 5, 10, 10, 0.3)])
    strict = build_scores(ann, pred, IMAGES).set_index("item_id")
    loose = build_scores(ann, pred, IMAGES, {"miss_min_conf": 0.2}).set_index("item_id")
    assert strict.loc["img:im1", "score"] == 0.0
    assert loose.loc["img:im1", "score"] == pytest.approx(0.3)


def test_output_has_every_annotation_and_every_image():
    ann = _ann([("a1", "im1", "car", 10, 10, 40, 20), ("a2", "im2", "car", 10, 10, 40, 20)])
    pred = _pred([("im1", "car", 10, 10, 40, 20, 0.9)])
    out = build_scores(ann, pred, IMAGES)
    assert set(out.item_id) == {"a1", "a2", "img:im1", "img:im2"}
    assert out.score.is_monotonic_decreasing
