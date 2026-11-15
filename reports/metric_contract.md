# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T11 · Post-label Error Review Prioritizer |
| Primary metric (tên đúng như evaluator in ra) | `Error Recall@B` |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | item = 1 annotation (`ann_id`) hoặc 1 ảnh thiếu box (`img:<image_id>`) |
| Công thức + numerator / denominator | số lỗi thật trong top-B (theo `score` giảm dần) / tổng số lỗi thật trong pool heldout. Oracle ceiling = min(1, B/#lỗi) = 0.909 |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | khớp chính xác theo `item_id` (không dùng IoU ở bước chấm) |
| Candidate pool cố định | mọi annotation + mọi `img:<image_id>` thuộc 17 ảnh split `heldout` (60 annotation + 17 ảnh = 77 item) |
| Review budget B / K | B = 10 (`configs/tracks/t11.yaml`); review queue người duyệt cũng B = 10 |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | 7 ngưỡng trong `configs/lab.yaml → params` chọn từ phân bố IoU/conf của 38 ảnh dev và kiểm bằng 20 nhãn tay (`reports/t11_dev_selfcheck.csv`, dev Precision@10 = 0.90); không chỉnh sau đó |
| Eval split (không dùng để tune) | `heldout` (17/55 ảnh ≈ 30%) |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | miss = lỗi thật không vào top-B (nặng nhất, nhất là class swap); false alarm = mục báo nhầm chiếm chỗ review (vd `ann_096`, `img:lbl_20`) |
| Slices sẽ báo cáo (≥2) | theo `error_type` (class_swap / box_shift / duplicate / missing) và theo class (car / pedestrian / motorbike / sign) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Wilson 95% CI cho tỷ lệ (N lỗi heldout ≈ 11, dev top-10 = 10 mẫu) |
| Ngày giờ chốt + người chốt | 2026-11-14 · Hoàng Quốc Mạnh |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
