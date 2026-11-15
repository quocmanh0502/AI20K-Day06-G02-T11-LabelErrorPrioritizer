# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G02 |
| Challenge | T11 · Post-label Error Review Prioritizer |
| Thành viên + vai trò | 1. Hoàng Quốc Mạnh (2A202603219) (A) — Lead/Integrator · Core algorithm · QA/metric · Report · 2. Nguyễn Mạnh Dũng (2A202603197) (B) — Review UI (contact sheet) · Human reviewer |
| Git URL | https://github.com/quocmanh0502/AI20K-Day06-G02-T11-LabelErrorPrioritizer |
| Tag `demo-180m` (commit) | xem `git rev-parse --short demo-180m` (tag tạo sau commit cuối của bản demo) |
| Tag `final-24h` (commit) | xem `git rev-parse --short final-24h` |
| Deadline (giờ VN) | +24h sau buổi lab ngày 2026-11-14 |

## 01 · Problem statement

- **Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai?** Đội train detector: nhãn sai (car ghi thành motorbike, box lệch, box trùng, vật bị bỏ sót) lọt vào tập train làm model học sai và metric đánh giá sai. Vendor gán nhãn chịu chi phí làm lại; Data QA lead (owner) chịu trách nhiệm quyết định release.
- **Điểm nghẽn cụ thể:** ~205 annotation, người kiểm tra chỉ duyệt được **B = 10** mục mỗi vòng → phải ưu tiên đúng mục có khả năng sai cao nhất.
- **Đầu vào → đầu ra:** `annotations.csv` (người gán, có lỗi) + `model_predictions.csv` (detector mô phỏng, có `conf`) + 55 ảnh → `ranking.csv` (`item_id, score`) → `review_queue.csv` top-10 → trang review → `corrected_annotations.csv` + `label_changes.csv` + `audit_log.jsonl` → `decision.json`.
- **Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval)** — `src/t11_review.py::decide_policy`:

  | Điều kiện | Quyết định | reason_code |
  |---|---|---|
  | chưa có kết quả review | HOLD | `REVIEW_PENDING` |
  | còn mục `escalate` | HOLD | `ESCALATIONS_OPEN` |
  | lỗi xác nhận / mục đã duyệt ≥ 30% | REWORK | `HIGH_CONFIRMED_ERROR_RATE` |
  | lỗi xác nhận > số đã sửa | HOLD | `CONFIRMED_NOT_FIXED` |
  | còn lại | GO | `ERRORS_FIXED` |

- **Giả định của nhóm (cần domain owner xác nhận):** prediction mô phỏng đủ tin cậy để làm "ý kiến thứ hai"; 4 loại lỗi trong đề là đủ (không có loại "box bịa"); ngưỡng REWORK 30% là chính sách nội bộ của nhóm, cần Data/ML owner phê duyệt.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 38 ảnh · 145 annotation · 140 prediction | `fixtures/t11_label_errors/` (`synthetic:tools/make_fixtures.py (seed 42)`) | CC0-1.0 | Có |
| eval / heldout | 17 ảnh · 60 annotation · 55 prediction (77 item tính cả `img:`) | như trên | CC0-1.0 | Có |

- **Schema & class mapping:** annotation `ann_id, image_id, class, x, y, w, h` (pixel, ảnh 320×180); prediction `image_id, class, x, y, w, h, conf`; 4 class `car, motorbike, pedestrian, sign`. Item chấm = `ann_id` hoặc `img:<image_id>`.
- **Seed:** 42 · **Cách chia split:** theo ảnh (cột `split` trong `images.csv`, có sẵn trong fixture) — mọi box của một ảnh nằm cùng split.
- **`manifest.csv` verified (sha256):** có — kiểm ở bước `load()` mỗi lần chạy (`strict_provenance: true`, sai hash → dừng).
- **Dữ liệu thật:** không dùng.

## 03 · Research & design

- **References & vì sao chọn kỹ thuật:** ý tưởng "bất đồng giữa nhãn người và detector" theo hướng của Cleanlab object detection (nhãn nghi ngờ khi prediction tự tin mâu thuẫn với annotation) và quy trình consensus/QA của CVAT (người duyệt xác nhận trước khi sửa). Chọn luật dựa trên IoU + conf vì: chạy offline trên CPU, không cần train, giải thích được từng điểm cho người duyệt (cột `reason`).
- **Sơ đồ data flow:**

  ```
  manifest.csv ─(sha256)─► 01 load ─► 02 process: match IoU → 4 detector → score = max ─► 03 rank top-10
        ─► 04 review: contact_sheet.html → review_queue_reviewed.csv → sửa bản sao + audit_log.jsonl
        ─► 05 decide: decide_policy → decision.json (status, reason_code, owner)
  ```

- **4 detector** (`src/t11_scoring.py`, tham số đóng băng trong `configs/lab.yaml`):

  | Detector | Điều kiện | Điểm |
  |---|---|---|
  | class_swap | IoU(annotation, prediction khớp nhất) ≥ 0.5 và khác class | conf của prediction |
  | box_shift | 0.05 < IoU < 0.6 | (0.6 − IoU)/0.6 × conf |
  | duplicate | 2 annotation cùng ảnh, cùng class, IoU ≥ 0.5 | IoU cho box khớp model kém hơn; box còn lại × 0.3 |
  | missing | prediction conf ≥ 0.5 không chạm annotation nào (IoU < 0.1) | conf, gán cho `img:<image_id>` |
  | (no_model_match) | IoU ≤ 0.05 (vật tí hon model bỏ sót) | 0.2 cố định |

  `score` = max các detector, `reason` = detector cho điểm cao nhất.
- **Thành phần offline:** không dùng model/weights ngoài; chỉ pandas, numpy, Pillow.
- **Trade-offs đã cân nhắc:** (1) gộp bằng `max` thay vì tổng có trọng số — đơn giản, mỗi điểm có một lý do rõ; (2) ngưỡng box_shift 0.6 lấy từ phân bố IoU dev (đa số box bình thường 0.6–0.95, trung vị 0.76); (3) loại IoU ≈ 0 khỏi box_shift vì 16/145 box dev là vật tí hon model bỏ sót, không phải lỗi; (4) duplicate chỉ nghi mạnh 1 box/cặp để không tốn 2 chỗ review; (5) **không** dùng tên ID (`ann_dXX` để lộ cách chèn lỗi) làm đặc trưng.

## 04 · Baseline vs PoC

- **Baseline:** random B mục, seed 42 — `src/examples/random_baseline.py` (có sẵn), chạy `python -m src.main --track t11 --pipeline src.examples.random_baseline --output out/baseline --offline --seed 42`.
- **PoC:** `src/pipeline.py` + `src/t11_scoring.py` + `src/t11_review.py`, config `configs/lab.yaml` (params FROZEN 2026-11-14).
- **Cùng budget / cùng pool / cùng seed:** xác nhận — B = 10, seed 42, cùng 260 item, so trên cùng pool dev.

**Bảng 1 · Baseline vs PoC** — sinh bởi `python tools/t11_dev_compare.py` (`reports/t11_dev_baseline_vs_poc.md`).
Không có `gt/` nên `src.compare` (heldout) chưa chạy được; bảng dưới dùng nhãn tự gán trên dev.

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary (dev proxy): Precision@10 (lỗi thật / 10 mục top dev) | 0.10 (1/10), Wilson 95% [0.02, 0.40] | 0.90 (9/10), Wilson 95% [0.60, 0.98] | +0.80; higher is better; N = 10/run |
| Loại lỗi phủ trong top-10 dev | 1/4 (box_shift) | 3/4 (class_swap, duplicate, missing) | box_shift của PoC nằm ở hạng 15–20 dev |
| Review@B trên queue thật (người duyệt) | — | 9/10 xác nhận lỗi, 1 báo nhầm (`ann_096`) | `reports/t11_decision.json` |
| Runtime / 260 items | 0.0749 s | 0.3349 s | CPU Intel64 Family 6 Model 141, Windows 11 |

Heldout chính thức (Error Recall@10, ceiling 0.909): `python -m src.evaluate --track t11 --pred out/poc/ranking.csv --out reports/metrics_poc.json` — cần overlay `gt/` của giảng viên.

## 05 · Evaluation protocol

- **Primary metric:** Error Recall@B = (lỗi thật trong top-B theo `score` giảm dần, chỉ tính item thuộc heldout) / (tổng lỗi thật trong pool heldout). Đơn vị: item. TP = lỗi thật trong top-B; FN = lỗi thật ngoài top-B; FP = item không lỗi trong top-B. Chi tiết: `reports/metric_contract.md`.
- **Threshold / budget:** B = 10; 7 ngưỡng chốt trên dev ngày 2026-11-14, ghi `FROZEN` trong `configs/lab.yaml`, lưu lại trong `out/poc/run_info.json → stats.t11_params`.
- **Độ bất định:** Wilson 95% CI (dev top-10: PoC [0.60, 0.98] vs baseline [0.02, 0.40] — không chồng lấn). Heldout chỉ ≈ 11 lỗi nên mỗi lỗi ≈ 9 điểm % recall.
- **Lệnh evaluate chính xác:** `python -m src.evaluate --track t11 --pred out/poc/ranking.csv --run-name poc --out reports/metrics_poc.json`

## 06 · Result slices

Precision theo `reason` trong top-20 dev của PoC (`reports/t11_dev_selfcheck.csv`, cột `source` = poc):

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| class_swap | 9 | — | 8/9 lỗi thật | 1 báo nhầm: `ann_096` |
| duplicate | 4 | — | 4/4 | |
| box_shift | 4 | 1/1 (`ann_176`) | 4/4 | baseline chỉ trúng nhờ ngẫu nhiên |
| missing | 3 | — | 2/3 | 1 báo nhầm: `img:lbl_20` |

- **Failure case 1 — `ann_096` (ảnh `lbl_24`, dev):** detector báo class_swap với conf 0.94 (model nói pedestrian, nhãn là motorbike). Nhìn ảnh: có bánh xe đen, không có đầu tròn → nhãn motorbike **đúng**, model sai. *Root cause:* pipeline tin conf của model như nhau cho mọi cặp class, trong khi model hay nhầm motorbike ↔ pedestrian. Người duyệt đã `reject` mục này ở vòng review thật.
- **Failure case 2 — `img:lbl_20` (dev):** model báo `sign` conf 0.57 ở góc dưới-trái trên bãi cỏ trống, không có vật thể. *Root cause:* `miss_min_conf = 0.5` thấp, conf của model không hiệu chỉnh. Không nâng ngưỡng sau khi thấy ca này để tránh overfit 20 mẫu.
- **Giới hạn đã biết — duplicate:** cặp `ann_032`/`ann_d02` khớp model như nhau (IoU 0.827), pipeline không phân biệt được box nào là bản sao và chọn box đứng trước; nếu GT coi box kia là lỗi thì trượt.
- **Latency / compute:** 0.33 s cho toàn pipeline 260 item (≈ 1.3 ms/item) trên CPU Intel64 Family 6 Model 141, Windows 11, Python 3.12.8.

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv`, B = 10 mục `score` cao nhất (mọi split), có `reason` và `evidence_path`.
- **Trang review:** `out/poc/contact_sheet.html` do bước `review()` tự sinh (template `src/templates/t11_review_workbench.html`, giao diện do Nguyễn Mạnh Dũng thiết kế): ảnh + box người (đỏ = đang nghi, cam = khác) + box model (xanh, kèm conf), chọn quyết định, sửa trực tiếp, bắt buộc tên reviewer, nút Export → `review_queue_reviewed.csv`.
- **Override + audit log:** quyết định `edit / approve / reject / escalate / skip`; cú pháp sửa `class=…`, `box=x,y,w,h`, `replace=cls,x,y,w,h`, `delete`, `add=cls,x,y,w,h`. Sửa trên **bản sao** → `corrected_annotations.csv`; diff → `label_changes.csv`; mỗi quyết định → 1 dòng `audit_log.jsonl` (append-only: ts, actor, action, item_id, before, after, reason). File gốc trong `fixtures/` không đổi; `ranking.csv` không đổi sau review.
- **Kết quả vòng review thật** (`reports/t11_review_queue_reviewed.csv`, `reports/t11_label_changes.csv`, `reports/t11_audit_log.jsonl`): 10 mục duyệt bởi Nguyễn Mạnh Dũng (B) → 9 `edit` (4 class, 3 delete duplicate, 2 add missing), 1 `reject` (`ann_096`). Annotation 205 → 204. Ảnh trước/sau: `demo/t11_before_after.png`.
- **Decision** (`reports/t11_decision.json`): **REWORK · `HIGH_CONFIRMED_ERROR_RATE`** — 9/10 mục là lỗi thật (90%) ≥ 30%; owner: Hoàng Quốc Mạnh (Data QA lead).
- **Khi nào HOLD/REWORK thay vì tự động release:** không bao giờ tự sửa nhãn chỉ dựa trên model (ca `ann_096` cho thấy model tự tin vẫn sai). Chưa review → HOLD; còn escalate → HOLD; tỷ lệ lỗi trong mẫu top-B ≥ 30% → REWORK vì lỗi mang tính hệ thống, sửa 10 box không đủ để release.

## 08 · Reproduce & tests

```powershell
python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
python -m src.main --track t11 --output out/poc --run-name poc --offline --seed 42          # lần 1: sinh contact sheet, HOLD
# duyệt trên out/poc/contact_sheet.html -> Export -> chép vào out/poc/review_queue_reviewed.csv
#   (bản đã duyệt: copy reports/t11_review_queue_reviewed.csv)
python -m src.main --track t11 --output out/poc --run-name poc --offline --seed 42          # lần 2: áp sửa, REWORK
python -m src.main --track t11 --pipeline src.examples.random_baseline --output out/baseline --run-name baseline --offline --seed 42
python tools/t11_dev_compare.py
python tools/t11_before_after.py
pytest -q
```

- **Output của `pytest -q`:** `74 passed, 39 skipped` (39 skipped = test cần `gt/` của giảng viên). Test của nhóm: `tests/test_t11_scoring.py` (7 test: IoU, 4 detector, ảnh không có prediction, conf thấp, override params) và `tests/test_t11_review.py` (20 test: 5 kiểu sửa, sửa trên bản sao, audit log, 8 override sai cú pháp, thiếu reviewer, 5 nhánh decision, trang review).
- **Tái lập:** chạy 2 lần cùng seed → `ranking.csv` giống hệt byte-by-byte.
- **Máy / CPU / OS / Python:** Intel64 Family 6 Model 141 · Windows 11 (10.0.26200) · Python 3.12.8 · git sha: lấy từ `out/poc/run_info.json → git_sha`.

## 09 · Limitations & ethics

- **Rủi ro pháp lý / PII / license:** dữ liệu synthetic CC0, không có người hay biển số thật; không dùng `data/real/`.
- **Điều chưa biết; false alarms; scale 10×:** prediction là detector mô phỏng — với model thật, conf có thể kém hiệu chỉnh hơn; nhãn dev do 1 người gán (không có đồng thuận nhiều người); N nhỏ (10 mẫu/run trên dev, ≈ 11 lỗi heldout). False alarm đã thấy: model nhầm motorbike ↔ pedestrian, model báo vật trên nền trống. Scale 10×: vòng lặp theo ảnh vẫn nhanh (tuyến tính theo số ảnh), nhưng B = 10 cố định sẽ phủ tỷ lệ nhỏ hơn → cần review theo lô hoặc tăng B theo 10%·N.
- **Không tuyên bố "đạt GDPR / luật Việt Nam"** từ checklist tự động; ngưỡng REWORK 30% là chính sách nội bộ cần Data/ML owner xác nhận.
- **Kế hoạch cải tiến:** hiệu chỉnh độ tin model theo từng cặp class (ma trận nhầm lẫn trên dev) để giảm ca như `ann_096`; dùng `cleanlab.object_detection` làm đối chứng; dùng 2 detector khác nhau và lấy bất đồng; phân biệt bản gốc/bản sao duplicate bằng thứ tự thời gian gán nhãn nếu có.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Hoàng Quốc Mạnh (A) | Quoc Manh · xem `git shortlog -sne --all` | Core algorithm + integration + QA + report: `src/t11_scoring.py`, `src/t11_review.py`, `src/pipeline.py`, `tools/t11_*.py`, `tests/test_t11_*.py`, `configs/lab.yaml`, `reports/` (trừ bằng chứng review), `demo/slides.md` | `fb96b66`, `bb96264`, `0e514ba` + commit phần review code/report (`git log --author="Quoc Manh"`) | Có |
| Nguyễn Mạnh Dũng (B) | github `ngmanhdungg` · xem `git shortlog -sne --all` | Review UI + human reviewer: `src/templates/t11_review_workbench.html`, vòng review thật → `reports/t11_review_queue_reviewed.csv`, `reports/t11_label_changes.csv`, `reports/t11_audit_log.jsonl`, `reports/t11_decision.json`, `demo/t11_before_after.png`, `demo/t11_contact_sheet.html` | `git log --author` của B (2 commit) | Có |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| https://docs.cleanlab.ai/ (object detection: `cleanlab.object_detection.filter.find_label_issues`) | 2026-11-14 | Tìm nhãn bbox nghi sai bằng cách so annotation với prediction có độ tin; là đối chứng gợi ý trong đề |
| https://docs.cvat.ai/ (Consensus, Quality control) | 2026-11-14 | Quy trình nhiều người gán/duyệt và đo chất lượng nhãn; cơ sở cho bước human review trước khi sửa |
| `docs/tracks/T11.md` (handout) | 2026-11-14 | Đề bài, output contract, metric Error Recall@B, oracle ceiling 0.909 |
