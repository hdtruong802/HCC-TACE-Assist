# T3_W3 — Grad-CAM Sanity Check (Reality-check gate #1)

> **Dự án:** Liver Cancer AI — phân loại slice CT có/không tổn thương gan → gộp patient.
> **Người thực hiện:** Hoàng Đức Trường · **Ngày:** 22/07/2026 · **Định vị:** Research Use Only (RUO), chưa kiểm định lâm sàng.
>
> **Tóm tắt:** Trước khi đầu tư sang Phase 2 / nâng cấp detect+classify, cần trả lời: model chính **ConvNeXt V2 nano** bám vào **đặc trưng tổn thương trong gan** hay học **"shortcut"** (artifact, viền, marker, vị trí cố định)? Dùng **Grad-CAM** soi định tính trên validation LiTS. Kết luận: **PASS** — heat khu trú trên vùng gan, **thay đổi vị trí theo từng bệnh nhân** (loại trừ location-prior), không bám text/không khí/viền. Đây là cửa kiểm chứng #1; cửa #2 (external IRCADb) xem `report/T3_W3_External_IRCADb.md`.

---

## 1. Mục tiêu
- Kiểm tra **model có học đúng bản chất** (đặc trưng tổn thương) hay khai thác tương quan giả (shortcut) — nguyên nhân phổ biến khiến AUROC nội bộ đẹp nhưng sập khi external.
- Là **phép thử rẻ** (không cần data mới) để quyết có nên tiếp tục hướng hiện tại trước khi đổi/augment data.

## 2. Phương pháp
- **Grad-CAM** generic cho backbone timm: lấy feature map từ `forward_features` (trước global-pool), trọng số = gradient trung bình không gian → tổ hợp kênh → ReLU → chuẩn hoá → overlay lên ảnh gốc.
- **Montage theo 3 nhóm** để soi cả đúng lẫn sai:
  - **TP** (dương đúng, tự tin nhất) → heat *phải* trùng ổ tổn thương.
  - **FP** (gan bình thường bị kêu dương) → model bám vào **cái gì**?
  - **FN** (u bị bỏ sót) → ổ nào bị lơ.
- **Đa dạng hoá:** chọn tối đa **1 slice/bệnh nhân** mỗi nhóm (`--cap 1`) → mỗi hàng 6 bệnh nhân khác nhau, tránh 1 ca "dễ" chiếm hết.

**Thiết lập:** ckpt `convnextv2_nano_fold0` (best theo slice_auroc), validation fold0, ngưỡng slice **0.20** (khóa từ val), 6 ví dụ/nhóm.

## 3. Kết quả

![Grad-CAM montage (6 bệnh nhân khác nhau mỗi hàng)](../image_eval/gradcam/gradcam_val_1.png)

**Hàng TP (6 ca: lits-26/109/93/31/52/113):** heat là **khối gọn, khu trú trên vùng gan** ở cả 6 ca; **vị trí thay đổi theo từng bệnh nhân** (vòm gan, thùy dưới, giữa, hai ổ...). → Model bám đặc trưng khu trú, **không** phải nền vị trí cố định.

**Hàng FP (lits-15/36/113/41/26/109):** dương-nhầm nhưng heat **vẫn nằm trên mô gan** (không bám artifact/marker/viền). → Gợi ý các "sai" này phần lớn là **nhãn biên do label-transfer** (ổ nhỏ dưới τ=20 hoặc slice kề u), không phải model học sai bản chất.

**Hàng FN (lits-21/59/0/113/77/67):** heat **tản mạn, lệch khỏi gan** (ruột/thận/cột sống/viền). → Trên ca u khó, model **không định vị được** → đúng kiểu recall yếu ở ca khó, không phải gian lận.

## 4. Kết luận
> ✅ **PASS** — Không có dấu hiệu shortcut. Model phát hiện dựa trên **đặc trưng khu trú trong gan**; kiểu thất bại (FP nhãn-biên, FN ca khó) là hợp lý và giải thích được. Kết luận này được **củng cố độc lập** bởi external IRCADb (slice-AUROC 0.807, Δ chỉ −0.07 — nếu là shortcut thì đã sập về ~0.5).

## 5. Hạn chế
- **Định tính (nhìn mắt)**, không đo IoU: cache đã **liver-crop** nên gần như cả ảnh là vùng gan → check ở đây là "heat có trúng ổ tổn thương *trong* gan", và cache **không lưu mask u** để overlay ground-truth. Muốn định lượng cần dựng lại mask u → nằm ngoài phạm vi sanity check.
- **Ngưỡng thấp (0.20):** model kêu dương dễ → nhiều FP tự tin → **Specificity là điểm yếu** (đã đo có nghĩa ở external, xem T3_W3_External).
- 1 fold / 1 checkpoint; chưa soi FastViT (finalist còn lại).

## 6. Tái lập
- **Lệnh (Kaggle, notebook 02 cell 6):**
  ```bash
  python -m src.interpretability.gradcam --config configs/train/base.yaml \
      --ckpt outputs/convnextv2_nano_fold0/best.ckpt --split val --fold 0 --n-per 6 --cap 1
  ```
- **Artifacts:** `outputs/convnextv2_nano_fold0/gradcam/gradcam_val.png` → lưu về `image_eval/gradcam/gradcam_val_1.png`.

---
> **Disclaimer y tế:** Research Use Only — chưa kiểm định lâm sàng. Grad-CAM là công cụ diễn giải định tính, không phải bằng chứng định vị tổn thương chính xác; không dùng để tuyên bố hiệu năng lâm sàng.
