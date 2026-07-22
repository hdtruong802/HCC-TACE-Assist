# T3_W3 — External Validation trên 3D-IRCADb-01 (Golden set, chạm 1 lần)

> **Dự án:** Liver Cancer AI — phân loại slice CT có/không tổn thương gan → gộp patient.
> **Người thực hiện:** Hoàng Đức Trường · **Ngày:** 22/07/2026 · **Định vị:** Research Use Only (RUO), chưa kiểm định lâm sàng.
>
> **Tóm tắt:** Đánh giá **generalization** của model chính **ConvNeXt V2 nano** (chọn trên LiTS-val) trên bộ **external hoàn toàn độc lập 3D-IRCADb-01** — khác máy chụp, khác annotator. **Chạm 1 lần**, threshold **khóa từ val** (không tinh chỉnh). Kết quả: **slice-AUROC = 0.807 [0.678, 0.902]** (internal 0.882 → external, Δ ≈ −0.07). Kèm với Grad-CAM sạch (không shortcut), đây là **cửa kiểm chứng #2 PASS** ở mức slice: khả năng phân biệt của model **tổng quát hóa được** sang miền dữ liệu mới.

---

## 1. Mục tiêu
- Kiểm chứng model có **tổng quát hóa** ngoài LiTS hay chỉ overfit đặc điểm máy chụp/nhãn LiTS.
- Đo **Δ domain-shift** = external − internal (spec sheet §4.6).
- Lần đầu có **ca âm thật đủ nhiều (25%)** → **Specificity** mới có ý nghĩa (LiTS chỉ 13/131 ca âm).
- **Ràng buộc:** external là *golden set*, chỉ chạm **1 lần**, **không** chọn model/không tinh chỉnh threshold ở đây.

## 2. Dữ liệu external

| Hạng mục | Giá trị |
|---|---|
| Nguồn | **3D-IRCADb-01** (IRCAD, 20 ca CT bụng có mask gan + u, DICOM) |
| Kaggle | `sarahelqersh/3dircadb1` (có nesting thừa 1 tầng — pipeline tự bóc) |
| Bệnh nhân | **20** — **15 có u / 5 không u** (ircad-5, 7, 11, 14, 20 không có folder `livertumor`) |
| Slice giữ lại (có gan) | **2,068** — **positive 556 / negative 1,512** (pos_ratio **0.269**) |
| Tiền xử lý | **Trùng khớp W1**: window gan WL50/WW350, liver-crop bbox, resize 256, uint8 |
| Label transfer | **Cùng τ**: `τ_area=20`, `τ_liver=50`; seg 0/1/2 (u đè gan) như LiTS |
| Đọc DICOM | HU = pixel·RescaleSlope + Intercept (kiểm tra HU ∈ [−1024, 1023] ✓); mask gộp mọi `livertumor*` |

> **Đối chiếu công bằng:** external đi qua **đúng** `preprocess`/`label_transfer` của LiTS; chỉ khác lớp đọc (DICOM thay NIfTI). Không thay đổi τ, window, size, cách gộp.

## 3. Thiết lập đánh giá
- **Model:** ConvNeXt V2 nano, checkpoint `convnextv2_nano_fold0` (best theo `slice_auroc` trên LiTS-val fold0).
- **Threshold KHÓA từ val:** patient = **0.9966**, slice = **0.2029** (Youden trên val, **không** đụng lại).
- **Suy luận:** toàn bộ 2,068 slice / 20 bệnh nhân; gộp patient mean-of-top-k (k=3).
- **CI:** cluster-bootstrap **theo bệnh nhân** (2,000 lần) cho cả slice & patient.

## 4. Kết quả

### 4.1. Bảng chính — Internal (val) vs External (IRCADb)

| Chỉ số | Internal val (ConvNeXt) | **External IRCADb** | Δ | Ghi chú |
|---|--:|--:|--:|---|
| **slice-AUROC** | 0.882 [0.815, 0.921] | **0.807 [0.678, 0.902]** | **−0.075** | CI loại trừ 0.5 → phân biệt **thật** |
| slice PR-AUC | 0.822 | 0.699 | −0.123 | prevalence 0.27 → 0.70 vẫn tốt |
| slice Sens/Spec @thr 0.20 | — | 0.741 / 0.708 | — | ngưỡng slice **chuyển được**, cân bằng |
| slice Sens@Spec=0.90 | 0.678 | 0.590 | −0.088 | vùng spec-cao suy giảm |
| patient-AUROC | 0.728 [0.51, 0.92] | 0.687 [0.379, 0.941] | −0.041 | **nhiễu** (20 ca, 5 âm) |
| patient PR-AUC | 0.955 | 0.854 | — | thổi phồng do prevalence 0.75 |
| patient Sens/Spec @thr 0.997 | — | 0.733 / 0.600 | — | ngưỡng patient **chuyển kém** |

*(TP=11, FP=2, TN=3, FN=4 ở mức bệnh nhân; slice_n=2,068, slice_pos=556.)*

### 4.2. Đường cong (slice-level — đáng tin nhất, 2,068 mẫu)

| ROC slice — AUROC 0.807 | PR slice — 0.699 (prevalence 0.27) |
|:--:|:--:|
| ![ROC slice](../image_eval/ircad_convnextv2/roc_slice.png) | ![PR slice](../image_eval/ircad_convnextv2/pr_slice.png) |

| ROC patient — 0.687 (20 ca) | Confusion patient @thr=0.997 |
|:--:|:--:|
| ![ROC patient](../image_eval/ircad_convnextv2/roc.png) | ![Confusion](../image_eval/ircad_convnextv2/confusion.png) |

## 5. Đánh giá

**Điểm tích cực (quyết định)**
- **Generalization có thật:** slice-AUROC **0.807** trên bộ **hoàn toàn độc lập** (khác máy chụp, khác annotator, DICOM thô). CI dưới **0.678 > 0.5** → không phải may rủi.
- **Δ ≈ −0.07 là mức bình thường** cho cross-dataset — **không** sụp về 0.5–0.6. Kết hợp Grad-CAM sạch → **loại trừ giả thuyết shortcut/overfit đặc điểm LiTS**.
- **Orientation không hỏng:** nếu xử lý DICOM sai chiều (lật) thì AUROC đã sập; 0.807 xác nhận hướng xử lý native đúng — gỡ được mối lo đã nêu.
- **Ngưỡng slice (0.20) chuyển tốt:** Sens 0.74 / Spec 0.71 cân bằng trên miền mới.

**Hạn chế phải nêu rõ**
- **Patient-level nhiễu:** 20 ca (5 âm) → patient-AUROC 0.687 với **CI rất rộng [0.38, 0.94]**; `Sens@Spec90(patient)=0` là **artifact mẫu nhỏ**, không kết luận được.
- **Threshold-transfer ở mức bệnh nhân kém:** ngưỡng patient khóa từ val (0.997, do val gần như toàn ca dương) **không chuyển tốt** → Spec chỉ 0.60. Đây là vấn đề **calibration/threshold**, *không phải* khả năng phân biệt (AUROC độc lập ngưỡng vẫn ổn). → Nếu triển khai, **cần calibrate lại ngưỡng patient** trên tập có đủ ca âm.
- **PR-AUC slice giảm** (0.82 → 0.70): còn nhiều slice khó (rìa tổn thương, ổ nhỏ, isoattenuating) ở miền mới.
- **Label-transfer khác annotator:** τ=20 áp cho nhãn IRCADb (chuẩn vẽ khác LiTS) → một phần "sai" có thể là biên nhãn, không phải model.

## 6. Kết luận & Milestone

> **Cả 2 cửa kiểm chứng đã PASS:** Grad-CAM (không shortcut) + External IRCADb (**slice-AUROC 0.807, Δ −0.07**). **Nền phát hiện mức-slice tổng quát hóa được và đáng tin cậy.** Điểm yếu duy nhất — patient-level & threshold-transfer — đúng bản chất hạn chế dữ liệu đã biết (thiếu ca âm), **không** phải lỗi năng lực model.

| Mốc | Trạng thái |
|---|---|
| **Reality-check gate #1 — Grad-CAM sanity** | ✅ PASS |
| **Reality-check gate #2 — External IRCADb (slice)** | ✅ PASS (slice-AUROC 0.807) |

→ **Có cơ sở vững** để (a) chốt Phase 2 hoàn thiện tầng phát hiện, và/hoặc (b) **nâng cấp sang detect + classify** (tầng phân loại đặt trên nền phát hiện đã kiểm chứng).

## 7. Tái lập
- **Code:** GitHub `hdtruong802/HCC-TACE-Assist`, branch `main`.
- **Data:** Kaggle `sarahelqersh/3dircadb1`; ckpt `marcohoang/convnextv2-nano-fold0`.
- **Lệnh (Kaggle, notebook 02 cell 7→9):**
  ```bash
  python scripts/build_ircad.py --data-root /kaggle/input --out-dir /kaggle/working
  python -m src.evaluation.eval_external --config configs/train/base.yaml \
      --ircad-config configs/data/ircad.yaml --ckpt <best.ckpt> \
      --data-root /kaggle/working --out /kaggle/working/eval_out/convnextv2_nano
  ```
- **Artifacts:** `eval_out/convnextv2_nano/eval_ircad/{metrics.json, roc_slice.png, pr_slice.png, roc.png, confusion.png, ...}`.

---
> **Disclaimer y tế:** Research Use Only — chưa kiểm định lâm sàng. External test chạm 1 lần trên 20 bệnh nhân (5 ca âm); patient-level mang tính tham khảo do cỡ mẫu nhỏ. Số liệu không dùng để tuyên bố hiệu năng lâm sàng.
