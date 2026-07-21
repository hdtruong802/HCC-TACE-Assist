# T3_W2_Phase1 — So sánh backbone (Pha 1 sàng lọc)

> **Dự án:** Liver Cancer AI · **Người thực hiện:** Hoàng Đức Trường · **Ngày:** 21/07/2026 · **RUO** (chưa kiểm định lâm sàng).
>
> **Mục đích:** Pha 1 của Comparison Protocol — chạy **mỗi backbone 1 lần (fold 0, 1 seed)**, cùng config, xếp hạng theo **slice-level AUROC** (đơn vị ổn định) để chọn **finalist** cho Pha 2 (5-fold). *Đây là sàng lọc sơ bộ, KHÔNG kết luận thắng/thua cuối cùng.*

## 1. Thiết lập (chung cho mọi backbone)
- Dataset `lits-processed` (LiTS, hash `8647d40`); **fold 0**: train 13,482 slice (89 bn) · val 2,555 slice (22 bn), pos 36%.
- Input 256×256, windowing gan + liver-crop; BCE+`pos_weight=1.85`; AdamW + discriminative LR + cosine/warmup; AMP + grad-clip.
- **Chọn best-ckpt/early-stop theo `slice_auroc`**; CI = **cluster-bootstrap theo bệnh nhân** (2000 lần).
- Chỉ đổi **backbone** (timm). Phần cứng: Kaggle T4.

## 2. Bảng xếp hạng Pha 1 (theo slice-level — metric quyết định)

| Hạng | Model | **slice AUROC [95% CI]** | slice PR-AUC | Sens@Spec90 | Sens/Spec @thr | best ep |
|:--:|---|:--:|--:|--:|:--:|:--:|
| 🥇 | **ConvNeXt V2 nano** | **0.882 [0.815, 0.921]** | **0.822** | 0.678 | 0.79 / 0.82 | 2 |
| 🥈 | **FastViT sa12** | **0.874 [0.815, 0.917]** | 0.808 | 0.669 | 0.78 / 0.83 | 4 |
| 🥉 | ResNet-50 *(baseline)* | 0.837 [0.757, 0.892] | 0.772 | 0.654 | 0.70 / 0.87 | 8 |
| 4 | EfficientNet-B0 | 0.826 [0.745, 0.881] | 0.753 | 0.543 | 0.78 / 0.74 | 11 |

## 3. Patient-level (chỉ tham khảo — KHÔNG đáng tin ở 1 fold)

| Model | patient AUROC [CI] | patient PR-AUC |
|---|:--:|--:|
| ConvNeXt V2 nano | 0.728 [0.51, 0.92] | 0.955 |
| FastViT sa12 | 0.842 [0.62, 1.00] | 0.975 |
| ResNet-50 | 0.860 [0.65, 1.00] | 0.978 |
| EfficientNet-B0 | 0.842 [0.45, 1.00] | 0.970 |

> ⚠️ **Nghịch lý minh hoạ vì sao bỏ patient-level 1 fold:** ConvNeXt V2 có **slice-AUROC cao nhất (0.882)** nhưng **patient-AUROC thấp nhất (0.728)**. Với val chỉ 3 ca âm, patient-level + threshold cực kỳ nhiễu (CI [0.51,0.92], PR-AUC ~0.97 "ảo"). → **Xếp hạng theo slice-level là đúng đắn.**

## 4. Nhận định
- **Hai kiến trúc hiện đại 2023 (ConvNeXt V2, FastViT) chiếm top**, đều **vượt baseline ResNet-50** (+0.045 / +0.037 slice-AUROC; PR-AUC thật +0.05). → Hướng SOTA có giá trị.
- **ConvNeXt V2 ≈ FastViT** (0.882 vs 0.874, CI gần trùng [0.815, ~0.92]) → **chưa tách bạch trên 1 fold**; cần Pha 2 để phân định.
- **EfficientNet-B0 dưới baseline** (0.826 < 0.837) → **loại**.
- **Mọi model hiện đại overfit rất nhanh** (best epoch 2–4; train_loss → ~0.07 nhưng val không tăng) trên 89 bn train → Pha 2 cần **tăng regularization** (drop_rate/weight_decay/aug) và/hoặc dừng sớm.

## 5. Quyết định finalist (vào Pha 2)
| Vai trò | Model |
|---|---|
| **Baseline anchor** | ResNet-50 |
| **Finalist 1 (main)** | **ConvNeXt V2 nano** |
| **Finalist 2** | **FastViT sa12** |
| Loại | EfficientNet-B0 |

## 6. Hạn chế
- **Chỉ 1 fold / 1 seed** → xếp hạng mang tính sàng lọc; CI slice còn rộng (~0.10, do val 22 bn).
- ConvNeXt vs FastViT **chưa có ý nghĩa thống kê** (CI chồng) → **bắt buộc Pha 2 (5-fold × seed + paired bootstrap)** mới kết luận.
- Chưa external, chưa Grad-CAM/error analysis (W4).

## 7. Bước tiếp — Pha 2
Chạy **ConvNeXt V2 + FastViT (+ ResNet-50 anchor)** trên **5 fold** (`--fold 0..4`) × ≥2 seed → gộp **OOF** → slice-AUROC + CI hẹp + **paired bootstrap ConvNeXt vs FastViT** → chốt model chính + khóa threshold. *(cân nhắc FPT H100 nếu Kaggle hết quota).*

## 8. Tái lập
- **Lệnh train (mỗi backbone):** `python -m src.training.train --config configs/train/base.yaml --arch <arch> --fold 0`
  (`resnet50` · `convnextv2_nano` · `fastvit_sa12` · `efficientnet_b0`).
- **Eval:** `python -m src.evaluation.evaluate --config configs/train/base.yaml --ckpt outputs/<arch>_fold0/best.ckpt --split val --fold 0`.
- Artifacts: `outputs/<arch>_fold0/{best.ckpt, val_metrics.json, eval_val/}` (Kaggle).

---
> **Disclaimer:** Research Use Only. Kết quả 1 fold trên val nhỏ (22 bn) mang tính sàng lọc; không dùng để tuyên bố hiệu năng lâm sàng. Kết luận cuối cần 5-fold CV + external validation + limitations.
