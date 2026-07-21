# Kế hoạch phối hợp W2 — Training (Liver Cancer AI)

## Context
Huấn luyện + đánh giá baseline → model chính trên dữ liệu W1 (`lits-processed`). Mục tiêu **Milestone M2**: pipeline train→eval chạy trọn, có **patient-level AUROC + 95% CI**. Bám `report/T3_W1_Spec_Sheet.md` (§3 Model Selection, Comparison Protocol 2 pha).

## Chú thích
👤 = **bạn tự làm** · 🤖 = **Claude (code/review)** · 👥 = cả hai · 💻 Local · ☁️ Kaggle

---

## BƯỚC 0 — Scaffold code · 🤖 · 💻 — ✅ ĐÃ XONG
Đã có + push: `configs/train/base.yaml`, `src/data/dataset.py`, `src/models/factory.py`, `src/training/train.py`, `src/evaluation/{metrics,evaluate}.py`, `notebooks/02_train_kaggle.ipynb`.
➡️ Bạn không cần làm gì.

## BƯỚC 1 — Đảm bảo có dataset `lits-processed` · 👤 · ☁️
- Nếu **chưa tạo**: mở lại notebook W1 → **Save & Run All** → tab Output → **New Dataset `lits-processed`** (Private) gồm `images_u8.npy` + `manifest.csv` + `splits/`.
- Nếu **đã có**: bỏ qua.
➡️ **Chỉ bạn làm được** (tài khoản Kaggle). Báo mình khi xong.

## BƯỚC 2 — Mở notebook train · 👤 · ☁️
1. Kaggle → New Notebook → **File → Import → GitHub**: `notebooks/02_train_kaggle.ipynb`.
2. Settings: **Internet ON**, **Accelerator: GPU T4 x1**.
3. **Add Data → `lits-processed`**.
➡️ Bạn: 3 thao tác. 🤖: notebook đã viết sẵn.

## BƯỚC 3 — Smoke train (chạy thử nhanh) · 👤 chạy / 🤖 code · ☁️
- Chạy cell 1–3 (cài lib, clone, **verify**: images khớp manifest).
- Cell 4 giảm tạm `--epochs 2` để test pipeline (~vài phút) → thấy `loss` giảm + in `val_auroc`.
➡️ Bạn: chạy. Lỗi gì (OOM, path…) dán cho 🤖 xử.

## BƯỚC 4 — Baseline ResNet-50 fold 0 (M2) · 👤 chạy / 🤖 review · ☁️
- Cell 4 full (`--arch resnet50 --fold 0`, epochs 20 + early stop). Cell 5 eval + curves.
- **Gửi 🤖**: `val_auroc` từng epoch (cell 4) + `metrics.json` (cell 5).
➡️ 🤖 review hội tụ → chỉnh LR/epochs/batch/pos_weight nếu cần. **Đạt M2 ở đây.**

## BƯỚC 5 — Model chính ConvNeXt V2 · 👤 chạy / 🤖 review · ☁️
- Đổi cell 4 → `--arch convnextv2_nano` (rồi thử `convnextv2_tiny`). Cùng fold 0.
- Gửi 🤖 kết quả để so với baseline.
➡️ Kỳ vọng ConvNeXt ≥ ResNet-50; nếu overfit → 🤖 chỉnh drop_rate/aug.

## BƯỚC 6 — Pha 1 sàng lọc backbone · 👥 · ☁️
- Chạy **mỗi arch 1 lần, fold 0** (cùng config): `resnet50`, `convnextv2_nano`, `efficientnet_b0`, `fastvit_sa12`, `swinv2_tiny_window8_256`.
- 🤖 tổng hợp **bảng xếp hạng val AUROC** → chọn **1–2 finalist** cho Pha 2.
➡️ Mỗi run tải `val_metrics.json` về (nhỏ) gửi 🤖.

## BƯỚC 7 — Ghi kết quả + commit · 👥 · 💻
- 🤖 gộp `reports/W2_model_comparison.md` (bảng AUROC/CI/Sens@Spec các backbone) + cập nhật WORKLOG → commit.
➡️ Bạn: gửi các `val_metrics.json`; 🤖: tổng hợp + commit.

---

## Bảng trách nhiệm
| Bước | Ai | Ở đâu | Ra cái gì |
|---|---|---|---|
| 0 Scaffold | 🤖 | 💻 | code train/eval (✅ xong) |
| 1 Dataset lits-processed | 👤 | ☁️ | dataset input cho train |
| 2 Mở notebook | 👤 | ☁️ | notebook + GPU + data |
| 3 Smoke train | 👤·🤖 | ☁️ | pipeline chạy thông |
| 4 Baseline ResNet-50 | 👤 chạy·🤖 review | ☁️ | **M2**: AUROC+CI val |
| 5 ConvNeXt V2 | 👤 chạy·🤖 review | ☁️ | so sánh main vs baseline |
| 6 Pha 1 sàng lọc | 👥 | ☁️ | bảng xếp hạng → finalist |
| 7 Commit kết quả | 👥 | 💻 | reports + WORKLOG |

## Verification (M2 pass khi)
- Cell 3 verify: `images_u8.npy` khớp manifest (19,094).
- Train: `val_auroc` tính được mỗi epoch, loss giảm, best ckpt lưu ở `outputs/<arch>_fold0/best.ckpt`.
- Eval: `metrics.json` có `auroc` + `ci_low/ci_high` (patient-level) + `sens_at_spec90`; có ROC/PR/calibration.

## Điểm cần quyết trong lúc làm
- **epochs / batch_size** (nếu OOM ở T4 → batch 16; ConvNeXt nặng hơn ResNet).
- **Finalist** sau Pha 1 (1 hay 2 model vào Pha 2).
- **Khi nào bật FPT H100**: để Pha 2 (5-fold × nhiều seed + CI) hoặc khi Swin V2 chậm/OOM ở T4 — KHÔNG dùng cho Pha 1/smoke.

## Ngoài phạm vi W2 (để W3–W4)
Pha 2 nghiêm ngặt (5-fold × seed + bootstrap CI gộp) · threshold khóa trên val · **internal test 1 lần** · external IRCADb · Grad-CAM · error analysis.
