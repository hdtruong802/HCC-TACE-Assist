# AGENTS.md — Liver Cancer AI

> **Nguồn context chung cho MỌI công cụ AI** (Claude Code, Codex, Antigravity, Cursor).
> File này cố tình **ngắn**: đọc file này trước, rồi mới nạp doc chi tiết khi cần. **Đừng đọc lại toàn bộ repo.**

## Dự án là gì
AI phát hiện tổn thương ung thư gan trên ảnh CT — **công cụ nghiên cứu/hỗ trợ (Research Use Only, chưa kiểm định lâm sàng)**. Nhân sự: 1 kỹ sư + AI agents · thời gian: 6 tuần.

## Scope đã chốt
**Phân loại nhị phân mức lát cắt (slice-level)**: lát cắt CT bụng **có tổn thương gan bất thường** vs **gan bình thường** → **gộp lên mức bệnh nhân**. Là *detection/triage proxy*, **KHÔNG** segmentation, **KHÔNG** phân biệt lành/ác, **KHÔNG** thay bác sĩ.

## Quyết định đã KHÓA (đừng suy diễn lại — sửa thì cập nhật ở đây + spec sheet)
| Hạng mục | Chốt |
|---|---|
| Task | Binary classification, slice → patient (mean-of-top-k / Attention-MIL) |
| Positive/Negative | Positive = có tổn thương (Sensitivity) · Negative = gan bình thường (Specificity) |
| Train / Internal Val | **LiTS** (patient-level split; val để tuning + chọn best model + khóa threshold) |
| External Test (Golden) | **3D-IRCADb-01** — chỉ đánh giá generalization, chạm **1 lần** |
| KHÔNG dùng | MSD Task03 = LiTS (trùng dữ liệu → không trộn, chỉ là mirror tải) |
| Baseline | ResNet-50 (ImageNet) — 1 mốc sàn |
| Main (SOTA-direction) | **ConvNeXt V2** (Nano/Tiny) |
| So sánh | FastViT/EfficientViT (2023) · Swin V2 (2022, cẩn thận overfit) |
| Metric quyết định | **patient-level ROC-AUC** (+ PR-AUC); phụ: Sens/Spec/F1/Sens@Spec=0.90/Brier/calibration |
| Threshold | Khóa trên **validation** (Youden J / Sens-priority), không đụng test |

## Conventions BẮT BUỘC (không được vi phạm)
- **Patient-level split** — không để slice cùng bệnh nhân rơi 2 tập; có unit test leakage (giao tập bệnh nhân = ∅).
- **CI tính ở mức bệnh nhân** (bootstrap resample bệnh nhân), không coi mỗi slice là mẫu độc lập.
- **Không** tối ưu threshold/chọn model trên test; external chỉ chạy 1 lần.
- **Không** commit ảnh y tế thô vào git (chỉ manifest/split/config/checksum).
- Mọi kết quả kèm 95% CI + limitations; **không** đặt mức AUC tuyệt đối tùy tiện.

## Index tài liệu (nạp khi cần, đừng đọc hết cùng lúc)
- **Chốt kỹ thuật (đọc trước tiên):** `report/T3_W1_Spec_Sheet.md`
- Kế hoạch 6 tuần chi tiết: `docs/Liver_Cancer_AI_6Week_Plan.md`
- Tổng quan nghiên cứu + benchmark (có trích dẫn): `report/T3_W1.md`
- **Nhật ký bàn giao giữa các tool:** `docs/WORKLOG.md` ← đọc phần cuối trước khi bắt đầu

## Quy ước làm việc đa công cụ (mọi tool tuân theo)
1. **Bắt đầu phiên:** đọc `AGENTS.md` + **tail `docs/WORKLOG.md`** → nắm trạng thái. Không đọc lại toàn repo.
2. **Có quyết định mới bị khóa/đổi:** cập nhật bảng trên + `report/T3_W1_Spec_Sheet.md`.
3. **Kết thúc phiên:** thêm entry mới lên **đầu** `docs/WORKLOG.md` (format: `[YYYY-MM-DD · tool] Done / Decisions / Next / Files`) rồi **git commit** với message rõ ràng.
4. Nguồn sự thật = **file đã commit**, không phải bộ nhớ riêng của từng tool.

## Trạng thái repo
Giai đoạn tài liệu/định nghĩa (chưa có code pipeline). Cấu trúc code dự kiến & lệnh chạy: xem `docs/Liver_Cancer_AI_6Week_Plan.md` (§15). Cập nhật mục "lệnh hay dùng" tại đây khi có code.
