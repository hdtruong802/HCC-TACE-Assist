# AGENTS.md — Liver Cancer AI

> **Nguồn context chung cho MỌI công cụ AI** (Claude Code, Codex, Antigravity, Cursor).
> File này cố tình **ngắn**: đọc file này trước, rồi mới nạp doc chi tiết khi cần. **Đừng đọc lại toàn bộ repo.**

## Dự án là gì
AI phát hiện tổn thương ung thư gan trên ảnh CT — **công cụ nghiên cứu/hỗ trợ (Research Use Only, chưa kiểm định lâm sàng)**. Nhân sự: 1 kỹ sư + AI agents · thời gian: 6 tuần.

## Scope đã chốt — NÂNG CẤP: hệ 2 tầng "phát hiện → phân loại" (22/07/2026)
- **Tầng 1 — PHÁT HIỆN (đã xong, đã kiểm chứng):** phân loại nhị phân mức slice CT (**có tổn thương gan** vs **gan bình thường**) → gộp patient. *Detection/triage proxy.* Đã qua 2 cửa kiểm chứng: Grad-CAM (không shortcut) + external IRCADb (slice-AUROC 0.807). Vẫn giữ nguyên mọi quyết định CT bên dưới.
- **Tầng 2 — PHÂN LOẠI (đang bắt đầu):** phân loại **loại tổn thương gan (7 lớp lành/ác)** ở mức tổn thương (ROI) trên **LLD-MMRI** (đa thì MRI). Two-stage khép kín trong LLD-MMRI (dataset có bbox) → làm được **oracle-ROI vs predicted-ROI** (đo lan truyền lỗi — đóng góp chính). Nhánh so sánh: joint/multi-task.
- **Lưu ý modality:** Tầng 1 = CT (LiTS/IRCADb), Tầng 2 = MRI (LLD-MMRI) → là **2 module** (không cascade trên cùng scan). Cascade + oracle-ROI chạy **nội bộ LLD-MMRI**.
- **RUO**, **KHÔNG** thay bác sĩ.

## Quyết định đã KHÓA (đừng suy diễn lại — sửa thì cập nhật ở đây + spec sheet)
**Tầng 1 — Phát hiện (CT):**
| Hạng mục | Chốt |
|---|---|
| Task | Binary classification, slice → patient (mean-of-top-k / Attention-MIL) |
| Positive/Negative | Positive = có tổn thương (Sensitivity) · Negative = gan bình thường (Specificity) |
| Train / Internal Val | **LiTS** (patient-level split; val để tuning + chọn best model + khóa threshold) |
| External Test (Golden) | **3D-IRCADb-01** — chỉ đánh giá generalization, chạm **1 lần** (đã chạy: slice-AUROC 0.807) |
| KHÔNG dùng | MSD Task03 = LiTS (trùng dữ liệu → không trộn, chỉ là mirror tải) |
| Baseline | ResNet-50 (ImageNet) — 1 mốc sàn |
| Main (SOTA-direction) | **ConvNeXt V2** (Nano/Tiny) |
| So sánh | FastViT/EfficientViT (2023) · Swin V2 (2022, cẩn thận overfit) |
| Metric quyết định | **slice-level ROC-AUC** (ổn định; patient-level nhiễu vì ít ca âm) + PR-AUC; phụ: Sens/Spec/Sens@Spec=0.90/calibration |
| Threshold | Khóa trên **validation** (Youden J / Sens-priority), không đụng test |

**Tầng 2 — Phân loại (MRI):**
| Hạng mục | Chốt |
|---|---|
| Task | Phân loại **7 lớp** loại tổn thương (lành/ác) mức ROI/tổn thương |
| Dataset | **LLD-MMRI** (498 bn, 8 thì MRI, có bbox; split challenge 316/78/104). Mask từ HF `wanglab/LLD-MMRI-MedSAM2` |
| 7 lớp | HCC · ICC (ung thư đường mật) · di căn · nang gan · u máu (hemangioma) · FNH · áp-xe |
| Kiến trúc | Two-stage (detect→classify) khép kín trong LLD-MMRI; **nhánh so sánh** joint/multi-task |
| Thí nghiệm lõi | **oracle-ROI vs predicted-ROI** (đo lan truyền lỗi tầng 1→2) |
| Metric | Macro-F1 / balanced-accuracy / per-class AUC (đa lớp, mất cân bằng) + CI |
| Split | **Patient-level** (như tầng 1); dùng split challenge để so benchmark (SDR-Former...) |

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
