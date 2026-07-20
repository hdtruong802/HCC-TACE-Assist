# DATA_CARD — LiTS (Liver Cancer AI)

## Nguồn & license
- **Dataset:** LiTS (Liver Tumor Segmentation Benchmark, ISBI/MICCAI 2017).
- **Truy cập (khuyến nghị):** Kaggle `andrewmvd/liver-tumor-segmentation` — 131 ca CT có nhãn, NIfTI.
- **Nguồn gốc:** CodaLab competition 17094 (Bilic et al., *Medical Image Analysis* 2023; arXiv:1901.04056).
- **License:** CC BY-NC-SA (phi thương mại — hợp mục đích **nghiên cứu**). Ghi công nguồn; **không** dùng thương mại.
- ⚠ **Không commit ảnh y tế thô/cache vào git** — chỉ manifest/split/config/report/DATA_CARD. Người khác tự tải qua Kaggle.

## Nội dung dùng trong dự án
- **Dùng:** 131 volume **có nhãn** (Batch 1: vol 0–27, Batch 2: vol 28–130).
- **KHÔNG dùng:** 70 volume test (không nhãn → không tính được AUC/Sens/Spec; leaderboard chỉ chấm segmentation).
- Nhãn gốc: voxel mask **0=nền, 1=gan, 2=u**. Định dạng NIfTI, in-plane 512×512, 74–987 slice/volume; spacing/slice-thickness không đồng nhất (in-plane 0.55–1.0mm, thickness 0.45–6.0mm).

## Chuyển nhãn (label transfer) — segmentation → phân loại slice
- Mỗi slice: `liver_area = |seg∈{1,2}|`, `tumor_area = |seg==2|`.
- `label = 1` nếu `tumor_area ≥ τ_area`; `= 0` nếu có gan (`liver_area ≥ τ_liver`) mà không đủ u; **loại** slice không đủ gan.
- **τ tạm:** `τ_area=10px, τ_liver=50px` → **CHỐT lại sau audit** (xem `reports/data_audit_report.md`).

## Tiền xử lý (W1)
Windowing gan WL=50/WW=350 → clip HU [−125,225] → scale [0,1] → **liver-ROI crop** (bbox mask gan + margin 16px) → resize **256×256** → cache **uint8 1 kênh** (nở 3ch + normalize lúc train). Orientation chuẩn RAS. Resample spacing: **tắt** (mặc định).

## Chia dữ liệu
- **Patient-level** StratifiedGroupKFold (k=5, group=`patient_id`) + hold-out internal test 15%, seed=42.
- Unit test `tests/test_leakage.py` đảm bảo giao tập bệnh nhân = ∅.
- **External test:** 3D-IRCADb-01 (xử lý ở W4, không thuộc W1).

## Hạn chế đã biết
- **Nhãn suy từ mask** → nhiễu nhẹ với tổn thương nhỏ/isoattenuating.
- LiTS gồm HCC **và di căn** → "positive" là *ác tính nói chung*, không thuần HCC; **không** phân biệt lành/ác.
- **Ít bệnh nhân "gan bình thường"** (đa số ca LiTS có u) → negative ở *patient-level* khan hiếm; Specificity mức bệnh nhân cần diễn giải thận trọng (dựa thêm slice-level + IRCADb có ~25% ca không u).
- **Infer-time gap:** liver-crop lúc train dùng mask gan; lúc suy luận không có mask → cần heuristic/liver-localizer (backlog).
- Mất cân bằng lớp ở slice-level (positive ≪ negative) → xử lý ở train (pos_weight/sampler/focal).

## Tái lập
Seed cố định; checksum file raw (khi tải từ CodaLab); config-driven (`configs/data/lits.yaml`); manifest lưu `cache_path` **tương đối** + `spacing/thickness` để truy vết.