# Liver Cancer AI - Core Spec Sheet (Chốt kỹ thuật)

>**Research Use Only (RUO)**, chưa kiểm định lâm sàng.

## 1. Problem Scope
- **Phân loại nhị phân mức slice** (detection proxy): lát cắt CT bụng **có tổn thương gan bất thường** vs **gan bình thường** → **gộp lên mức bệnh nhân**.
- Positive = có tổn thương (Sensitivity); Negative = gan bình thường (Specificity). **Không** phân biệt lành/ác; **không** segmentation.

## 2. Dataset Strategy
- **Train set:** LiTS (huấn luyện mô hình).
- **Internal Validation:** trích từ LiTS (**patient-level split**),  theo dõi train, hyperparameter tuning, **chọn best model nội bộ**, khóa threshold.
- **External Test (Golden set):** 3D-IRCADb-01, chỉ đánh giá generalization **sau khi đã chốt best model**.
- **Bắt buộc:** split ở **mức bệnh nhân** + unit test leakage (giao tập bệnh nhân = ∅); CI tính ở mức bệnh nhân.

## 3. Model Selection

> **Định hướng SOTA có kiểm soát:** dataset nhỏ (~131 bn) khiến transformer lớn dễ **overfit**, *SOTA trên ImageNet ≠ SOTA trên dữ liệu y tế nhỏ*. Chọn kiến trúc hiện đại *vừa phải* + pretrain mạnh + aggregation tốt.

- **Baseline (sàn):** **ResNet-50** (ImageNet pretrained), mốc tham chiếu chuẩn để đo lợi ích của kiến trúc SOTA.
- **Main Model (Chính, SOTA-direction):** **ConvNeXt V2** (Nano/Tiny, pretrain ImageNet-22k qua `timm`), CNN hiện đại (2023), hợp dữ liệu nhỏ, **Grad-CAM chạy trực tiếp** (demo phụ thuộc Grad-CAM), nhẹ GPU.
- **Comparison Models (So sánh):**
> - **FastViT / EfficientViT** (2023): hybrid conv+attention, **inference nhanh** (tốt cho latency demo).
> - **Swin Transformer V2** (2022): transformer thuần, kiểm định giả thuyết "attention có giúp không?"; chỉ dùng kèm pretrain mạnh + aug (rủi ro overfit).
- **SOTA levers (giá trị nghiên cứu > đổi backbone):**
> - **Medical pretraining** (vd RadImageNet) thay vì chỉ ImageNet, thường vượt trên bài toán y tế.
> - **Attention-MIL / TransMIL** cho aggregation slice→patient (hiện đại hơn max/mean-pool).
<!-- - **2D-to-Patient Aggregation Strategy**: [Attention-MIL / TransMIL / Max-Pooling / 2.5D Slices] → Patient-level ROC-AUC. -->
- **Training Strategy**: Freeze backbone $\rightarrow$ Gradual Unfreezing với Discriminative Learning Rates.
- **Class Imbalance Handling**: WeightedSampler / Focal Loss / pos_weight tùy tỷ lệ mất cân bằng thực tế của tập Train.
- **XAI note**: CNN/ConvNeXt → Grad-CAM trực tiếp; ViT/Swin thuần → cần attention-rollout.

### Comparison Protocol (so sánh model - 2 phases)
- **Nguyên tắc controlled comparison:** mọi model chạy **cùng split / aug / epoch-budget / seed / cách gộp patient**, chỉ đổi **backbone**. Fine-tune từ pretrained (`timm`), **không** train from scratch.
- **Phase 1 - Sàng lọc:** train **mỗi model 1 lần** (1 seed, 1 fold, early stopping) → xếp hạng theo **patient-level AUC** trên validation.
- **Phase 2 - Nghiêm ngặt:** **chỉ 1–2 model top** mới chạy full **5-fold CV × nhiều seed + bootstrap CI + calibration**.
- **Chốt:** best model → **external test IRCADb 1 lần**; threshold khóa trên val (không đụng test).
- **Compute:** AMP + ảnh 256px + early stopping; ưu tiên biến thể nhẹ (ConvNeXt V2 **Nano/Tiny**, EfficientViT, FastViT); **Swin V2 để cuối** (nặng nhất).

## 4. Evaluation Metrics
**4.1. Metric chính (Quyết định Best Model)**
Patient-level ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
> - **ROC-AUC**: Đánh giá khả năng "xếp hạng" của mô hình. Giả sử lấy ngẫu nhiên 1 bệnh nhân có bệnh và 1 bệnh nhân không có bệnh, ROC-AUC là xác suất mô hình chấm điểm rủi ro cho người có bệnh cao hơn người không bệnh. AUC = 1.0 là hoàn hảo, 0.5 là đoán mò.
> - **Patient-level**: Đánh giá trên tổng thể bệnh nhân (ví dụ: bệnh nhân này có u không?), chứ không phải trên từng pixel hay từng lát cắt (slice).
<!-- > - **Threshold-independent**: Metric này đánh giá dựa trên xác suất thô (raw probability) từ 0 đến 1 do mô hình xuất ra, không phụ thuộc vào việc bạn chọn ngưỡng nào (ví dụ $>0.5$) để kết luận là có bệnh. Điều này giúp đánh giá năng lực thực sự của mạng nơ-ron trước khi con người can thiệp. -->

**4.2. Các Metric phụ (Secondary Metrics)**
> - **PR-AUC (Precision-Recall AUC)**: Rất quan trọng khi dữ liệu bị mất cân bằng (ví dụ: số ca có bệnh hiếm hơn số ca bình thường). Nó tập trung vào việc: trong những ca mô hình dự đoán là có bệnh, tỷ lệ thực sự có bệnh là bao nhiêu.
> - **Accuracy (Độ chính xác)**: Tỷ lệ đoán đúng trên tổng số ca. (Ít có ý nghĩa nhất trong y tế nếu dữ liệu mất cân bằng).
> - **Sensitivity (Độ nhạy / Recall):**: Khả năng bắt bệnh. Trong 100 người thực sự CÓ bệnh, mô hình tìm ra được bao nhiêu người? Trong y tế, bỏ sót bệnh (False Negative) rất nguy hiểm, nên Sensitivity thường được ưu tiên cao.
> - **Specificity (Độ đặc hiệu)**: Khả năng loại trừ bệnh. Trong 100 người KHÔNG có bệnh, mô hình chẩn đoán đúng bao nhiêu người là khỏe mạnh? Specificity cao giúp giảm báo động giả (False Positive), tiết kiệm thời gian cho bác sĩ.
> - **F1-Score**: Trung bình điều hòa giữa Precision và Recall (Sensitivity). Giúp cân bằng khi muốn cả hai chỉ số này đều tốt.
> - **Sens @ Spec = 0.90**: Đây là một metric mang tính ứng dụng lâm sàng cực cao. Nó có nghĩa là: "Nếu tôi ép hệ thống phải giữ tỷ lệ báo động giả ở mức thấp (cố định Specificity ở mức 90%), thì mô hình còn giữ lại được khả năng bắt bệnh (Sensitivity) là bao nhiêu?".

**4.3. Các Metric về Độ tin cậy (Calibration)**
Trong y tế, xác suất mô hình đưa ra phải có ý nghĩa thực tế. (Ví dụ: Nếu mô hình báo 80% nguy cơ ác tính, thì trong thực tế 100 ca y hệt vậy, phải có đúng 80 ca ác tính).
> - **Brier score**: Đo sai số bình phương giữa dự đoán (xác suất) và thực tế (0 hoặc 1). Điểm càng thấp càng tốt. Nó đánh giá xem mô hình có bị quá tự tin hoặc quá tự ti hay không.
> - **Calibration slope**: Độ dốc của đường chuẩn chuẩn. Đường hoàn hảo có slope = 1.0. Nếu slope $< 1$, mô hình đang bị quá tự tin (overconfident - dự đoán 90% nhưng thực tế chỉ 70%). Nếu slope $> 1$, mô hình đang bị kém tự tin (underconfident).

**4.4. Chiến lược Ngưỡng (Threshold)**
> - **Khóa trên validation / Không tối ưu trên test**: Để biến xác suất thành kết luận "Có/Không", cần một ngưỡng cắt. Cần "khóa" ngưỡng này trên tập Validation và mang áp dụng mù (blindly) lên tập Test, nếu bạn chỉnh lại ngưỡng trên tập Test để ra kết quả đẹp hơn, đó gọi là gian lận (Data Leakage).
> - **Youden J ($Sens + Spec - 1$)**: Cách chọn ngưỡng bằng toán học để cân bằng tối ưu giữa khả năng bắt bệnh và khả năng giảm báo động giả.
> - **Sens-priority**: Cách chọn ngưỡng thiên vị lâm sàng, chấp nhận báo động giả nhiều hơn một chút để tuyệt đối không bỏ sót bệnh nhân.


**4.5. Thống kê**
> - **95% CI bootstrap (resample bệnh nhân)**: Dùng kỹ thuật lấy mẫu lại (có hoàn lại) để tạo ra Khoảng tin cậy 95%. Nó trả lời câu hỏi: "Nếu áp dụng mô hình này cho quần thể bệnh nhân khác, kết quả tệ nhất/tốt nhất có thể dao động trong khoảng nào?".
> - **Mean±std qua 5 fold/nhiều seed**: Đánh giá tính ổn định của việc huấn luyện. Mô hình tốt không được phép thay đổi hiệu năng quá lớn chỉ vì chia dữ liệu kiểu khác (fold) hay khởi tạo ngẫu nhiên khác (seed).

**4.6. Chiến lược với Golden set (IRCADb)** 
> - Đo **$\Delta$ (Domain shift)**: $\Delta$ là độ chênh lệch hiệu năng giữa tập Internal và tập IRCADb. Thuật ngữ này đo lường "Domain shift" (Sự dịch chuyển miền dữ liệu).
> - Ví dụ: Nếu ROC-AUC nội bộ là 0.95, nhưng sang IRCADb tụt xuống 0.70 ($\Delta = -0.25$), điều này chứng tỏ mô hình đã bị Overfit vào đặc điểm máy chụp của bệnh viện LiTS và thất bại khi gặp máy chụp của bệnh viện khác.

**4.7. Kết quả external (đã chạy 22/07/2026 — chạm 1 lần)** — chi tiết `report/T3_W3_External_IRCADb.md`
> - ConvNeXt V2 nano trên **3D-IRCADb-01** (20 ca: 15 u / 5 âm, 2,068 slice): **slice-AUROC 0.807 [0.678, 0.902]** (internal 0.882 → **Δ −0.07**). Slice Sens/Spec @thr-khóa 0.20 = 0.74/0.71.
> - **Gate #2 PASS (mức slice):** generalization thật, không shortcut (khớp Grad-CAM), orientation OK. Điểm yếu = patient-level nhỏ + threshold-transfer patient kém (cần calibrate ngưỡng patient trên tập đủ ca âm).

---
**Tiêu chí thành công tổng:** vượt baseline + ổn định qua fold/seed + protocol đúng (no leakage, threshold từ val) + demo chạy + báo cáo trung thực có CI & limitations. *Không đặt mức AUC tuyệt đối tùy tiện.*