Bạn là Principal AI Research Scientist chuyên về Computer Vision y tế, Medical Imaging, MLOps và triển khai sản phẩm AI trong bệnh viện.

Hãy xây dựng một kế hoạch nghiên cứu và phát triển hoàn chỉnh trong 6 tuần cho bài toán:

“Xây dựng mô hình AI phát hiện và phân loại ung thư gan từ ảnh y tế, đánh giá bằng Accuracy, Sensitivity, Specificity, AUC và triển khai một demo hoàn chỉnh.”

## 1. Bối cảnh dự án

* Thời gian thực hiện: 6 tuần.
* Nhân sự chính: 1 AI Engineer/Researcher.
* Có sự hỗ trợ của các coding agent và AI agent.
* Dữ liệu: tự tìm từ các nguồn dữ liệu public.
* Đầu ra cuối cùng:

  1. Dataset đã được thu thập, chuẩn hóa và phân tích.
  2. Pipeline tiền xử lý và augmentation.
  3. Baseline CNN.
  4. Mô hình được huấn luyện, tối ưu và đánh giá đầy đủ.
  5. Demo cho phép tải ảnh lên và nhận kết quả dự đoán.
  6. Báo cáo nghiên cứu và tài liệu kỹ thuật.
  7. Repository có cấu trúc rõ ràng, có thể tái lập kết quả.

Kế hoạch sơ bộ được giao:

* Giai đoạn 1, tuần 1–2: Thu thập dữ liệu ảnh, tiền xử lý, augmentation, xây dựng baseline CNN.
* Giai đoạn 2, tuần 3–4: Huấn luyện mô hình, tối ưu và đánh giá hiệu năng.
* Giai đoạn 3, tuần 5–6: Triển khai mô hình, xây dựng demo, kiểm thử và hoàn thiện tài liệu.

## 2. Yêu cầu quan trọng

Ung thư gan có thể được nghiên cứu trên nhiều loại ảnh như CT, MRI, siêu âm hoặc ảnh mô bệnh học. Với giới hạn 6 tuần, không được lựa chọn phạm vi quá rộng.

Trước khi lập kế hoạch chi tiết, hãy:

1. Phân tích ít nhất 3 hướng bài toán khả thi, ví dụ:

   * Phân loại ảnh hoặc lát cắt CT/MRI thành bình thường, tổn thương lành tính và ung thư.
   * Phân loại khối u gan từ ROI đã được xác định.
   * Phát hiện hoặc phân đoạn tổn thương gan, sau đó phân loại tổn thương.
   * Phân loại ảnh siêu âm gan.
   * Phân loại ảnh mô bệnh học ung thư gan.

2. So sánh các hướng theo:

   * Mức độ phù hợp với tiêu chí “phát hiện/phân loại ung thư gan”.
   * Khả năng tìm dữ liệu public.
   * Độ khó tiền xử lý.
   * Yêu cầu phần cứng.
   * Khả năng hoàn thành trong 6 tuần.
   * Khả năng đạt được Accuracy, Sensitivity, Specificity và AUC đáng tin cậy.
   * Khả năng xây dựng demo trực quan.
   * Rủi ro data leakage và sai lệch đánh giá.

3. Chọn một hướng chính phù hợp nhất để làm MVP trong 6 tuần.

4. Chỉ đề xuất thêm segmentation, object detection, multimodal learning hoặc mô hình 3D khi chúng thực sự khả thi. Không mở rộng phạm vi chỉ để dự án trông phức tạp hơn.

5. Nếu bài toán “phát hiện ung thư gan” chưa đủ rõ, hãy tự đưa ra giả định hợp lý, ghi rõ giả định và không hỏi lại người dùng.

## 3. Nghiên cứu dữ liệu public

Hãy tìm kiếm và đề xuất các bộ dữ liệu public phù hợp với hướng bài toán đã chọn.

Với mỗi dataset, cần cung cấp:

* Tên dataset.
* Đường dẫn nguồn chính thức.
* Loại ảnh: CT, MRI, siêu âm, mô bệnh học hoặc loại khác.
* Định dạng dữ liệu: DICOM, NIfTI, PNG, JPEG...
* Số bệnh nhân.
* Số study, volume hoặc hình ảnh.
* Nhãn có sẵn.
* Các loại tổn thương hoặc lớp bệnh.
* Có mask, bounding box, ROI hay chỉ có nhãn phân loại.
* Điều kiện truy cập.
* Giấy phép sử dụng.
* Có được phép dùng cho nghiên cứu và demo hay không.
* Ưu điểm.
* Hạn chế.
* Rủi ro mất cân bằng lớp.
* Rủi ro trùng bệnh nhân hoặc data leakage.
* Mức độ phù hợp với dự án 6 tuần.

Ưu tiên nguồn chính thức và đáng tin cậy như:

* The Cancer Imaging Archive.
* Grand Challenge.
* Kaggle, nhưng phải truy ngược về nguồn gốc dữ liệu.
* Zenodo.
* Figshare.
* PhysioNet.
* Các repository chính thức đi kèm bài báo khoa học.

Không được bịa tên dataset, số lượng dữ liệu hoặc đường dẫn. Nếu thông tin không chắc chắn, phải ghi rõ cần xác minh.

Sau khi so sánh, hãy chọn:

* Một dataset chính.
* Một dataset dự phòng.
* Nếu khả thi, một dataset external test để đánh giá khả năng tổng quát hóa.

Nếu dataset gốc không có nhãn trực tiếp “ung thư/không ung thư”, hãy giải thích cách chuyển nhãn hợp lý và các hạn chế khoa học của việc chuyển nhãn đó.

## 4. Định nghĩa bài toán học máy

Hãy xác định rõ:

* Input của mô hình.
* Output của mô hình.
* Đơn vị dự đoán:

  * Một ảnh.
  * Một lát cắt.
  * Một ROI.
  * Một volume.
  * Một bệnh nhân.
* Số lượng lớp.
* Ý nghĩa từng lớp.
* Lớp positive dùng để tính Sensitivity.
* Lớp negative dùng để tính Specificity.
* Có sử dụng binary classification hay multiclass classification.
* Cách chuyển kết quả multiclass thành các chỉ số Sensitivity và Specificity.
* Cách tổng hợp dự đoán từ slice-level lên patient-level nếu cần.
* Cách xử lý trường hợp một bệnh nhân có nhiều tổn thương.
* Các tiêu chí inclusion và exclusion cho dữ liệu.

Đưa ra một problem statement cuối cùng, cụ thể và có thể triển khai, ví dụ theo cấu trúc:

“Cho đầu vào là ..., mô hình dự đoán ..., tại mức ..., với mục tiêu hỗ trợ ..., không thay thế quyết định của bác sĩ.”

Phải định vị hệ thống là công cụ nghiên cứu hoặc hỗ trợ quyết định, không tuyên bố là thiết bị chẩn đoán lâm sàng đã được kiểm định.

## 5. Thiết kế protocol dữ liệu

Xây dựng một protocol chi tiết gồm:

### 5.1. Kiểm tra dữ liệu

* Phân bố lớp.
* Số lượng bệnh nhân.
* Số ảnh trên mỗi bệnh nhân.
* Kích thước ảnh.
* Pixel spacing.
* Slice thickness.
* Loại máy chụp hoặc scanner.
* Có sử dụng thuốc cản quang hay không.
* Các phase chụp nếu là CT hoặc MRI.
* Missing data.
* Corrupted files.
* Duplicate images.
* Near-duplicate images.
* Metadata có khả năng làm lộ nhãn.
* Chữ, marker, tên bệnh viện hoặc annotation xuất hiện trên ảnh.
* Các yếu tố gây shortcut learning.

### 5.2. Chia tập dữ liệu

Phải chia dữ liệu ở mức bệnh nhân, không chia ngẫu nhiên ở mức ảnh nếu nhiều ảnh thuộc cùng bệnh nhân.

Đề xuất cụ thể:

* Train.
* Validation.
* Internal test.
* External test nếu có.

Giải thích lựa chọn tỷ lệ chia.

Nếu dataset nhỏ, hãy đề xuất:

* Stratified group split.
* Group K-fold cross-validation.
* Repeated cross-validation.
* Bootstrap confidence interval.

Đưa ra pseudocode hoặc Python skeleton cho quy trình patient-level split.

### 5.3. Tiền xử lý

Tùy theo loại ảnh đã chọn, phân tích và lựa chọn:

* DICOM/NIfTI loading.
* Windowing cho CT.
* Intensity clipping.
* HU normalization.
* MRI intensity normalization.
* Bias field correction nếu cần.
* Resampling.
* Resize.
* Padding.
* Crop.
* Liver crop hoặc lesion ROI crop.
* Loại bỏ ảnh không chứa gan.
* Chuẩn hóa kênh ảnh.
* Chuyển grayscale thành ba kênh khi dùng pretrained model.
* Xử lý các phase ảnh.
* Xử lý nhiều slice hoặc nhiều series.
* Cache dữ liệu đã xử lý.

Nêu rõ bước nào bắt buộc, bước nào tùy chọn và bước nào không phù hợp trong phạm vi 6 tuần.

### 5.4. Data augmentation

Đề xuất augmentation phù hợp với ảnh y tế, ví dụ:

* Horizontal flip khi hợp lý về mặt giải phẫu.
* Small rotation.
* Translation.
* Scaling.
* Random crop.
* Contrast hoặc brightness adjustment.
* Gaussian noise.
* Blur nhẹ.
* Elastic transformation khi phù hợp.

Không sử dụng augmentation có thể làm sai ý nghĩa giải phẫu hoặc làm thay đổi nhãn.

Đề xuất thư viện triển khai như Albumentations, MONAI hoặc TorchIO và giải thích lựa chọn.

## 6. Chiến lược mô hình

Đề xuất một lộ trình mô hình theo mức độ tăng dần.

### 6.1. Baseline đơn giản

Phải có ít nhất:

* Một CNN tự xây dựng đơn giản.
* Một pretrained CNN phổ biến như ResNet, DenseNet hoặc EfficientNet.
* Loss function.
* Optimizer.
* Learning rate.
* Batch size.
* Số epoch.
* Early stopping.
* Learning-rate scheduler.
* Class weighting hoặc sampling strategy.

Baseline phải đủ đơn giản để hoàn thành trong tuần 2.

### 6.2. Mô hình chính

Đề xuất tối đa 2–3 mô hình chính để tránh thử nghiệm dàn trải.

So sánh các lựa chọn như:

* ResNet.
* DenseNet.
* EfficientNet.
* ConvNeXt.
* Vision Transformer.
* Swin Transformer.
* 2.5D CNN.
* 3D CNN.

Lựa chọn mô hình dựa trên:

* Kích thước dataset.
* Loại dữ liệu.
* GPU dự kiến.
* Thời gian huấn luyện.
* Khả năng giải thích.
* Khả năng triển khai demo.

Nếu lựa chọn mô hình 2D trên từng slice, hãy nêu cách tổng hợp kết quả lên patient-level.

Nếu lựa chọn 2.5D, hãy nêu cách lấy các slice lân cận.

Nếu lựa chọn 3D, phải giải thích rõ vì sao khả thi trong 6 tuần và đưa ra phương án fallback.

### 6.3. Transfer learning

Phân tích:

* ImageNet pretrained weights.
* Medical pretrained weights nếu có.
* Freeze backbone.
* Gradual unfreezing.
* Fine-tuning toàn bộ mô hình.
* Learning rate khác nhau cho backbone và classifier.

### 6.4. Loss và mất cân bằng dữ liệu

So sánh:

* Cross-entropy.
* Weighted cross-entropy.
* Focal loss.
* Label smoothing.
* Oversampling.
* WeightedRandomSampler.

Đưa ra lựa chọn chính và điều kiện chuyển sang phương án khác.

## 7. Thiết kế thí nghiệm

Tạo một bảng experiment matrix có kiểm soát, tránh thử nghiệm quá nhiều.

Mỗi thí nghiệm cần có:

* Experiment ID.
* Mục tiêu.
* Dataset version.
* Input representation.
* Model.
* Pretrained weights.
* Augmentation.
* Loss.
* Sampling.
* Learning rate.
* Batch size.
* Seed.
* Evaluation level.
* Metric chính.
* Điều kiện giữ hoặc loại bỏ thí nghiệm.

Đề xuất khoảng 6–10 thí nghiệm quan trọng nhất trong 6 tuần, ví dụ:

1. CNN baseline.
2. Pretrained CNN không augmentation.
3. Pretrained CNN có augmentation.
4. Weighted loss.
5. Focal loss hoặc sampling.
6. Fine-tuning toàn bộ backbone.
7. Mô hình thứ hai để so sánh.
8. Threshold optimization.
9. Ensemble nhỏ nếu còn thời gian.
10. External test nếu có dữ liệu.

Mỗi thay đổi chỉ nên thay một hoặc một nhóm yếu tố có lý do rõ ràng.

## 8. Evaluation protocol

Phải giải thích và cung cấp công thức cho:

* Accuracy.
* Sensitivity/Recall/True Positive Rate.
* Specificity/True Negative Rate.
* Precision.
* F1-score.
* ROC-AUC.
* PR-AUC.
* Confusion matrix.

Không chỉ báo cáo Accuracy vì dữ liệu y tế thường mất cân bằng.

Xác định:

* Metric chính để lựa chọn mô hình.
* Metric phụ.
* Ngưỡng phân loại mặc định.
* Cách tối ưu threshold trên validation set.
* Không được tối ưu threshold trên test set.
* Cách lựa chọn threshold ưu tiên Sensitivity.
* Cách lựa chọn threshold theo Youden’s J statistic.
* Cách báo cáo Specificity tại một mức Sensitivity mục tiêu.
* Macro, micro và weighted average cho multiclass.
* One-vs-rest AUC nếu là multiclass.

Đề xuất báo cáo:

* Mean và standard deviation qua nhiều seed hoặc fold.
* 95% confidence interval bằng bootstrap.
* ROC curve.
* Precision-recall curve.
* Confusion matrix.
* Calibration curve.
* Brier score nếu phù hợp.

Giải thích cách tránh:

* Test-set overfitting.
* Data leakage.
* Model selection dựa trên test set.
* Báo cáo kết quả quá lạc quan do slice-level split.
* Báo cáo hàng nghìn slice như hàng nghìn mẫu độc lập trong khi chỉ có ít bệnh nhân.

## 9. Explainability và phân tích lỗi

Đề xuất các phương pháp:

* Grad-CAM.
* Grad-CAM++.
* Saliency map.
* Integrated Gradients.

Giải thích:

* Explainability chỉ hỗ trợ kiểm tra hành vi mô hình, không chứng minh mô hình có lý luận giống bác sĩ.
* Cách kiểm tra mô hình có tập trung vào gan hoặc tổn thương hay không.
* Cách phát hiện shortcut learning.
* Cách hiển thị heatmap trong demo.

Xây dựng error analysis framework gồm:

* False positive.
* False negative.
* Sai theo kích thước tổn thương.
* Sai theo loại tổn thương.
* Sai theo phase chụp.
* Sai theo scanner.
* Sai theo chất lượng ảnh.
* Sai theo bệnh nhân có nhiều bệnh đồng mắc.
* Sai do ROI không phù hợp.
* Sai do artefact.

Đề xuất một mẫu bảng để bác sĩ hoặc người đánh giá có thể review các ca lỗi.

## 10. Demo sản phẩm

Thiết kế một demo khả thi trong tuần 5–6.

Đề xuất stack phù hợp, ưu tiên đơn giản:

* Backend: FastAPI hoặc Flask.
* Frontend: Streamlit, Gradio hoặc web frontend riêng.
* Model serving: PyTorch hoặc ONNX Runtime.
* Containerization: Docker.
* Tracking: MLflow, Weights & Biases hoặc TensorBoard.
* Deployment: local, Hugging Face Spaces, cloud VM hoặc nền tảng phù hợp.

Demo tối thiểu cần có:

1. Upload ảnh hoặc volume phù hợp với dataset.
2. Kiểm tra định dạng file.
3. Hiển thị ảnh đầu vào.
4. Hiển thị kết quả:

   * Nhãn dự đoán.
   * Xác suất.
   * Ngưỡng đang sử dụng.
   * Heatmap hoặc Grad-CAM nếu khả thi.
5. Cảnh báo rằng đây là hệ thống nghiên cứu, không dùng để thay thế bác sĩ.
6. Hiển thị phiên bản mô hình.
7. Hiển thị thông tin preprocessing chính.
8. Xử lý lỗi khi file không hợp lệ.
9. Không lưu trữ dữ liệu người dùng ngoài ý muốn.
10. Có một số sample case để thử nhanh.

Nếu input là DICOM hoặc NIfTI, hãy mô tả flow upload và cách chọn series, slice hoặc volume.

Phân biệt rõ:

* Demo kỹ thuật dùng để chứng minh pipeline.
* Sản phẩm y tế thực tế cần kiểm định, bảo mật, quản lý dữ liệu, giám sát và phê duyệt pháp lý.

## 11. Kiến trúc hệ thống

Đưa ra sơ đồ kiến trúc dạng Mermaid thể hiện:

Dataset → Validation → Preprocessing → Training → Evaluation → Model Registry → Inference API → Demo UI.

Mô tả rõ:

* Offline training pipeline.
* Online inference pipeline.
* Lưu model checkpoint.
* Lưu metadata.
* Cấu hình preprocessing.
* Logging.
* Experiment tracking.
* Model versioning.
* Reproducibility.

## 12. Cấu trúc repository

Đề xuất repository structure chi tiết, ví dụ:

* configs/
* data/
* notebooks/
* src/

  * data/
  * preprocessing/
  * augmentation/
  * models/
  * training/
  * evaluation/
  * explainability/
  * inference/
* app/
* tests/
* scripts/
* reports/
* docs/
* checkpoints/
* outputs/
* Dockerfile
* docker-compose.yml
* requirements.txt hoặc pyproject.toml
* README.md
* MODEL_CARD.md
* DATA_CARD.md

Giải thích trách nhiệm của từng thư mục.

Đưa ra naming convention cho:

* Dataset version.
* Experiment.
* Checkpoint.
* Report.
* Model version.

## 13. Kế hoạch chi tiết trong 6 tuần

Lập kế hoạch cho từng tuần, từng ngày làm việc hoặc từng nhóm ngày.

Mỗi tuần phải có:

* Mục tiêu.
* Công việc chính.
* Công việc kỹ thuật.
* Công việc nghiên cứu.
* Deliverable.
* Tiêu chí hoàn thành.
* Rủi ro.
* Phương án fallback.

### Tuần 1

Tập trung vào:

* Xác định phạm vi bài toán.
* Tìm dataset.
* Kiểm tra license.
* Download dữ liệu mẫu.
* Phân tích cấu trúc dữ liệu.
* Data audit.
* Xây dựng patient-level split.
* Xây dựng preprocessing prototype.
* Tạo data card ban đầu.

### Tuần 2

Tập trung vào:

* Hoàn thiện preprocessing.
* Augmentation.
* Dataset loader.
* CNN baseline.
* Pretrained baseline.
* Training pipeline.
* Evaluation pipeline.
* Experiment tracking.
* Kết quả baseline đầu tiên.

### Tuần 3

Tập trung vào:

* Huấn luyện mô hình chính.
* Fine-tuning.
* Xử lý mất cân bằng lớp.
* So sánh augmentation.
* Tối ưu learning rate.
* Kiểm tra lỗi pipeline.
* Phân tích kết quả validation.

### Tuần 4

Tập trung vào:

* Hoàn thiện experiment matrix.
* Chọn mô hình cuối.
* Threshold optimization.
* Internal test.
* Confidence interval.
* Calibration.
* Grad-CAM.
* Error analysis.
* External validation nếu khả thi.

### Tuần 5

Tập trung vào:

* Đóng gói model.
* Inference pipeline.
* FastAPI/Gradio/Streamlit.
* Upload và validate dữ liệu.
* Hiển thị prediction và heatmap.
* Docker.
* Kiểm thử chức năng.
* Kiểm thử thời gian inference.

### Tuần 6

Tập trung vào:

* Hoàn thiện demo.
* Regression testing.
* Viết báo cáo.
* Hoàn thiện README.
* Model card.
* Data card.
* API documentation.
* User guide.
* Video demo hoặc slide demo.
* Chuẩn bị kết quả và limitations.
* Đóng version release.

Hãy lập thêm bảng timeline theo ngày hoặc theo work package, với:

* Task.
* Owner.
* Thời lượng.
* Dependency.
* Output.
* Priority.
* Trạng thái dự kiến.
* Definition of Done.

## 14. Milestone và tiêu chí hoàn thành

Đề xuất milestone cuối mỗi tuần.

Ví dụ:

* M1: Dataset và problem definition được khóa.
* M2: Baseline chạy end-to-end.
* M3: Mô hình chính vượt baseline.
* M4: Đánh giá test hoàn chỉnh.
* M5: Demo chạy local bằng Docker.
* M6: Release hoàn chỉnh.

Xây dựng Definition of Done cho toàn dự án, bao gồm:

* Pipeline chạy từ raw data đến prediction.
* Split ở mức bệnh nhân.
* Không có leakage đã biết.
* Có baseline và mô hình chính.
* Có báo cáo Accuracy, Sensitivity, Specificity, ROC-AUC.
* Có confusion matrix và ROC curve.
* Có threshold rõ ràng.
* Có ít nhất một phương pháp explainability.
* Demo chạy được.
* Có unit test cơ bản.
* Có README.
* Có hướng dẫn tái lập kết quả.
* Có model card.
* Có data card.
* Có limitations.
* Có disclaimer y tế.

Đưa ra hai cấp độ:

1. Minimum Success Criteria: bắt buộc phải đạt.
2. Stretch Goals: chỉ làm khi tiến độ cho phép.

Không đặt một mức Accuracy hoặc AUC tùy ý nếu chưa biết độ khó dataset. Thay vào đó, tiêu chí thành công nên dựa trên:

* Vượt baseline.
* Kết quả ổn định qua nhiều seed hoặc fold.
* Không có lỗi nghiêm trọng trong evaluation protocol.
* Demo hoạt động đầy đủ.
* Kết quả được báo cáo trung thực với confidence interval và limitations.

## 15. Rủi ro và phương án dự phòng

Tạo risk register gồm:

* Dataset không truy cập được.
* License không phù hợp.
* Dataset quá nhỏ.
* Nhãn không đúng với bài toán.
* Mất cân bằng lớp.
* Dữ liệu có nhiều series hoặc phase phức tạp.
* DICOM preprocessing lỗi.
* Data leakage.
* Training không hội tụ.
* GPU không đủ.
* 3D model quá nặng.
* Mô hình học shortcut.
* Validation tốt nhưng test kém.
* Không có external dataset.
* Demo xử lý file quá chậm.
* Grad-CAM không hợp lý.
* Thiếu thời gian viết tài liệu.

Mỗi rủi ro cần có:

* Xác suất.
* Mức ảnh hưởng.
* Cách phát hiện sớm.
* Biện pháp giảm thiểu.
* Phương án fallback.
* Thời điểm quyết định chuyển sang fallback.

Ví dụ:

* Nếu xử lý volume 3D mất quá nhiều thời gian sau tuần 1, chuyển sang 2D hoặc 2.5D.
* Nếu không tìm được dataset có nhãn phân loại đáng tin cậy, chuyển sang bài toán phân loại ROI hoặc phân loại có/không có tổn thương.
* Nếu external validation không khả thi, thực hiện group cross-validation và bootstrap confidence interval.
* Nếu segmentation làm trễ tiến độ, dùng ROI hoặc crop có sẵn và ghi rõ giới hạn.

## 16. Phần cứng và công nghệ

Đưa ra cấu hình tối thiểu và khuyến nghị:

* GPU.
* VRAM.
* RAM.
* Storage.
* CPU.

Đề xuất technology stack cụ thể:

* Python version.
* PyTorch.
* MONAI.
* pydicom.
* nibabel.
* SimpleITK.
* Albumentations.
* scikit-learn.
* pandas.
* NumPy.
* MLflow hoặc Weights & Biases.
* FastAPI.
* Gradio hoặc Streamlit.
* Docker.
* pytest.
* pre-commit.

Chỉ chọn những thư viện thực sự cần thiết.

## 17. Tài liệu cuối dự án

Đề xuất cấu trúc báo cáo cuối cùng gồm:

1. Executive summary.
2. Clinical/research background.
3. Problem definition.
4. Dataset.
5. Data inclusion/exclusion.
6. Preprocessing.
7. Model architecture.
8. Training protocol.
9. Evaluation protocol.
10. Results.
11. Error analysis.
12. Explainability.
13. Demo architecture.
14. Limitations.
15. Ethical considerations.
16. Future work.
17. Reproducibility instructions.

Đề xuất nội dung cho:

* README.
* Model card.
* Data card.
* API documentation.
* User guide.
* Demo script.
* Slide thuyết trình 10–12 trang.

## 18. Câu hỏi nghiên cứu

Đề xuất 3–5 câu hỏi nghiên cứu phù hợp, ví dụ:

* Transfer learning cải thiện kết quả bao nhiêu so với CNN baseline?
* Data augmentation có cải thiện khả năng tổng quát hóa hay không?
* Weighted loss và focal loss ảnh hưởng thế nào đến Sensitivity của lớp ung thư?
* Mô hình có ổn định ở mức patient-level hay không?
* Threshold ưu tiên Sensitivity ảnh hưởng thế nào đến Specificity?

Mỗi câu hỏi cần gắn với một thí nghiệm và metric cụ thể.

## 19. Yêu cầu về cách trình bày câu trả lời

Câu trả lời phải có cấu trúc rõ ràng và đủ chi tiết để có thể bắt đầu triển khai ngay.

Hãy trả lời theo thứ tự:

1. Executive summary.
2. Phân tích và lựa chọn phạm vi bài toán.
3. Problem statement cuối cùng.
4. So sánh dataset public.
5. Dataset được lựa chọn.
6. Input, output và labels.
7. Data protocol.
8. Preprocessing và augmentation.
9. Mô hình.
10. Experiment matrix.
11. Evaluation protocol.
12. Explainability và error analysis.
13. Kiến trúc hệ thống.
14. Thiết kế demo.
15. Repository structure.
16. Kế hoạch 6 tuần.
17. Timeline chi tiết.
18. Milestones.
19. Definition of Done.
20. Risk register.
21. Phần cứng và technology stack.
22. Cấu trúc tài liệu cuối.
23. Các câu hỏi nghiên cứu.
24. Checklist công việc có thể đưa trực tiếp vào GitHub Issues hoặc project board.

Sử dụng:

* Bảng so sánh khi phù hợp.
* Mermaid diagram cho kiến trúc.
* Checklist cho task.
* Pseudocode cho các bước quan trọng.
* Công thức cho metric.
* Ví dụ configuration YAML cho một experiment.
* Ví dụ command chạy training, evaluation và demo.
* Ước lượng thời gian cho từng task.
* Phân biệt rõ Must-have và Nice-to-have.

## 20. Nguyên tắc chất lượng

* Không bịa dataset, paper, metric hoặc số liệu.
* Không tuyên bố hiệu năng dự kiến nếu chưa có bằng chứng.
* Không dùng Accuracy làm metric duy nhất.
* Không chia dữ liệu ở mức slice nếu các slice cùng bệnh nhân có thể rơi vào nhiều tập.
* Không dùng test set để lựa chọn mô hình hoặc threshold.
* Không làm quá nhiều mô hình trong 6 tuần.
* Không mở rộng sang hệ thống chẩn đoán lâm sàng hoàn chỉnh.
* Không giả định heatmap là bằng chứng lâm sàng.
* Nêu rõ các hạn chế về dữ liệu, domain shift và generalization.
* Mọi lựa chọn phải đi kèm lý do.
* Khi có nhiều lựa chọn, phải đưa ra một lựa chọn khuyến nghị chính thay vì chỉ liệt kê.
* Ưu tiên một pipeline hoàn chỉnh, đúng protocol và tái lập được hơn một mô hình quá phức tạp.
* Kết quả cuối phải đủ cụ thể để AI coding agent có thể chuyển từng đầu việc thành code và GitHub Issue.

Cuối câu trả lời, hãy cung cấp:

1. Một bảng “Recommended Final Scope” tóm tắt phạm vi được chọn.
2. Một bảng “6-Week Critical Path”.
3. Danh sách 10 việc cần thực hiện ngay trong 48 giờ đầu tiên.
4. Danh sách các quyết định phải được khóa trước khi kết thúc tuần 1.
5. Một checklist nghiệm thu cuối dự án.
6. Một danh sách rõ ràng các hạng mục nên loại bỏ nếu tiến độ bị chậm.
