# WORKLOG — Nhật ký bàn giao giữa các công cụ

> Cơ chế bàn giao context khi switch giữa Claude Code / Codex / Antigravity / Cursor.
> **Bắt đầu phiên:** đọc vài entry mới nhất (trên cùng). **Kết thúc phiên:** thêm entry mới **lên trên cùng** rồi git commit.
> Format mỗi entry:
> ```
> ## [YYYY-MM-DD · <tool>]
> - **Done:** …
> - **Decisions:** … (nếu có quyết định bị khóa/đổi → cập nhật cả AGENTS.md + report/T3_W1_Spec_Sheet.md)
> - **Next:** …
> - **Files:** …
> ```

---

## [2026-07-21 · Claude Code] — W2 Pha 1: sàng lọc 4 backbone (fold 0)
- **Done:** Train+eval 4 backbone (fold 0, slice-selected). Xếp hạng **slice-AUROC**: ConvNeXt V2 nano **0.882 [0.815,0.921]** > FastViT sa12 **0.874 [0.815,0.917]** > ResNet-50 (baseline) 0.837 [0.757,0.892] > EfficientNet-B0 0.826 [0.745,0.881]. Báo cáo: `report/T3_W2_Phase1.md`.
- **Decisions:** Finalist Pha 2 = **ConvNeXt V2 + FastViT** (+ ResNet-50 anchor); loại EfficientNet-B0. Nghịch lý minh hoạ: ConvNeXt slice cao nhất nhưng patient thấp nhất → khẳng định dùng slice-level. Mọi model hiện đại overfit nhanh (best ep 2–4) → Pha 2 tăng regularization.
- **Next:** Pha 2 = 5-fold × seed OOF + paired bootstrap ConvNeXt vs FastViT → chốt main + khóa threshold.

## [2026-07-21 · Claude Code] — W2: thêm slice-level metrics (ổn định hơn patient)
- **Done:** `metrics.full_report` nay trả về CẢ patient-level và **slice-level** (auroc/pr/sens/spec/f1 + threshold Youden slice) với **CI cluster-bootstrap theo bệnh nhân** (không phóng đại). Train chọn best-ckpt/early-stop theo `slice_auroc` (config `train.select_metric`), in cả slice + patient AUROC/epoch. Eval bật slice-bootstrap CI + vẽ roc_slice/pr_slice. Test synthetic: patient CI thoái hoá [1,1] vs slice CI hẹp có nghĩa.
- **Why:** val 1 fold chỉ 3 ca âm → patient metrics vô nghĩa; slice (2,555) ổn định để so sánh backbone công bằng.
- **Next:** chạy lại eval ResNet-50 (có slice metrics) → Pha 1: ConvNeXt V2 + backbone khác, xếp hạng theo slice_auroc.

## [2026-07-21 · Claude Code] — W2 M2: baseline ResNet-50 chạy trọn ✅
- **Done:** Train baseline **ResNet-50 fold0** trên Kaggle T4: hội tụ (loss 0.88→0.15), **best val patient-AUROC=0.895 @ep8** (early-stop ep13), PR-AUC~0.98. Pipeline train→eval→ckpt OK. Fix dọc đường: DATA_ROOT auto-detect, split resolve, AMP GradScaler mới + grad-clip (hết crash scaler.update), tqdm progress + full metric/epoch.
- **Caveat:** val fold 22 bn (~3 âm) → **CI rộng [0.67,1.0]**, spec bước 1/3, PR-AUC thổi phồng. Số baseline "thật" cần 5-fold CV gộp + slice-level + IRCADb.
- **Next:** ConvNeXt V2 (main) cùng fold0 → Pha 1 sàng lọc các backbone → chọn finalist.
- **Files:** outputs/resnet50_fold0/ (ckpt+metrics) trên Kaggle.

## [2026-07-21 · Claude Code] — W2 training scaffold
- **Done:** Pipeline train/eval config-driven, chạy Kaggle GPU. `src/data/dataset.py` (memmap dataset + Albumentations + patient split), `src/models/factory.py` (timm backbone + discriminative LR + freeze/unfreeze), `src/training/train.py` (AMP, cosine+warmup, BCE/Focal+pos_weight, early-stop theo **patient AUROC**, MLflow, best ckpt), `src/evaluation/metrics.py` (gộp patient mean-topk, AUROC/PR, Sens/Spec, threshold Youden/Sens-priority, **bootstrap CI mức bệnh nhân**), `evaluate.py` (ROC/PR/CM/calibration), `configs/train/base.yaml`, `notebooks/02_train_kaggle.ipynb`. Syntax OK.
- **Next:** (nếu chưa) tạo dataset `lits-processed` → chạy `02_train_kaggle`: baseline **ResNet-50** fold0 (M2) → **ConvNeXt V2** → Pha 1 sàng lọc các backbone.
- **Files:** configs/train/, src/{data,models,training,evaluation}/, notebooks/02_train_kaggle.ipynb, requirements.txt

## [2026-07-21 · Claude Code] — W1 data build DONE (Kaggle)
- **Done:** Chạy full pipeline trên Kaggle. Attach 2 dataset (part1+part2) = 131 vol. Fix dọc đường: repo→public (git clone), cell4 dùng f-string. Build cache+manifest (**19,094 slice có gan**), audit → chốt **τ_area=20** (pos **37%**; patient **118 u / 13 không u**), split patient-level (**test=20, 5-fold, seed=42, hash 8647d40**), **leakage test 4/4 PASS**.
- **Next:** Tạo Kaggle dataset `lits-processed`; commit `data/manifest.csv` + `data/splits/lits_v1.json`; bắt đầu **W2 (training)**.
- **Caveat:** chỉ 13/131 ca không u → Specificity patient-level CI rộng (dựa thêm slice-level + IRCADb).
- **Files:** manifest/split (Kaggle output), DATA_CARD cập nhật số liệu.

## [2026-07-21 · Claude Code]
- **Done:** Scaffold pipeline data W1 (Bước 0). Tạo `configs/data/lits.yaml`, `src/data/{io,preprocess,label_transfer,audit}.py`, `scripts/{build_manifest,make_split}.py`, `tests/test_leakage.py`, `notebooks/01_data_kaggle.ipynb`, `DATA_CARD.md`, `requirements.txt`. Đã test logic split+leakage (synthetic → PASS); syntax/JSON/YAML hợp lệ.
- **Decisions:** Dùng dataset Kaggle có sẵn `andrewmvd/liver-tumor-segmentation` (bỏ tải CodaLab). Chốt: 256px · liver-crop ON · cache uint8 1-kênh · τ tạm 10/50 (chốt sau audit) · bỏ 70 vol test.
- **Next (Bước 1→7):** Bạn tạo Kaggle token + notebook → Add Data → chạy `notebooks/01_data_kaggle.ipynb` (verify → manifest → audit chốt τ → split → leakage) → tạo dataset `lits-processed` → tải manifest/split/DATA_CARD về commit.
- **Files:** configs/, src/data/, scripts/, tests/, notebooks/01_data_kaggle.ipynb, DATA_CARD.md, requirements.txt

## [2026-07-20 · Claude Code]
- **Done:** Thiết lập hệ shared-context đa công cụ: tạo `AGENTS.md` (canonical), `CLAUDE.md` (@import), `docs/WORKLOG.md`, `.gitignore`, rules cho Cursor/Antigravity; commit đầu tiên vào git.
- **Decisions:** Chọn `AGENTS.md` làm nguồn context chung (chuẩn xuyên tool); nguồn sự thật = file đã commit, không dùng bộ nhớ riêng của tool.
- **Next:** Khi bắt đầu code pipeline → cập nhật mục "lệnh hay dùng" + cấu trúc `src/` trong AGENTS.md; bắt đầu W1 (data & setup) theo `docs/Liver_Cancer_AI_6Week_Plan.md`.
- **Files:** `AGENTS.md`, `CLAUDE.md`, `docs/WORKLOG.md`, `.gitignore`, `.cursor/rules/project.mdc`, `.agents/rules/project.md`
