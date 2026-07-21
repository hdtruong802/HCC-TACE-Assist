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
