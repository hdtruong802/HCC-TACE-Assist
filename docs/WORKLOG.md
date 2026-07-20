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

## [2026-07-20 · Claude Code]
- **Done:** Thiết lập hệ shared-context đa công cụ: tạo `AGENTS.md` (canonical), `CLAUDE.md` (@import), `docs/WORKLOG.md`, `.gitignore`, rules cho Cursor/Antigravity; commit đầu tiên vào git.
- **Decisions:** Chọn `AGENTS.md` làm nguồn context chung (chuẩn xuyên tool); nguồn sự thật = file đã commit, không dùng bộ nhớ riêng của tool.
- **Next:** Khi bắt đầu code pipeline → cập nhật mục "lệnh hay dùng" + cấu trúc `src/` trong AGENTS.md; bắt đầu W1 (data & setup) theo `docs/Liver_Cancer_AI_6Week_Plan.md`.
- **Files:** `AGENTS.md`, `CLAUDE.md`, `docs/WORKLOG.md`, `.gitignore`, `.cursor/rules/project.mdc`, `.agents/rules/project.md`
