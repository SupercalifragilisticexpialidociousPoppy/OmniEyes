# OmniEyes: iteration 3 plans (working notes)

Working notes for ideas discussed after iteration 2. Iteration 3 implemented and evaluated the Omni migration (01, steps 1–4), the frame-selection fixes (02, ideas 2–3) and the timing fix (06); each file's status line says what is done. The rest is open, and `reports/Iteration3_Report.md` §9 lists what comes next.

The project goal, from the proposal: make **omni models** (video + audio) understand streaming video better, more cheaply, or both. Iterations 1–2 used Qwen2-VL-2B (no audio) as a test bed. Iteration 3 moves to the real target.

| File | Topic | Priority |
|---|---|---|
| `01_omni_migration.md` | Switch to Qwen2.5-Omni, add causal audio, align time positions | 1 |
| `02_frame_selection.md` | Fix which frames the gate keeps and sends, choose τ properly, better similarity | 2 |
| `03_resolution.md` | Gate at low resolution, encode kept frames at high resolution | 3 |
| `04_feeding_frames.md` | How kept frames reach the language model: real time positions, frame pairing, prompt | 4 |
| `05_evaluation.md` | Baselines (vanilla Omni), grader, forward questions, sample size, labelling | runs alongside |
| `06_engineering_fixes.md` | `resume_from_k` crash (timing fixed in iteration 3), fake 50 ms, code/results mismatch, small code bugs | do first (cheap) |

Background for every item: `reports/Iteration2_Report.md` (results and drawbacks) and `reports/ITERATION2_REVIEW.md` (code review).

## Suggested order
1. Engineering fixes (06): small, and they make every later measurement trustworthy.
2. Frame selection (02, items 1–2): always keep the current frame, minimum frame budget. One-line-scale changes with the largest expected effect.
3. Omni migration (01), including time alignment (04, item 1).
4. Resolution split (03).
5. Audio-triggered keeps (01, item 6), as the multimodal contribution.
6. Evaluation upgrades (05) for the final run.
