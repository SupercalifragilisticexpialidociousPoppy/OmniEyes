# Iteration 2 code review: suggestions (not implemented)

*File paths in this document are relative to the repository root.*

These are notes from merging `h18_2/` into the base folder. None of them is applied in the code. The shared iteration 2 code is exactly the teammate's version plus the merges listed at the end. Items are ordered by how much they affect the results.

## 1. Must fix or state in the iteration 2 report

### 1.1 `resume_ms` is a constant 50.0 ms in `results/iteration2/results_v2_k3.jsonl`
All 1950 rows have `resume_ms = 50.0` exactly. A real timing measurement never repeats to the decimal, so the file was not produced by the `src/run.py` shared here, which measures it with `gate.resume_from_k`. It looks like a hard-coded fallback, e.g. in a Colab-edited copy after `resume_from_k` raised an error.
- **Effect:** the "ideal" ViT cost of every gate row (`vit_ms` = N·gate_ms + m·resume_ms) rests on a made-up number. Only `vit_ms_actual` is a real measurement.
- **Suggestion:** report only `vit_ms_actual` for iteration 2, or rerun with the real `resume_from_k` timing. Ask whoever ran it which `src/run.py` produced the file.

### 1.2 Forward questions are probably unanswerable in this setup
74 of the 150 questions (49%) are `forward`, e.g. *"Tell me when he navigates to the subscription options…"*, with a reference starting *"At this moment, the 'Available Plans' screen…"*. The event happens **after** the question time `t`, but the model only sees the window `[t−16 s, t]`. The model cannot have seen the answer, so forward scores mostly measure guessing.
- **Suggestion:** report instant + backward as the headline (`src/analyze.py --task instant backward`), and forward separately with this caveat. A proper forward evaluation needs the response timestamp from the PhoStream annotation, and a window ending there. That is iteration 3 work.

### 1.3 `reports/Iteration2_Report.md` does not match `results/iteration2/results_v2_k3.jsonl`
**Status: resolved.** The report has been rewritten from the new results and the NLI grader.

The report says n = 106 questions, gate@0.90 keeps 9.8 frames, LLM-judge scores ≈ 14/100. The new file has 150 questions × 13 conditions, and gate@0.90 keeps **3.7** frames. The report's tables come from an earlier run and must be regenerated from the new file and `src/semantic_judge.py`.

Some wording also overstates the evidence. "Fully supported", "proving", and "superior" are all used where every CI includes 0. Use "no detectable difference" unless a CI excludes 0 (`src/analyze.py` now marks those with `*`).

### 1.4 Frame budgets are much lower than in iteration 1
At the same τ, the gate keeps far fewer frames than before:

| τ | Iteration 1 | Iteration 2 |
|---|---|---|
| 0.90 | 9.5 | 3.7 |
| 0.95 | 15.9 | 6.7 |
| 0.88 | – | 3.2 |
| 0.93 | – | 5.3 |

The gate code is unchanged, and only 7 of the 150 questions overlap with iteration 1, so the new windows are probably more static. With 3–4 frames, Qwen2-VL sees only 2 temporal patches (it pairs frames), so every condition is starved of frames. That makes gate vs uniform harder to separate.
- **Suggestion:** mention this in the report. For iteration 3, choose τ by target frame count (e.g. ~8 and ~16 frames) on calibration data from the same question set.

### 1.5 The gate often keeps only the *oldest* frame, so the methods see the same frames
Measured on the 76 instant + backward questions of `results/iteration2/results_v2_k3.jsonl`:

| τ | Gate keeps exactly 1 frame | Gate picks the same frames as pixel | Gate and pixel give the identical answer |
|---|---|---|---|
| 0.88 | 31 / 76 | 62 / 76 | 68 / 76 |
| 0.90 | 30 / 76 | 57 / 76 | 69 / 76 |
| 0.93 | 26 / 76 | 59 / 76 | 73 / 76 |
| 0.95 | 15 / 76 | 51 / 76 | 69 / 76 |

- `select_by_gate` always keeps frame 0, the **oldest** frame, 16 s before the question. If no later frame falls below τ, that is the only frame the model sees. The model then never sees the current moment, which is exactly what instant questions ask about.
- Because so many questions get 1–3 frames, gate, uniform and pixel usually pick the same frames. Their answers are then identical, so the paired differences are close to 0 by construction (e.g. gate@0.93 vs pixel@0.93 = −0.0 [−0.0, +0.0]). The comparison mostly cannot tell the methods apart.
- 10 of the 76 questions get the identical answer under all 13 conditions, including `full`.
- **Suggestions:** (a) always keep the **newest** frame (the one at time `t`), or select backwards from it; (b) set a minimum frame budget, e.g. at least 4 frames; (c) find out why block-3 similarity stays above 0.95 for 16 s of video, e.g. with a histogram from `src/calibrate.py`. Mean token cosine at an early layer may be dominated by features that are similar for every image.

### 1.6 What the error lists show (`results/iteration2/tables/*_errors.csv`)
- **The drops come from a few questions.** The 40 instant + backward error rows cover only 12 questions, and the same ones fail at every τ.
- **The gate almost never keeps the current frame.** Frame 32, the moment of the question, is kept in only 2 of 40 error rows in each group. One clear case: *"What action is currently being performed in the Mission Control interface?"* The gate kept frames 0, 1, 4, 7 and 8, and answered *"Mission Control is not being used in the current frame."*
- **Many "−99" drops are the grader punishing a specific wrong answer harder than a vague wrong answer.** Example: for the yoga instructor, full said *"wearing glasses… white wall"* (neutral, about 0) and the gate said *"wearing a white shirt… white wall"* (contradicts the pink top, −99.9). Both answers are wrong, but the delta is −99.5. The grader prefers vague answers. That adds noise to every paired difference.
- **Several questions need audio**, e.g. the "inner voice", the narrator's explanation, or *"Can I hold you?"*. The model only gets frames, so no condition can answer them.
- **Forward: `full` gets high scores for answers that only restate the question**, e.g. *"She switches to 'Back Turns' after she has extended her arm and rotated her torso"* scores 99.6. These are grader false positives, which makes forward scores even less trustworthy.

### 1.7 Timestamp prompting makes the model answer with timestamps
With `F0=…s` labels in the prompt, many answers, especially to "Tell me when…" questions, become timestamps instead of content: *"The navigation voice starts speaking at the timestamp F1"*, *"The phone is placed on the stand at F14.5s"*, *"The video ends at F1"*. Those answers carry no content for the grader to support. This is a side effect of the iteration 2 timestamp change worth reporting.
- **Suggestion:** tell the model to use the timestamps for ordering only, and to answer the question's content. Or compare with and without timestamps on the same questions.

### 1.8 Grader agreement by question type (`results/iteration2/labels_v2.csv`, 60 rows)
Spearman ρ of `ent_con` with the human labels:

| Group | n | ρ | 95% CI | AUC* |
|---|---|---|---|---|
| all | 60 | 0.34 | 0.09 – 0.55 | 0.75 |
| **instant + backward** | 38 | **0.47** | 0.17 – 0.69 | **0.86** |
| instant | 22 | 0.57 | 0.19 – 0.80 | – |
| backward | 16 | 0.31 (n.s.) | −0.22 – 0.70 | – |
| forward | 22 | 0.13 (n.s.) | −0.31 – 0.52 | 0.60 |

*AUC = the probability that the grader scores a randomly chosen answer the human called right (label 1 or 2) higher than one the human called wrong (label 0). 0.5 = coin flip, 1.0 = perfect.

- **The grader works reasonably on instant + backward and is close to chance on forward.** Headline claims should use instant + backward only.
- **The labels are very unbalanced.** 50 of 60 rows are 0 (wrong), and backward has a single non-zero label, so its ρ means nothing. Three of the ten "right" rows are the same answer to the same question under three conditions.
- **Labelling was inconsistent for answers that only restate the question.** *"The goat eventually gets off the device"* was labelled 2, but *"The layout of the note application has been changed"* was labelled 0. Both add nothing beyond the question.
- **Suggestions for the next labelling round:** (a) write a one-line rule for restating answers (e.g. always 0 or 1); (b) sample one condition per question, so the same answer isn't labelled several times; (c) sample the task types evenly, with more rows (~100), so every group has enough right answers.

## 2. Code issues

### 2.1 `src/calibrate.py` simulates other τ values from τ=0.90 similarities
`src/run.py --calibrate_only` logs `sims` only for `taus[0]`. These are similarities to the *last kept frame*, and which frame was last kept depends on τ. Re-thresholding those numbers at 0.88 / 0.93 / 0.95 is therefore only an approximation. The "Recommended tau sweep" text at the end is also hard-coded, and its frame counts (~8 / 11 / 17 / 20) don't match what the iteration 2 run actually kept (3.2 / 3.7 / 5.3 / 6.7).
- **Suggestion:** have `src/calibrate.py` re-run `select_by_gate` on stored features, or log `sims` for every τ. Then print the counts it computes instead of fixed text.

### 2.2 `src/test_gate_cpu.py` checks only the *shape* of `resume_from_k`'s output
It asserts that the resumed output has the same shape as a full ViT pass, not the same values. A wrong `rotary_pos_emb` or `cu_seqlens` would still pass.
- **Suggestion:** add `torch.allclose(resumed_out, full_out.float(), atol=1e-3)` (fp32 on CPU), so the "ideal" cost is known to describe a pass that gives the correct output.

### 2.3 Resumption is still only timed, never used
Kept frames are still encoded from block 1 by `model.generate`. `resume_from_k` exists only to time the ideal path. This is fine as long as the report says so, but "Compute savings" claims should use `vit_ms_actual`.

### 2.4 Full-ViT time is averaged over the first 3 frames only
`full_t` is measured on the first 3 frames of each window, right after the gate pass. That's better than 1 frame, but the first call can include warm-up. Timing frames 2–4, or the median of 5, would be steadier.

### 2.5 Timestamps are absolute seconds from the video start
For example `F0=634.0s … F32=650.0s`. Relative offsets from the question time (`F0=-16.0s … F32=0.0s`) are shorter, and probably easier for a 2B model to use. The teammate's report already lists this as an iteration 3 idea.

## 3. Merge decisions made in the base folder

| File | Action |
|---|---|
| `src/gate.py`, `src/run.py`, `src/test_gate_cpu.py`, `src/calibrate.py`, `src/judge.py`, `reports/Iteration2_Report.md` | Copied from `h18_2/` unchanged |
| `results/iteration2/results_v2_k3.jsonl` | Copied from `h18_2/` |
| `src/data.py` | **Merged.** The teammate's copy had lost the base fix that flattens `verified_responses` and the extra field names (`user_query`, `response`, `time_type`); without them the PhoStream annotations don't load. Kept those, and added the teammate's timestamp return from `read_window`. |
| `src/analyze.py` | **Merged.** Kept everything from the teammate's version (multi-file, K ablation, ideal/actual ViT, `--errors`, `*` significance marker). Added `--metric` (default `ent_con` when present), `--task` filtering, and two bug fixes: rows were dropped whenever `score < 0`, which would have discarded every negative `ent_con` row (now only `src/judge.py`'s −1 failure code is skipped); and `--errors` printed scores with `%d`, which crashes on decimal scores. |
| `README.md` | **Merged.** The teammate's iteration 2 structure, with the LLM judge cells replaced by the `src/semantic_judge.py` grading and validation steps. |
| `reports/Iteration1_Report.md` | Kept the base version (the copy in `h18_2/` is the old one with all-zero scores). |
