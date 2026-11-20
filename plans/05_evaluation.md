# 05 · Evaluation

## Keep two roles separate
- **Baseline (what we compare against):** vanilla Qwen2.5-Omni with all frames + audio, i.e. Omni's `full` condition. Gated Omni must match it on accuracy while being cheaper. This is the main comparison for iteration 3.
- **Grader (who scores answers):** stays the local NLI grader (`src/semantic_judge.py`, `ent_con`, DeBERTa-v3-large). It must grade every condition the same way. Using Omni itself as the judge would be expensive and adds nothing, since the grader doesn't need to see the video.
- **External reference:** check whether the PhoStream paper reports Qwen2.5-Omni numbers. If so, reproduce its protocol for one run to get a published reference point. Their 235B judge is out of reach, so treat any comparison with their numbers as rough.

## Forward questions
- They ask about an event *after* the question time, so a window ending at `t` can't contain the answer. The grader is also near chance on them (ρ = 0.13, AUC 0.60).
- **Either** evaluate them with a window ending at the annotated response time (find that field in the PhoStream annotation), **or** keep them out of accuracy claims entirely.

## Sample size
- With 76 instant + backward questions, paired CIs are about ±12 points (vs. `full`).
- Aim for **~300 instant + backward questions**, which roughly halves the CIs. Download more archives if needed. Iteration 2 used EgoBlind, phone_class, record_file and youtube_dl.

## Grader validation (improve the labelling protocol)
Iteration 2: ρ = 0.47, AUC 0.86 on instant + backward; ρ = 0.13 on forward. The labels were unbalanced (10 of 60 non-zero), with repeats and inconsistent rules.
- Label **~100 rows**, **one condition per question**, balanced across question types.
- **Write a rule** for answers that only restate the question (e.g. always 0).
- **Two labellers** on the same rows, to report human–human agreement as a ceiling for the grader.
- Optionally grade a small subset with a strong LLM judge, as a second reference.

## Report hygiene
- Use "no difference detected" unless a CI excludes 0. With many comparisons, expect about 1 in 20 to cross by chance.
- Report measured ViT cost (`vit_ms_actual`) unless `resume_ms` is a real measurement (see `06_engineering_fixes.md`).
- Track per-question flags ("needs audio", "text-reading", "needs current frame") to explain *where* each change helps.
