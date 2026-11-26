# Iteration 3 analysis tables (Qwen2.5-Omni-3B, K=3)

Made from the corrected runs (each question answered from its own video). Settings for both runs: 150 instant + backward questions (seed 0), τ ∈ {0.88, 0.90, 0.93, 0.95}, `--keep_current --min_keep 4`, 336 px frames, 16 s window. Graded by `src/semantic_judge.py` with DeBERTa-v3-large NLI. The score is `ent_con` × 100, from −100 (contradicted) to +100 (supported). Results and interpretation: `reports/Iteration3_Report.md`.

Regenerate with (from the repository root):
```bash
python src/analyze.py results/iteration3/judged_v3_k3.jsonl --errors --csv results/iteration3/tables/audio > results/iteration3/tables/audio.txt
python src/analyze.py results/iteration3/judged_v3_k3_noaudio.jsonl --errors --csv results/iteration3/tables/noaudio > results/iteration3/tables/noaudio.txt
python src/compare_audio.py results/iteration3/judged_v3_k3.jsonl results/iteration3/judged_v3_k3_noaudio.jsonl --csv results/iteration3/tables/audio_vs_noaudio.csv > results/iteration3/tables/audio_vs_noaudio.txt
python src/validate_grader.py results/iteration3/judged_v3_k3.jsonl results/iteration3/labels_v3.csv > results/iteration3/tables/grader_validation.txt
python src/analyze.py results/iteration3/judged_v3_k3.jsonl --metric p_con --csv results/iteration3/tables/audio_pcon > results/iteration3/tables/audio_pcon.txt
python src/analyze.py results/iteration3/judged_v3_k3_noaudio.jsonl --metric p_con --csv results/iteration3/tables/noaudio_pcon > results/iteration3/tables/noaudio_pcon.txt
```

| File | One row per | Columns |
|---|---|---|
| `audio_summary.csv` / `noaudio_summary.csv` | condition | mean score, frames kept, ViT ms and % of full (ideal and actual), prompt tokens, LLM latency |
| `*_by_task.csv` | condition × task | mean score per question type |
| `*_paired.csv` | gate vs. one baseline | mean difference (gate − baseline) on the same questions, 95% bootstrap CI, `significant` = CI excludes 0 |
| `*_errors.csv` | one of the 10 largest gate-vs-full drops per τ | both scores, question, reference, both answers, which frames the gate kept (0 = oldest, 32 = the question moment) |
| `audio_vs_noaudio.csv` | condition | mean of (audio score − no-audio score) on the same question |
| `grader_validation.txt` | grade × question type | agreement of `ent_con` and `1 − p_con` with the 60 hand labels in `../labels_v3.csv`: Spearman ρ with 95% CI, AUC, median grade per label |
| `audio_pcon_*`, `noaudio_pcon_*` | as above | the same summary / by-task / paired tables with `p_con` × 100 as the score (**lower is better**, so a positive gate − baseline difference means the gate is worse) |
| `*.txt` | — | the full terminal output, including gate vs full by task, frame overlap between methods and latency (`audio_vs_noaudio.txt`) |

Notes:
- `vit_pct_ideal` now uses a measured `resume_ms` (about 83 ms per frame), not the iteration 2 placeholder. Generation still encodes kept frames in full, so `vit_pct_actual` is the cost this code pays.
- All 150 sampled videos have an audio track, so `compare_audio.py`'s "no audio track" control column is empty.
- Grader agreement with the hand labels: ρ = 0.40 for `ent_con`, 0.70 for `1 − p_con` (`grader_validation.txt`; `reports/Iteration3_Report.md` §6.6).
