# Iteration 2 analysis tables (K=3)

Made from `results/iteration2/judged_v2_k3.jsonl`, graded by `src/semantic_judge.py` with DeBERTa-v3-large NLI. The score is `ent_con` × 100: −100 means the reference contradicts the answer, +100 means it supports it.

Regenerate with (from the repository root):
```bash
python src/analyze.py results/iteration2/judged_v2_k3.jsonl --task instant backward --errors --csv results/iteration2/tables/instant_backward
python src/analyze.py results/iteration2/judged_v2_k3.jsonl --task forward --errors --csv results/iteration2/tables/forward
```

There are two groups of files: `instant_backward_*` (76 questions, the headline results) and `forward_*` (74 questions, reported separately).

| File | One row per | Columns |
|---|---|---|
| `*_summary.csv` | condition | mean score, frames kept, ViT ms and % of full (ideal and actual), prompt tokens, LLM latency |
| `*_by_task.csv` | condition × task | mean score per question type |
| `*_paired.csv` | gate vs. one baseline | mean difference (gate − baseline) on the same questions, 95% bootstrap CI, `significant` = CI excludes 0 |
| `*_errors.csv` | one of the 10 largest gate-vs-full drops per τ | both scores, question, reference, both answers, which frames the gate kept (0 = oldest, 32 = the question moment) |
| `*.txt` | — | the full terminal output |

Use `vit_pct_actual`, not `vit_pct_ideal`: the ideal figure uses `resume_ms`, which is a constant 50.0 placeholder in this run (see `reports/ITERATION2_REVIEW.md`, item 1.1).
