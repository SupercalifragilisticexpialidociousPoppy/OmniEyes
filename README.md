# OmniEyes: Early-Layer Variance Gating (iteration 2)

OmniEyes (idea H18 in the 2C2/Final Submission ranking) tests whether a streaming video assistant can skip redundant frames cheaply. Every incoming frame goes through only the first K layers of the vision encoder (ViT). A frame is kept only if those early features differ enough from the last kept frame. Only the kept frames are sent to the vision-language model (Qwen2-VL-2B) to answer a question about the video.

Each question is answered under 13 conditions. The gate is compared with all frames (`full`), and with two cheap ways of picking the same number of frames: evenly spaced (`uniform`) and by pixel change (`pixel`). Answers are graded locally with an NLI model, with no API calls.

- **Results and conclusions:** `reports/Iteration2_Report.md` (iteration 1: `reports/Iteration1_Report.md`)
- **Known problems and suggested fixes (not yet implemented):** `reports/ITERATION2_REVIEW.md`
- **Ideas for iteration 3 (moving to an Omni model with audio, better frame selection, and more):** `plans/`

Hardware: the VLM runs need a **Colab T4** (16 GB). An RTX 3050 (4–8 GB) is too small for the 2B model with ~33 frames at fp16. Grading and analysis run on any machine (grading is much faster on a GPU).

---

## Repository layout

```
OmniEyes/
├── README.md                     this file
├── plans/                        working notes for iteration 3 (ideas, not implemented)
├── src/                          all code (run every command from the repository root)
│   ├── run.py                    main experiment: answers each question under every condition
│   ├── gate.py                   the early-layer gate and the three frame selectors
│   ├── data.py                   PhoStream annotation loader and video frame reader
│   ├── calibrate.py              summarises gate similarities to help choose τ
│   ├── semantic_judge.py         local NLI grader (used for all reported scores)
│   ├── analyze.py                tables, paired confidence intervals, error lists, CSV export
│   ├── judge.py                  LLM-as-judge over an API (kept for reference, not used)
│   └── test_gate_cpu.py          CPU smoke test of gate.py on a tiny random model
├── reports/
│   ├── Iteration1_Report.md
│   ├── Iteration2_Report.md
│   └── ITERATION2_REVIEW.md      code review notes and iteration 3 suggestions
├── results/
│   ├── iteration1/               τ ∈ {0.90, 0.95}, instant + backward, no timestamps
│   │   ├── calib.jsonl           gate similarities from the calibration run (30 questions)
│   │   ├── results.jsonl         model answers, 150 questions × 7 conditions
│   │   ├── judged_ite1.jsonl     first LLM-judge pass (every score 0: the failure described in the report)
│   │   ├── judged_sem.jsonl      NLI-graded, base DeBERTa model
│   │   ├── judged_sem_large.jsonl  NLI-graded, large DeBERTa model (used in the report)
│   │   └── labels.csv            60 hand labels used to validate the grader
│   └── iteration2/               K=3, τ ∈ {0.88, 0.90, 0.93, 0.95}, all tasks, timestamps in the prompt
│       ├── results_v2_k3.jsonl   model answers, 150 questions × 13 conditions
│       ├── judged_v2_k3.jsonl    the same rows, NLI-graded (large model)
│       ├── labels_v2.csv         60 fresh hand labels for re-validating the grader
│       └── tables/               CSV and text output of analyze.py (see tables/README.md)
└── archive/                      superseded one-off files, kept for the record
```

When you run on Colab, two more folders appear at the root: `PhoStream/` (the cloned annotation repository) and `videos/` (the downloaded video archives). They are not part of this repository.

---

## What each file does

### `src/run.py`: the experiment
For each question:
1. Reads the 16 s of video before the question time at 2 frames per second (~33 frames).
2. Runs every frame through the first K ViT blocks to get gate features.
3. Builds the frame lists for every condition: `full`, and `gate@τ`, `uniform@τ`, `pixel@τ` for each τ.
4. Asks Qwen2-VL-2B the question once per condition, with each kept frame's timestamp in the prompt.

It writes one JSON line per (question, condition), containing:
- the answer (`pred`) and the reference (`answer`);
- which frames were kept;
- the ViT cost, ideal (`vit_ms`) and measured (`vit_ms_actual`);
- prompt tokens and LLM latency.

If Colab disconnects, re-running the same command skips finished (question, condition) pairs.

Main options:

| Option | Default | Meaning |
|---|---|---|
| `--ann` | required | PhoStream annotation JSON (or a folder of them) |
| `--video_root` | required | folder containing the extracted videos |
| `--out` | `results.jsonl` | output file |
| `--k` | 3 | number of ViT blocks the gate uses |
| `--taus` | 0.88 0.90 0.93 0.95 | similarity thresholds; a frame is kept if its similarity is below τ |
| `--tasks` | instant backward forward | question types to run |
| `--n` | 150 | number of questions (sampled with `--seed`, default 0) |
| `--window` / `--fps` / `--size` | 16 s / 2 / 336 px | causal window length, sampling rate, frame size |
| `--calibrate_only` | off | only log gate similarities; no LLM answers |

### `src/gate.py`: the gate
- **`EarlyLayerGate`** wraps the model's vision encoder.
  - `early_features()` runs only blocks 1..K, stopping early with a hook.
  - `full_time()` times a complete ViT pass.
  - `resume_from_k()` runs the remaining blocks plus the merger on stored block-K features. It is used to time the "ideal" cost; generation itself still encodes kept frames in full.
- **`select_by_gate(feats, τ)`** keeps frame 0, then every frame whose mean per-token cosine similarity to the last kept frame is below τ.
- **`select_uniform`** and **`select_by_pixel_diff`** are the baselines. They pick the same number of frames, evenly spaced or by largest pixel change.

### `src/data.py`: data loading
- `load_annotations()` reads the PhoStream JSON, flattens the nested `verified_responses`, and maps field names (question, answer, timestamp, task, video) from lists of candidate names.
- `read_window()` returns the frames, as PIL images and NumPy arrays, plus their timestamps in seconds.
- Run it directly to check that the annotation fields are detected: `python src/data.py <annotation json>`.

### `src/calibrate.py`: choosing τ
Reads a `--calibrate_only` output file. It prints the distribution of gate similarities and a simulated frame-retention table for τ = 0.80 … 0.99. Known limitation: the similarities are logged for the first τ only, and the closing "recommended" text is fixed (see `reports/ITERATION2_REVIEW.md` 2.1).

### `src/semantic_judge.py`: grading (no API)
Compares each answer with the reference and adds these columns:

| Column | Meaning |
|---|---|
| `ent_con` | P(reference supports the answer) − P(reference contradicts it), from an NLI model. **The grade used in the reports**; `analyze.py` shows it ×100, from −100 to +100. |
| `p_ent` / `p_con` / `p_neu` | the three NLI probabilities: supports / contradicts / neither |
| `sim` | Sentence-BERT cosine similarity. Measures topic closeness, not correctness. |
| `echo` | P(the question alone supports the answer), i.e. the answer just restates the question |
| `p_ent_rev` | reverse direction, P(answer supports reference). Not useful. |
| `bertscore` | BERTScore F1, only with `--bertscore` |
| `score` | older combined formula, kept for reference |
| `nli_model` | which NLI model graded the row |

Other modes:
- `--sample N CSV` writes N random rows for hand labelling: fill the `human` column with 0 / 1 / 2 (wrong / partial / correct).
- `--validate CSV` prints the Spearman correlation of every column with those labels.

Use `--nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; the default base model agrees with humans less well.

### `src/analyze.py`: results tables
Reads one or more graded files and prints three tables:
1. per condition: score, frames, ViT % of full (ideal and actual), prompt tokens, LLM latency;
2. score per question type;
3. paired gate-minus-baseline differences with 95% bootstrap intervals, where `*` means the interval excludes 0.

Options:

| Option | Meaning |
|---|---|
| `--task T …` | only these question types, e.g. `--task instant backward` |
| `--metric NAME` | column to score by (default `ent_con` if present, else `score`) |
| `--errors` | list the 10 largest gate-vs-full drops per τ, with both answers and the kept frames |
| `--csv PREFIX` | also write `PREFIX_summary.csv`, `_by_task.csv`, `_paired.csv` and `_errors.csv` |
| several files | combine runs, e.g. a K=1 / 3 / 6 ablation; conditions are labelled by K |

### `src/judge.py`: LLM judge (not used for results)
Grades answers 0–100 with any OpenAI-compatible chat API (`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`). It rate-limits requests (`--rps`) and marks unparseable replies as `-1`. It was replaced by `semantic_judge.py` after the iteration 1 failure and API rate limits. If you use it: `JUDGE_MODEL` must be a chat model, not a speech model such as `whisper-*`. Keep keys in Colab's Secrets panel, never in the notebook.

### `src/test_gate_cpu.py`: smoke test
Builds a tiny random Qwen2-VL vision tower on CPU. It checks early exit, `resume_from_k` (output shape only), and that the three selectors pick the expected frames. Runs in seconds without a GPU.

---

## Reproducing iteration 2 on Colab

Clone the repository into Colab (or upload this folder as a zip), then work from inside it:

```bash
!git clone https://github.com/<your-account>/OmniEyes.git   # use the repository's real URL
%cd /content/OmniEyes
```

**1. Install and download data**
```bash
!pip -q install "transformers>=4.45" accelerate decord qwen-vl-utils sentence-transformers scipy
!git clone https://github.com/Lucky-Lance/PhoStream
!huggingface-cli download lucky-lance/PhoStream phone_class_reencoded_2fps.tar.gz --repo-type dataset --local-dir videos
!cd videos && tar -xf phone_class_reencoded_2fps.tar.gz
```
Iteration 2 used questions from four archives: EgoBlind, phone_class, record_file and youtube_dl. Download the others from the same dataset page in the same way, if you want the same question set.

**2. Check the setup**
```bash
!python src/data.py PhoStream/dataset/phostream_annotations.json
!python src/test_gate_cpu.py
```

**3. Calibration (optional, ~10 min, no LLM calls)**
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 30 --calibrate_only --out results/iteration2/calib.jsonl
!python src/calibrate.py results/iteration2/calib.jsonl
```
Iteration 2 itself reused `results/iteration1/calib.jsonl`.

**4. Main run (~1–2 h on a T4)**
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward forward \
  --out results/iteration2/results_v2_k3.jsonl
```
Save the output to Drive. If the run is interrupted, re-run the same command.

**5. Grade (a few minutes on a T4)**
```bash
!python src/semantic_judge.py results/iteration2/results_v2_k3.jsonl results/iteration2/judged_v2_k3.jsonl \
  --nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

**6. Analyse**
```bash
# Headline results: instant + backward
!python src/analyze.py results/iteration2/judged_v2_k3.jsonl --task instant backward --errors --csv results/iteration2/tables/instant_backward
# Forward questions, reported separately
!python src/analyze.py results/iteration2/judged_v2_k3.jsonl --task forward --errors --csv results/iteration2/tables/forward
```
Forward questions ("tell me when X happens") ask about an event *after* the question time, outside the frames the model sees, and the grader is near chance on them. Keep them out of the headline numbers.

**7. Validate the grader (once per new results file)**
```bash
!python src/semantic_judge.py results/iteration2/judged_v2_k3.jsonl --sample 60 results/iteration2/labels_v2.csv
# download labels_v2.csv, fill "human" with 0 / 1 / 2, upload it back, then:
!python src/semantic_judge.py results/iteration2/judged_v2_k3.jsonl --validate results/iteration2/labels_v2.csv
```
Always label fresh rows for a new results file. Question IDs repeat between runs but the answers change, so old labels would be matched to the wrong answers without any error. Iteration 2 agreement: ρ = 0.47 (AUC 0.86) on instant + backward, near chance on forward.

**Optional: gate depth ablation**
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 150 --k 1 --taus 0.90 0.95 --out results/iteration2/results_v2_k1.jsonl
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 150 --k 6 --taus 0.90 0.95 --out results/iteration2/results_v2_k6.jsonl
# grade each file with step 5, then compare:
!python src/analyze.py results/iteration2/judged_v2_k1.jsonl results/iteration2/judged_v2_k3.jsonl results/iteration2/judged_v2_k6.jsonl --task instant backward
```

---

## Results at a glance (iteration 2, instant + backward, n = 76)

- **Speed:** with the gate, LLM latency falls from 2.17 s to 0.87–1.00 s, and prompt tokens from 2,758 to 356–624.
- **Accuracy:** no condition differs detectably from `full`. But all conditions score low (about −42 to −47), and the gate, pixel and uniform selectors usually received the same frames. So the run cannot tell the methods apart.
- **Gate cost:** the gate's measured ViT cost is 2–3× that of uniform or pixel sampling at the same frame count.

Details, examples and the iteration 3 plan are in `reports/Iteration2_Report.md`.
