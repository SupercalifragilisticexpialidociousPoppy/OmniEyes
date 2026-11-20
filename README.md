# OmniEyes: Early-Layer Variance Gating (iteration 3)

OmniEyes (idea H18 in the 2C2/Final Submission ranking) tests whether a streaming video assistant can skip redundant frames cheaply. Every incoming frame goes through only the first K layers of the vision encoder (ViT). A frame is kept only if those early features differ enough from the last kept frame. Only the kept frames are sent to the model to answer a question about the video. Iterations 1–2 used Qwen2-VL-2B (video only). From iteration 3 the default is **Qwen2.5-Omni-3B**, which also hears the audio of the same window.

Each question is answered under 13 conditions. The gate is compared with all frames (`full`), and with two cheap ways of picking the same number of frames: evenly spaced (`uniform`) and by pixel change (`pixel`). Answers are graded locally with an NLI model, with no API calls.

- **Results and conclusions:** `reports/Iteration2_Report.md` (iteration 1: `reports/Iteration1_Report.md`)
- **Known problems and suggested fixes (not yet implemented):** `reports/ITERATION2_REVIEW.md`
- **Ideas for iteration 3 (moving to an Omni model with audio, better frame selection, and more):** `plans/`

Hardware: the model runs need a Colab GPU: a **T4** (16 GB, fp16) works, and an **L4 or A100** (bf16) is better. An RTX 3050 (4–8 GB) is too small, and Colab's TPU runtime is not supported. Grading and analysis run on any machine (grading is much faster on a GPU).

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
│   ├── iteration2/               K=3, τ ∈ {0.88, 0.90, 0.93, 0.95}, all tasks, timestamps in the prompt
│   │   ├── results_v2_k3.jsonl   model answers, 150 questions × 13 conditions
│   │   ├── judged_v2_k3.jsonl    the same rows, NLI-graded (large model)
│   │   ├── labels_v2.csv         60 fresh hand labels for re-validating the grader
│   │   └── tables/               CSV and text output of analyze.py (see tables/README.md)
│   └── iteration3/               Qwen2.5-Omni-3B runs (copied in from Drive; see step 11)
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
4. Asks the model the question once per condition, with each kept frame's timestamp in the prompt. With an Omni model, the audio of the same 16 s window is passed too. Audio is never gated and never extends past the question time.

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
| `--model` | Qwen/Qwen2.5-Omni-3B | Omni checkpoint (video + audio, Thinker only) or Qwen2-VL checkpoint (video only) |
| `--no_audio` | off | with an Omni model, answer from frames alone (video-only ablation) |
| `--keep_current` | off | always send the newest frame (the question moment), in every condition |
| `--min_keep` | 1 | the gate keeps at least this many frames; uniform and pixel get the same budget |

### `src/gate.py`: the gate
- **`EarlyLayerGate`** wraps the model's vision encoder.
  - `early_features()` runs only blocks 1..K, stopping early with a hook.
  - `full_time()` times a complete ViT pass.
  - `split_time()` times one real full pass and marks the clock when block K finishes. The part after block K (`resume_ms`) is the "ideal" cost of a kept frame that could continue from its block-K features. Generation itself still encodes kept frames in full.
- **`select_by_gate(feats, τ)`** keeps frame 0, then every frame whose mean per-token cosine similarity to the last kept frame is below τ. Two iteration 3 options, both off by default so iterations 1–2 stay reproducible:
  - `keep_current` also keeps the newest frame (the question moment). Without it, the gate often sends only the oldest frame, from 16 s before the question.
  - `min_keep` tops the selection up to a minimum number of frames. It adds the frames least similar to anything already kept, using the gate's own block-K features. With only 1–2 frames, gate, uniform and pixel are forced onto the same frames, and the comparison is meaningless.
- **`select_uniform`** and **`select_by_pixel_diff`** are the baselines. They pick the same number of frames, evenly spaced or by largest pixel change. With `keep_current`, they also always include the newest frame.

### `src/data.py`: data loading
- `load_annotations()` reads the PhoStream JSON, flattens the nested `verified_responses`, and maps field names (question, answer, timestamp, task, video) from lists of candidate names.
- `read_window()` returns the frames, as PIL images and NumPy arrays, plus their timestamps in seconds.
- `read_audio_window()` returns the mono audio of the same causal window at the model's sampling rate (via ffmpeg), or nothing if the video has no audio track.
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
Builds a tiny random Qwen2-VL vision tower on CPU. It checks early exit, the split timing (the part after block K must be shorter than the full pass), and that the three selectors pick the expected frames. Runs in seconds without a GPU.

---

## Running on Colab (iteration 3: Qwen2.5-Omni-3B with audio)

Run every command from the repository root. Lines starting with `!` or `%` go in Colab cells.

### 0. Pick a GPU runtime
**Runtime → Change runtime type → T4 GPU** (or L4 / A100 if your plan has them). The TPU runtime will not work: the code is written for NVIDIA GPUs.

### 1. Get the code
```bash
!git clone https://github.com/SupercalifragilisticexpialidociousPoppy/OmniEyes.git
%cd /content/OmniEyes
```
In a new session after a disconnect, run `%cd /content/OmniEyes` and `!git pull` instead of cloning again.

### 2. Install
```bash
!pip -q install -U transformers accelerate
!pip -q install decord sentence-transformers scipy
```
- `transformers` must be recent: Qwen2.5-Omni support is new. If you see `KeyError: 'qwen2_5_omni'`, the upgrade didn't take effect. Use **Runtime → Restart session**, then continue from step 3.
- `decord` reads video frames. `sentence-transformers` and `scipy` are only for grading (steps 8–10).
- `ffmpeg`, used to read the audio, is already installed on Colab.

### 3. Download the dataset
```bash
!git clone https://github.com/Lucky-Lance/PhoStream
!hf download lucky-lance/PhoStream phone_class_reencoded_2fps.tar.gz --repo-type dataset --local-dir videos
!cd videos && tar -xf phone_class_reencoded_2fps.tar.gz
```
On an older `huggingface_hub`, the command is `huggingface-cli download` instead of `hf download`. Iteration 2 also used the EgoBlind, record_file and youtube_dl archives. Download them the same way if you want the same question mix.

### 4. Check the setup
```bash
!python src/data.py PhoStream/dataset/phostream_annotations.json
!python src/test_gate_cpu.py
!ffprobe -v error -select_streams a -show_entries stream=codec_name,sample_rate -of csv=p=0 "$(find videos -name '*.mp4' | head -1)"
```
- `data.py` should print the number of parsed questions and the task types.
- `test_gate_cpu.py` should end with `ALL TESTS PASSED`.
- `ffprobe` prints the audio track, e.g. `aac,44100`. **If it prints nothing, the videos have no audio.** Omni runs then fall back to video only, and every output row shows `audio_s = 0`.

### 5. Save results to Google Drive
Colab deletes `/content` when the session ends. Write run outputs straight to Drive, so an interrupted run can resume:
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/OmniEyes/iteration3
```

### 6. Smoke test: 3 questions, 1 τ
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 3 --taus 0.90 --tasks instant backward \
  --out /content/drive/MyDrive/OmniEyes/iteration3/omni_smoke.jsonl
```
The first run downloads the model (about 10 GB). Check:
1. The startup line says `model: Qwen/Qwen2.5-Omni-3B | audio: on, 16000 Hz`.
2. It prints one `[i/3]` progress line per question.
3. The answers look sensible:
```python
import json
for l in open('/content/drive/MyDrive/OmniEyes/iteration3/omni_smoke.jsonl'):
    r = json.loads(l); print(r['cond'], r['n_kept'], r['audio_s'], r['llm_latency_s'], '|', r['pred'][:100])
```
`audio_s` should be about 16 (seconds of audio used). Note the seconds per question: the full run (step 7) has 13 conditions per question instead of 4.

### 7. Full run
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward \
  --keep_current --min_keep 4 \
  --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3.jsonl
```
- `--keep_current --min_keep 4` turn on the frame-selection fixes: every condition sees the question moment, and each method gets at least 4 frames to choose. Every output row records both settings.
- Roughly 20 s per question on a T4, so about 50 minutes for 150 questions (estimated from the smoke test).
- If Colab disconnects, repeat steps 1–3 and 5, then re-run this exact command: finished (question, condition) pairs are skipped. Use a new `--out` file whenever you change any option, so rows made with different settings never end up in the same file.
- Forward questions are left out: their answer happens after the question time, outside the window the model sees (see `reports/Iteration2_Report.md` §5.4).
- **Video-only ablation** (what audio adds), same questions:
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward --no_audio \
  --keep_current --min_keep 4 \
  --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3_noaudio.jsonl
```

### 8. Grade (a few minutes on a GPU)
```bash
!python src/semantic_judge.py /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3.jsonl \
  /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl \
  --nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

### 9. Analyse
```bash
!mkdir -p /content/drive/MyDrive/OmniEyes/iteration3/tables
!python src/analyze.py /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl --task instant backward --errors \
  --csv /content/drive/MyDrive/OmniEyes/iteration3/tables/instant_backward
```
How to read the output: `results/iteration2/tables/README.md` and `reports/Iteration2_Report.md` §5. A `*` after a paired difference means its 95% interval excludes 0.

### 10. Validate the grader on fresh labels
```bash
!python src/semantic_judge.py /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl --sample 60 /content/drive/MyDrive/OmniEyes/iteration3/labels_v3.csv
# open labels_v3.csv from Drive, fill the "human" column with 0 / 1 / 2 (wrong / partial / correct), then:
!python src/semantic_judge.py /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl --validate /content/drive/MyDrive/OmniEyes/iteration3/labels_v3.csv
```
Never reuse labels from an earlier run: question IDs repeat, but the answers change.

### 11. Bring the results into the repository
Copy the Drive folder's files into `results/iteration3/` in your local clone, then commit them.

### Troubleshooting
| Symptom | Likely cause and fix |
|---|---|
| `KeyError: 'qwen2_5_omni'` or `cannot import name 'Qwen2_5OmniThinkerForConditionalGeneration'` | `transformers` is too old: repeat step 2, then **Runtime → Restart session** |
| `CUDA out of memory` | Try `--no_audio`, then a shorter `--window` (e.g. 8), then fewer τ values |
| Empty, repeated or garbled answers | fp16 on a T4. On an L4 / A100, change `torch.float16` to `torch.bfloat16` in `load_model()` in `src/run.py` |
| `ffmpeg not found` | `!apt-get -qq install ffmpeg`, or run with `--no_audio` |
| `audio_s` is 0 in every row | the videos have no audio track (check with the `ffprobe` command in step 4) |
| `AttributeError` mentioning `image_processor` or `visual` | the Omni processor or model is laid out differently than expected: copy the full error into an issue |

### Reproducing iteration 2 (Qwen2-VL-2B, video only)
Same steps, with `--model Qwen/Qwen2-VL-2B-Instruct` and the iteration 2 settings:
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward forward \
  --out results/iteration2/results_v2_k3.jsonl
```
Then grade and analyse `results/iteration2/…` as in steps 8–9. Iteration 2's own outputs are already in `results/iteration2/`.

### Optional: gate depth ablation
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 150 --k 1 --taus 0.90 0.95 --tasks instant backward --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k1.jsonl
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 150 --k 8 --taus 0.90 0.95 --tasks instant backward --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k8.jsonl
# grade each file as in step 8, then compare:
!python src/analyze.py /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k1.jsonl /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k8.jsonl --task instant backward
```
K = 8 is the first block after a full-attention layer in Omni's vision encoder (see `plans/01_omni_migration.md`).

---

## Results at a glance (iteration 2, instant + backward, n = 76)

- **Speed:** with the gate, LLM latency falls from 2.17 s to 0.87–1.00 s, and prompt tokens from 2,758 to 356–624.
- **Accuracy:** no condition differs detectably from `full`. But all conditions score low (about −42 to −47), and the gate, pixel and uniform selectors usually received the same frames. So the run cannot tell the methods apart.
- **Gate cost:** the gate's measured ViT cost is 2–3× that of uniform or pixel sampling at the same frame count.

Details, examples and the iteration 3 plan are in `reports/Iteration2_Report.md`.
