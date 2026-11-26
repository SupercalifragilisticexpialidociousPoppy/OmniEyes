# OmniEyes: Early-Layer Variance Gating for Streaming Omni-Modal Assistants

A phone-camera stream is mostly near-identical frames, yet streaming assistants encode every frame in full. **OmniEyes** runs each frame through only the first 3 of the 32 vision-encoder blocks, keeps it only if those early features changed, and sends just the kept frames, with the audio of the same window, to **Qwen2.5-Omni-3B**. We test it on the [PhoStream](https://huggingface.co/datasets/lucky-lance/PhoStream) benchmark against sending every frame and against two cheap selectors given the same frame budget.

BITS F471 Introduction to Large Language Models, Group 9: Aryan Rao, Mohammad Zaid Agha, Shiva P, Kavya Paras Shah, Soumitra Ashish Kulkarni, and Saksham Agarwal (Tasks 1–3).

---

## Read this first

| What | Where |
|---|---|
| **Paper** | [paper/main.pdf](paper/main.pdf); source [main.tex](paper/main.tex) + [refs.bib](paper/refs.bib); also zipped as [paper.zip](paper.zip) |
| **Final report** (the project journey, Tasks 1–5) | [reports/Final_Report.pdf](reports/Final_Report.pdf); source [journey_report.tex](reports/journey_report.tex) |
| **Iteration 3 report** (final results and full detail) | [reports/Iteration3_Report.md](reports/Iteration3_Report.md) |

---

## Key results

Qwen2.5-Omni-3B, 150 PhoStream instant and backward questions, 13 conditions per question. Score = `ent_con` × 100 from the NLI grader, from −100 (contradicted by the reference) to +100 (supported).

| | `full` | `gate@0.88` | `uniform@0.88` | `pixel@0.88` |
|---|---|---|---|---|
| Frames sent | 32.5 | 14.0 | 14.0 | 14.0 |
| Score, with audio | **26.9** | **25.4** | 20.1 | 23.3 |
| Score, video only | **19.9** | 12.5 | −0.4 | 7.3 |
| Prompt tokens (with audio) | 3,254 | 1,675 | 1,675 | 1,675 |
| Time per question (vision + LLM) | 6.46 s | 3.82 s | 3.34 s | 3.33 s |

- **Cheaper than sending every frame.** With audio, the gate sends 43% of the frames, halves the prompt and cuts the time per question by 41%. Its score is 1.5 points below `full`, which is not significant.
- **Audio is the main lever.** Audio adds 10–20 points to every pruned condition. Without audio, gating loses 7–14 points against `full`.
- **Not a better selector than simple sampling.** The gate scores 1.5–5 points above uniform and pixel selection at the same frame count, but no difference is significant. Its fixed per-frame cost also makes it 1.2–1.4× more expensive than those baselines in the vision encoder.
- **The grader agrees moderately with humans** (ρ = 0.40, AUC 0.83 on 60 hand labels). A stricter "not contradicted" grade (ρ = 0.70) gives the same conclusions.

> **Validity note.** Iterations 1–2 and the first iteration 3 run answered every question from the *same* video. Every PhoStream file is named `output_1.mp4`, and the old loader matched on the file name. Their accuracy numbers are therefore invalid; their timing and token numbers still hold. Fixing the loader moved `full` from −31.9 to +26.9. Details: [Iteration3_Report.md §3](reports/Iteration3_Report.md).

---

## How it works

For each question asked at time *t* in a video:
1. **Window:** take the 16 s before *t*: about 33 frames at 2 frames per second, plus the audio of the same 16 s. Nothing after *t* is used.
2. **Gate:** run every frame through vision-encoder blocks 1–3 only (about 16% of a full pass). Keep a frame if its mean per-token cosine similarity to the last kept frame is below τ.
3. **Selection fixes:** the newest frame (the question moment) is always kept. If fewer than 4 frames are kept, the frames most different from those already kept are added.
4. **Answer:** Qwen2.5-Omni-3B gets the kept frames with their timestamps, the whole audio clip (never gated) and the question.

Each question is answered with all frames (`full`), and for τ ∈ {0.88, 0.90, 0.93, 0.95} with the gate's frames (`gate@τ`) and with the same number of frames chosen evenly spaced (`uniform@τ`) or by largest pixel change (`pixel@τ`). The whole set is run with and without audio.

---

## Project timeline

1. **Proposal:** started from the Mini-Omni papers and DuplexOmni (full-duplex omni models).
2. **Literature review:** generated with a SurveyGen-style pipeline over DuplexOmni's references.
3. **Research gap:** read the 21 papers ourselves and found that vision in real-time systems is processed continuously, with no decision about *when* to look. Narrowed the project to vision.
4. **Ideation:** an evidence ledger of vision-efficiency papers, then LLM-generated ideas. Chose early-layer gating (H18) because it needs no training and fits a Colab T4.
5. **Codification**, in three iterations:
   - Qwen2-VL-2B, with a local NLI grader replacing an LLM judge that failed;
   - more measurements, which showed the gate kept only the oldest frame;
   - Qwen2.5-Omni-3B with audio and the selection fixes, which is also when the same-video bug was found and fixed.

What we observed at each step is in the [final report](reports/Final_Report.pdf).

---

## Repository layout

```
OmniEyes/
├── README.md
├── paper/                        the research paper: main.tex, refs.bib, compiled main.pdf
├── paper.zip                     the paper folder, zipped for upload
├── reports/
│   ├── Final_Report.pdf          project journey report (Tasks 1–5); source: journey_report.tex
│   ├── Iteration3_Report.md      Omni + audio, the video-bug fix, final results, grader validation
│   ├── Iteration1_Report.md      Qwen2-VL-2B (kept as submitted)
│   ├── Iteration2_Report.md      Qwen2-VL-2B, more measurements (kept as submitted)
│   └── ITERATION2_REVIEW.md      code review of iteration 2
├── src/                          all code; run every command from the repository root
├── results/
│   ├── iteration1/               Qwen2-VL-2B answers, gradings, 60 hand labels
│   ├── iteration2/               Qwen2-VL-2B, 13 conditions, 60 fresh hand labels, tables/
│   ├── iteration3/               final runs: with and without audio, gradings, labels_v3.csv,
│   │                             the Colab notebook Task4_Ite3.ipynb, and tables/ (see its README)
│   └── iteration3-scrapped/      the first iteration 3 run, made with the video bug (invalid, kept for the record)
├── plans/                        ideas for future work (frame selection, resolution, time alignment, evaluation)
└── archive/                      Task 1–4 submissions (proposal, literature review and its documentation,
                                  research gap analysis, evidence ledger, ideation), an earlier paper build,
                                  and superseded one-off files
```

Running on Colab adds two folders at the root that are not part of the repository: `PhoStream/` (annotations) and `videos/` (the extracted video archives).

---

## Code

| File | What it does | GPU? |
|---|---|---|
| [`src/run.py`](src/run.py) | The experiment: answers each question under every condition and writes one JSON line per (question, condition) | yes |
| [`src/gate.py`](src/gate.py) | The early-exit gate (`EarlyLayerGate`), its timing, and the three frame selectors | yes (CPU for the test) |
| [`src/data.py`](src/data.py) | PhoStream annotation loader, frame reader, causal audio reader (ffmpeg) | no |
| [`src/semantic_judge.py`](src/semantic_judge.py) | Local NLI grader (used for every reported score); also samples rows for hand labelling | faster on GPU |
| [`src/analyze.py`](src/analyze.py) | Summary tables, scores by question type, paired bootstrap CIs, error lists, CSV export | no |
| [`src/compare_audio.py`](src/compare_audio.py) | Compares an audio run with its `--no_audio` twin | no |
| [`src/validate_grader.py`](src/validate_grader.py) | Grader-vs-hand-label agreement: ρ with CI, AUC, per question type | no |
| [`src/test_gate_cpu.py`](src/test_gate_cpu.py) | CPU smoke test of the gate on a tiny random model | no |
| [`src/calibrate.py`](src/calibrate.py) | Summarises gate similarities to help choose τ (known limitation: logs the first τ only) | no |
| [`src/run_plain.py`](src/run_plain.py) | Plain-Omni baseline (all frames, audio interleaved into the video). **Written but never run** (dropped for time) | yes |
| [`src/judge.py`](src/judge.py) | LLM-as-judge over an OpenAI-compatible API. Replaced by the NLI grader; kept for reference | no |

**Video matching.** Every PhoStream video is named `output_1.mp4`, so `resolve_video()` in `run.py` matches on the folder path and never guesses. Questions whose video is missing or ambiguous are skipped before sampling, and each output row records the file used (`video_file`).

### `run.py` options

| Option | Default | Meaning |
|---|---|---|
| `--ann` | required | PhoStream annotation JSON (or a folder of them) |
| `--video_root` | required | folder containing the extracted videos |
| `--out` | `results.jsonl` | output file; re-running the same command skips finished (question, condition) pairs |
| `--model` | Qwen/Qwen2.5-Omni-3B | Omni checkpoint (video + audio, Thinker only) or Qwen2-VL checkpoint (video only) |
| `--k` | 3 | number of vision-encoder blocks the gate uses |
| `--taus` | 0.88 0.90 0.93 0.95 | similarity thresholds; a frame is kept if its similarity is below τ |
| `--tasks` | instant backward forward | question types to run |
| `--n` / `--seed` | 150 / 0 | number of questions, and the sampling seed |
| `--window` / `--fps` / `--size` | 16 s / 2 / 336 px | causal window length, sampling rate, frame size |
| `--keep_current` | off | always send the newest frame, in every condition |
| `--min_keep` | 1 | the gate keeps at least this many frames; uniform and pixel get the same budget |
| `--no_audio` | off | with an Omni model, answer from frames alone |
| `--calibrate_only` | off | only log gate similarities; no answers |

Each row holds the answer (`pred`) and reference (`answer`), the kept frames, the vision cost (ideal `vit_ms`, measured `vit_ms_actual`), prompt tokens and LLM latency.

### Grader columns (`semantic_judge.py`)

| Column | Meaning |
|---|---|
| `ent_con` | P(reference supports the answer) − P(reference contradicts it). **The grade used in the reports and paper**; shown ×100 |
| `p_ent` / `p_con` / `p_neu` | the three NLI probabilities; `1 − p_con` is the stricter robustness grade |
| `sim` | Sentence-BERT similarity: topic closeness, not correctness |
| `echo` | P(the question alone supports the answer), i.e. the answer restates the question |
| `p_ent_rev`, `score`, `bertscore` | reverse direction, an older combined formula, BERTScore (`--bertscore`): kept for reference |

Always pass `--nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; the default base model agrees with humans less well. `--sample N CSV` writes rows for hand labelling (0 / 1 / 2 = wrong / partial / correct).

### `analyze.py` options
`--task T …` (filter question types) · `--metric NAME` (column to score by; default `ent_con`) · `--errors` (the 10 largest gate-vs-full drops per τ) · `--csv PREFIX` (write `_summary`, `_by_task`, `_paired`, `_errors` CSVs) · several input files (combine runs, e.g. a K ablation). In the paired table, `*` means the 95% interval excludes 0.

---

## Reproducing the results (Colab)

**Hardware:** the model runs need an NVIDIA GPU. A Colab **T4** (16 GB, fp16) works; an **L4 or A100** is better. The TPU runtime is not supported. Grading is faster on a GPU, and analysis runs anywhere. Run every command from the repository root; lines starting with `!` or `%` go in Colab cells.

**From the notebook:** [results/iteration3/Task4_Ite3.ipynb](results/iteration3/Task4_Ite3.ipynb) produced the iteration 3 results (about 3–4 hours on a T4). Its cells map to the steps below:

| Notebook cells | Step |
|---|---|
| 0–2 | 1. Get the code (cell 2 must print `1`: the video-lookup fix is present) |
| 3 | 2. Install |
| 4, 6–10 | 3. Videos. **Cell 4's extraction line is broken** (the loop was pasted twice), so nothing is extracted there; cell 7 extracts and cell 9 deletes the archives. Step 3 below avoids the detour. |
| 5 | 4. Check the setup |
| 11 | 5. Mount Google Drive |
| 12–13 | 6. Smoke test |
| 14–15 | 7. Full runs, with and without audio |
| 16–17 | 8. Grade both runs |

Steps 9–11 were done locally after downloading the Drive folder.

### 0. Pick a GPU runtime
**Runtime → Change runtime type → T4 GPU** (or L4 / A100).

### 1. Get the code
```bash
!git clone https://github.com/SupercalifragilisticexpialidociousPoppy/OmniEyes.git
%cd /content/OmniEyes
```
After a disconnect, run `%cd /content/OmniEyes` and `!git pull` instead.

### 2. Install
```bash
!pip -q install -U transformers accelerate
!pip -q install decord sentence-transformers scipy
```
- If you see `KeyError: 'qwen2_5_omni'`, the upgrade didn't take effect: **Runtime → Restart session**, then continue from step 3.
- `ffmpeg` (for audio) is already on Colab.
- `transformers` is not pinned. Record the version you used (`!pip show transformers | head -2`); newer versions may change how many tokens a frame uses (the run logs a `cap_pixels_per_frame` warning).

### 3. Download the dataset (four video archives, about 12 GB)
```bash
!git clone https://github.com/Lucky-Lance/PhoStream
# see which archives the dataset has
from huggingface_hub import list_repo_files
print([f for f in list_repo_files("lucky-lance/PhoStream", repo_type="dataset") if f.endswith(".tar.gz")])
# download every archive, then extract each one and delete the .tar.gz to save disk
!hf download lucky-lance/PhoStream --repo-type dataset --include "*.tar.gz" --local-dir videos
!cd videos && for f in *.tar.gz; do echo "extracting $f"; tar -xzf "$f" && rm "$f"; done
!rm -rf videos/.cache
!find videos -name '*.mp4' | wc -l          # should print 579
```
- Run the extraction loop as its own line. If `find` prints 0, run the loop again.
- `Ignoring unknown extended header keyword 'SCHILY.fflags'` / `'LIBARCHIVE.xattr…'` warnings are harmless.
- To use only some sources, replace `"*.tar.gz"` with one archive name; `run.py` then samples only from those.
- On an older `huggingface_hub`, use `huggingface-cli download` instead of `hf download`.
- Colab deletes `videos/` when the session ends; re-downloading takes a few minutes.

### 4. Check the setup
```bash
!python src/data.py PhoStream/dataset/phostream_annotations.json
!python src/test_gate_cpu.py
!ffprobe -v error -select_streams a -show_entries stream=codec_name,sample_rate -of csv=p=0 "$(find videos -name '*.mp4' | head -1)"
!find videos -name '*.mp4' | wc -l          # how many videos were extracted
!find videos -name '*.mp4' | head -3        # what the extracted paths look like
```
- `data.py` prints the number of questions and the task types. `test_gate_cpu.py` ends with `ALL TESTS PASSED`.
- `ffprobe` prints the audio track (e.g. `aac,44100`). If it prints nothing, the videos have no audio, and every row will show `audio_s = 0`.
- At startup, `run.py` prints how many questions found their own video. If it says 0, compare a `find` path with the "no file for:" example it prints.

### 5. Save results to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/OmniEyes/iteration3
```
Writing to Drive lets an interrupted run resume.

### 6. Smoke test: 3 questions, 1 τ
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 3 --taus 0.90 --tasks instant backward --keep_current --min_keep 4 \
  --out /content/drive/MyDrive/OmniEyes/iteration3/omni_smoke.jsonl
```
The first run downloads the model (about 10 GB). Check that:
1. the startup line says `model: Qwen/Qwen2.5-Omni-3B | audio: on, 16000 Hz`, and about 2,760 of 2,770 questions found their video;
2. the gate's frame count varies between questions (e.g. 14, 4, 31);
3. each question uses a different video and the answers make sense:
```python
import json
from collections import defaultdict
rows = defaultdict(list)
for l in open('/content/drive/MyDrive/OmniEyes/iteration3/omni_smoke.jsonl'):
    r = json.loads(l); rows[r['qid']].append(r)
for qid, rs in rows.items():
    print('=' * 100)
    print(f"[{rs[0]['task']}] Q:   {rs[0]['question']}")
    print(f"        REF: {rs[0]['answer']}")
    print(f"        FILE: {rs[0]['video_file']}   audio {rs[0]['audio_s']:.1f}s")
    for r in rs:
        kept = 'all' if r['cond'] == 'full' else r['kept']
        print(f"  {r['cond']:<12} kept={str(kept):<20} {r['llm_latency_s']:.2f}s | {r['pred']}")
```
`audio_s` should be about 16.

### 7. Full runs (about 1–1.5 hours each on a T4)
With audio:
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward \
  --keep_current --min_keep 4 \
  --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3.jsonl
```
Video only (same questions):
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward --no_audio \
  --keep_current --min_keep 4 \
  --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3_noaudio.jsonl
```
- If Colab disconnects, repeat steps 1–3 and 5, then re-run the same command; finished rows are skipped. Use a new `--out` file whenever you change an option.
- Forward questions are left out: their answer happens after the question time, outside the window.

### 8. Grade both runs
```bash
!python src/semantic_judge.py /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3.jsonl \
  /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl \
  --nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
!python src/semantic_judge.py /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k3_noaudio.jsonl \
  /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3_noaudio.jsonl \
  --nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

### 9. Analyse (any machine)
Download the Drive `iteration3` folder into `results/iteration3/`, then:
```bash
mkdir -p results/iteration3/tables
python src/analyze.py results/iteration3/judged_v3_k3.jsonl --errors --csv results/iteration3/tables/audio > results/iteration3/tables/audio.txt
python src/analyze.py results/iteration3/judged_v3_k3_noaudio.jsonl --errors --csv results/iteration3/tables/noaudio > results/iteration3/tables/noaudio.txt
python src/compare_audio.py results/iteration3/judged_v3_k3.jsonl results/iteration3/judged_v3_k3_noaudio.jsonl \
  --csv results/iteration3/tables/audio_vs_noaudio.csv > results/iteration3/tables/audio_vs_noaudio.txt
```
How to read the output: [results/iteration3/tables/README.md](results/iteration3/tables/README.md) and [Iteration3_Report.md §6](reports/Iteration3_Report.md).

### 10. Validate the grader
```bash
!python src/semantic_judge.py /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl --sample 60 /content/drive/MyDrive/OmniEyes/iteration3/labels_v3.csv
# open labels_v3.csv from Drive, fill the "human" column with 0 / 1 / 2 (wrong / partial / correct), then:
```
Then, on any machine:
```bash
python src/validate_grader.py results/iteration3/judged_v3_k3.jsonl results/iteration3/labels_v3.csv > results/iteration3/tables/grader_validation.txt
python src/analyze.py results/iteration3/judged_v3_k3.jsonl --metric p_con --csv results/iteration3/tables/audio_pcon > results/iteration3/tables/audio_pcon.txt
python src/analyze.py results/iteration3/judged_v3_k3_noaudio.jsonl --metric p_con --csv results/iteration3/tables/noaudio_pcon > results/iteration3/tables/noaudio_pcon.txt
```
The last two repeat the analysis with `p_con` (lower is better) as a robustness check. Never reuse labels from an earlier run: question IDs repeat, but the answers change.

### 11. Bring the results into the repository
Copy the Drive folder's files into your local machine or wherever for further analysis.

### Troubleshooting
Some frequent issues we ran into:

| Symptom | Likely cause and fix |
|---|---|
| `KeyError: 'qwen2_5_omni'` or `cannot import name 'Qwen2_5OmniThinkerForConditionalGeneration'` | `transformers` is too old: repeat step 2, then **Runtime → Restart session** |
| `CUDA out of memory` | Try `--no_audio`, then a shorter `--window` (e.g. 8), then fewer τ values |
| Empty, repeated or garbled answers | fp16 on a T4. On an L4 / A100, change `torch.float16` to `torch.bfloat16` in `load_model()` in `src/run.py` |
| `ffmpeg not found` | `!apt-get -qq install ffmpeg`, or run with `--no_audio` |
| `audio_s` is 0 in every row | the videos have no audio track (check with `ffprobe`, step 4) |
| `AttributeError` mentioning `image_processor` or `visual` | the Omni processor or model layout changed: copy the full error into an issue |

### Iteration 2 (Qwen2-VL-2B, video only)
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --n 150 --k 3 --taus 0.88 0.90 0.93 0.95 --tasks instant backward forward \
  --out results/iteration2/results_v2_k3.jsonl
```
Then grade and analyse as in steps 8–9. These are the iteration 2 settings with the corrected video lookup.

### Optional: gate depth ablation
```bash
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 150 --k 1 --taus 0.90 0.95 --tasks instant backward --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k1.jsonl
!python src/run.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos --n 150 --k 8 --taus 0.90 0.95 --tasks instant backward --out /content/drive/MyDrive/OmniEyes/iteration3/results_v3_k8.jsonl
# grade each file as in step 8, then compare:
!python src/analyze.py /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k1.jsonl /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k3.jsonl /content/drive/MyDrive/OmniEyes/iteration3/judged_v3_k8.jsonl --task instant backward
```
K = 8 is the first block after a full-attention layer in Omni's vision encoder (see [plans/01_omni_migration.md](plans/01_omni_migration.md)).

---
