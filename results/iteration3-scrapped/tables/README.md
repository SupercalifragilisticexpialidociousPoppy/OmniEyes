# Iteration 3 analysis tables (Qwen2.5-Omni-3B, K=3)

> **⚠ Invalid: do not use these numbers.** Because of a video-lookup bug in `src/run.py`, every question in these runs was answered from the **same** video, not its own (see `plans/06_engineering_fixes.md`, item 0). The bug is fixed. The runs must be repeated before any accuracy result can be reported. Timing and token counts are not affected.

Settings for both runs: 150 instant + backward questions, τ ∈ {0.88, 0.90, 0.93, 0.95}, `--keep_current --min_keep 4`, 336 px frames, 16 s window. Graded by `src/semantic_judge.py` with DeBERTa-v3-large NLI. The score is `ent_con` × 100, from −100 (contradicted) to +100 (supported).

| File | Made by | Contents |
|---|---|---|
| `audio.txt`, `audio_*.csv` | `src/analyze.py results/iteration3/judged_v3_k3.jsonl --errors --csv results/iteration3/tables/audio` | with audio: summary, by task, paired CIs, worst gate-vs-full drops |
| `noaudio.txt`, `noaudio_*.csv` | same, on `judged_v3_k3_noaudio.jsonl` | the same tables for the video-only run |
| `audio_vs_noaudio.txt`, `audio_vs_noaudio.csv` | `src/compare_audio.py results/iteration3/judged_v3_k3.jsonl results/iteration3/judged_v3_k3_noaudio.jsonl --csv results/iteration3/tables/audio_vs_noaudio.csv` | audio minus no-audio per condition, gate vs full with and without audio, by task, frame overlap between methods, latency |

**Audio coverage:** 111 of 150 questions got the full 16 s of audio, 7 got less (questions in the first seconds of a video), and 32 videos have no audio track. Those 32 are a built-in control: their audio and no-audio answers are identical (416 / 416), as they should be.

**Grader caveat:** the grader has not yet been validated on Omni's answers (it was ρ = 0.47 on iteration 2's instant + backward answers). It also rewards vague answers that restate the question. Several "gains" of gated conditions over `full` on backward questions come from that: see `audio_errors.csv`.
