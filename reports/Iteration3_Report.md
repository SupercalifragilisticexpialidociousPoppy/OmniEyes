# Task 4 — Iteration 3: OmniEyes on an Omni Model, with Audio

*File paths in this document are relative to the repository root.*

## 1. Summary
Iteration 3 moves OmniEyes (early-layer variance gating; idea H18 in the 2C2/Final Submission ranking) from Qwen2-VL-2B, which sees video only, to **Qwen2.5-Omni-3B**, which also hears the audio of the same window. It also fixes the two frame-selection problems found in iteration 2: the gate now always sends the current frame and keeps at least 4 frames.

While analysing the first iteration 3 run, we found a **bug that invalidates the accuracy results of every earlier run**: every question had been answered from the same video, not its own (§3). After fixing it, we reran everything. This report covers only the corrected runs.

Main findings, on 150 instant + backward questions:
- **The fix changed everything.** With the same model and settings, the `full` score went from −31.9 (wrong videos) to **+26.9** (correct videos). Scores are now positive for the first time. The negative scores in iterations 1–2 mostly reflect the bug, not the model or the gate.
- **Audio is the biggest lever for pruned inputs.** Earlier iterations had no audio at all. Adding it raises every pruned condition by +7 to +20 points. 11 of the 12 gains are significant. Without audio, every gate setting loses 7–14 points against `full`. With audio, the loss shrinks to 1.5–4.6 points and is not significant.
- **The gate is cheaper than `full`, but not more accurate than simpler selectors.** At τ = 0.88, the gate sends 14 of ~33 frames. That halves prompt tokens (3,254 → 1,675) and cuts LLM time by 40% (3.37 s → 2.01 s), with no significant accuracy loss when audio is on. The gate scores above uniform and pixel selection at every τ, but only one of 24 comparisons is significant. And the gate costs more vision-encoder time than either.

---

## 2. Issues from iteration 2 and changes made

| # | Issue in iteration 2 | Change in iteration 3 | Status |
|---|---|---|---|
| 1 | No audio: several questions depended on speech (§7, drawback 5 of the iteration 2 report) | Qwen2.5-Omni-3B (Thinker only, no speech output), with the causal audio of the same 16 s window (`read_audio_window()` in `src/data.py`) | Done |
| 2 | The gate kept only the oldest frame in 30–40% of questions | `--keep_current`: every condition includes the newest frame (the question moment) | Done |
| 3 | With 1–2 frames, gate, uniform and pixel picked the same frames | `--min_keep 4`: the gate tops up to 4 frames using farthest-point selection on its block-K features. The baselines get the same budget | Done |
| 4 | `resume_from_k()` crashed; `resume_ms` was a fake constant 50 ms | Replaced by `EarlyLayerGate.split_time()`: a hook marks the end of block K inside one real full pass, so `resume_ms` is measured | Done (timing only; generation still encodes kept frames in full) |
| 5 | Forward questions can't be answered from the window, and the grader is near chance on them | Forward questions dropped from the runs | Done |
| 6 | Omni's answers ran on, inventing new "Human:" turns, or hedged ("I'm not sure…") | Explicit stop tokens; no system prompt; the prompt asks for direct, concise answers "based only on what you have seen and heard" | Done |
| 7 | **Every question used the same video** (found during iteration 3, §3) | `resolve_video()` matches the full folder path, never guesses, and logs `video_file` in every row | Done |
| — | τ chosen by calibration, top-k similarity, timestamp ablation, K ablation, larger sample, fresh grader labels | — | Not done (§8) |

---

## 3. The video bug: every earlier run answered from one video

### What happened
Every PhoStream video file is named `output_1.mp4`; only the folder path tells them apart. The annotation paths also differ slightly from the extracted folder names (the archives add a `_2fps` suffix). The old `resolve_video()` tried the annotation path, failed, and fell back to **"the first file found with the same name"**. So every question in iterations 1 and 2, and in the first iteration 3 run, was answered from the same ~10-minute video.

### How it was found
In the first iteration 3 run, questions about different videos asked at the same timestamp received **identical frame selections** (19 of 19, 23 of 23 and 19 of 19 pairs). Also, no question after ~615 s received any audio, which is the length of the one video being used.

### The fix
- `resolve_video()` now compares the end of the folder path, after removing the `_2fps` suffix, and returns nothing if the match is missing or ambiguous.
- `run.py` drops questions without their own video **before** sampling. It prints how many were found per source.
- Every output row records `video_file`.
- On the full dataset (four archives, 578 videos), 2,761 of 2,770 instant + backward questions found their own video. The 9 skipped questions point to files missing from the archives.

### Check on the new run
- The 150 questions cover **114 different videos**, from all four sources.
- Different videos at the same timestamp now get different gate selections (12 of 13 pairs).
- All 150 questions received audio: 16 s for 146 of them, 3–15 s only for questions in a video's first seconds.

### Effect on earlier results
`full` condition, score = `ent_con` × 100 (−100 contradicted, +100 supported):

| Run | Model | Audio | Videos | `full` | `gate@0.90` |
|---|---|---|---|---|---|
| Iteration 1 | Qwen2-VL-2B | no | one video for all | −28.1 | −29.0 |
| Iteration 2 (instant + backward) | Qwen2-VL-2B | no | one video for all | −44.2 | −43.4 |
| Iteration 3, first run (scrapped) | Qwen2.5-Omni-3B | yes | one video for all | −31.9 | −31.1 |
| **Iteration 3** | Qwen2.5-Omni-3B | **yes** | **own video** | **+26.9** | **+23.8** |
| **Iteration 3, video only** | Qwen2.5-Omni-3B | no | **own video** | **+19.9** | **+6.5** |

The scrapped run and the final run differ only in the video lookup. The fix alone moves `full` by about **+59 points**. So:
- The low, flat scores in iterations 1–2 came mainly from the model describing a video unrelated to the question. A specific but wrong answer scores about −100.
- Every iteration 1–2 accuracy comparison (gate vs. uniform vs. pixel vs. `full`) was made on the wrong video and **is not valid**. The timing, token and frame-count measurements are not affected, because they don't depend on which video was read.
- The iteration 1 and 2 reports are left as submitted, as records of what was run at the time.
- Qwen2-VL-2B was not rerun on the correct videos. We can't say how much of the remaining gap between iterations is the model and how much is the bug.

---

## 4. Method (v3)

For each question asked at video time $t$:
1. **Causal window:** frames from $[t - 16\text{ s}, t]$ at 2 Hz ($N \approx 33$), resized to 336 × 336, with their timestamps. The audio of the same window, mono at 16 kHz, never extends past $t$.
2. **Gate features:** run ViT blocks 1..K (K = 3) of Omni's vision encoder on all $N$ frames. Measured cost: **14.8 ms per frame vs. 95.2 ms** for a full pass (≈ 16%).
3. **Selection:**
   - Keep frame 0. Keep frame $i$ if its mean per-token cosine similarity to the last kept frame is below τ.
   - `keep_current`: also keep the newest frame.
   - `min_keep = 4`: if fewer than 4 frames are kept, add the frame least similar to everything kept, until there are 4.
4. **Answer:** send the kept frames, their timestamps and the audio clip to Qwen2.5-Omni-3B (Thinker only, fp16, greedy decoding, ≤ 64 new tokens). The audio is never gated: 16 s is about 400 tokens.
5. **Cost accounting (vision encoder):**
   - ideal cost $= N\,t_{\text{gate}} + m\,t_{\text{resume}}$;
   - actual cost $= N\,t_{\text{gate}} + m\,t_{\text{full}}$;

   where $m$ is the number of kept frames. $t_{\text{resume}}$ is now measured: 82.7 ms. Generation still encodes kept frames from block 1, so the actual cost is what this code pays.

**Conditions (13 per question):** `full` (all frames), and for each τ ∈ {0.88, 0.90, 0.93, 0.95}: `gate@τ`, and `uniform@τ` and `pixel@τ` with the same number of frames. With `keep_current`, the baselines also always include the newest frame.

---

## 5. Setup
- **Data:** PhoStream, all four video archives. 150 questions (seed 0): 101 instant and 49 backward, from youtube_dl (81), phone_class (40), record_file (16) and EgoBlind (13).
- **Runs:** the same 150 questions and settings twice, with audio and with `--no_audio` (1,950 answers each).
- **Hardware:** Colab T4, fp16. Notebook: `results/iteration3/Task4_Ite3.ipynb`.
- **Grader:** `ent_con` from DeBERTa-v3-large NLI (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`), as in iteration 2. It has **not** been re-validated on Omni's answers. The iteration 2 estimate for instant + backward questions is ρ = 0.47, AUC = 0.86.
- **Statistics:** paired bootstrap (2,000 resamples, 95% CI) over the same questions. `*` means the interval excludes 0.

---

## 6. Results (`results/iteration3/tables/`)

### 6.1 With audio (main result)

| Condition | Score ↑ | Frames | ViT % of full (actual) ↓ | Prompt tokens ↓ | LLM s ↓ |
|---|---|---|---|---|---|
| `full` | **26.9** | 32.5 | 100% | 3254 | 3.37 |
| `gate@0.88` | **25.4** | 14.0 | 58% | 1675 | 2.01 |
| `uniform@0.88` | 20.1 | 14.0 | 43% | 1675 | 2.02 |
| `pixel@0.88` | 23.3 | 14.0 | 43% | 1675 | 2.01 |
| `gate@0.90` | 23.8 | 15.9 | 64% | 1835 | 2.13 |
| `uniform@0.90` | 19.9 | 15.9 | 49% | 1835 | 2.15 |
| `pixel@0.90` | 21.5 | 15.9 | 49% | 1835 | 2.14 |
| `gate@0.93` | 23.2 | 19.0 | 74% | 2102 | 2.36 |
| `uniform@0.93` | 20.6 | 19.0 | 58% | 2102 | 2.34 |
| `pixel@0.93` | 20.4 | 19.0 | 58% | 2102 | 2.37 |
| `gate@0.95` | 22.3 | 21.6 | 82% | 2322 | 2.55 |
| `uniform@0.95` | 20.9 | 21.6 | 66% | 2322 | 2.55 |
| `pixel@0.95` | 20.6 | 21.6 | 66% | 2322 | 2.56 |

Paired differences, gate − baseline:

| τ | vs. uniform | vs. pixel | vs. full |
|---|---|---|---|
| 0.88 | +5.3 [−2.7, +13.0] | +2.1 [−4.3, +7.9] | −1.5 [−10.1, +7.0] |
| 0.90 | +3.9 [−2.6, +10.6] | +2.3 [−4.2, +8.5] | −3.2 [−10.7, +4.7] |
| 0.93 | +2.6 [−5.9, +11.0] | +2.8 [−2.5, +8.2] | −3.7 [−11.6, +3.9] |
| 0.95 | +1.5 [−3.6, +6.8] | +1.7 [−2.1, +5.8] | −4.6 [−11.5, +2.0] |

- The gate is ahead of both baselines in all 8 comparisons, by 1.5–5.3 points, but no interval excludes 0.
- No gate setting is significantly worse than `full`.
- **Fewer frames did not mean lower scores:** τ = 0.88 keeps the fewest frames and scores highest. The differences between τ values are within noise.

### 6.2 Without audio

| Condition | Score | vs. `full` | gate − uniform | gate − pixel |
|---|---|---|---|---|
| `full` | 19.9 | — | — | — |
| `gate@0.88` | 12.5 | −7.4 [−17.2, +2.3] | **+12.8 [+3.8, +22.6] \*** | +5.2 [−2.4, +12.9] |
| `gate@0.90` | 6.5 | **−13.4 [−22.1, −4.7] \*** | +4.0 [−4.1, +12.7] | −1.0 [−8.4, +6.2] |
| `gate@0.93` | 6.0 | **−13.9 [−22.4, −6.4] \*** | −5.1 [−13.2, +2.8] | −7.0 [−14.3, +0.3] |
| `gate@0.95` | 11.6 | **−8.3 [−16.0, −0.9] \*** | +2.3 [−7.0, +11.4] | +2.2 [−2.8, +7.9] |

Uniform scores −0.4 to 11.1 and pixel 7.3 to 13.0, across the τ values. Frame counts, ViT cost and LLM time are the same as with audio. Prompt tokens are ~410 lower, because there are no audio tokens.
- Without audio, dropping frames clearly hurts: 3 of 4 gate settings are significantly below `full`.
- `gate@0.88` beats `uniform@0.88` significantly (+12.8). It is the only significant gate-vs-baseline result in the whole iteration, and it's 1 of 16 such comparisons across both runs. At τ = 0.93 the gate trails both baselines. We treat it as a hint, not a finding.

### 6.3 What audio adds (same questions, same frames)

| Condition | Audio − no audio |
|---|---|
| `full` | +7.0 [−1.3, +15.9] |
| `gate@0.88` / `0.90` / `0.93` / `0.95` | **+13.0 / +17.2 / +17.2 / +10.8** \* |
| `uniform@0.88` / `0.90` / `0.93` / `0.95` | **+20.5 / +17.4 / +9.5 / +11.6** \* |
| `pixel@0.88` / `0.90` / `0.93` / `0.95` | **+16.0 / +14.0** \* / +7.4 / **+11.3** \* |

- **Audio helps most where frames were dropped.** With all frames, its gain (+7) is not significant. With 14–22 frames, it adds 10–20 points. Speech and sound carry information that the dropped frames would have shown, so audio makes pruning much safer.
- Typical examples, gate@0.88:
  - "What specific control is she adding to the Control Center list?" With audio: "Screen recording" (+100). Without: "the 'Airpods Pro' control" (−100).
  - "What is the final outcome of the procedure?" (reference: failure, flatline sound). With audio: "The procedure was unsuccessful." (+100). Without: "successful" (−100).
  - "What ingredient adds a 'briny' flavor?" With audio: "oysters" (+100). Without: "salt" (−100).
- No control group was available: all 150 sampled videos have an audio track.

### 6.4 By question type and video source (with audio)

| Condition | Instant (n = 101) | Backward (n = 49) |
|---|---|---|
| `full` | 30.7 | 19.0 |
| `gate@0.88` / `0.90` / `0.93` / `0.95` | 29.3 / 30.3 / 30.9 / 27.2 | 17.4 / 10.2 / 7.4 / 12.3 |
| `uniform@0.88` / … / `0.95` | 23.4 / 24.8 / 25.1 / 23.0 | 13.5 / 9.9 / 11.4 / 16.5 |
| `pixel@0.88` / … / `0.95` | 26.5 / 26.0 / 25.5 / 25.8 | 16.7 / 12.1 / 9.9 / 10.0 |

- On **instant** questions, the gate matches `full` (−1.4 to +0.2 at τ ≤ 0.93) and leads uniform by 4–6 points. Keeping the current frame (§2, change 2) is what makes this work.
- On **backward** questions, all pruned conditions lose 2–12 points. Those questions ask about something earlier in the window, which a dropped frame may have shown.

| Source | n | `full`, audio | `full`, no audio | `gate@0.88`, audio | `gate@0.88`, no audio |
|---|---|---|---|---|---|
| phone_class (narrated phone tutorials) | 40 | 46.6 | 31.1 | 41.4 | 7.7 |
| youtube_dl | 81 | 24.3 | 22.3 | 22.3 | 18.3 |
| record_file | 16 | 22.7 | −3.2 | 27.6 | −1.8 |
| EgoBlind | 13 | −12.0 | −1.4 | −7.1 | 8.3 |

- Audio matters most for the narrated phone tutorials. There, gating without audio drops the score from 31.1 to 7.7, while with audio the gate stays close to `full` (41.4 vs. 46.6).
- EgoBlind (13 questions) is the hardest source in every condition. The sample is small.

### 6.5 Cost

| Per question (audio run) | `full` | `gate@0.88` | `uniform@0.88` / `pixel@0.88` |
|---|---|---|---|
| Frames sent | 32.5 | 14.0 | 14.0 |
| Vision encoder, actual | 3.09 s | 1.79 s | 1.32 s |
| LLM (prefill + answer) | 3.37 s | 2.01 s | 2.02 s |
| **Total** | **6.46 s** | **3.80 s (−41%)** | **3.34 s (−48%)** |
| Prompt tokens | 3,254 | 1,675 (−49%) | 1,675 |

- **Against `full`, the gate roughly halves the cost of a question** with no significant accuracy loss when audio is on.
- **Against the baselines, the gate costs more:** every frame pays the block-1..3 pass (14.8 ms), even frames that are dropped. If kept frames resumed from block 3 instead of being re-encoded, the gate's vision cost would fall from 58% to 53% of `full`. That is still above uniform and pixel (43%).
- Selection is adaptive: at τ = 0.88, the gate kept the minimum of 4 frames on 49 questions (mostly static screens) and all frames on 7 (fast camera motion). Uniform and pixel only received these counts because the gate chose them first.

---

## 7. Hypothesis check

- **Compute savings: supported against `full`.** The gate cuts prompt tokens by 29–49%, LLM time by 24–40% and total time per question by about 41% (τ = 0.88). It is still more expensive than uniform or pixel selection at the same frame count.
- **Accuracy retention: supported with audio, not without.** With audio, no gate setting is significantly worse than `full`. The worst estimate is −4.6, with intervals reaching −10 to −12. Without audio, 3 of 4 settings lose significantly (−8 to −14). This is not a proof of equivalence: the intervals are wide and the grader is noisy.
- **Gate better than cheap selection: suggestive, not shown.** With audio, the gate is ahead of uniform and pixel in all 8 comparisons, by +1.5 to +5.3 points, but none is significant. Unlike iteration 2, the methods now receive different frames: gate and uniform picked identical frames in only 9–42 of 150 questions. On 88–107 of 150 questions, however, gate, uniform and pixel still produce word-for-word identical answers. Frame choice only matters on a minority of questions, which limits the power of the comparison.
- **Summary:** the gate is a cheaper way to run the Omni model, at a small accuracy cost that is not significant with audio. It is not yet shown to be better than simpler selectors.

---

## 8. Drawbacks and limitations

### Drawback 1: The grader is not validated on Omni's answers
The NLI grader was validated on Qwen2-VL answers (ρ = 0.47 on instant + backward). Omni's answers are shorter and more direct, and the grader's known biases still apply. A vague wrong answer scores about 0 and a specific wrong one about −100.
- *Question:* "What flavor name is printed on the label?" (reference: "Vanilla Cupcake").
- `full`: "Vanilla" (+99). `uniform@0.88`: "vanilla" (−99). Nearly the same answer, opposite grades.
- `gate@0.88`: "date" (−100), a genuine error.

### Drawback 2: The gate's fixed cost
Every frame pays ~16% of a full vision pass before the gate can decide. So at equal frame counts, the gate's vision cost is 1.2–1.4× that of uniform or pixel selection. Resuming kept frames from block K (now timed correctly, §2 change 4) would only partly close this gap, and it is not yet implemented in generation.

### Drawback 3: Backward questions suffer most from pruning
Pruned conditions lose up to 12 points on backward questions (§6.4), against at most 3.5 on instant questions. Mean-token similarity at block 3 can drop the frame that shows a brief earlier event. Example: "What occurs immediately after the hand interacts with the phone screen?" `full` describes the screen after the tap. Gate and uniform (4 frames) both answer "the phone rings" (−99).

### Drawback 4: Small differences, many comparisons
150 questions give paired intervals of about ±7–10 points, while the gate–baseline differences are 1.5–5 points. With 24 gate-vs-baseline comparisons, about one "significant" result is expected by chance. The one we found (§6.2) should be read that way.

### Drawback 5: Reproducibility
- `transformers` is installed unpinned. During the run it warned that video processing will change in v5.22 (`cap_pixels_per_frame`), which could change the number of video tokens.
- In the notebook, the first video-extraction cell failed with a shell syntax error, and a later cell extracted the archives correctly. The README gives the corrected steps.

---

## 9. Plan for iteration 4

1. **Validate the grader on Omni's answers.** Label ~100 rows, one condition per question, balanced across instant and backward, with two labellers. Without this, differences of a few points cannot be trusted.
2. **More questions.** 300–500 instant + backward questions would halve the intervals. The full dataset has 2,761 eligible questions.
3. **Gate–baseline separation.** Top-k token similarity, so small local changes such as on-screen text count. And audio-triggered keeps: keep the nearest frame when speech starts or a loud event happens (`plans/01_omni_migration.md`, item 6).
4. **Lower the gate's fixed cost.** K = 1 ablation, gating at low resolution (`plans/03_resolution.md`), and feeding resumed block-K features into generation.
5. **Time alignment for Omni.** Make sure the pruned frames' temporal positions match their real times, so audio stays paired with the right frames (`plans/04_feeding_frames.md`, item 1).
6. **Pin `transformers`** and log its version in every output row.
7. **Qwen2-VL-2B on the correct videos** (optional): separates the effect of the model change from the bug fix.
