# Task 4 — Iteration 2: Codifying OmniEyes (Early-Layer Variance Gating)

*File paths in this document are relative to the repository root.*

## 1. Idea selected and why
We continue with **OmniEyes** (early-layer variance gating; idea **H18** in the 2C2/Final Submission ranking, where it ranked #1). OmniEyes needs no fine-tuning: the vision-language model (VLM) stays frozen, and a gate built from the first few layers of the vision encoder drops video frames that look like the last frame kept.

Iteration 1 built the pipeline and showed large latency savings. Its accuracy evaluation first failed: the LLM judge scored every answer 0, and was then replaced by a local NLI grader. Iteration 2 makes eight changes to the method, measurement and tooling (§2), runs a larger sweep (150 questions × 13 conditions), and re-validates the grader on fresh human labels.

---

## 2. Issues identified in iteration 1 and refinements adopted

| # | Issue identified | Type | Refinement in iteration 2 |
|---|---|---|---|
| 1 | LLM judge scored all 1,038 answers 0; Groq rate limits (HTTP 429) | Measurement / infrastructure | Grading moved to a **local NLI grader** (`src/semantic_judge.py`, DeBERTa-v3-large, no API). It was validated against human labels in iteration 1 and re-validated here on fresh labels (§4.1). `src/judge.py` was also hardened (`max_tokens=200`, `<score>` tags, `-1` failure code), but it is not used for the results below. |
| 2 | ViT overhead from the duplicate forward pass | Compute / implementation gap | `resume_from_k()` in `src/gate.py` runs blocks K..N plus the patch merger on stored block-K features. Every row now logs an **ideal** ViT cost (kept frames resume from block K) and an **actual** cost (kept frames re-run from block 1). Answer generation still encodes kept frames from block 1. |
| 3 | Full-ViT time measured on a single frame | Profiling noise | `full_vit_ms` is averaged over 3 frames per window. |
| 4 | Irregular gaps between kept frames | Confounder | `src/data.py` returns each frame's time, and `src/run.py` adds them to the prompt (`F0=634.0s, F1=634.5s, …`). |
| 5 | Ad-hoc τ values | Empirical ambiguity | `src/calibrate.py` summarises the gate's similarity distribution. The τ sweep was widened to {0.88, 0.90, 0.93, 0.95}. |
| 6 | Forward questions not evaluated | Scope gap | `forward` added to the default task set. |
| 7 | Resuming after a disconnect could lose partial questions | Data integrity | `src/run.py` tracks finished `(question, condition)` pairs. |
| 8 | No failure diagnostics | Analysis gap | `src/analyze.py` gained `--errors` (largest gate-vs-full drops), `--task` filtering, `--metric`, multi-file input for K ablations, and `--csv` export. |

---

## 3. Refined method (v2)

For each question asked at video time $t$:
1. **Causal window:** sample frames from $[t - 16\text{ s}, t]$ at 2 Hz ($N \approx 33$), resized to $336 \times 336$, with their timestamps $t_0, \dots, t_{N-1}$.
2. **Gate features:** run ViT blocks $1..K$ ($K = 3$) on all $N$ frames. This is the fixed per-frame cost of gating: measured at **10.5 ms vs. 67.4 ms** for a full ViT pass, i.e. ≈ 16%.
3. **Greedy selection:** keep frame 0. Keep frame $i$ if its mean per-token cosine similarity to the last kept frame is below τ:
   $$\text{sim}(f_i, f_{\text{last}}) = \frac{1}{|P|} \sum_{p} \frac{f_{i,p} \cdot f_{\text{last},p}}{\|f_{i,p}\|\,\|f_{\text{last},p}\|} < \tau$$
4. **Cost accounting:**
   - ideal ViT cost $= N\, t_{\text{gate}} + m\, t_{\text{resume}}$;
   - actual ViT cost $= N\, t_{\text{gate}} + m\, t_{\text{full}}$;

   where $m$ is the number of kept frames.
5. **Answer:** give the kept frames plus their timestamps to Qwen2-VL-2B (greedy decoding, ≤ 64 new tokens).

**Conditions** (13 per question):
- `full`: all ~33 frames.
- `gate@τ`: frames chosen by the gate.
- `uniform@τ`: the same number of frames, evenly spaced.
- `pixel@τ`: the same number of frames, chosen by pixel difference.

τ ∈ {0.88, 0.90, 0.93, 0.95}.

---

## 4. Setup

- **Data:** PhoStream, 150 questions (seed 0) from four archives: 48 instant, 28 backward and 74 forward. Every question was run under all 13 conditions (1,950 answers).
- **VLM:** Qwen2-VL-2B-Instruct, fp16, greedy decoding, `max_new_tokens=64`, Colab T4.
- **Grader:** `ent_con` = P(entail) − P(contradict) from DeBERTa-v3-large NLI (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli). The premise is the reference answer and the hypothesis is the model's answer. Scores are shown ×100, from −100 (confidently contradicted) to +100 (confidently supported).
- **Statistics:** paired bootstrap (2,000 resamples, 95% CI) of gate minus each baseline, over the same questions.
- **Headline set: instant + backward (n = 76).** Forward questions are reported separately (§5.4), for two reasons. Their answer event happens *after* time $t$, outside the window the model sees. And the grader does not agree with humans on them (§4.1).

### 4.1 Grader validation on fresh labels
We labelled 60 randomly sampled rows of the iteration 2 output as 0 / 1 / 2 (wrong / partial / correct). Agreement of `ent_con` with these labels:

| Question type | Rows | Spearman ρ | 95% CI | AUC* |
|---|---|---|---|---|
| All | 60 | 0.34 (p = 0.008) | 0.09 – 0.55 | 0.75 |
| **Instant + backward** | 38 | **0.47** (p = 0.003) | 0.17 – 0.69 | **0.86** |
| Instant | 22 | 0.57 (p = 0.006) | 0.19 – 0.80 | – |
| Backward | 16 | 0.31 (n.s.) | −0.22 – 0.70 | – |
| Forward | 22 | 0.13 (n.s.) | −0.31 – 0.52 | 0.60 |

*AUC: the probability that the grader scores an answer the human judged right (label ≥ 1) above one judged wrong (label 0). 0.5 = chance, 1.0 = perfect.

- **The grader is usable on instant + backward and near chance on forward.** On instant + backward, answers humans judged wrong have a median grade of −91, and answers judged right +86. On forward, many answers are timestamps ("…at F0=249.0s"), and the grader rewards answers that restate the question.
- **The iteration 1 figure of ρ = 0.59 was optimistic,** because the metric was selected on the same labels it was evaluated on. ρ = 0.47 on instant + backward is the honest out-of-sample estimate. Sentence-BERT similarity remained weaker (ρ = 0.27 overall), and the older combined score was not significant.
- **Limitations:** only 10 of 60 labels were non-zero (just one for backward), 3 of those 10 are the same answer under different conditions, and answers that restate the question were labelled inconsistently. The ρ estimates are therefore uncertain.

---

## 5. Results (`results/iteration2/tables/`)

### 5.1 Main table: instant + backward (n = 76)

| Condition | Score ↑ | Frames | ViT % of full (actual) ↓ | Prompt tokens ↓ | LLM s ↓ |
|---|---|---|---|---|---|
| `full` | −44.2 | 31.6 | 100% | 2758 | 2.17 |
| `gate@0.88` | −41.8 | 2.8 | 24.5% | 356 | 0.87 |
| `uniform@0.88` | −41.8 | 2.8 | 8.8% | 356 | 0.87 |
| `pixel@0.88` | −45.0 | 2.8 | 8.8% | 356 | 0.87 |
| `gate@0.90` | −43.4 | 3.2 | 26.1% | 393 | 0.90 |
| `uniform@0.90` | −44.0 | 3.2 | 10.3% | 393 | 0.89 |
| `pixel@0.90` | −44.1 | 3.2 | 10.3% | 393 | 0.88 |
| `gate@0.93` | −44.1 | 4.6 | 30.4% | 505 | 0.95 |
| `uniform@0.93` | −45.6 | 4.6 | 14.6% | 505 | 0.95 |
| `pixel@0.93` | −44.1 | 4.6 | 14.6% | 505 | 0.95 |
| `gate@0.95` | −43.5 | 6.1 | 35.0% | 624 | 1.00 |
| `uniform@0.95` | −41.9 | 6.1 | 19.2% | 624 | 1.00 |
| `pixel@0.95` | −47.2 | 6.1 | 19.2% | 624 | 1.01 |

The table reports **actual** ViT cost only. The ideal cost depends on `resume_ms`, which is a constant 50.0 ms in every row of this run, i.e. a placeholder rather than a measurement (§7).

### 5.2 Paired differences, gate − baseline (instant + backward, n = 76)

| τ | vs. uniform | vs. pixel | vs. full |
|---|---|---|---|
| 0.88 | +0.1 [−6.2, +7.5] | **+3.3 [+0.1, +8.9]** | +2.5 [−8.7, +13.8] |
| 0.90 | +0.6 [−4.5, +7.6] | +0.7 [−3.0, +3.9] | +0.9 [−11.2, +13.5] |
| 0.93 | +1.4 [−4.5, +9.1] | −0.0 [−0.0, +0.0] | +0.1 [−13.0, +12.0] |
| 0.95 | −1.6 [−9.2, +7.3] | +3.8 [−0.0, +10.2] | +0.8 [−9.9, +12.1] |

Only one of 12 intervals excludes 0 (gate@0.88 vs. pixel, lower bound +0.1). With 12 comparisons at 95%, about one such result is expected by chance.

### 5.3 Score by question type

| Condition | Instant (n = 48) | Backward (n = 28) |
|---|---|---|
| `full` | −50.6 | −33.3 |
| `gate@0.88` / `0.90` / `0.93` / `0.95` | −51.4 / −52.0 / −51.7 / −50.7 | −25.2 / −28.6 / −31.1 / −31.0 |
| `uniform@0.88` / `0.90` / `0.93` / `0.95` | −54.9 / −55.4 / −53.6 / −53.7 | −19.4 / −24.4 / −31.8 / −21.6 |
| `pixel@0.88` / `0.90` / `0.93` / `0.95` | −52.4 / −52.9 / −51.7 / −50.8 | −32.4 / −28.9 / −31.1 / −41.2 |

Instant scores are flat across all conditions (−50.6 to −55.4). Backward scores vary more (−19.4 to −41.2), but with n = 28 that spread is within noise.

### 5.4 Forward questions (n = 74, reported separately)
Scores range from −25.4 to −38.3. The gate's differences from `full` (+5.8 to +11.5) are not significant. The only interval excluding 0 (gate@0.95 vs. uniform@0.95, +9.9 [+0.9, +20.0]) comes from uniform@0.95 scoring unusually low (−38.3) compared with uniform at the other τ values. Given that the grader is near chance on forward questions (AUC 0.60), and the answer event lies outside the model's window, we draw no accuracy conclusions from forward questions.

---

## 6. Hypothesis check (from Final Submission §F/G)

- **Compute savings: partially supported.**
  - LLM latency falls from 2.17 s to 0.87–1.00 s (−54% to −60%), and prompt tokens from 2,758 to 356–624 (−77% to −87%).
  - The gate's measured ViT cost (24.5–35.0% of `full`) is still **about 2–3×** that of uniform or pixel sampling at the same frame count (8.8–19.2%). Every frame pays the block-1..3 gating cost (≈ 16% of a full pass), and kept frames are still re-encoded from block 1.
- **Accuracy retention: not distinguishable from `full`; not a demonstration of equivalence.**
  - No gate condition differs significantly from `full`. But the intervals are wide (about ±12 points), and the grader is noisy (ρ = 0.47).
  - Accuracy is also flat across *every* condition, including those with 2.8 frames: the 2B model scores about −44 with 31.6 frames and −42 with 2.8. At this level, the input frames barely change its answers, so the experiment has little power to detect a loss.
- **Gate better than cheap frame selection: not shown.**
  - The gate never beats uniform significantly, and beats pixel in one of four comparisons, at the edge of significance.
  - More importantly, the methods mostly received **the same frames**. At τ = 0.90 the gate kept a single frame in 30 of 76 questions, gate and pixel selected identical frames in 57 of 76, and gate, pixel and uniform produced word-for-word identical answers in 51 of 76. These comparisons cannot separate the methods (§7, drawback 1).

---

## 7. Drawbacks, failure modes and observed limitations

Sources: `src/analyze.py --errors` (`results/iteration2/tables/*_errors.csv`) and direct inspection of `results/iteration2/judged_v2_k3.jsonl`.

### Drawback 1: The gate keeps the oldest frame and often nothing else
The gate always keeps frame 0, the oldest frame, from 16 s before the question. Later frames are kept only if they differ from it by more than τ. For many windows nothing does:

| τ | Gate keeps exactly 1 frame | Gate keeps the current frame (frame 32) |
|---|---|---|
| 0.88 | 31 / 76 | 3 / 76 |
| 0.90 | 30 / 76 | 5 / 76 |
| 0.93 | 26 / 76 | 10 / 76 |
| 0.95 | 15 / 76 | 16 / 76 |

For instant questions, which ask about the current moment, this is the main failure.
- *Question:* "What action is currently being performed in the Mission Control interface?"
- *Reference:* "The cursor is moving to the top right corner… clicking on a plus (+) button…"
- *Gate@0.95* (kept frames 0, 1, 4, 7, 8): "The Mission Control interface is not being used in the current frame." (−99.8)
- *Full* (all frames): "The user is currently using the Mission Control interface to stream a video." (−0.1)

Frame counts were also much lower than in iteration 1 at the same τ (3.2 vs. 9.5 frames at τ = 0.90), on a mostly different question set (7 of 150 overlap). The block-3 mean-token similarity stays above τ across most of a 16 s window.

### Drawback 2: The fixed gating cost
Every frame passes through blocks 1..3 before the gate can decide (10.5 ms of a 67.4 ms full pass ≈ 16%). Uniform and pixel sampling pay nothing for dropped frames, so at equal frame counts the gate costs 2–3× more ViT time (§6). Resuming kept frames from block 3 would remove the re-encoding part of this cost. In this run, however, `resume_ms` was logged as a constant 50.0 ms, so the ideal-cost column cannot be trusted. Generation also still re-encodes kept frames from block 1.

### Drawback 3: Timestamp prompting changes the answer format
With frame times in the prompt, 127 forward answers (out of 962 forward rows) answer with a time instead of content, e.g. "The navigation voice starts speaking at the timestamp F1", "The phone is placed on the stand at F14.5s", "The video ends at F1". These carry nothing the grader can check against the reference. Whether timestamps help backward questions cannot be separated in this run: iteration 1 used different questions and no timestamps.

### Drawback 4: Grader behaviour inflates some per-question differences
The NLI grader treats a vague wrong answer as neutral (≈ 0) and a specific wrong answer as a contradiction (≈ −100). Several of the largest "drops" are therefore both answers being wrong:
- *Question:* "Describe the visual appearance of the instructor…"
- *Reference:* a woman in a **pink** long-sleeved top on a blue mat.
- *Full:* "wearing glasses… in front of a white wall" (−0.5).
- *Gate:* "wearing a **white shirt**… in front of a white wall" (−99.9). Delta −99.5.

This adds noise to every paired difference. It does not favour any condition, but it widens the CIs.

### Drawback 5: Questions the model cannot answer under any condition
- Several questions depend on audio (an inner monologue, a narrator's explanation, a spoken "Can I hold you?"), and the model receives frames only.
- Forward questions ask about events after the window.

All conditions fail these questions equally, which pushes scores toward the floor and further reduces the power to compare methods.

---

## 8. Plan for iteration 3

### 1. Always keep the current frame, and set a minimum frame budget (fixes drawback 1)
- **Change:** select backwards from the newest frame (time $t$), or always add it. Require at least ~4 frames per window.
- **Expected effect:** instant questions see the current moment, and gate, uniform and pixel choose different frames, so the comparison becomes informative.
- **Risk:** a fixed minimum lowers the compression ratio on static scenes.

### 2. Choose τ by target frame count on matching calibration data
- **Change:** log similarities for every τ (not only the first) and fix `src/calibrate.py`'s simulation. Then pick τ values that give ~8 and ~16 frames on a calibration set drawn from the same archives and question types.
- **Risk:** τ chosen this way may not transfer across video sources (screen recordings vs. handheld camera).

### 3. Spatially aware similarity (top-k tokens) for small UI and text changes
- **Change:** compare frames on their most-changed tokens rather than the mean over all tokens:
  $$\text{sim}_{\text{top-}k}(f_i, f_{\text{ref}}) = \frac{1}{k} \sum_{p \in \text{Top-}k} \cos(f_{i,p}, f_{\text{ref},p})$$
  This reacts to localized changes such as a number updating on a phone screen.
- **Risks:** handheld jitter, autofocus and compression noise can trigger false keeps, and the top-k step adds work to the always-on loop.
- **Mitigation:** require a minimum cluster of adjacent changed tokens.

### 4. Real resumption timing, then real resumption
- **Change:** verify that `resume_from_k` matches a full pass numerically (in `src/test_gate_cpu.py`, compare values, not just shapes), and log real `resume_ms`. Then feed resumed features into generation, so the ideal cost becomes the actual cost.
- **Risk:** injecting precomputed vision features into `generate()` couples the code to `transformers` internals (M-RoPE positions, vision placeholder tokens).

### 5. Timestamp format ablation (drawback 3)
- **Change:** run with and without timestamps on the same questions. Use relative offsets (`-16.0s … 0.0s`) and instruct the model to use times for ordering only, not as the answer.

### 6. Gate depth ablation (K ∈ {1, 3, 6})
- **Question:** can K = 1 cut the fixed cost from ~16% to ~5% without hurting selection?
- **Risks:** block-1 features are close to pixel statistics (sensitive to lighting and panning), and K = 6 raises the fixed cost to ~25%.

### 7. Evaluation upgrades
- **Forward questions:** evaluate them with a window ending at the annotated response time, or drop them from accuracy claims.
- **More data:** increase the instant + backward set from 76 to ~300 questions, so the paired CIs shrink by about half.
- **Better grader validation:** label ~100 rows, one condition per question, balanced across question types. Write down a rule for answers that only restate the question, and have two people label to measure human–human agreement.
