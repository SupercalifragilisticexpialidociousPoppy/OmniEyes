# Task 4 — Iteration 1: Codifying OmniEyes (Early-Layer Variance Gating)

*File paths in this document are relative to the repository root.*

## 1. Idea selected and why
We chose **OmniEyes** (early-layer variance gating; idea **H18** in the 2C2/Final Submission ranking, where it ranked #1). It needs no training: the VLM stays frozen and we only add an inference wrapper. That makes it the only top-ranked idea we can run on our hardware (a Colab T4 and an RTX 3050). H16 needs SFT plus a simulated-user harness, and H19 needs changes to the KV cache.

## 2. Issues found before codification (spec as written vs. what can be run)

| # | Issue in the original OmniEyes spec | Type | Refinement adopted |
|---|---|---|---|
| 1 | "Qwen-VL or LLaVA" does not name a model; PhoStream baselines are 7B–30B | Missing detail / compute | **Qwen2-VL-2B-Instruct** (32-block ViT, fp16, fits a T4) |
| 2 | Acquisition at "5 Hz", but PhoStream videos are re-encoded at **2 fps** | Inconsistency | Acquisition fixed at **2 Hz** |
| 3 | PhoStream videos average 13.3 min, too long to feed to a 2B model on a T4 | Compute | **Causal window of the last 16 s** (33 frames) before the question timestamp |
| 4 | The similarity measure was not specified (per-token vs. pooled), and τ had no value | Ambiguity | **Mean per-token cosine** on block-K output vs. the **last kept** frame; τ ∈ {0.90, 0.95}, set from a calibration run |
| 5 | "Reuse the KV cache of the previous frame" is undefined for streaming | Ambiguity | Skipped frames are **dropped** from the context; the last kept frame stands in for them |
| 6 | C++/LibTorch wrapper | Unnecessary effort | PyTorch **forward hook** that raises an exception to stop the ViT after block K (works across model versions) |
| 7 | Gate overhead might cancel the savings (Handoff2 Q1) | Feasibility risk | Measure gate ms and full-ViT ms per frame and report ViT cost per condition |
| 8 | Forward questions test response *timing* (Early Response), which OmniEyes does not change | Scope mismatch | Iteration 1 evaluates **Instant + Backward** questions only; Forward is left for iteration 2 |
| 9 | PhoStream's judge is Qwen3-235B, which we cannot run | Compute | A local, API-free **NLI-based grader** (§4.1), validated against human labels, scores **all conditions identically**; we compare conditions against each other, not against published absolute scores |
| 10 | Is the ViT needed at all, or would plain frame differencing work just as well? | Missing baseline | Added a **pixel-difference gate** baseline at the same frame budget |
| 11 | The annotation schema is not documented on the dataset card | Missing detail | The loader probes and maps the fields (`src/data.py`) |
| 12 | Dropping frames leaves irregular gaps, but Qwen2-VL encodes time by frame index | Confounder | Noted; fix in iteration 2 (add timestamps to the prompt) |

## 3. Refined method (v1)
For each question at time t:
1. Sample frames from [t−16 s, t] at 2 Hz and resize them to 336².
2. Run ViT blocks 1..K (K=3) on every frame to get block-3 features. This is the cost paid on every frame.
3. Keep frame i only if cos(feat_i, feat_last_kept) < τ.
4. Give the kept frames to Qwen2-VL-2B and answer the question.

**Conditions:** `full` (all frames, the upper bound) · `gate@τ` · `uniform@τ` (same number of frames as the gate, evenly spaced) · `pixel@τ` (same number, chosen by pixel difference).
**ViT cost of the gate:** N·gate_ms + kept·(full_ms − gate_ms). This assumes kept frames resume from block K; the current code re-runs them from the start, so the figure is the ideal cost.

## 4. Setup
- **Data:** a PhoStream subset (one archive), 150 Instant + Backward QAs, seed 0
- **Model:** Qwen2-VL-2B-Instruct, fp16, greedy decoding, max 64 new tokens
- **Metrics:** grader score (§4.1) · frames passed to the LLM · ViT ms (and % of `full`) · prompt tokens · LLM latency
- **Statistics:** paired bootstrap 95% CI for gate minus each baseline (same questions in every condition)

### 4.1 Grading answers without an LLM judge
Our first grading pass used an LLM judge over the Groq API, and it failed: every answer scored 0 (see §7). Running a large LLM judge on all 1038 answers was also too costly for us. We therefore replaced it with a local, deterministic grader built from small encoder models (`src/semantic_judge.py`, no API calls, a few minutes on a T4). It compares each prediction with the reference answer. We tested several candidate signals:

| Signal | Model | What it measures |
|---|---|---|
| `sim` | Sentence-BERT `all-mpnet-base-v2`, cosine similarity | Closeness in meaning / topic |
| `bertscore` | BERTScore F1 (roberta-large, rescaled) | Token-level semantic overlap |
| `p_ent`, `p_con` | NLI cross-encoder, premise = reference, hypothesis = prediction | P(reference supports / contradicts the prediction) |
| `echo` | Same NLI model, premise = question | Whether the prediction only restates the question |
| `ent_con` | `p_ent − p_con`, scaled to −100…+100 | Single NLI grade |

**Validation against human labels.** We hand-labelled 60 randomly sampled (question, reference, prediction) rows as 0 / 1 / 2 (wrong / partial / correct) and computed the Spearman rank correlation ρ between each signal and the labels (labels: `results/iteration1/labels.csv`; graded files: `results/iteration1/judged_sem.jsonl` for the base NLI model and `results/iteration1/judged_sem_large.jsonl` for the large one):

| Signal | NLI = DeBERTa-v3-base | NLI = DeBERTa-v3-large (MNLI+FEVER+ANLI+LingNLI+WANLI) |
|---|---|---|
| `sim` (Sentence-BERT) | +0.21 (p=0.12) | +0.21 (p=0.12) |
| `bertscore` | +0.18 (p=0.17) | — |
| `p_ent` | +0.41 | +0.56 |
| `p_con` | −0.39 | −0.55 |
| `p_ent_rev` (reverse direction) | +0.04 | −0.05 |
| **`ent_con`** | +0.47 | **+0.59** (p < 10⁻⁶) |

Findings:
- **Embedding similarity does not measure correctness here.** Sentence-BERT and BERTScore were not significantly correlated with the human labels. The model's wrong answers are usually on-topic: "the building is predominantly *white*" vs. the reference "predominantly *pink*" still has high similarity.
- **NLI captures contradiction.** It catches wrong colours, yes/no flips and wrong menu items. It roughly triples the agreement with humans.
- **The larger, adversarially trained NLI model helps** (ρ 0.47 → 0.59). The base model was fooled by word overlap, e.g. it judged "select the *Edit* menu option" as supported by "He selects *Markup and Reply*… opens the PDF in an *editing* mode" with p = 0.99.
- **The echo penalty and the reverse direction were dropped.** Correct answers often reuse the question's wording, so `echo` correlated *positively* with correctness. A short answer can never entail a long, detailed reference, so `p_ent_rev` was uninformative.

**Grader used below: `ent_con` from DeBERTa-v3-large**, range −100 (confidently contradicted) to +100 (confidently supported). Caveat: we selected this metric using the same 60 labels, so ρ = 0.59 is slightly optimistic. A fresh label set would confirm it.

## 5. Results
Score = `ent_con` (DeBERTa-v3-large NLI), mean over ~148 questions, from `results/iteration1/judged_sem_large.jsonl`.

| Condition | Score ↑ | Frames | ViT % of full ↓ | Prompt tokens ↓ | LLM s ↓ |
|---|---|---|---|---|---|
| full | −28.1 | 32.1 | 100% | 2461 | 2.73 |
| gate@0.90 | −29.0 | 9.5 | 40% | 799 | 1.28 |
| uniform@0.90 | −32.1 | 9.4 | 29% | 795 | 1.27 |
| pixel@0.90 | −34.0 | 9.4 | 29% | 795 | 1.27 |
| gate@0.95 | **−23.9** | 15.9 | 57% | 1270 | 1.68 |
| uniform@0.95 | −28.2 | 15.9 | 50% | 1270 | 1.68 |
| pixel@0.95 | −24.8 | 15.9 | 50% | 1270 | 1.68 |

The negative absolute scores mean that the reference more often contradicts than supports the 2B model's answers. Qwen2-VL-2B is weak on this task in absolute terms, which is expected at this scale and is not what OmniEyes tests.

Paired differences, gate − baseline (95% bootstrap CI):

| Comparison | Δ | 95% CI | n |
|---|---|---|---|
| gate@0.90 vs uniform@0.90 | +2.3 | [−4.5, +9.6] | 148 |
| gate@0.90 vs pixel@0.90 | +4.2 | [−0.4, +9.6] | 148 |
| gate@0.90 vs full | −0.9 | [−11.4, +9.7] | 149 |
| gate@0.95 vs uniform@0.95 | +4.3 | [−2.2, +11.5] | 148 |
| gate@0.95 vs pixel@0.95 | +0.9 | [−2.0, +3.7] | 148 |
| gate@0.95 vs full | +5.1 | [−3.0, +14.0] | 148 |

Score by task type:

| Condition | Backward (n=46) | Instant (n≈103) |
|---|---|---|
| full | −6.5 | −37.7 |
| gate@0.90 | −18.3 | −33.7 |
| uniform@0.90 | −21.8 | −36.8 |
| pixel@0.90 | −23.9 | −38.5 |
| gate@0.95 | −3.5 | −33.1 |
| uniform@0.95 | −10.9 | −36.0 |
| pixel@0.95 | −5.6 | −33.4 |

**Hypothesis check (from Final Submission §F/G):**
- **Compute savings: partially supported.** At τ = 0.90 the gate passes 9.5 of 32.1 frames to the LLM. This cuts prompt tokens by 68% (2461 → 799) and LLM latency by 53% (2.73 s → 1.28 s). However, the ViT cost is 40% of `full`, vs. 29% for uniform and pixel at the same frame budget. The extra 11 points come from running blocks 1..K on every frame and then re-running kept frames from block 1 (§7).
- **Accuracy retention: supported within our statistical power.** Neither gate setting is significantly worse than `full`: gate@0.95 is +5.1 [−3.0, +14.0] with half the frames, and gate@0.90 is −0.9 [−11.4, +9.7] with 30% of the frames.
- **Gate better than cheap frame selection: not shown.** The gate's mean is higher than uniform and pixel in all four comparisons (+0.9 to +4.3), but every CI includes 0. The closest is vs. pixel@0.90 (+4.2 [−0.4, +9.6]). With n ≈ 148 the CIs span about ±7 points, so differences of this size cannot be resolved.
- **Task breakdown (exploratory, small n).** Backward questions, which need information from earlier in the window, lose the most under aggressive pruning: −6.5 for full vs. −18.3 to −23.9 at τ = 0.90. At τ = 0.95 the gate fully recovers (−3.5). Instant questions score low under every condition. Many ask about small on-screen text (menu names, labels), which the 2B model struggles to read at 336² regardless of frame selection.

## 6. Comparison to state of the art
Published PhoStream Instant/Backward results (Table 3) evaluate larger 7B–30B backbone architectures using a 235B parameter judge (Qwen3-235B), so our absolute scores with a local 2B model are not directly comparable.

Instead, OmniEyes targets relative efficiency vs. performance scaling:

- Frame Reduction: At τ = 0.90, gate prunes average frame count down to 9.5 frames/stream (from 32.1), matching the visual sampling budget of uniform@0.90 (9.4 frames) and pixel@0.90 (9.4 frames).
- Latency Scaling: LLM decode latency scales linearly with frame budget, dropping from 2.73s (full, 32.1 frames) down to 1.28s (gate@0.90, 9.5 frames) and 1.68s (gate@0.95, 15.9 frames).
- Accuracy under pruning: under the same grader, neither gate setting is significantly below `full` (§5). The latency and token savings therefore come without a detectable accuracy cost at this sample size.


## 7. Issues found after the trial run → plan for iteration 2

**Observed problems (iteration 1)**
- **LLM judge failure (zero-score artifact).** The judge LLM wrote a description of the two answers before the score and ran out of its token budget, so no score was parsed and every answer defaulted to 0. A later retry also failed because a speech-to-text model (`whisper-large-v3-turbo`) had been configured as the judge, and it does not support chat completions. Together with Groq free-tier rate limits (HTTP 429), this made API-based judging impractical. **Resolved** by the local NLI grader in §4.1.
- **Grader noise.** The NLI grader agrees only moderately with humans (ρ = 0.59). A noisy grader does not favour any condition, but it widens the CIs and makes small differences harder to detect.
- **ViT execution overhead (duplicate forward pass).** gate@0.90 uses 40% of the full ViT time vs. 29% for uniform@0.90. Every frame runs blocks 1..K for gating, and kept frames then re-run blocks 1..32 from scratch instead of reusing the block-K hidden states.
- **Statistical power.** With ~148 questions, paired CIs are about ±7 points wide on the −100…+100 scale. Halving them needs roughly 4× the questions (~600).

**Plan for iteration 2**
- **Larger evaluation set.** Increase to ~600 Instant + Backward QAs (or more archives), so that gate-vs-baseline differences of ~4 points become detectable.
- **Confirm the grader.** Label ~40 fresh rows (ideally by two annotators, to also report human–human agreement as a ceiling) and check that `ent_con` stays near ρ ≈ 0.6.
- **Feature resumption.** Modify `src/gate.py` to cache block-K representations for kept frames, avoiding re-computation of the early ViT blocks.
- **Timestamp prompting.** Inject frame timestamps into the prompt, so the model knows how much time the gaps between kept frames span. This matters most for backward questions, which lost the most under aggressive pruning (§5).
- **K-block ablation.** Sweep K ∈ {1, 3, 6} to measure the trade-off between gating quality and gate cost.
- **Longer window (`--window 32`).** In a 16 s window almost any 9 frames are near-duplicates, which may explain why frame *selection* made little difference. A longer window should separate gate from uniform more clearly.
