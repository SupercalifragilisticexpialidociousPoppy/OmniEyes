# 06 · Engineering fixes (cheap, do first)

## 0. Every question was answered from the same video (iterations 1–3) — fixed, runs must be redone
- **What happened:** all PhoStream videos are named `output_1.mp4`. The old `resolve_video()` tried the annotation path, then fell back to "any file with the same name", taking the first hit. The annotation path never matched the extracted folders, so **every question in iterations 1, 2 and 3 used one and the same (~10-minute) video**.
- **Evidence:** at the same timestamp, different videos always got identical frame selections (19/19, 23/23, 19/19 pairs), and no question after ~615 s got audio (the stand-in video's length).
- **Not affected:** timing and token counts (they depend on frame counts), and the code-level findings (e.g. the gate keeping only the oldest frame).
- **Fix (iteration 3):** `resolve_video()` matches the end of the folder path, never guesses, and returns nothing if the match is missing or ambiguous. `run.py` drops such questions before sampling and logs `video_file` in every row.
- **Done:** all four archives downloaded and both iteration 3 runs repeated (`results/iteration3/`; the bugged run is in `results/iteration3-scrapped/`). Iterations 1–2 reports are left as submitted; `reports/Iteration3_Report.md` §3 explains the effect.


## 1. `resume_from_k` crash and the fake 50 ms
**Status: resolved for timing in iteration 3.** `resume_from_k()` crashed again on Qwen2.5-Omni (`position_embeddings` was None). It was replaced by `EarlyLayerGate.split_time()`: a timing hook marks the end of block K inside a real full pass, so `resume_ms` is measured with the model's own code path, in any `transformers` version. Items 2–3 below still apply if resumed features are ever fed to generation.

- **What happened:** `resume_from_k()` (`src/gate.py`) crashed on the GPU. Newer `transformers` versions pass M-RoPE position information to the vision blocks in a different format: precomputed `position_embeddings` (cos, sin) instead of `rotary_pos_emb`. The Colab copy of the code caught the error and **wrote 0.05 s**, so every iteration 2 row has `resume_ms = 50.0`. The "ideal" ViT cost is therefore fake.
- **The repository does not contain the fix or the fallback.** `src/gate.py` still passes `rotary_pos_emb`, and `src/run.py` has no 50 ms fallback. The code in the repo is not the code that produced `results/iteration2/results_v2_k3.jsonl`.
- **To do:**
  1. Get the teammate's fixed `gate.py` / `run.py` and commit them.
  2. On failure, record **missing** (`null` / NaN), never a plausible number. `src/analyze.py` should then skip missing values and say how many were skipped.
  3. Pin the `transformers` version used for each run, and log it in the output rows.

## 2. Test `resume_from_k` by value, not just shape
`src/test_gate_cpu.py` only checks that the resumed output has the same *shape* as a full pass. Add `torch.allclose(resumed, full, atol=1e-3)` (fp32 on CPU), so the ideal cost describes a pass that gives the correct output.

## 3. Use resumption for real
Once item 2 passes, feed the resumed features into generation. Then the ideal cost becomes the actual cost, removing the duplicate block-1..K pass for kept frames. This requires injecting precomputed vision embeddings into `generate()`, which is fragile across `transformers` versions; keep it behind a flag.

## 4. Calibration logging
`src/run.py --calibrate_only` writes similarities for the first τ only (`sims_log[taus[0]]`). Log them for every τ, or log the features' pairwise similarities, so `src/calibrate.py` can simulate any τ correctly. Remove the hard-coded "recommended" text in `calibrate.py`.

## 5. Timing
- `full_vit_ms` is averaged over the first 3 frames of each window, right after the gate pass. The first call may include warm-up.
- Use the median of ~5 frames, skipping the first.

## 6. Reproducibility
- Record the git commit hash in every output row (or a sidecar file), so results can always be traced to the exact code.
- Keep all Colab edits in the repository: no hot-patching in notebook cells.
