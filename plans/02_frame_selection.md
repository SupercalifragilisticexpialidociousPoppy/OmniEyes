# 02 · Which frames the gate keeps and sends

**Status: ideas 2 and 3 implemented and evaluated in iteration 3** (`reports/Iteration3_Report.md` §6: the gate now matches `full` on instant questions; gate, uniform and pixel get different frames).
- `--keep_current` always sends the newest frame, in every condition.
- `--min_keep N` tops the gate up to N frames by farthest-point selection on its block-K features. The baselines get the same budget.
- Both are off by default, so iterations 1–2 stay reproducible. See `select_by_gate()` in `src/gate.py`.

## Problems seen in iteration 2
- `select_by_gate` (`src/gate.py`) always keeps **frame 0, the oldest frame** (16 s before the question). It keeps a later frame only if it differs from the last kept one by more than τ.
- In 30–40% of instant + backward questions the gate kept **only frame 0**. The current frame (the question moment) was kept in just 3–16 of 76 questions.
- With 1 kept frame, `uniform` and `pixel` also return `[0]`, so all three methods gave identical inputs and answers. The comparison can't separate them.
- The similarity is the **mean over all 576 patch tokens**, so small local changes (a number on a screen, a moving cursor) barely move it.

## Ideas

### 1. Separate the anchor from the frame that gets sent
Discussed alternative: update the reference to every new frame. That compares consecutive frames, so slow, smooth change (panning, gradients, a slowly moving object) never accumulates enough to trigger a keep, and no history survives.

**Better:** keep two roles apart.
- **Anchor (what we compare against):** stays the last kept frame, as now, so slow drift still adds up.
- **Frame we send:** when a run of similar frames ends, send the **newest frame of that run**, not the anchor. It carries the latest state (e.g. updated text).

### 2. Always include the current frame
The frame at time `t` is always sent, whatever the gate says. Instant questions are about now.

### 3. Minimum frame budget
At least ~4 frames per window, so gate, uniform and pixel actually choose *different* frames and the comparison is informative. Report the compression ratio honestly alongside it.

### 4. Spatially aware similarity (teammate's idea)
Compare on the most-changed tokens instead of the mean:
$$\text{sim}_{\text{top-}k}(f_i, f_{\text{ref}}) = \frac{1}{k} \sum_{p \in \text{Top-}k} \cos(f_{i,p}, f_{\text{ref},p})$$
- **Risk:** handheld jitter, autofocus and compression noise can trigger false keeps.
- **Mitigation:** require a minimum cluster of adjacent changed tokens (e.g. ≥ 4).

### 5. Choose τ by target frame count
- Iteration 2 kept far fewer frames than iteration 1 at the same τ (3.2 vs 9.5 at τ = 0.90), on a different question mix.
- Pick τ values that give e.g. ~8 and ~16 frames, on calibration data from the same archives and question types.
- **Fix `src/calibrate.py` first:** it only gets similarities for the first τ (`src/run.py` logs `sims_log[taus[0]]`), and its "recommended" text is hard-coded. Either log similarities for every τ, or re-run `select_by_gate` on stored features.

### 6. Gate depth (K) sweep
- K ∈ {1, 3, 6} on Qwen2-VL, or {3, 8} on Omni (see `01_omni_migration.md`, step 4).
- Trade-off: K = 1 is cheaper (~5% of a full pass?) but close to pixel statistics. A deeper K costs more per frame (≈ 16% at K = 3, ≈ 25% at K = 6).

### 7. Investigate why block-3 similarity stays so high
Plot the similarity distribution per video type with `calibrate.py`. If early-layer mean cosine is > 0.9 for nearly everything, the features may be dominated by components shared by all images. Normalising or centring the features (subtracting a per-video mean) before the cosine may help.
