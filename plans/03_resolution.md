# 03 · Low-resolution gating, high-resolution kept frames

## Problem
- `src/run.py` resizes every frame to **336 × 336** (`--size 336`). That was our choice in iteration 1, to fit 33 frames on a T4; it is not a model limit.
- The cost:
  - **Text and fine detail are unreadable.** Many failures were on-screen text: menu names, "1.1 GB", subscription prices.
  - **Phone videos are portrait (9:16)**, so squashing them into a square distorts them.
- **Qwen2-VL / Qwen2.5-VL / Omni support native resolution and aspect ratio.** They just use more 14-px patches. The 2B/3B models are trained to read documents and screens, so higher resolution should help a lot.

## Cost of higher resolution
- Doubling the side length gives 4× the patches. ViT attention cost grows roughly with patches² (full-attention blocks), and the language model gets 4× the tokens per frame.
- 33 frames at 672 px would be about 10k image tokens, too much for a T4.

## Idea: split the two jobs
1. **The gate runs at low resolution** (e.g. 336 or even 224). It only needs to decide "did something change?", so this stays cheap.
2. **Kept frames are encoded at high resolution, in native aspect ratio** (e.g. longest side 672–896, capped by a total-pixel budget).

With 4–6 kept frames, the total tokens are similar to 33 frames at 336 px. The savings from gating get spent on **readability**, which turns compute savings into accuracy: a much stronger result for the project.

## Implementation notes
- `src/data.py` `read_window()`: return the original-size frames as well as the resized ones (or re-read only the kept indices at high resolution).
- `src/run.py`: gate on the small frames, then pass the kept *large* frames to the processor. Set a pixel budget (Qwen processors accept `min_pixels` / `max_pixels`).
- **Fairness:** `uniform` and `pixel` must also get high-resolution versions of *their* frames, and `full` must stay the same as before (or also be tested at high resolution with fewer frames), so the comparison stays matched.
- **Report:** tokens, latency and accuracy per resolution setting. Also hand-tag "text-reading" questions, to show where resolution matters.
