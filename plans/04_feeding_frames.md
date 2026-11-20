# 04 · How kept frames reach the language model

## How it works now
1. Each frame goes through the ViT, and the merger squashes each 2 × 2 group of patches into one token.
2. Qwen also fuses **every 2 consecutive input frames into one set of tokens** (temporal patch size 2). So 2 frames become 144 tokens at 336 px.
3. Every token gets a 3-part position stamp (M-RoPE): **when** (time), and **where** (row, column).
4. All image tokens plus the question go into the language model. Its attention compares tokens across positions and times. "Something changed here" is inferred from tokens with similar *where* but different *when*.

This part isn't a black box we must accept: two things about *how we feed it* are wrong after gating, and both are fixable without changing the model.

## Ideas

### 1. Put real time into the position stamps
- **Qwen2-VL** assigns time by frame index (1, 2, 3, …). Kept frames at 0 s, 10 s and 16 s look *evenly spaced* to it.
- **Qwen2.5-VL / Omni** build real time into the temporal position (absolute-time-aligned, e.g. via seconds-per-grid values). They assume uniform spacing within one video, though.
- **Plan:** for pruned frames, set each kept frame's temporal position to its real timestamp. This likely means overriding the position-index computation (`get_rope_index` or equivalent) for video inputs. The same work also aligns audio with video in Omni (`01_omni_migration.md`, step 5).
- **Fallback if that's too invasive:** pass each contiguous kept run as a separate short video, with its own correct start time.

### 2. Don't fuse frames across time gaps
- After gating, a "pair" can be two frames 10 s apart, and the model blends them into one token set as if they were 0.5 s apart.
- **Fix:** feed each kept frame **paired with itself**, the way single images are handled. That's 144 tokens per kept frame instead of per pair, so it costs 2× tokens per kept frame, but each frame stays clean.
- Run it as an ablation: paired-with-self vs. naive pairing, at the same kept frames.

### 3. Rethink timestamps in the prompt text
- Iteration 2 added `F0=634.0s, …` to the prompt. 127 forward answers then answered *with* a timestamp ("at F1", "at F14.5s") instead of content.
- If item 1 works, drop the text timestamps.
- Otherwise, use relative offsets (`-16.0s … 0.0s`) and tell the model to use times only for ordering, not as the answer. Test with vs. without on the same questions.

### 4. (Long shot) a different model
Streaming-specific research models and other omni models (e.g. MiniCPM-o) exist. The proposal targets omni models, though, and fixing how we feed Qwen2.5-Omni (items 1–3) is cheaper and easier to defend. Only revisit this if Omni can't be made to work on our hardware.
