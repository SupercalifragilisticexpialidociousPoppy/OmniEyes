# 01 · Move to an Omni model with audio

**Status: steps 1–4 done and evaluated in iteration 3 (`reports/Iteration3_Report.md`).**
- `src/run.py` loads Qwen2.5-Omni-3B (Thinker only) by default.
- `src/data.py` `read_audio_window()` reads the causal audio with ffmpeg.
- The audio is passed as its own clip (`use_audio_in_video=False`).
- `src/gate.py` finds the Thinker's vision encoder.

Still open: time alignment (step 5), audio-triggered keeps (step 6), K = 8 test.

## Why
- The proposal targets omni models. Qwen2-VL can't hear.
- Several iteration 2 failures needed audio: an inner monologue, a narrator's explanation, a spoken "Can I hold you?".
- Questions like these are unanswerable for any condition, which pushes all scores to the floor and hides differences between methods.

## Model choice
- **Qwen2.5-Omni-3B** is the realistic target on a Colab T4: fp16, with the speech-output part disabled.
- **7B** likely needs 4-bit quantisation on a T4. **Qwen3-Omni** is too large for a T4.
- How the model is built:
  - The **Thinker** reads video, audio and text and writes text. That's the only part we need.
  - The **Talker** generates speech. Disable it to save several GB.
  - The **audio encoder** is Whisper-style (from Qwen2-Audio): 16 kHz mono input, roughly 25 tokens per second of audio.
  - **TMRoPE** interleaves audio and video tokens in time order (about 2 s chunks), using real timestamps, so the model knows which sound goes with which frames.

> **Verify before coding:** these details are from memory. Check them against the Qwen2.5-Omni model card and `transformers` docs: class names (`Qwen2_5OmniForConditionalGeneration` / Thinker-only class, `Qwen2_5OmniProcessor`), the `qwen-omni-utils` helper (`process_mm_info`, `use_audio_in_video`), how to disable the Talker, and fp16 behaviour on a T4.

## Implementation steps
1. **Audio acquisition (`src/data.py`).** Add `read_audio_window(video_path, t_end, window_s, sr=16000)`. It extracts the audio for exactly `[t − window, t]`, mono at 16 kHz (ffmpeg or librosa). **Causal**: never include audio after `t`. Return the waveform and its start time.
2. **Don't gate the audio.** 16 s of audio is about 400 tokens, which is cheap, and dropping audio loses speech. The gate only prunes video, the expensive part.
3. **Model loading and prompting (`src/run.py`).**
   - Load the Thinker only.
   - Build the conversation with the kept frames + the audio clip + the question.
   - Pass our own audio clip explicitly, rather than letting the helper pull audio from the whole video file. Our clip is already causal and windowed.
4. **Gate on Omni's vision encoder (`src/gate.py`).**
   - Point `EarlyLayerGate` at the Thinker's ViT (`model.thinker.visual` or similar; `_vision_blocks()` should find `.blocks`).
   - Omni's ViT (Qwen2.5-VL design) uses **window attention** in most blocks and **full attention every 8th block** (indices to verify, e.g. 7/15/23/31). So blocks 1–3 only see local neighbourhoods, and the features are even more texture-like than with Qwen2-VL.
   - Test **K = 8**, the first block after a full-attention layer, against K = 3.
5. **Time alignment of pruned frames (main risk).** TMRoPE assumes evenly spaced frames. After gating, the kept frames' temporal positions must equal their *real* times, or audio will be paired with the wrong pictures. See `04_feeding_frames.md`, item 1. **Prototype this first, on one video.**
6. **Audio-triggered keeps (novel contribution).** When speech starts or a loud event happens (simple energy or voice-activity detection on the clip), keep the nearest frame even if the picture barely changed. That makes the gate genuinely multimodal, and it fits the "omni" framing.

## Measurements to add
- Audio tokens and audio-encoder time per question.
- An ablation: video-only vs. video + audio, at the same frame budgets.
- Per-question flag for "needs audio" (hand-tagged on a subset), to show where audio helps.

## Risks
- T4 memory with 3B and 33 frames + audio. Fallback: fewer frames for `full`, or a 4-bit model.
- fp16 numerical issues on a T4 (no bf16).
- `transformers` version churn: the iteration 2 `resume_from_k` crash was caused by exactly this (see `06_engineering_fixes.md`).
