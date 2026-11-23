"""Plain Qwen2.5-Omni baseline: video and audio fed the way the model was trained.

Same questions, window, frames and prompt as src/run.py (same --seed, --n and --tasks give the
same questions, with the same qid), but with one difference in how the input is packed:

  run.py         video tokens, then the audio clip as a separate input (use_audio_in_video=False)
  run_plain.py   the audio interleaved into the video in 2 s chunks, on one time axis
                 (use_audio_in_video=True), as in Qwen2.5-Omni's own examples

No gating: every frame in the window is sent. Rows have cond = "plain".
The audio is still the causal clip [t - window, t], never the video file's whole track. A check
on every prompt stops the run if the audio was not interleaved, or if it has more tokens than
the clip can give (which would mean the processor read the file's audio instead).

Usage (from the repository root, same options as run.py):
  python src/run_plain.py --ann PhoStream/dataset/phostream_annotations.json --video_root videos \
      --n 150 --tasks instant backward --out results_plain.jsonl
"""
import argparse, collections, inspect, json, os, random, shutil, time
import numpy as np
import torch

from gate import EarlyLayerGate
from data import load_annotations, read_window, read_audio_window
from run import PROMPT, index_videos, resolve_video, load_model

AUDIO_TOKENS_PER_S = 25       # Omni's audio encoder: about 25 tokens per second of audio


def _accepts(proc, name):
    """True if any __call__ in the processor's class hierarchy has a parameter called name."""
    for cls in type(proc).__mro__:
        fn = cls.__dict__.get("__call__")
        if fn is not None:
            try:
                if name in inspect.signature(fn).parameters:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def check_interleaved(proc, ids, max_audio_tokens):
    """Stop if the audio was not interleaved with the video, or has too many tokens."""
    tok = proc.tokenizer
    a_id = tok.convert_tokens_to_ids(getattr(proc, "audio_token", "<|AUDIO|>"))
    v_id = tok.convert_tokens_to_ids(getattr(proc, "video_token", "<|VIDEO|>"))
    ids = ids.tolist()
    a_pos = [i for i, x in enumerate(ids) if x == a_id]
    v_pos = [i for i, x in enumerate(ids) if x == v_id]
    if not a_pos or not v_pos:
        raise SystemExit(f"prompt has {len(a_pos)} audio and {len(v_pos)} video tokens; expected both")
    if not (v_pos[0] < a_pos[0] < v_pos[-1]):
        raise SystemExit("audio tokens are not interleaved with the video tokens: this transformers "
                         "version handles use_audio_in_video differently")
    if len(a_pos) > max_audio_tokens:
        raise SystemExit(f"{len(a_pos)} audio tokens, more than the causal clip allows ({max_audio_tokens}): "
                         "the processor probably read the video file's whole audio track")
    return len(a_pos)


@torch.no_grad()
def answer_plain(model, proc, frames, question, timestamps, audio, max_audio_tokens, max_new=64):
    fr, ts = list(frames), list(timestamps)
    if len(fr) % 2:                       # temporal patch = 2 frames
        fr.append(fr[-1]); ts.append(ts[-1])
    vid = np.stack([np.asarray(f) for f in fr])
    ts_str = ", ".join(f"F{i}={t:.1f}s" for i, t in enumerate(ts[:len(frames)]))
    has_audio = audio is not None
    text_prompt = PROMPT.format(q=question, ts=ts_str,
                                audio_note="You also hear the audio of the same time span.\n" if has_audio else "",
                                perceived="seen and heard" if has_audio else "seen")
    # One video item: with use_audio_in_video=True its audio comes with it, no separate audio item.
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text_prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tok = proc.tokenizer
    stop_ids = sorted({i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
                       if isinstance(i, int) and i >= 0 and i != tok.unk_token_id})
    gen_kwargs = {"eos_token_id": stop_ids, "use_audio_in_video": has_audio,
                  "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]}
    proc_kwargs = {"use_audio_in_video": has_audio}
    if has_audio:
        proc_kwargs["audio"] = [audio]
        # Newer transformers (OmniModalProcessorMixin): say that this audio belongs to videos[0].
        # With one audio per slot, the processor uses our causal clip as given and never reads the
        # file. Older versions don't know the argument and pair one audio with one video themselves.
        if _accepts(proc, "audio_from_video_indices"):
            proc_kwargs["audio_from_video_indices"] = [0]
    inp = proc(text=[text], videos=[vid], return_tensors="pt", padding=True, **proc_kwargs)
    n_audio_tok = check_interleaved(proc, inp.input_ids[0], max_audio_tokens) if has_audio else 0
    inp = inp.to(model.device).to(model.dtype)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, **gen_kwargs)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new_ids = out[:, inp.input_ids.shape[1]:]
    ans = proc.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return ans, dt, int(inp.input_ids.shape[1]), int(new_ids.shape[1]), n_audio_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--out", default="results_plain.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--window", type=float, default=16.0)
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--tasks", nargs="+", default=["instant", "backward"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fps = 2.0   # the Omni processor times video frames at 2 fps; frames must be sampled at that rate

    if "omni" not in a.model.lower():
        raise SystemExit("run_plain.py needs a Qwen2.5-Omni checkpoint")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found: it is needed to read the audio")

    # Same question selection as run.py, so the qids match its output rows.
    recs = [r for r in load_annotations(a.ann) if any(t in r["task"] for t in a.tasks)]
    index = index_videos(a.video_root)
    for r in recs:
        r["video_file"] = resolve_video(a.video_root, r["video"], index)
    n_missing = sum(r["video_file"] is None for r in recs)
    recs = [r for r in recs if r["video_file"] is not None]
    by_src = collections.Counter(r["video"].split("/")[0] for r in recs)
    print(f"{len(index)} video files; {len(recs)} eligible QAs have their video ({dict(by_src)}), {n_missing} skipped")
    if not recs:
        raise SystemExit("no question has its video under --video_root")
    random.Random(a.seed).shuffle(recs)

    model, proc = load_model(a.model)
    sr = proc.feature_extractor.sampling_rate
    timer = EarlyLayerGate(model, k=1)            # only for timing a full vision pass
    max_audio_tokens = int(AUDIO_TOKENS_PER_S * a.window) + 8
    print(f"model: {a.model} | plain: video + audio interleaved (use_audio_in_video=True), {sr} Hz")

    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["qid"] for l in open(a.out, encoding="utf-8") if l.strip()}

    fout = open(a.out, "a", encoding="utf-8")
    n_run = 0
    for r in recs:
        if n_run >= a.n:
            break
        qid = f"{r['video']}|{r['t']}|{r['question'][:40]}"
        if qid in done:
            n_run += 1
            continue
        vp = r["video_file"]
        try:
            frames, _, frame_ts = read_window(vp, r["t"], a.window, fps, a.size)
        except Exception as e:
            print("skip", vp, e); continue
        if len(frames) < 2:
            continue
        audio = read_audio_window(vp, r["t"], a.window, sr)

        # Vision cost, measured like run.py: mean full pass over up to 3 frames, times N frames
        ips = [proc.image_processor(images=[f], return_tensors="pt") for f in frames[:3]]
        full_t = float(np.mean([timer.full_time(ip["pixel_values"], ip["image_grid_thw"]) for ip in ips]))
        N = len(frames)

        pred, lat, ntok, ngen, n_audio_tok = answer_plain(model, proc, frames, r["question"], frame_ts,
                                                          audio, max_audio_tokens)
        row = dict(qid=qid, video=r["video"], video_file=vp, t=r["t"], task=r["task"], question=r["question"],
                   answer=r["answer"], n_frames=N, model=a.model, cond="plain", kept=list(range(N)), n_kept=N,
                   audio_s=(len(audio) / sr) if audio is not None else 0.0, audio_tokens=n_audio_tok,
                   full_vit_ms=full_t * 1e3, vit_ms=N * full_t * 1e3, vit_ms_actual=N * full_t * 1e3,
                   llm_latency_s=lat, prompt_tokens=ntok, gen_tokens=ngen, pred=pred)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
        n_run += 1
        print(f"[{n_run}/{a.n}] N={N} audio {row['audio_s']:.1f}s ({n_audio_tok} tokens) | {pred[:80]}")


if __name__ == "__main__":
    main()
