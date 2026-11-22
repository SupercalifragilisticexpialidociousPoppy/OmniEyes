"""OmniEyes experiment on PhoStream (Instant + Backward + Forward questions).

For each QA: take the causal window [t-W, t] at 2 Hz, compute block-K gate
features for every frame, then answer under these conditions:
  full            all frames in window (upper bound)
  gate@tau        frames kept by the early-layer gate
  uniform@tau     uniform subsampling, same #frames as gate@tau
  pixel@tau       pixel-difference gate, same #frames as gate@tau

Iteration 2 updates:
  - Timestamp prompting (per-frame time injection into the LLM prompt)
  - Vision feature resumption timing (resume_ms) and ideal vs actual vit_ms accounting
    (iteration 3: resume_ms is measured inside a real full pass, split at block K)
  - Multi-frame averaging for baseline full_vit_ms
  - Expanded tasks: instant, backward, and forward
  - Granular resume-from-interrupt tracking (qid, cond)

Iteration 3 updates:
  - Default model is Qwen2.5-Omni-3B (Thinker only: text answers, no speech output).
    Qwen2-VL checkpoints still work via --model.
  - With an Omni model, the audio of the same causal window [t-W, t] is passed with the
    frames (never gated, never after t). --no_audio turns it off for a video-only ablation.
  - Frame-selection fixes (off by default, so iterations 1-2 stay reproducible):
    --keep_current always sends the newest frame (the question moment) in every condition;
    --min_keep N tops the gate up to at least N frames, so the baselines get the same budget.
"""
import argparse, collections, glob, json, os, random, re, shutil, time
import numpy as np
import torch
from transformers import AutoProcessor

from gate import EarlyLayerGate, select_by_gate, select_uniform, select_by_pixel_diff
from data import load_annotations, read_window, read_audio_window

PROMPT = ("You are watching a live phone camera stream. The frames shown end at the current moment.\n"
          "{audio_note}"
          "Frame timestamps (seconds): {ts}\n"
          "Answer the user's question directly and concisely, based only on what you have {perceived}. "
          "Do not ask follow-up questions.\nQuestion: {q}")
# No system prompt on purpose: Omni's default "virtual human" system prompt is only needed for
# speech output, and in the iteration 3 smoke test it made answers hedging and chatty.


def _norm_path(p):
    """Annotation paths omit the '_2fps' suffix that the extracted archive folders carry
    (e.g. 'task-x_reencoded/' vs 'task-x_reencoded_2fps/'), so drop it from every folder name."""
    parts = p.replace("\\", "/").strip("/").split("/")
    return "/".join([re.sub(r"_2fps$", "", q) for q in parts[:-1]] + parts[-1:])


def index_videos(root):
    """All .mp4 files under root, as root-relative paths with '/' separators."""
    files = glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True)
    return [os.path.relpath(f, root).replace(os.sep, "/") for f in files]


def resolve_video(root, v, index):
    """Find the file for annotation path v, or return None. Never guesses.

    Every PhoStream video is named output_1.mp4, so matching on the file name alone picks
    an unrelated video. (Iterations 1-3 did exactly that: every question was answered from
    the same video.) A file only matches if the end of its path equals the end of the
    annotation path, including at least the video's own folder, and exactly one file does.
    Folder names are compared without their '_2fps' suffix (see _norm_path). That tolerates
    the archives' folder naming, but never a different video.
    """
    v = v.replace("\\", "/").strip("/")
    for cand in (v, os.path.join(root, v)):
        if os.path.isfile(cand):
            return cand
    parts = _norm_path(v).split("/")
    norm_index = [(f, _norm_path(f)) for f in index]
    for k in range(0, len(parts) - 1):          # drop leading folders, keep >= folder + file
        suffix = "/".join(parts[k:])
        hits = [f for f, nf in norm_index if nf == suffix or nf.endswith("/" + suffix)]
        if len(hits) == 1:
            return os.path.join(root, hits[0])
        if len(hits) > 1:
            return None                         # ambiguous: refuse to guess
    return None


def is_omni(model_name):
    return "omni" in model_name.lower()


def load_model(model_name):
    """Load the VLM and its processor. Omni: Thinker only (text output, no Talker)."""
    if is_omni(model_name):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration as cls
    else:
        from transformers import Qwen2VLForConditionalGeneration as cls
    model = cls.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(model_name)
    return model, proc


@torch.no_grad()
def answer(model, proc, frames, question, timestamps=None, audio=None, omni=False, max_new=64):
    fr = list(frames)
    ts = list(timestamps) if timestamps is not None else None
    if len(fr) % 2:                       # Qwen2-VL temporal patch = 2 frames
        fr.append(fr[-1])
        if ts is not None and len(ts) > 0:
            ts.append(ts[-1])
    vid = np.stack([np.asarray(f) for f in fr])
    if ts is not None and len(ts) > 0:
        ts_str = ", ".join(f"F{i}={t:.1f}s" for i, t in enumerate(ts[:len(frames)]))
    else:
        ts_str = "not available"
    audio_note = "You also hear the audio of the same time span.\n" if audio is not None else ""
    content = [{"type": "video"}]
    if audio is not None:
        content.append({"type": "audio"})
    content.append({"type": "text", "text": PROMPT.format(q=question, ts=ts_str, audio_note=audio_note,
                                                      perceived="seen and heard" if audio is not None else "seen")})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Stop at the end of the assistant's turn. Without this, the Omni Thinker keeps writing
    # invented "Human: ..." turns until max_new_tokens.
    tok = proc.tokenizer
    stop_ids = sorted({i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id)
                       if isinstance(i, int) and i >= 0 and i != tok.unk_token_id})
    proc_kwargs = {}
    gen_kwargs = {"eos_token_id": stop_ids, "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]}
    if audio is not None:
        proc_kwargs["audio"] = [audio]
    if omni:
        # audio is passed as its own clip, not extracted from the video file
        proc_kwargs["use_audio_in_video"] = False
        gen_kwargs["use_audio_in_video"] = False
    inp = proc(text=[text], videos=[vid], return_tensors="pt", padding=True, **proc_kwargs)
    inp = inp.to(model.device).to(model.dtype)   # casts float inputs (pixels, audio features) to fp16
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, **gen_kwargs)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new_ids = out[:, inp.input_ids.shape[1]:]
    ans = proc.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return ans, dt, int(inp.input_ids.shape[1]), int(new_ids.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B",
                    help="Qwen2.5-Omni (video + audio) or Qwen2-VL (video only) checkpoint")
    ap.add_argument("--no_audio", action="store_true",
                    help="Omni only: answer from frames alone (video-only ablation)")
    ap.add_argument("--keep_current", action="store_true",
                    help="always include the newest frame (the question moment) in every condition")
    ap.add_argument("--min_keep", type=int, default=1,
                    help="gate keeps at least this many frames (topped up by its own features)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.88, 0.90, 0.93, 0.95])
    ap.add_argument("--window", type=float, default=16.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--tasks", nargs="+", default=["instant", "backward", "forward"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate_only", action="store_true", help="only log gate similarities, no LLM calls")
    a = ap.parse_args()

    recs = [r for r in load_annotations(a.ann) if any(t in r["task"] for t in a.tasks)]
    # Keep only questions whose own video is on disk, before sampling, so --n questions are
    # really answered from their videos. Which questions are sampled therefore depends on
    # which archives are downloaded: download all of them for the full question mix.
    index = index_videos(a.video_root)
    for r in recs:
        r["video_file"] = resolve_video(a.video_root, r["video"], index)
    missing = [r for r in recs if r["video_file"] is None]
    recs = [r for r in recs if r["video_file"] is not None]
    by_src = collections.Counter(r["video"].split("/")[0] for r in recs)
    print(f"{len(index)} video files under {a.video_root}; {len(recs)} eligible QAs have their video "
          f"({dict(by_src)}), {len(missing)} skipped (video not downloaded)")
    if missing:
        print(f"  e.g. no file for: {missing[0]['video']}")
    if not recs:
        raise SystemExit("no question has its video under --video_root: check the archive was extracted there")
    random.Random(a.seed).shuffle(recs)
    print(f"running {min(a.n, len(recs))}")

    omni = is_omni(a.model)
    use_audio = omni and not a.no_audio
    if use_audio and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found: install it, or pass --no_audio for a video-only run")
    model, proc = load_model(a.model)
    audio_sr = proc.feature_extractor.sampling_rate if use_audio else None
    gate = EarlyLayerGate(model, k=a.k)
    print(f"model: {a.model} | audio: {'on, %d Hz' % audio_sr if use_audio else 'off'} | "
          f"vision blocks: {gate.n_blocks}, gate at K={a.k}")

    done_conds = set()
    cond_counts = {}
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            key = d["qid"]
            done_conds.add((key, d.get("cond")))
            cond_counts[key] = cond_counts.get(key, 0) + 1

    n_expected_conds = 1 + 3 * len(a.taus)
    done_qids = {k for k, count in cond_counts.items() if count >= n_expected_conds}

    fout = open(a.out, "a", encoding="utf-8")
    n_run = 0
    for qi, r in enumerate(recs):
        if n_run >= a.n:
            break
        qid = f"{r['video']}|{r['t']}|{r['question'][:40]}"
        if a.calibrate_only:
            if (qid, "calib") in done_conds:
                n_run += 1
                continue
        else:
            if qid in done_qids:
                n_run += 1
                continue

        vp = r["video_file"]
        try:
            frames, frames_np, frame_ts = read_window(vp, r["t"], a.window, a.fps, a.size)
        except Exception as e:
            print("skip", vp, e); continue
        if len(frames) < 2:
            continue
        # causal audio for the same window; it is not gated (cheap, and dropping it loses speech)
        audio = read_audio_window(vp, r["t"], a.window, audio_sr) if use_audio else None

        # --- gate pass: block-K features for every frame (this is the per-frame always-on cost)
        feats, gate_t, ip_cache = [], [], []
        for f in frames:
            ip = proc.image_processor(images=[f], return_tensors="pt")
            ft, dt = gate.early_features(ip["pixel_values"], ip["image_grid_thw"])
            feats.append(ft); gate_t.append(dt)
            ip_cache.append(ip)

        # Full ViT time and the "resume" part of it (blocks K+1..N + merger), averaged over
        # up to 3 frames. Both come from one real pass, split at block K by a timing hook.
        sample_ips = ip_cache[:min(3, len(ip_cache))]
        splits = [gate.split_time(sip["pixel_values"], sip["image_grid_thw"]) for sip in sample_ips]
        full_t = float(np.mean([f for f, _ in splits]))
        resume_t = float(np.mean([r for _, r in splits]))

        g_mean = float(np.mean(gate_t[1:] if len(gate_t) > 1 else gate_t))
        N = len(frames)

        conds = {"full": list(range(N))}
        sims_log = {}
        for tau in a.taus:
            kept, sims = select_by_gate(feats, tau, min_keep=a.min_keep, keep_current=a.keep_current)
            sims_log[tau] = sims
            conds[f"gate@{tau}"] = kept
            conds[f"uniform@{tau}"] = select_uniform(N, len(kept), keep_current=a.keep_current)
            conds[f"pixel@{tau}"] = select_by_pixel_diff(frames_np, len(kept), keep_current=a.keep_current)

        base = dict(qid=qid, video=r["video"], video_file=vp, t=r["t"], task=r["task"], question=r["question"],
                    answer=r["answer"], n_frames=N, k=a.k, gate_ms=g_mean * 1e3,
                    full_vit_ms=full_t * 1e3, resume_ms=resume_t * 1e3, model=a.model,
                    keep_current=a.keep_current, min_keep=a.min_keep,
                    audio_s=(len(audio) / audio_sr) if audio is not None else 0.0)

        if a.calibrate_only:
            if (qid, "calib") not in done_conds:
                fout.write(json.dumps({**base, "cond": "calib", "sims": sims_log[a.taus[0]]}, ensure_ascii=False) + "\n")
                fout.flush()
                done_conds.add((qid, "calib"))
            n_run += 1
            continue

        for name, idx in conds.items():
            if (qid, name) in done_conds:
                continue
            pred, lat, ntok, ngen = answer(model, proc, [frames[i] for i in idx], r["question"],
                                     timestamps=[frame_ts[i] for i in idx], audio=audio, omni=omni)
            m = len(idx)
            if name.startswith("gate"):
                # IDEAL cost: gate runs on all N; kept frames resume from block K
                vit_ms_ideal = N * g_mean * 1e3 + m * resume_t * 1e3
                # ACTUAL cost: gate runs on all N; full re-run on kept frames
                vit_ms_actual = N * g_mean * 1e3 + m * full_t * 1e3
            else:
                vit_ms_ideal = m * full_t * 1e3
                vit_ms_actual = m * full_t * 1e3

            record = {**base, "cond": name, "kept": idx, "n_kept": m,
                      "vit_ms": vit_ms_ideal, "vit_ms_actual": vit_ms_actual,
                      "llm_latency_s": lat, "prompt_tokens": ntok, "gen_tokens": ngen, "pred": pred}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done_conds.add((qid, name))

        n_run += 1
        print(f"[{n_run}/{a.n}] N={N} " + " ".join(f"{k}:{len(v)}" for k, v in conds.items() if k.startswith('gate')))


if __name__ == "__main__":
    main()
