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
"""
import argparse, glob, json, os, random, shutil, time
import numpy as np
import torch
from transformers import AutoProcessor

from gate import EarlyLayerGate, select_by_gate, select_uniform, select_by_pixel_diff
from data import load_annotations, read_window, read_audio_window

# Omni's documented default system prompt; the model behaves more reliably with it
OMNI_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
               "perceiving auditory and visual inputs, as well as generating text and speech.")

PROMPT = ("You are watching a live phone camera stream. The frames shown end at the current moment.\n"
          "{audio_note}"
          "Frame timestamps (seconds): {ts}\n"
          "Answer the user's question concisely based only on what you have seen.\nQuestion: {q}")


def resolve_video(root, v):
    if os.path.isfile(v):
        return v
    p = os.path.join(root, v)
    if os.path.isfile(p):
        return p
    for ext in ("", ".mp4"):
        hits = glob.glob(os.path.join(root, "**", os.path.basename(v) + ext), recursive=True)
        if hits:
            return hits[0]
    hits = glob.glob(os.path.join(root, "**", v.strip("/"), "**", "*.mp4"), recursive=True)
    return hits[0] if hits else None


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
    content.append({"type": "text", "text": PROMPT.format(q=question, ts=ts_str, audio_note=audio_note)})
    msgs = [{"role": "user", "content": content}]
    if omni:
        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM}]})
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
    random.Random(a.seed).shuffle(recs)
    print(f"{len(recs)} eligible QAs; running {min(a.n, len(recs))}")

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

        vp = resolve_video(a.video_root, r["video"])
        if vp is None:
            continue
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
            kept, sims = select_by_gate(feats, tau)
            sims_log[tau] = sims
            conds[f"gate@{tau}"] = kept
            conds[f"uniform@{tau}"] = select_uniform(N, len(kept))
            conds[f"pixel@{tau}"] = select_by_pixel_diff(frames_np, len(kept))

        base = dict(qid=qid, video=r["video"], t=r["t"], task=r["task"], question=r["question"],
                    answer=r["answer"], n_frames=N, k=a.k, gate_ms=g_mean * 1e3,
                    full_vit_ms=full_t * 1e3, resume_ms=resume_t * 1e3, model=a.model,
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
