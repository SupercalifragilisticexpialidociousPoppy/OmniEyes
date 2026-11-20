"""H18 iteration-2 experiment on PhoStream (Instant + Backward + Forward questions).

For each QA: take the causal window [t-W, t] at 2 Hz, compute block-K gate
features for every frame, then answer under these conditions:
  full            all frames in window (upper bound)
  gate@tau        frames kept by the early-layer gate
  uniform@tau     uniform subsampling, same #frames as gate@tau
  pixel@tau       pixel-difference gate, same #frames as gate@tau

Iteration 2 updates:
  - Timestamp prompting (per-frame time injection into the LLM prompt)
  - Vision feature resumption timing (resume_ms) and ideal vs actual vit_ms accounting
  - Multi-frame averaging for baseline full_vit_ms
  - Expanded tasks: instant, backward, and forward
  - Granular resume-from-interrupt tracking (qid, cond)
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from gate import EarlyLayerGate, select_by_gate, select_uniform, select_by_pixel_diff
from data import load_annotations, read_window

PROMPT = ("You are watching a live phone camera stream. The frames shown end at the current moment.\n"
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


@torch.no_grad()
def answer(model, proc, frames, question, timestamps=None, max_new=64):
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
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT.format(q=question, ts=ts_str)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], videos=[vid], return_tensors="pt").to(model.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return ans, dt, int(inp.input_ids.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
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

    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    gate = EarlyLayerGate(model, k=a.k)
    print(f"vision blocks: {gate.n_blocks}, gate at K={a.k}")

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

        # --- gate pass: block-K features for every frame (this is the per-frame always-on cost)
        feats, gate_t, ip_cache = [], [], []
        for f in frames:
            ip = proc.image_processor(images=[f], return_tensors="pt")
            ft, dt = gate.early_features(ip["pixel_values"], ip["image_grid_thw"])
            feats.append(ft); gate_t.append(dt)
            ip_cache.append(ip)

        # Average full ViT timing across up to 3 frames (fixing single-frame measurement bug)
        sample_ips = ip_cache[:min(3, len(ip_cache))]
        full_times = [gate.full_time(sip["pixel_values"], sip["image_grid_thw"]) for sip in sample_ips]
        full_t = float(np.mean(full_times))

        # Time the resume path (blocks K..N + merger) for cost accounting
        resume_times = []
        for sip, sfeat in zip(sample_ips, feats[:len(sample_ips)]):
            _, rt = gate.resume_from_k(sfeat, sip["image_grid_thw"])
            resume_times.append(rt)
        resume_t = float(np.mean(resume_times))

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
                    full_vit_ms=full_t * 1e3, resume_ms=resume_t * 1e3)

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
            pred, lat, ntok = answer(model, proc, [frames[i] for i in idx], r["question"],
                                     timestamps=[frame_ts[i] for i in idx])
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
                      "llm_latency_s": lat, "prompt_tokens": ntok, "pred": pred}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done_conds.add((qid, name))

        n_run += 1
        print(f"[{n_run}/{a.n}] N={N} " + " ".join(f"{k}:{len(v)}" for k, v in conds.items() if k.startswith('gate')))


if __name__ == "__main__":
    main()
