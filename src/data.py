"""PhoStream loader. The annotation schema is not documented on the dataset card,
so field names are resolved from candidate lists. Run
`python src/data.py PhoStream/dataset/phostream_annotations.json` (from the repository root)
first to print the detected schema and fix KEYS below if needed."""
import json, os, sys, glob
import numpy as np
from PIL import Image

KEYS = {
    "question": ["question", "Question", "query", "user_query"],
    "answer": ["answer", "Answer", "gt_answer", "ground_truth", "response"],
    "t_question": ["timestamp_question", "Timestamp Question", "question_time", "timestamp", "time"],
    "task": ["task", "temporal_scope", "type", "task_type", "scope", "time_type"],
    "video": ["video", "video_path", "video_id", "path", "video_name"],
}

def _get(rec, key):
    for k in KEYS[key]:
        if k in rec:
            return rec[k]
    return None


def _to_sec(t):
    if isinstance(t, (int, float)):
        return float(t)
    parts = [float(p) for p in str(t).split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s

def load_annotations(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)) if os.path.isdir(path) else [path]
    recs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            d = d.get("data", d.get("annotations", list(d.values())))
        recs.extend(d if isinstance(d, list) else [])
    
    # NEW BLOCK: Flatten the nested verified_responses
    flat_recs = []
    for r in recs:
        if "verified_responses" in r:
            for vr in r["verified_responses"]:
                flat_item = vr.copy()
                flat_item["video_path"] = r.get("video_path")
                flat_recs.append(flat_item)
        else:
            flat_recs.append(r)

    out = []
    # Process the flattened list
    for r in flat_recs:
        if not isinstance(r, dict):
            continue
        q, a, t, task, v = (_get(r, k) for k in ("question", "answer", "t_question", "task", "video"))
        if q is None or a is None or t is None or v is None:
            continue
        out.append(dict(question=q, answer=str(a), t=_to_sec(t), task=str(task or "unknown").lower(), video=str(v), raw=r))
    return out


def read_window(video_path, t_end, window_s, fps=2.0, size=336):
    """Return (PIL frames, numpy frames, timestamps_seconds) sampled at `fps` in
    [t_end-window_s, t_end]. Causal: nothing after t_end.
    Timestamps are absolute seconds from video start for each frame."""
    import decord
    vr = decord.VideoReader(video_path)
    vfps = vr.get_avg_fps()
    t0 = max(0.0, t_end - window_s)
    ts = np.arange(t0, t_end + 1e-6, 1.0 / fps)
    idx = np.clip((ts * vfps).round().astype(int), 0, len(vr) - 1)
    arr = vr.get_batch(idx.tolist()).asnumpy()
    frames = [Image.fromarray(x).resize((size, size)) for x in arr]
    return frames, [np.asarray(f) for f in frames], ts.tolist()


def read_audio_window(video_path, t_end, window_s, sr=16000):
    """Return the mono float32 waveform of [t_end-window_s, t_end] at `sr` Hz, or None if
    the video has no audio track. Causal: nothing after t_end. Needs ffmpeg on PATH."""
    import subprocess
    t0 = max(0.0, t_end - window_s)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t0:.3f}", "-t", f"{t_end - t0:.3f}",
           "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    wav = np.frombuffer(res.stdout, dtype=np.float32)
    if res.returncode != 0 or wav.size < sr // 10:  # no audio stream, or under 0.1 s decoded
        return None
    return wav.copy()


if __name__ == "__main__":
    p = sys.argv[1]
    files = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)) if os.path.isdir(p) else [p]
    print("json files:", files[:10])
    with open(files[0]) as fh:
        d = json.load(fh)
    first = d[0] if isinstance(d, list) else d
    print("first record keys / sample:\n", json.dumps(first, indent=1, ensure_ascii=False)[:1500])
    recs = load_annotations(p)
    print(f"parsed {len(recs)} records; tasks:", {r['task'] for r in recs})
