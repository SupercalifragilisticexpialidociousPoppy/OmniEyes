"""Compare an audio run with its --no_audio twin (same questions, same settings).

Usage (from the repository root):
  python src/compare_audio.py results/iteration3/judged_v3_k3.jsonl results/iteration3/judged_v3_k3_noaudio.jsonl \
      --csv results/iteration3/tables/audio_vs_noaudio.csv

Questions whose video has no audio track (audio_s == 0) are a built-in control: their two
runs get identical input, so audio-minus-no-audio there must be ~0.
"""
import argparse, csv, json, numpy as np, collections
ap = argparse.ArgumentParser()
ap.add_argument("audio_file"); ap.add_argument("noaudio_file"); ap.add_argument("--csv", default=None)
args = ap.parse_args()
A = {(r['qid'], r['cond']): r for r in map(json.loads, open(args.audio_file))}
N = {(r['qid'], r['cond']): r for r in map(json.loads, open(args.noaudio_file))}
qids = sorted({q for q, _ in A})
has = {q: A[(q, 'full')]['audio_s'] > 0 for q in qids}
task = {q: A[(q, 'full')]['task'] for q in qids}
conds = ['full'] + [f'{m}@{t}' for t in ['0.88', '0.9', '0.93', '0.95'] for m in ['gate', 'uniform', 'pixel']]
rng = np.random.default_rng(0)
def ci(d):
    d = np.array(d, float); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return f"{d.mean():+6.1f} [{lo:+6.1f}, {hi:+6.1f}]{' *' if lo > 0 or hi < 0 else '  '}"
s = lambda r: 100 * r['ent_con']
rows = []
print("=== 1. Audio minus no-audio, same question & condition ===")
n_has = sum(has.values())
print(f"{'condition':<14}{f'has audio track (n={n_has})':>30}{f'no audio track (n={len(qids) - n_has}, control)':>34}")
for c in conds:
    a = [s(A[(q, c)]) - s(N[(q, c)]) for q in qids if has[q]]
    b = [s(A[(q, c)]) - s(N[(q, c)]) for q in qids if not has[q]]
    print(f"{c:<14}{ci(a):>30}{ci(b):>34}")
    rows.append(dict(condition=c, audio_minus_noaudio_with_track=round(np.mean(a), 2), audio_minus_noaudio_no_track=round(np.mean(b), 2)))
print("\nidentical answers audio vs no-audio, no-track questions:",
      sum(A[(q, c)]['pred'] == N[(q, c)]['pred'] for q in qids if not has[q] for c in conds), "/", (len(qids) - sum(has.values())) * len(conds))
print("\n=== 2. Gate minus full, questions WITH an audio track ===")
for name, D in [('with audio', A), ('no audio', N)]:
    for t in ['0.88', '0.9', '0.93', '0.95']:
        d = [s(D[(q, f'gate@{t}')]) - s(D[(q, 'full')]) for q in qids if has[q]]
        print(f"  {name:<11} gate@{t:<5} vs full {ci(d)}")
print("\n=== 3. Gate minus full by task, audio run, all questions ===")
for tk in ['instant', 'backward']:
    for t in ['0.9', '0.95']:
        d = [s(A[(q, f'gate@{t}')]) - s(A[(q, 'full')]) for q in qids if task[q] == tk]
        print(f"  {tk:<9} gate@{t:<5} vs full {ci(d)}  n={len(d)}")
print("\n=== 4. Do the methods now get different frames? ===")
for t in ['0.88', '0.9', '0.93', '0.95']:
    g = lambda m, q: A[(q, f'{m}@{t}')]['kept']
    gp = sum(g('gate', q) == g('pixel', q) for q in qids); gu = sum(g('gate', q) == g('uniform', q) for q in qids)
    same_ans = sum(len({A[(q, f'{m}@{t}')]['pred'] for m in ['gate', 'uniform', 'pixel']}) == 1 for q in qids)
    cur = sum(A[(q, f'gate@{t}')]['n_frames'] - 1 in g('gate', q) for q in qids)
    nk = collections.Counter(len(g('gate', q)) for q in qids)
    print(f"  tau {t}: gate=pixel frames {gp}/{len(qids)}, gate=uniform {gu}/{len(qids)}, all-3 identical answers {same_ans}/{len(qids)}, current frame kept {cur}/{len(qids)}, gate n_kept {dict(sorted(nk.items()))}")
print("\n=== 5. Latency (s) and prompt tokens, mean ===")
for name, D in [('audio', A), ('no audio', N)]:
    for c in ['full', 'gate@0.9']:
        R = [D[(q, c)] for q in qids]
        print(f"  {name:<9} {c:<9} LLM {np.mean([r['llm_latency_s'] for r in R]):.2f}s  tokens {np.mean([r['prompt_tokens'] for r in R]):.0f}  gen {np.mean([r['gen_tokens'] for r in R]):.1f}")
if args.csv:
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
