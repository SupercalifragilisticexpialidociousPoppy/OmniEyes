"""Aggregate judged results: accuracy, frames kept, ViT cost, latency per condition,
plus paired bootstrap CI for gate vs matched baselines, multi-file support, K-ablation,
per-task breakdown and error analysis.

Usage (from the repository root):
  python src/analyze.py results/iteration2/judged_v2_k3.jsonl [more judged files ...]
         [--metric NAME] [--task T ...] [--errors] [--csv PREFIX]
  e.g. --csv results/iteration2/tables/instant_backward

  --metric NAME  column to score by. Default: ent_con if the rows have it (semantic_judge.py),
                 otherwise score (judge.py). 0-1 / -1..1 metrics from semantic_judge.py
                 (ent_con, p_ent, p_con, sim, ...) are shown x100.
  --task T ...   only use these task types, e.g. --task instant backward
  --errors       print the 10 largest drops of each gate condition vs. full
  --csv PREFIX   also write every table to CSV: PREFIX_summary.csv, PREFIX_by_task.csv,
                 PREFIX_paired.csv (and PREFIX_errors.csv with --errors)
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
import numpy as np


def load_rows(paths):
    rows = []
    for p in paths:
        matched = glob.glob(p) if any(ch in p for ch in ("*", "?")) else [p]
        for f in matched:
            if not os.path.exists(f):
                print(f"Warning: file not found: {f}", file=sys.stderr)
                continue
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--task", nargs="+", default=None)
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--csv", metavar="PREFIX", default=None)
    a = ap.parse_args()
    tables = defaultdict(list)  # name -> list of dict rows, written as CSV at the end

    rows = [r for r in load_rows(a.files) if r.get("cond") != "calib"]
    if a.task:
        rows = [r for r in rows if r.get("task") in a.task]
    if not rows:
        print("No valid rows loaded.", file=sys.stderr)
        return

    metric = a.metric or ("ent_con" if "ent_con" in rows[0] else "score")
    if metric not in rows[0]:
        sys.exit(f"column '{metric}' not found; available: {sorted(rows[0])}")
    llm_judge = metric == "score" and "raw_judge" in rows[0]  # judge.py output, 0-100, -1 = parse failure
    scale = 1.0 if metric == "score" else 100.0
    print(f"Metric: {metric}" + (" (x100)" if scale != 1 else "")
          + (f"   Tasks: {', '.join(a.task)}" if a.task else ""))

    def valid(r):
        # Only judge.py marks failures with -1; semantic_judge metrics can be legitimately negative.
        return not (llm_judge and r["score"] == -1)

    for r in rows:
        r["_s"] = scale * r[metric]

    # Check for judge parse failures (score == -1)
    parse_fails = [r for r in rows if not valid(r)]
    n_total = len(rows)
    if parse_fails:
        print(f"\n[Notice] Judge parse failures: {len(parse_fails)}/{n_total} ({100 * len(parse_fails) / n_total:.1f}%)")
        print("Sample unparseable response:")
        print("  " + parse_fails[0].get("raw_judge", "N/A")[:200].replace("\n", " "))
        print("Note: Rows with score -1 are excluded from accuracy averages.\n")

    # Detect multi-K dataset
    k_vals = sorted(set(r.get("k", 3) for r in rows))
    multi_k = len(k_vals) > 1

    by = defaultdict(list)
    for r in rows:
        cond_label = f"K{r.get('k', 3)}_{r['cond']}" if multi_k else r["cond"]
        by[cond_label].append(r)

    # Reference full ViT timing
    full_vit_map = {}
    if multi_k:
        for k in k_vals:
            k_full = by.get(f"K{k}_full", [])
            full_vit_map[k] = np.mean([r["vit_ms"] for r in k_full]) if k_full else 1.0
    else:
        full_vit = np.mean([r["vit_ms"] for r in by["full"]]) if "full" in by else 1.0

    print(f"{'condition':<18}{'n':>5}{'score':>8}{'frames':>8}{'ViT ms':>9}{'ViT %ideal':>12}{'ViT %act':>10}{'tokens':>8}{'LLM s':>7}")
    for c in sorted(by, key=lambda x: ("full" not in x, x.split("@")[-1] if "@" in x else "", x)):
        R = by[c]
        k = R[0].get("k", 3)
        ref_full_vit = full_vit_map[k] if multi_k else full_vit

        valid_scores = [r["_s"] for r in R if valid(r)]
        score_avg = np.mean(valid_scores) if valid_scores else 0.0

        vit_ideal = np.mean([r["vit_ms"] for r in R])
        vit_actual = np.mean([r.get("vit_ms_actual", r["vit_ms"]) for r in R])
        pct_ideal = 100 * vit_ideal / max(1e-6, ref_full_vit)
        pct_actual = 100 * vit_actual / max(1e-6, ref_full_vit)

        tables["summary"].append(dict(
            condition=c, n=len(R), score=round(score_avg, 2),
            frames=round(np.mean([r["n_kept"] for r in R]), 2),
            vit_ms_ideal=round(vit_ideal, 1), vit_pct_ideal=round(pct_ideal, 1),
            vit_ms_actual=round(vit_actual, 1), vit_pct_actual=round(pct_actual, 1),
            prompt_tokens=round(np.mean([r["prompt_tokens"] for r in R]), 1),
            llm_latency_s=round(np.mean([r["llm_latency_s"] for r in R]), 3)))
        print(f"{c:<18}{len(R):>5}{score_avg:>8.1f}{np.mean([r['n_kept'] for r in R]):>8.1f}"
              f"{vit_ideal:>9.0f}{pct_ideal:>11.0f}%{pct_actual:>9.0f}%"
              f"{np.mean([r['prompt_tokens'] for r in R]):>8.0f}{np.mean([r['llm_latency_s'] for r in R]):>7.2f}")

    # Per-task score breakdown
    print("\nScore by task (excluding parse failures):")
    for c in sorted(by):
        t = defaultdict(list)
        for r in by[c]:
            if valid(r):
                t[r["task"]].append(r["_s"])
        for task, v in sorted(t.items()):
            tables["by_task"].append(dict(condition=c, task=task, n=len(v), score=round(np.mean(v), 2)))
        print(f"  {c:<18}" + "  ".join(f"{k}={np.mean(v):.1f}(n={len(v)})" for k, v in sorted(t.items())))

    # Paired bootstrap: gate - baseline
    rng = np.random.default_rng(0)
    print("\nPaired diff (gate - baseline), 95% bootstrap CI   (* = CI excludes 0):")
    for c in sorted(by):
        if "@" not in c or not any(prefix in c for prefix in ("gate@", "_gate@")):
            continue
        prefix, tau = c.rsplit("@", 1)
        base_prefix = prefix.replace("gate", "")
        g = {r["qid"]: r["_s"] for r in by[c] if valid(r)}

        baselines = [f"{base_prefix}uniform@{tau}", f"{base_prefix}pixel@{tau}", f"{base_prefix}full"]
        for b in baselines:
            if b not in by:
                continue
            bb = {r["qid"]: r["_s"] for r in by[b] if valid(r)}
            ks = [k for k in g if k in bb]
            if not ks:
                continue
            d = np.array([g[k] - bb[k] for k in ks], float)
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            ci_low = np.percentile(bs, 2.5)
            ci_high = np.percentile(bs, 97.5)
            sig = "*" if (ci_low > 0 or ci_high < 0) else " "
            tables["paired"].append(dict(
                gate=c, baseline=b, diff=round(d.mean(), 2), ci_low=round(ci_low, 2),
                ci_high=round(ci_high, 2), n=len(d), significant=sig == "*"))
            print(f"  {c:<18} vs {b:<18} {d.mean():+6.1f} [{ci_low:+6.1f}, {ci_high:+6.1f}] n={len(d):<4} {sig}")

    # Error analysis
    if a.errors:
        print("\n=== ERROR ANALYSIS: 10 WORST DROPS VS FULL ===")
        for c in sorted(by):
            if "@" not in c or not any(prefix in c for prefix in ("gate@", "_gate@")):
                continue
            prefix, tau = c.rsplit("@", 1)
            base_prefix = prefix.replace("gate", "")
            full_key = f"{base_prefix}full"
            if full_key not in by:
                continue

            g = {r["qid"]: r for r in by[c] if valid(r)}
            f = {r["qid"]: r for r in by[full_key] if valid(r)}

            diffs = sorted((g[q]["_s"] - f[q]["_s"], q) for q in g if q in f)

            print(f"\n--- Condition {c} vs {full_key} (showing largest drops) ---")
            top_drops = [item for item in diffs if item[0] < 0][:10]
            if not top_drops:
                print("  No cases where gate scored lower than full.")
                continue

            for delta, qid in top_drops:
                gr = g[qid]
                fr = f[qid]
                tables["errors"].append(dict(
                    gate=c, baseline=full_key, delta=round(delta, 1),
                    gate_score=round(gr["_s"], 1), full_score=round(fr["_s"], 1),
                    task=gr.get("task"), question=gr.get("question"), reference=gr.get("answer"),
                    pred_full=fr.get("pred"), pred_gate=gr.get("pred"),
                    n_kept=gr.get("n_kept"), n_frames=gr.get("n_frames"),
                    kept=" ".join(map(str, gr.get("kept", [])))))
                print(f"  Delta: {delta:+.1f} (Gate: {gr['_s']:.1f} | Full: {fr['_s']:.1f}) [Task: {gr.get('task')}]")
                print(f"  Question: {gr.get('question')}")
                print(f"  Ref:      {gr.get('answer')}")
                print(f"  Full:     {fr.get('pred')}")
                print(f"  Gate:     {gr.get('pred')}")
                print(f"  Frames:   {gr.get('n_kept')}/{gr.get('n_frames')} kept: {gr.get('kept')}")
                print()

    if a.csv:
        for name, trows in tables.items():
            path = f"{a.csv}_{name}.csv"
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(trows[0]))
                w.writeheader()
                w.writerows(trows)
            print(f"wrote {path} ({len(trows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
