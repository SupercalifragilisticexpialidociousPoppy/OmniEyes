"""Agreement of the NLI grader with hand labels (no GPU, no model: reads the graded file).

Like `semantic_judge.py --validate`, plus a bootstrap CI for Spearman's rho, an AUC, a split by
question type, and the alternative grade 1 - p_con ("not contradicted").

Usage (from the repository root):
  python src/validate_grader.py results/iteration3/judged_v3_k3.jsonl results/iteration3/labels_v3.csv
"""
import argparse, collections, csv, json
import numpy as np
from scipy.stats import spearmanr

GRADES = {"ent_con": lambda r: r["ent_con"], "1-p_con": lambda r: 1 - r["p_con"]}


def auc(pos, neg):
    """P(a random positive scores above a random negative); ties count half."""
    return float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]))


def stats(rows, grade, rng):
    s = np.array([grade(r) for r, _ in rows]); h = np.array([y for _, y in rows])
    rho, p = spearmanr(s, h)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(s), len(s))
        if len(set(h[i])) > 1:
            boots.append(spearmanr(s[i], h[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    a = auc(s[h >= 1], s[h == 0]) if (h == 0).any() and (h >= 1).any() else float("nan")
    med = {k: round(100 * float(np.median(s[h == k])), 1) for k in (0, 1, 2) if (h == k).any()}
    return rho, p, lo, hi, a, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged"); ap.add_argument("labels")
    a = ap.parse_args()
    R = {(r["qid"].strip(), r["cond"].strip()): r for r in map(json.loads, open(a.judged, encoding="utf-8"))}
    rows = [(R[(l["qid"].strip(), l["cond"].strip())], int(float(l["human"])))
            for l in csv.DictReader(open(a.labels, encoding="utf-8-sig")) if l["human"].strip()]
    print(f"{len(rows)} labelled rows, {len({r['qid'] for r, _ in rows})} distinct questions; "
          f"labels {dict(sorted(collections.Counter(y for _, y in rows).items()))}; "
          f"conditions {dict(collections.Counter(r['cond'].split('@')[0] for r, _ in rows))}")
    print("AUC = P(an answer labelled 1-2 grades above one labelled 0); median grade x100 per label\n")
    rng = np.random.default_rng(0)
    for name, grade in GRADES.items():
        for task in (None, "instant", "backward", "forward"):
            sub = [x for x in rows if task is None or x[0]["task"] == task]
            if len(sub) < 5:
                continue
            rho, p, lo, hi, au, med = stats(sub, grade, rng)
            print(f"{name:<8} {task or 'all':<9} n={len(sub):>3}  rho={rho:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3g}  "
                  f"AUC={au:.2f}  median by label {med}")
        print()


if __name__ == "__main__":
    main()
