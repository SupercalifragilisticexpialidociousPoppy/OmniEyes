"""Cheap, deterministic, local judge for the results files written by run.py (no API calls).

Scores every (question, reference answer, prediction) row with small encoder models:

  sim     cosine similarity of sentence embeddings (bi-encoder, all-mpnet-base-v2)
          -> "how close in meaning/topic". Fast, but blind to negation and to wrong
             attribute values ("white" vs "pink" still score high).
  p_ent   NLI cross-encoder (DeBERTa-v3): P(reference ENTAILS prediction)
  p_con   P(reference CONTRADICTS prediction)  -> catches "yes" vs "no", wrong colour, etc.
  p_neu   P(neutral)                            -> prediction neither supported nor refuted
  echo    P(question ENTAILS prediction)        -> prediction only restates the question
                                                   ("The slider has a cue that shows its function")
  bertscore  (optional, --bertscore) BERTScore F1, rescaled with baseline
  ent_con P(ent) - P(con), range -1..1         -> simpler NLI-only grade
  p_ent_rev  P(prediction ENTAILS reference)    -> reverse direction: does the answer
                                                   actually state what the reference says?
                                                   ("select Edit" does not entail
                                                   "selects Markup and Reply")

The NLI model can be swapped with --nli_model, e.g. the larger adversarially trained
  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
Score with it into a separate file and --validate both files on the same labels file.

Combined score (0-100), written to "score" so analyze.py works unchanged:

  score = 100 * (1 - echo) * (p_ent + p_neu * max(sim, 0))

  i.e. full credit when the reference supports the answer, partial credit (scaled by
  similarity) when it is neither supported nor refuted, ~0 when contradicted, and a
  penalty when the answer carries no information beyond the question itself.

Usage (run from the repository root; a GPU is much faster with the large NLI model):
  pip -q install sentence-transformers scipy     # + bert-score if you use --bertscore
  python src/semantic_judge.py results/iteration2/results_v2_k3.jsonl results/iteration2/judged_v2_k3.jsonl \
      --nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  python src/analyze.py results/iteration2/judged_v2_k3.jsonl              # scores by ent_con
  python src/analyze.py results/iteration2/judged_v2_k3.jsonl --metric p_ent   # any other column

Validating the metric against humans (once per new results file, with fresh labels):
  python src/semantic_judge.py results/iteration2/judged_v2_k3.jsonl --sample 60 results/iteration2/labels_v2.csv
      -> fill the "human" column with 0 / 1 / 2 (wrong / partial / correct)
  python src/semantic_judge.py results/iteration2/judged_v2_k3.jsonl --validate results/iteration2/labels_v2.csv
      -> Spearman correlation of every metric with your labels
"""
import argparse, csv, json, random, sys

import numpy as np
import torch

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
METRICS = ["score", "sim", "p_ent", "p_con", "p_neu", "echo", "ent_con", "p_ent_rev", "bertscore"]


def load_rows(path):
    return [d for d in (json.loads(l) for l in open(path)) if d.get("cond") != "calib"]


def nli_probs(pairs, device, model_name, bs=32):
    """Return {(premise, hypothesis): {"entailment": p, "contradiction": p, "neutral": p}}."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}  # don't hard-code the order
    if set(id2label.values()) != {"entailment", "contradiction", "neutral"}:
        sys.exit(f"{model_name} is not a 3-way NLI model (labels: {id2label})")
    pairs = list(dict.fromkeys(pairs))
    out = {}
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], padding=True,
                  truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1).cpu().numpy()
        for pair, pr in zip(chunk, probs):
            out[pair] = {id2label[j]: float(pr[j]) for j in range(len(pr))}
    return out


def judge(inp, out, use_bertscore, nli_model):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(inp)
    preds = [r["pred"].strip() or "(empty)" for r in rows]
    refs = [r["answer"] for r in rows]
    qs = [r["question"] for r in rows]

    emb = SentenceTransformer(EMB_MODEL, device=device)
    texts = list(dict.fromkeys(preds + refs))
    vecs = dict(zip(texts, emb.encode(texts, batch_size=64, normalize_embeddings=True,
                                      show_progress_bar=True)))
    sims = [float(vecs[p] @ vecs[a]) for p, a in zip(preds, refs)]

    # (ref, pred): forward; (q, pred): echo; (pred, ref): reverse
    nli = nli_probs(list(zip(refs, preds)) + list(zip(qs, preds)) + list(zip(preds, refs)),
                    device, nli_model)

    bs_f1 = [None] * len(rows)
    if use_bertscore:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True,
                              device=device, batch_size=32)
        bs_f1 = [float(x) for x in f1]

    with open(out, "w") as fo:
        for r, p, a, q, s, b in zip(rows, preds, refs, qs, sims, bs_f1):
            ra, qa, rev = nli[(a, p)], nli[(q, p)], nli[(p, a)]
            r.update(sim=s, p_ent=ra["entailment"], p_con=ra["contradiction"],
                     p_neu=ra["neutral"], echo=qa["entailment"],
                     ent_con=ra["entailment"] - ra["contradiction"],
                     p_ent_rev=rev["entailment"], nli_model=nli_model)
            if b is not None:
                r["bertscore"] = b
            r["score"] = 100 * (1 - r["echo"]) * (r["p_ent"] + r["p_neu"] * max(s, 0.0))
            fo.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    summary(load_rows(out))


def summary(rows):
    cols = [m for m in METRICS if m in rows[0]]
    print("\nmean per condition:")
    print(f"{'cond':<14}" + "".join(f"{c:>10}" for c in cols))
    for cond in sorted({r["cond"] for r in rows}):
        R = [r for r in rows if r["cond"] == cond]
        print(f"{cond:<14}" + "".join(f"{np.mean([r[c] for r in R]):>10.3f}" for c in cols))


def sample(inp, n, path, seed=0):
    rows = load_rows(inp)
    random.Random(seed).shuffle(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "cond", "question", "answer", "pred", "human"])
        for r in rows[:n]:
            w.writerow([r["qid"], r["cond"], r["question"], r["answer"], r["pred"], ""])
    print(f"wrote {min(n, len(rows))} rows to {path}; fill 'human' with 0/1/2 and run --validate")


def validate(inp, path):
    from scipy.stats import spearmanr

    # strip(): spreadsheet editors trim trailing spaces, and qids can end in one
    rows = {(r["qid"].strip(), r["cond"].strip()): r for r in load_rows(inp)}
    lab = [(rows[(l["qid"].strip(), l["cond"].strip())], float(l["human"]))
           for l in csv.DictReader(open(path, encoding="utf-8-sig")) if l["human"].strip()]
    print(f"{len(lab)} labelled rows")
    h = [y for _, y in lab]
    for m in METRICS:
        if m in lab[0][0]:
            rho, pv = spearmanr([r[m] for r, _ in lab], h)
            print(f"  {m:<10} spearman={rho:+.3f}  p={pv:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out", nargs="?", help="output jsonl (judge mode)")
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--sample", nargs=2, metavar=("N", "CSV"))
    ap.add_argument("--validate", metavar="CSV")
    ap.add_argument("--nli_model", default=NLI_MODEL,
                    help="any 3-way NLI model on Hugging Face (default: %(default)s)")
    a = ap.parse_args()
    if a.sample:
        sample(a.inp, int(a.sample[0]), a.sample[1])
    elif a.validate:
        validate(a.inp, a.validate)
    elif a.out:
        judge(a.inp, a.out, a.bertscore, a.nli_model)
    else:
        sys.exit("give an output path, --sample N CSV, or --validate CSV")
