> **Archived.** One-off message sent to a teammate during iteration 2. Its commands use the old flat layout and an older `analyze.py`; the current steps are in the main `README.md`.

I'm sending two files for grading the new `results.jsonl`: **`semantic_judge.py`** and **`analyze.py`**.

**Why we're not using the LLM judge anymore**
The Groq judge kept hitting rate limits, and only scored 106 of 150 questions last time. `semantic_judge.py` grades answers **locally**: no API, no keys, no limits. It uses an NLI model (DeBERTa-v3-large), which reads the reference answer and our model's answer and outputs:
- the probability that the reference **supports** our answer, and
- the probability that it **contradicts** it.

The grade we use is **`ent_con` = P(support) − P(contradict)**, on a scale from −100 (confidently wrong) to +100 (confidently right). I checked it against 60 hand-labelled answers: it agreed with the human labels much better than BERT similarity did (Spearman 0.59 vs 0.21).

**How to run it (Colab, use a T4 GPU runtime)**

1. Install:
```bash
pip -q install sentence-transformers scipy
```
2. Grade every answer. The first run downloads about 2 GB of models; after that it takes a few minutes.
```bash
python semantic_judge.py results.jsonl judged_sem_large.jsonl --nli_model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```
3. Get the results table and the paired confidence intervals:
```bash
python analyze.py judged_sem_large.jsonl ent_con
```
The second argument picks which column to score by.

**Checking the grader on the new answers**
Please **don't reuse my old `labels.csv`**. The question IDs are the same, but the answers changed, so the old labels would be matched to the wrong answers without any error.

1. Sample 60 rows:
```bash
python semantic_judge.py judged_sem_large.jsonl --sample 60 labels_new.csv
```
2. Download `labels_new.csv` and fill the `human` column with **0 / 1 / 2** (wrong / partial / correct). Re-upload it.
3. Validate:
```bash
python semantic_judge.py judged_sem_large.jsonl --validate labels_new.csv
```
Send me the output. If `ent_con` comes out around 0.5–0.6 again, the grader is confirmed.

**⚠️ `analyze.py` needs merging**
We each changed `analyze.py` separately:
- **Your version** added `--errors` and the Ideal/Actual ViT columns.
- **My version** added the optional second argument for choosing a column (e.g. `ent_con`).

Whichever file gets uploaded last will be missing the other's features. Please send me your `analyze.py`, and I'll merge them into one file that has everything. Until then:
- use **my** `analyze.py` for the `ent_con` results table, and
- use **yours** for `--errors` and the Ideal/Actual ViT columns.

Rename one of them (e.g. `analyze_errors.py`) so they don't overwrite each other.

**Two quick checks on the new run**
- **Forward questions:** if they're included, report them separately. They test response *timing*, which this grader can't judge, and mixing them in would blur the instant/backward results.
- **All 150 questions:** make sure every question finishes under every condition. With no rate limits, we can score the full set this time.

