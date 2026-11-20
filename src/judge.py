"""LLM-as-judge scoring for OmniEyes (0–100 scale).
Any OpenAI-compatible endpoint works (set JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL).
Iteration 2 fixes: max_tokens=200, XML-structured output, raw_judge audit trail,
fallback score=-1 for parse failures, rate-limiting."""
import argparse, json, os, re, sys, time

RUBRIC = """You are grading an AI assistant's answer about a live phone-camera video stream.

**Question asked:** {q}
**Reference (ground-truth) answer:** {a}
**Assistant's answer:** {p}

Score the assistant's answer for correctness and specificity on a 0–10 scale:
- 10 = fully correct, specific, matches reference
- 7–9 = mostly correct, minor omissions
- 4–6 = partially correct, significant gaps
- 1–3 = mostly wrong, only trivially relevant
- 0 = completely wrong, irrelevant, or refusal

First, reason briefly (2–3 sentences max). Then give your score inside XML tags.
Example: <score>7</score>"""


def parse_score(text):
    """Extract score from judge output. Returns 0-100 int, or -1 if unparseable."""
    if not text:
        return -1
    # Primary: XML tags
    m = re.search(r"<score>\s*(\d+)\s*</score>", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return min(10, val) * 10 if val <= 10 else min(100, val)
    # Fallback: look for <score> tag without closing or explicit pattern
    m_tag = re.search(r"<score>\s*(\d+)", text, re.IGNORECASE)
    if m_tag:
        val = int(m_tag.group(1))
        return min(10, val) * 10 if val <= 10 else min(100, val)
    # Fallback: last standalone integer between 0 and 10 in the text
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        for candidate in reversed(matches):
            val = int(candidate)
            if 0 <= val <= 10:
                return val * 10
            elif val in (20, 30, 40, 50, 60, 70, 80, 90, 100):
                return val
    return -1


def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge scoring for PhoStream OmniEyes experiment")
    ap.add_argument("input", help="results.jsonl from run.py")
    ap.add_argument("output", help="output judged JSONL path")
    ap.add_argument("--rps", type=float, default=1.5,
                    help="requests per second (rate limit, default 1.5)")
    ap.add_argument("--max_tokens", type=int, default=200,
                    help="max tokens for judge response (default 200)")
    a = ap.parse_args()

    from openai import OpenAI
    cli = OpenAI(base_url=os.environ["JUDGE_BASE_URL"],
                 api_key=os.environ["JUDGE_API_KEY"])
    model = os.environ["JUDGE_MODEL"]

    # Load already-judged rows to avoid re-doing them
    cache = {}
    if os.path.exists(a.output):
        for line in open(a.output, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cache[(d["qid"], d["cond"])] = d

    fo = open(a.output, "a", encoding="utf-8")
    n_done, n_skip, n_fail = 0, 0, 0

    for line in open(a.input, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("cond") == "calib":
            continue
        if (d["qid"], d["cond"]) in cache:
            n_skip += 1
            continue

        success = False
        for attempt in range(5):
            try:
                time.sleep(1.0 / max(0.01, a.rps))  # rate limiting
                r = cli.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=a.max_tokens,
                    messages=[{"role": "user",
                               "content": RUBRIC.format(
                                   q=d["question"], a=d["answer"], p=d["pred"])}]
                )
                raw = r.choices[0].message.content or ""
                d["raw_judge"] = raw
                d["score"] = parse_score(raw)
                success = True
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  retry {attempt+1}/5: {e}, waiting {wait}s")
                time.sleep(wait)

        if not success:
            d["score"] = -1
            d["raw_judge"] = "FAILED_ALL_RETRIES"
            n_fail += 1

        fo.write(json.dumps(d, ensure_ascii=False) + "\n")
        fo.flush()
        n_done += 1
        if n_done % 50 == 0:
            print(f"  judged {n_done} (skipped {n_skip}, failed {n_fail})")

    print(f"Done: judged {n_done}, skipped {n_skip}, failed {n_fail}")
    fo.close()


if __name__ == "__main__":
    main()
