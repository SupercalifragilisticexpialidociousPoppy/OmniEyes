# ARCHIVED: iteration 1 helper for the LLM-judge output; superseded by src/analyze.py.
# Reads a file named "judged(3).jsonl" in the current directory.
import json

scores = []
with open("judged(3).jsonl") as f:
    for line in f:
        data = json.loads(line)
        scores.append(data.get("score", 0))

print(f"Total evaluated: {len(scores)}")
print(f"Non-zero scores: {sum(1 for s in scores if s > 0)}")
print(f"Average score: {sum(scores) / len(scores):.2f}")
