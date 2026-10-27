"""Evaluate retrieval: for each test question, is the expected page in the top-k results?

Run:  python evaluate.py          (uses k=4)
      python evaluate.py 8        (uses k=8)
"""
import csv
import sys

from rag import retrieve

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CSV_PATH = "eval/questions.csv"

# 1) Read the test questions
with open(CSV_PATH, encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["question"].strip()]

if not rows:
    print(f"{CSV_PATH} is empty. Write your test questions first.")
    sys.exit(1)

file_hits = 0
page_hits = 0
reciprocal_ranks = []

# 2) For each question, search and check if the right file/page is in the results
for r in rows:
    expected_file = r["expected_file"].strip()
    expected_page = int(r["expected_page"])
    results = retrieve(r["question"], K)

    found_file = any(s["file"] == expected_file for s in results)

    # position (1, 2, 3...) of the first result with the right file AND page
    rank = None
    for position, s in enumerate(results, start=1):
        if s["file"] == expected_file and s["page"] == expected_page:
            rank = position
            break

    file_hits += found_file
    page_hits += rank is not None
    reciprocal_ranks.append(1 / rank if rank else 0)

    status = f"OK (rank {rank})" if rank else "MISS"
    got = ", ".join(f"p.{s['page']}" for s in results)
    print(f"{status:12} {r['question'][:60]:60} expected p.{expected_page} | got {got}")

# 3) Print the summary
n = len(rows)
print(f"\nQuestions: {n}   k = {K}")
print(f"Hit@{K} (right paper): {file_hits / n:.1%}")
print(f"Hit@{K} (right page):  {page_hits / n:.1%}")
print(f"MRR (right page):     {sum(reciprocal_ranks) / n:.3f}")