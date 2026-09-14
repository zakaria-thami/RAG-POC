"""
Sanity check for Step 1: does every article cited in data/questions.json
resolve to a heading in its regulation file?

Run from the project root:
    ./venv/Scripts/python.exe scripts/check_questions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.questions import load_questions
from rag.regulations import get_articles, regulation_exists

ok = 0
questions = load_questions()
for q in questions:
    target = q["regulation_target"]
    if not regulation_exists(target):
        print(f"[X] {q['question_id']:<12} regulation '{target}' missing")
        continue
    try:
        articles = get_articles(target, q["regulation_articles"])
    except KeyError as e:
        print(f"[X] {q['question_id']:<12} {e}")
        continue
    ok += 1
    found = ", ".join(f"{a['heading']} ({len(a['text'])} chars)" for a in articles)
    print(f"[OK] {q['question_id']:<12} {target:<16} {found}")

print(f"\n{ok}/{len(questions)} questions resolved")
