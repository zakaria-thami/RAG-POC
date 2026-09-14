"""
The questionnaire: data/questions.json.

Each entry looks like
    {
      "question_id": "GDPR-05",
      "query_text": "E' stata redatta ... una valutazione d'impatto (DPIA) ...?",
      "regulation_target": "gdpr",
      "regulation_articles": ["art_35"]
    }
regulation_target names the file data/Regulations/<target>.md and
regulation_articles the headings inside it (see rag/regulations.py).
"""
import json

from rag import config


def load_questions() -> list[dict]:
    return json.loads(config.QUESTIONS_FILE.read_text(encoding="utf-8"))


def get_question(question_id: str) -> dict:
    for q in load_questions():
        if q["question_id"] == question_id:
            return q
    raise KeyError(f"Unknown question_id '{question_id}'")
