"""
note: one call of answer_question() = up to 2 OpenRouter calls
(router + answer) plus 1 Gemini embedding of the question.
"""
import json

from rag import config
from rag.llm import ask, parse_json_reply
from rag.prompts import SYSTEM_ANSWER, SYSTEM_ROUTER, build_answer_message, build_router_message
from rag.questions import get_question, load_questions
from rag.regulations import get_articles, regulation_exists
from rag.retrieval import search


def load_documents() -> list[dict]:
    """The uploaded documents as listed in documentation.json (filename, Type, preview)."""
    if not config.DOCS_METADATA_FILE.exists():
        return []
    return json.loads(config.DOCS_METADATA_FILE.read_text(encoding="utf-8"))


def _na(question: dict, reason: str, trace: dict) -> dict:
    return {"question_id": question["question_id"], "status": "NA", "na_reason": reason,
            "answer": None, "trace": trace}


async def select_documents(question: dict, articles: list[dict], documents: list[dict]) -> dict:
    """Prompt 1: ask the model which uploaded documents could answer the question.

    Returns {"relevant_documents": [...], "reason": str, "seconds", "tokens"}.
    Filenames the model invents (not in `documents`) are dropped.
    """
    message = build_router_message(question, articles, documents)
    reply = await ask(SYSTEM_ROUTER, message, label=f"ROUTE {question['question_id']}")
    parsed = parse_json_reply(reply["text"])

    known = {d["filename"] for d in documents}
    chosen = [f for f in parsed.get("relevant_documents", []) if f in known]
    return {
        "relevant_documents": chosen,
        "reason": parsed.get("reason", ""),
        "prompt_chars": len(message),
        "seconds": round(reply["seconds"], 1),
        "tokens": reply["tokens"],
    }


async def answer_question(question_id: str) -> dict:
    question = get_question(question_id)
    trace: dict = {"model": config.MODEL_NAME}

    # --- "Requirement regulation in uploaded corpus?" ---
    target = question["regulation_target"]
    if not regulation_exists(target):
        return _na(question, f"regulation '{target}' not uploaded", trace)

    # --- Lexical Retrieval A ---
    try:
        articles = get_articles(target, question["regulation_articles"])
    except KeyError as e:
        return _na(question, str(e.args[0]), trace)
    trace["articles"] = [{"heading": a["heading"], "chars": len(a["text"])} for a in articles]

    # --- Prompt 1: which documents? ---
    documents = load_documents()
    if not documents:
        return _na(question, "no documents uploaded", trace)

    filenames = None                       # None = search every document
    if config.USE_DOC_ROUTER:
        routing = await select_documents(question, articles, documents)
        trace["routing"] = routing
        filenames = routing["relevant_documents"]
        if not filenames:
            return _na(question, "no uploaded document is relevant to this question", trace)

    # --- Semantic Retrieval B + "Strong similarity score?" ---
    chunks = search(question["query_text"], filenames)
    trace["chunks"] = [{"filename": c["filename"], "page_label": c["page_label"], "score": c["score"]}
                       for c in chunks]
    trace["similarity_cutoff"] = config.SIMILARITY_CUTOFF
    if not chunks:
        return _na(question, f"no chunk above similarity cutoff {config.SIMILARITY_CUTOFF}", trace)

    # --- Prompt 2: answer ---
    message = build_answer_message(question, articles, chunks)
    reply = await ask(SYSTEM_ANSWER, message, label=f"ANSWER {question['question_id']}")
    trace["answer_call"] = {"prompt_chars": len(message), "seconds": round(reply["seconds"], 1),
                            "tokens": reply["tokens"]}
    try:
        answer = parse_json_reply(reply["text"])
    except ValueError:
        # The model ignored "JSON only". Keep the raw text so nothing is lost.
        answer = {"answer": "ABSTAIN", "evidence": [], "confidence": "LOW",
                  "note": "model reply was not valid JSON", "raw": reply["text"]}

    return {"question_id": question_id, "status": "answered", "na_reason": None,
            "answer": answer, "trace": trace}


async def answer_all() -> list[dict]:
    """Run every question in order and save the results to data/answers.json.

    Sequential on purpose: easier to follow in chat.log, and no burst of
    parallel calls against the OpenRouter account.
    """
    results = [await answer_question(q["question_id"]) for q in load_questions()]
    config.ANSWERS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
