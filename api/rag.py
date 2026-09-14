"""
HTTP adapter for the rag/ package.

Every endpoint here is a thin wrapper: parse the request, call one function
in rag/, return its result. No retrieval logic lives in this file - that is
what keeps rag/ portable. The endpoints are ordered like the flowchart in
pipeline.jpg, so each pipeline stage can be tried on its own from the browser
(http://localhost:8000/docs) or with curl.
"""
import json

from fastapi import APIRouter, HTTPException

from api.documentation import sync_docs_metadata
from rag import config, indexing, pipeline, regulations, retrieval
from rag.questions import get_question, load_questions

router = APIRouter(prefix="/api")


# ---------- Step 1: Lexical Retrieval A ----------
@router.get("/regulations/{target}/articles/{article_id}")
async def get_regulation_article(target: str, article_id: str):
    """The text of one article, e.g. /api/regulations/aiact/articles/art_2"""
    try:
        return regulations.get_articles(target, [article_id])[0]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


# ---------- Step 2: the questionnaire + meta evaluation ----------
@router.get("/questions")
async def list_questions():
    return {"questions": load_questions()}


@router.get("/questions/{question_id}/context")
async def question_context(question_id: str):
    """The question plus the regulation article(s) it cites.

    This is the first diamond of the flowchart ("requirement regulation in
    uploaded corpus?"): if the regulation file is not there the answer is a
    pipeline-level NA, before any document search or model call.
    """
    try:
        question = get_question(question_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))

    target = question["regulation_target"]
    if not regulations.regulation_exists(target):
        return {"status": "NA", "na_reason": f"regulation '{target}' not uploaded", "question": question}

    try:
        articles = regulations.get_articles(target, question["regulation_articles"])
    except KeyError as e:
        return {"status": "NA", "na_reason": str(e.args[0]), "question": question}

    return {"status": "ok", "question": question, "articles": articles}


# ---------- Step 3: the vector index over the user's PDFs ----------
@router.get("/index/status")
async def get_index_status():
    return indexing.index_status()


@router.post("/index/rebuild")
async def rebuild_index():
    """Re-read and re-embed every PDF in data/Documentations.

    Costs one embedding request per batch of chunks; use it after changing
    CHUNK_SIZE / EMBED_MODEL, or if data/index/ was deleted.
    """
    result = indexing.build_index()
    sync_docs_metadata()
    return result


# ---------- Step 4: Semantic Retrieval B ----------
@router.get("/search")
async def search_documents(
    q: str,
    docs: str | None = None,
    top_k: int | None = None,
    cutoff: float | None = None,
):
    """Try the vector search on its own and look at the scores.

    /api/search?q=valutazione d'impatto&docs=dpia_talentmatch.pdf,altro.pdf&top_k=5&cutoff=0.6
    `docs` is a comma-separated list of filenames; omit it to search everything.
    """
    filenames = [d.strip() for d in docs.split(",") if d.strip()] if docs else None
    return {"query": q, "filenames": filenames, "results": retrieval.search(q, filenames, top_k, cutoff)}


# ---------- Step 5: Prompt 1 on its own ----------
@router.post("/route/{question_id}")
async def route_question(question_id: str):
    """Ask the model which uploaded documents matter for one question. 1 LLM call."""
    try:
        question = get_question(question_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))
    articles = regulations.get_articles(question["regulation_target"], question["regulation_articles"])
    return await pipeline.select_documents(question, articles, pipeline.load_documents())


# ---------- Step 6: the whole pipeline ----------
@router.post("/answer/{question_id}")
async def answer_question(question_id: str):
    """Run the full flowchart for one question. Up to 2 LLM calls."""
    try:
        return await pipeline.answer_question(question_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


@router.post("/answer-all")
async def answer_all_questions():
    """Run every question and write data/answers.json. Up to 2 LLM calls per question."""
    return {"answers": await pipeline.answer_all()}


@router.get("/answers")
async def saved_answers():
    """The last answer-all run, as saved in data/answers.json."""
    if not config.ANSWERS_FILE.exists():
        return {"answers": []}
    return {"answers": json.loads(config.ANSWERS_FILE.read_text(encoding="utf-8"))}
