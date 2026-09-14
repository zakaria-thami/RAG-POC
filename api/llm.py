"""
/api/chat - send one free-text question straight to the model.

This is the "no retrieval" baseline kept from phase 1: the model only sees
the question and the answer prompt, no documents. Useful to compare against
the full pipeline in api/rag.py. The model client itself lives in rag/llm.py.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from rag import config
from rag.llm import ask
from rag.prompts import SYSTEM_ANSWER

router = APIRouter(prefix="/api")


# Structure of the request body: {"question": "..."}
class Question(BaseModel):
    question: str


@router.post("/chat")
async def chat(body: Question):
    reply = await ask(SYSTEM_ANSWER, body.question, label="CHAT")
    return {"answer": reply["text"], "model": config.MODEL_NAME}
