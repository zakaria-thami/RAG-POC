import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------- where the data lives ----------
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", "data/Documentations"))      # user PDFs
REGULATIONS_DIR = Path(os.getenv("RAG_REGULATIONS_DIR", "data/Regulations"))  # regulation .md files
QUESTIONS_FILE = Path(os.getenv("RAG_QUESTIONS_FILE", "data/questions.json"))
DOCS_METADATA_FILE = Path(os.getenv("RAG_DOCS_METADATA_FILE", "data/documentation.json"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "data/index"))            # persisted vector index
ANSWERS_FILE = Path(os.getenv("RAG_ANSWERS_FILE", "data/answers.json"))


# ---------- chat model (OpenRouter) ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


# ---------- embedding model (Gemini) ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")


# ---------- chunking ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


# ---------- retrieval ----------
# How many chunks we hand to the model for the final answer.
TOP_K = int(os.getenv("TOP_K", "5"))
# Chunks whose similarity to the question is below this are dropped. 0.0
# means "keep everything". The right value depends on the embedding model and
# is chosen by looking at real scores (see /api/search). Calibration with
# gemini-embedding-001 on the test PDFs (Sept 2026):
#     off-topic documents (HR policy, printer guide)   0.54 - 0.60
#     on-topic document, no evidence for the question  0.60 - 0.69
#     on-topic document, contains the evidence         0.70 - 0.79
# 0.62 rejects off-topic text and leaves "no evidence -> ABSTAIN" to the model.
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.62"))
# Prompt 1 (ask the LLM which documents matter) can be switched off to compare
# results with plain semantic search over every document.
USE_DOC_ROUTER = os.getenv("USE_DOC_ROUTER", "true").lower() == "true"
# How many characters of each document we show the router so it can judge
# what the document is about, even when the filename says nothing.
DOC_PREVIEW_CHARS = int(os.getenv("DOC_PREVIEW_CHARS", "300"))
