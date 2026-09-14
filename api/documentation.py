import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile

from rag import indexing

# APIRouter works like a mini FastAPI app: routes are declared on it here,
# then plugged into the real app with app.include_router() in main.py.
router = APIRouter(prefix="/api")

# Folder where uploaded files get saved. Created now if it doesn't exist yet.
UPLOAD_DIR = Path("data/Documentations")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# JSON file that tracks metadata (filename, size, upload time, ...) for every
# uploaded document. "List docs" reads only from this file instead of
# re-scanning the folder, so it's also the place to add more metadata
# (e.g. risk tags) later without touching the upload logic.
METADATA_FILE = Path("data/documentation.json")


def load_docs_metadata() -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    return json.loads(METADATA_FILE.read_text(encoding="utf-8"))


def save_docs_metadata(docs: list[dict]) -> None:
    METADATA_FILE.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")


def sync_docs_metadata() -> list[dict]:
    """Bring documentation.json in line with the files actually on disk.

    - entries whose PDF was deleted from the folder are dropped
    - every remaining entry gets a fresh "preview": the first lines of the
      PDF, which the document router (Prompt 1) reads together with the
      filename and Type to decide whether a document matters for a question
    Called after a full index rebuild.
    """
    docs = [d for d in load_docs_metadata() if (UPLOAD_DIR / d["filename"]).exists()]
    for d in docs:
        d["preview"] = indexing.document_preview(UPLOAD_DIR / d["filename"])
    save_docs_metadata(docs)
    return docs


@router.get("/documentations")
async def list_docs():
    return {"documentations": load_docs_metadata()}


@router.post("/upload")
async def upload_file(file: UploadFile, doc_type: str | None = Form(default=None)):
    # Read the uploaded file's contents and write them to data/Documentations
    contents = await file.read()
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(contents)

    # Split, embed and store the new file so it is searchable right away.
    indexed = indexing.add_document(save_path)

    # Record/update this file's metadata in documentation.json. "Type" is a
    # free-text label the user gives on upload (e.g. "DPIA", "Policy"); it
    # and "preview" are what the document router reads.
    docs = load_docs_metadata()
    docs = [d for d in docs if d["filename"] != file.filename]
    docs.append({
        "filename": file.filename,
        "size_bytes": len(contents),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "Type": doc_type or "",
        "preview": indexing.document_preview(save_path),
    })
    save_docs_metadata(docs)

    return {"status": "success", "filename": file.filename, "indexed": indexed}
