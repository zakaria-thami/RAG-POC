import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, UploadFile

# Same idea as api/documentation.py, but for regulation documents. Keeping them in a
# separate router, folder and JSON file means the two kinds of documents
# never get mixed up, and each one can grow its own metadata later.
router = APIRouter(prefix="/api")

# Folder where uploaded regulation files get saved.
UPLOAD_DIR = Path("data/Regulations")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# JSON file that tracks metadata for every uploaded regulation.
# "List regulations" reads only from this file.
METADATA_FILE = Path("data/regulations.json")


def load_regulations_metadata() -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    return json.loads(METADATA_FILE.read_text())


def save_regulations_metadata(regulations: list[dict]) -> None:
    METADATA_FILE.write_text(json.dumps(regulations, indent=2))


@router.get("/regulations")
async def list_regulations():
    return {"regulations": load_regulations_metadata()}


@router.post("/upload-regulation")
async def upload_regulation(file: UploadFile):
    # Read the uploaded file's contents and write them to data/Regulations
    contents = await file.read()
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(contents)

    # Record/update this file's metadata in regulations.json
    regulations = load_regulations_metadata()
    regulations = [r for r in regulations if r["filename"] != file.filename]
    regulations.append({
        "filename": file.filename,
        "size_bytes": len(contents),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    save_regulations_metadata(regulations)

    return {"status": "success", "filename": file.filename}
