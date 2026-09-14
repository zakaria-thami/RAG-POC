"""
The "vector database" is LlamaIndex's built-in SimpleVectorStore, saved as a
few JSON files under data/index/. No server to install; for a POC of a few
dozen PDFs it is more than enough. Swapping it for Chroma/Qdrant later means
changing the StorageContext in _new_index() and nothing else.

Every chunk keeps two pieces of metadata:
    filename    which PDF it came from   (used to search only some documents)
    page_label  which page               (becomes the "location" in evidence)

Public functions:
    build_index()        rebuild from every PDF in data/Documentations
    add_document(path)   index one (new or re-uploaded) PDF
    remove_document(fn)  drop one PDF from the index
    index_status()       counts, for the UI
    get_index()          the loaded index, used by rag/retrieval.py
    read_pages(path)     the raw page texts, used for the document preview
"""
from pathlib import Path

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from rag import config

if not config.GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing (needed for embeddings).")

# LlamaIndex reads these globals whenever an index needs to embed or split
# text, so setting them once here configures every index operation below.
Settings.embed_model = GoogleGenAIEmbedding(
    model_name=config.EMBED_MODEL,
    api_key=config.GEMINI_API_KEY,
)
Settings.node_parser = SentenceSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)

# The one index in memory. Loaded from disk on first use, kept for the life of
# the server process, saved back to disk after every change.
_index: VectorStoreIndex | None = None


def read_pages(pdf_path: Path) -> list[Document]:
    """Read one PDF into one Document per page, with clean metadata.

    SimpleDirectoryReader (via pypdf) already splits by page and gives us
    page_label. We keep only the two fields we need so the metadata stored
    with every chunk stays small and readable. The document id is
    "<filename>::page<N>" so all chunks of a file can be found and removed
    together (see remove_document).
    """
    pages = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    for page in pages:
        page_label = page.metadata.get("page_label", "?")
        page.metadata = {"filename": pdf_path.name, "page_label": page_label}
        page.id_ = f"{pdf_path.name}::page{page_label}"
    return pages


def _new_index() -> VectorStoreIndex:
    return VectorStoreIndex([], storage_context=StorageContext.from_defaults())


def _persist(index: VectorStoreIndex) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(config.INDEX_DIR))


def get_index() -> VectorStoreIndex:
    """The index, loaded from data/index/ on first call (or empty if absent)."""
    global _index
    if _index is None:
        if (config.INDEX_DIR / "docstore.json").exists():
            storage = StorageContext.from_defaults(persist_dir=str(config.INDEX_DIR))
            _index = load_index_from_storage(storage)
        else:
            _index = _new_index()
    return _index


def _filenames_in(index: VectorStoreIndex) -> set[str]:
    # ref_doc_info maps "<filename>::page<N>" -> the chunks made from that page
    return {ref_id.split("::")[0] for ref_id in index.ref_doc_info}


def remove_document(filename: str) -> int:
    """Delete every chunk of one PDF. Returns how many pages were removed."""
    index = get_index()
    page_ids = [ref_id for ref_id in index.ref_doc_info if ref_id.startswith(f"{filename}::")]
    for ref_id in page_ids:
        index.delete_ref_doc(ref_id, delete_from_docstore=True)
    if page_ids:
        _persist(index)
    return len(page_ids)


def add_document(pdf_path: Path) -> dict:
    """Index one PDF (replacing any previous version of the same filename).

    This is the only function that calls the embedding API when a user
    uploads a file: one request per batch of chunks.
    """
    pdf_path = Path(pdf_path)
    remove_document(pdf_path.name)
    index = get_index()
    pages = read_pages(pdf_path)
    before = len(index.docstore.docs)
    for page in pages:
        index.insert(page)      # split into chunks + embed + store
    _persist(index)
    return {"filename": pdf_path.name, "pages": len(pages), "chunks": len(index.docstore.docs) - before}


def build_index() -> dict:
    """Throw the index away and rebuild it from every PDF in data/Documentations."""
    global _index
    _index = _new_index()
    results = [add_document(pdf) for pdf in sorted(config.DOCS_DIR.glob("*.pdf"))]
    _persist(_index)
    return {"indexed": results, **index_status()}


def index_status() -> dict:
    index = get_index()
    return {
        "documents": len(_filenames_in(index)),
        "chunks": len(index.docstore.docs),
        "filenames": sorted(_filenames_in(index)),
        "index_dir": str(config.INDEX_DIR),
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


def document_preview(pdf_path: Path) -> str:
    """The first DOC_PREVIEW_CHARS of text, shown to the document router (Prompt 1)."""
    text = " ".join(page.text for page in read_pages(Path(pdf_path)))
    return " ".join(text.split())[: config.DOC_PREVIEW_CHARS]
