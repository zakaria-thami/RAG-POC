"""
"Semantic Retrieval B" from pipeline.jpg: find the chunks of the user's
documents that are closest in meaning to a question.

How it works: the question is embedded with the same model used at indexing
time, then compared (cosine similarity) with every stored chunk vector. The
closest TOP_K chunks come back with a score between 0 and 1.

Two filters sit on top of that:
  filenames          only look inside these documents (the ones Prompt 1
                     picked). None = search every document.
  similarity_cutoff  drop chunks below this score. This is the "Strong
                     similarity score?" diamond of the flowchart: if nothing
                     survives, the pipeline answers NA instead of asking the
                     model to reason about irrelevant text.
"""
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from rag import config
from rag.indexing import get_index


def search(
    query: str,
    filenames: list[str] | None = None,
    top_k: int | None = None,
    similarity_cutoff: float | None = None,
) -> list[dict]:
    """Return [{"filename", "page_label", "score", "text"}], best score first."""
    top_k = top_k if top_k is not None else config.TOP_K
    cutoff = similarity_cutoff if similarity_cutoff is not None else config.SIMILARITY_CUTOFF

    # Restrict the search to the chosen documents by matching the "filename"
    # metadata stored on every chunk (see rag/indexing.py).
    filters = None
    if filenames:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="filename", value=filenames, operator=FilterOperator.IN)
        ])

    retriever = get_index().as_retriever(similarity_top_k=top_k, filters=filters)
    nodes = retriever.retrieve(query)

    # A "postprocessor" runs on the retrieved list; this one simply removes
    # nodes whose score is under the cutoff.
    nodes = SimilarityPostprocessor(similarity_cutoff=cutoff).postprocess_nodes(nodes)

    return [
        {
            "filename": n.metadata.get("filename"),
            "page_label": n.metadata.get("page_label"),
            "score": round(n.score, 4) if n.score is not None else None,
            "text": n.get_content(),
        }
        for n in nodes
    ]
