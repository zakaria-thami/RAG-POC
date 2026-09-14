"""
rag - the retrieval-augmented-generation core of this POC.

This package is deliberately self-contained: nothing in here imports from
api/ or main.py, and nothing in here knows about FastAPI. That is what makes
it a "micro-service": copy the rag/ folder into another project, point
rag/config.py at that project's folders, and call rag.pipeline.answer_question()
from whatever web framework (or plain script) that project uses.

Reading order for someone new:
  config.py       -> where files live, which models are used, the tunable numbers
  llm.py          -> how we talk to the chat model (OpenRouter)
  regulations.py  -> "Lexical Retrieval A": pull one article out of a regulation
  questions.py    -> load the questionnaire (data/questions.json)
  indexing.py     -> turn PDFs into searchable chunks (embeddings + vector index)
  retrieval.py    -> "Semantic Retrieval B": find the chunks closest to a question
  prompts.py      -> the two system prompts and how their user messages are built
  pipeline.py     -> the flowchart in pipeline.jpg, one function
"""
