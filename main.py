from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from api.documentation import router as docs_router
from api.llm import router as llm_router
from api.regulations import router as regulations_router
from api.rag import router as rag_router

app = FastAPI(title="RAG POC")

# Serve files (like our CSS) that live in the "static" folder at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Plug in the routes
app.include_router(docs_router)
app.include_router(llm_router)
app.include_router(regulations_router)
app.include_router(rag_router)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def serve_ui(r: Request):
    return templates.TemplateResponse(request=r, name="index.html", context={"message": "home"})
