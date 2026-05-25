"""FastAPI app: SSE endpoints for RAG and LLM Wiki pipelines + demo questions."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from backend import rag, wiki

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources once per process."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    openai_client = AsyncOpenAI(api_key=api_key)

    # Chroma collection: we'll provide query embeddings directly, so no embedding
    # function is registered here (the build_index script handles the build side).
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_coll = chroma_client.get_collection(name="wiki")

    rag.init(openai_client, chroma_coll)
    wiki.init(openai_client)
    n_files = wiki.preload_corpus()
    print(f"[startup] OpenAI client ready, Chroma collection loaded, {n_files} corpus files in memory")

    yield

    await openai_client.close()


app = FastAPI(title="RAG vs LLM Wiki", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/demo-questions")
def demo_questions():
    path = ROOT / "data" / "demo_questions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@app.get("/api/rag")
async def rag_endpoint(question: str, request: Request):
    async def gen():
        async for ev in rag.rag_stream(question, request):
            yield {"data": json.dumps(ev)}
    return EventSourceResponse(gen())


@app.get("/api/wiki")
async def wiki_endpoint(question: str, request: Request):
    async def gen():
        async for ev in wiki.wiki_stream(question, request):
            yield {"data": json.dumps(ev)}
    return EventSourceResponse(gen())


@app.get("/api/health")
def health():
    corpus = ROOT / "corpus"
    return {
        "corpus_files": len(list(corpus.glob("*.md"))) if corpus.exists() else 0,
        "corpus_in_memory": len(wiki._CORPUS),
        "index_built": CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()),
        "model": rag.GENERATION_MODEL,
    }
