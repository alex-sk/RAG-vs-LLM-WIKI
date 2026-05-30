"""FastAPI app: SSE endpoints for RAG and LLM Wiki pipelines + demo questions."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

from backend import agentic_rag, graph_rag, judge, rag, wiki

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"
GRAPH_DIR = ROOT / "data" / "graph"


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
    agentic_rag.init(openai_client, chroma_coll)
    judge.init(openai_client)
    n_files = wiki.preload_corpus()
    # graph_rag.init() must run after wiki.preload_corpus() — it does not
    # itself load the corpus, but the runtime fetch_evidence step reads from
    # wiki._CORPUS, so the corpus must already be in memory.
    graph_rag.init(openai_client, GRAPH_DIR)
    # Build the entity-embedding matrix in the background so the first Graph RAG
    # query that needs the NN seed fallback doesn't pay the embed cost inline.
    emb_warm_task = asyncio.create_task(graph_rag.warm_entity_embeddings())
    print(f"[startup] OpenAI client ready, Chroma collection loaded, {n_files} corpus files in memory")

    yield

    emb_warm_task.cancel()
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
async def rag_endpoint(question: str, request: Request, rerank: str = "none"):
    mode = rerank if rerank in ("none", "cross-encoder", "llm") else "none"

    async def gen():
        async for ev in rag.rag_stream(question, request, rerank_mode=mode):
            yield {"data": json.dumps(ev)}
    return EventSourceResponse(gen())


@app.get("/api/wiki")
async def wiki_endpoint(question: str, request: Request):
    async def gen():
        async for ev in wiki.wiki_stream(question, request):
            yield {"data": json.dumps(ev)}
    return EventSourceResponse(gen())


@app.get("/api/agentic-rag")
async def agentic_rag_endpoint(question: str, request: Request, rerank: str = "none"):
    mode = rerank if rerank in ("none", "cross-encoder", "llm") else "none"

    async def gen():
        async for ev in agentic_rag.agentic_rag_stream(question, request, rerank_mode=mode):
            yield {"data": json.dumps(ev)}
    return EventSourceResponse(gen())


@app.get("/api/graph-rag")
async def graph_rag_endpoint(question: str, request: Request):
    async def gen():
        async for ev in graph_rag.graph_rag_stream(question, request):
            yield {"data": json.dumps(ev)}
    return EventSourceResponse(gen())


class JudgeRequest(BaseModel):
    question: str
    gold: str
    answer: str


@app.post("/api/judge")
async def judge_endpoint(req: JudgeRequest):
    return await judge.judge(req.question, req.gold, req.answer)


@app.get("/api/health")
def health():
    corpus = ROOT / "corpus"
    return {
        "corpus_files": len(list(corpus.glob("*.md"))) if corpus.exists() else 0,
        "corpus_in_memory": len(wiki._CORPUS),
        "index_built": CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()),
        "graph": graph_rag.graph_stats(),
        "model": rag.GENERATION_MODEL,
    }
