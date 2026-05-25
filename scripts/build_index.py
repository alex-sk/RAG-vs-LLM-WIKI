"""Build a Chroma vector index over corpus/*.md for the RAG pipeline.

Chunks each article into ~400-token windows with 50-token overlap, embeds
with OpenAI text-embedding-3-small, persists to data/chroma/.

Run: uv run scripts/build_index.py
"""
from __future__ import annotations

import os
from pathlib import Path

import chromadb
import tiktoken
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
CHROMA_DIR = ROOT / "data" / "chroma"

ENC = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50


def chunk_text(text: str) -> list[str]:
    toks = ENC.encode(text)
    if len(toks) <= CHUNK_TOKENS:
        return [text]
    chunks: list[str] = []
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(toks), step):
        chunk = ENC.decode(toks[start : start + CHUNK_TOKENS])
        chunks.append(chunk)
        if start + CHUNK_TOKENS >= len(toks):
            break
    return chunks


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (copy .env.example to .env)")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset collection for clean rebuilds
    try:
        client.delete_collection("wiki")
    except Exception:
        pass

    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    coll = client.create_collection(name="wiki", embedding_function=embed_fn)

    docs: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    files = sorted(CORPUS_DIR.glob("*.md"))
    print(f"Indexing {len(files)} files...")
    for f in files:
        text = f.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            ids.append(f"{f.stem}::{i}")
            metadatas.append({"source": f.name, "slug": f.stem, "chunk": i})

    print(f"Embedding {len(docs)} chunks (batched)...")
    BATCH = 200
    for i in range(0, len(docs), BATCH):
        coll.add(
            documents=docs[i : i + BATCH],
            ids=ids[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )
        print(f"  {min(i + BATCH, len(docs))} / {len(docs)}")

    print(f"Index built at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
