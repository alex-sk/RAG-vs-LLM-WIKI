"""Smoke test all four pipelines against the curated questions.

Prints a short pass/fail-ish report — useful for verifying the demo before
running it live in front of an exec audience. Graph RAG is skipped with a
warning if the graph hasn't been built yet (run scripts/build_graph.py).

Run: uv run scripts/smoke_test.py [--limit 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import agentic_rag, graph_rag, rag, wiki
from backend.agentic_rag import agentic_rag_stream
from backend.graph_rag import graph_rag_stream
from backend.rag import rag_stream
from backend.wiki import wiki_stream


def init_shared() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    openai_client = AsyncOpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
    chroma_coll = chroma_client.get_collection(name="wiki")
    rag.init(openai_client, chroma_coll)
    wiki.init(openai_client)
    agentic_rag.init(openai_client, chroma_coll)
    wiki.preload_corpus()
    graph_rag.init(openai_client, ROOT / "data" / "graph")


def looks_correct(answer: str, gold: str) -> bool:
    a = answer.lower()
    g = gold.lower().strip()
    if not g:
        return False
    return g in a


async def run_one(q: dict, pipeline_name: str, stream_fn) -> dict:
    answer_parts: list[str] = []
    tool_calls: list[dict] = []
    final = None
    async for ev in stream_fn(q["question"]):
        if ev["event"] == "token":
            answer_parts.append(ev["text"])
        elif ev["event"] == "tool_call":
            tool_calls.append({"tool": ev["tool"], "args": ev["args"]})
        elif ev["event"] == "done":
            final = ev
    answer = "".join(answer_parts).strip()
    correct = looks_correct(answer, q["answer"])
    return {
        "pipeline": pipeline_name,
        "answer": answer,
        "correct": correct,
        "gold": q["answer"],
        "metrics": final or {},
        "tool_calls": tool_calls,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    init_shared()
    questions = json.loads((ROOT / "data" / "demo_questions.json").read_text())[: args.limit]

    for q in questions:
        print(f"\n{'=' * 80}\nQ [{q['type']}/{q['level']}]: {q['question']}")
        print(f"Gold: {q['answer']}")

        rag_result = await run_one(q, "RAG", rag_stream)
        agent_result = await run_one(q, "Agent", agentic_rag_stream)
        results = [rag_result, agent_result]

        if graph_rag.is_built():
            results.append(await run_one(q, "Graph", graph_rag_stream))
        else:
            print("  (Graph RAG skipped — run scripts/build_graph.py first)")

        results.append(await run_one(q, "Wiki", wiki_stream))

        for r in results:
            mark = "✓" if r["correct"] else "✗"
            m = r["metrics"]
            print(f"\n  {mark} {r['pipeline']:6s} ({m.get('t_ms', '?')}ms, ${m.get('cost_usd', 0):.5f})")
            print(f"      {r['answer']}")
            if r["tool_calls"]:
                print(f"      tools: {[tc['tool'] + ':' + str(list(tc['args'].values())[0])[:30] for tc in r['tool_calls']]}")


if __name__ == "__main__":
    asyncio.run(main())
