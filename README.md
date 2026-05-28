# RAG vs Agentic RAG vs Graph RAG vs LLM Wiki — Side-by-Side PoC

A small, hands-on comparison of four retrieval architectures answering the same question over the same Wikipedia corpus:

- **RAG** — text embeddings → top-5 chunks from a Chroma vector store → LLM answer. Optional re-ranking stage (toggle on the RAG panel): pull 20 candidates, then narrow to 5 with either a local cross-encoder (`ms-marco-MiniLM-L-6-v2`) or an LLM-as-reranker (`gpt-4o-mini`).
- **Agentic RAG** — the same agent loop as LLM Wiki, but with a `vector_search` tool over the same Chroma index alongside `read_file` / `grep` / `glob`. The first vector search is pre-seeded into the opening prompt; the agent then decides what to read, whether to follow a `[[wiki-link]]`, or whether to re-search with a refined query.
- **Graph RAG** — a knowledge graph of typed entities and typed relationships is extracted from the corpus offline by an LLM (one-time build). At query time, a `gpt-4o-mini` pass extracts seed entities from the question, the graph expands a 2-hop neighborhood around the seeds, supporting article snippets are fetched, and `gpt-4o` answers over the combined graph + evidence.
- **LLM Wiki** (Karpathy-style) — an agent navigates the corpus directly via `glob` / `read_file` / `grep` tools over markdown files, following `[[wiki-links]]` between articles. No embeddings, no graph.

All four pipelines use the same pinned generation model (`gpt-4o-2024-08-06`) — only retrieval differs. The frontend renders them as side-by-side panels with live tool-call / chunk-retrieval / neighborhood-expansion traces so the contrast is visible. A chip row above the panels lets you pick which methods to compare; default visible set is RAG + Agentic RAG + Graph RAG (LLM Wiki off by default to keep the side-by-side roomy — toggle it back on with one click).

## Why HotpotQA

[HotpotQA](https://hotpotqa.github.io/) is a multi-hop QA dataset over Wikipedia paragraphs. It ships with ground-truth supporting paragraphs for each question, which makes it usable as a public stand-in for a private corpus (e.g., internal policy documents) where:

- Ground-truth Q's exist, so any claim about answer correctness is verifiable.
- Data sensitivity is not a barrier to demoing the comparison publicly.
- Cross-document reasoning patterns (multi-hop "bridge" questions) map directly to real cross-referencing tasks like policy lookup.

## Setup

```bash
# Backend
uv sync
cp .env.example .env             # add OPENAI_API_KEY

# Build the corpus + index + dropdown (one-time, ~5 minutes)
uv run scripts/build_corpus.py --num-questions 300 --seed 42
uv run scripts/build_index.py
uv run scripts/curate_questions.py                # 7 cherry-picked questions
uv run scripts/curate_questions.py --add-strata   # +8 stratified-random → 15 total

# Build the knowledge graph for Graph RAG (one-time, ~20 minutes in gpt-4o-mini)
uv run scripts/build_graph.py --max-cost-usd 3

# Frontend
cd frontend && npm install && cd ..
```

## Run

```bash
# terminal 1 — backend
uv run uvicorn backend.app:app --reload

# terminal 2 — frontend
cd frontend && npm run dev
```

Then visit http://localhost:5173.

> First-run note: the first time you select **cross-encoder** rerank, the model (~80 MB) is downloaded into the Hugging Face cache. Subsequent runs load it from disk in ~1 s. The backend kicks off the load in a background thread at startup so the first request stays snappy.

## Project structure

```
backend/
  app.py         FastAPI app, lifespan-managed shared resources, SSE endpoints
  rag.py         Direct embeddings call → Chroma query → optional rerank → streamed answer
  rerank.py      Cross-encoder + LLM-as-reranker strategies
  wiki.py        Agent loop with glob / read_file / grep over an in-memory corpus
  agentic_rag.py Agent loop with pre-seeded vector_search + read_file / grep / glob over the shared Chroma index
  graph_rag.py   Seed extraction → NetworkX neighborhood expansion → evidence fetch → streamed answer

corpus/          Generated: <slug>.md per Wikipedia entity, with [[wiki-links]]
data/
  chroma/        Generated: vector store
  graph/         Generated: entities.jsonl + edges.jsonl + by_file.jsonl + manifest.json
  all_questions.json    All 300 sampled HotpotQA questions
  demo_questions.json   15 dropdown questions (7 cherry-picks + 8 stratified)

frontend/
  src/App.tsx                          Side-by-side UI shell, method-picker integration
  src/components/MethodPicker.tsx      Chip row that toggles which panels are visible (localStorage-persisted)
  src/components/MitigationsBanner.tsx Production-hardening caveat callout
  src/components/PipelinePanel.tsx     Streaming chunk / tool / graph-neighborhood trace + answer
  src/components/QueryBar.tsx          Input + benchmark-question dropdown

scripts/         build_corpus, build_index, build_graph, curate_questions, smoke_test
```

## Smoke test

```bash
uv run scripts/smoke_test.py --limit 3
```

Runs both pipelines against the first N curated questions and prints a short pass/fail-ish report. Uses the same substring check the UI uses, with the same caveats.
