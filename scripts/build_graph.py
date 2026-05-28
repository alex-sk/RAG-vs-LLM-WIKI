"""Build a knowledge graph over corpus/*.md for the Graph RAG pipeline.

For each markdown file, runs an LLM extraction pass (gpt-4o-mini) that returns
typed entities and typed relationships. Appends one record per file to
data/graph/by_file.jsonl so a crashed/interrupted run can resume cheaply.

After all per-file extractions are done, a merge pass canonicalises entity
ids (slugify(name)), fuzzy-merges alias variants within the same type, and
writes the final entities.jsonl + edges.jsonl + manifest.json.

Run: uv run scripts/build_graph.py [--limit N] [--max-cost-usd 3]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
GRAPH_DIR = ROOT / "data" / "graph"
BY_FILE_PATH = GRAPH_DIR / "by_file.jsonl"
ENTITIES_PATH = GRAPH_DIR / "entities.jsonl"
EDGES_PATH = GRAPH_DIR / "edges.jsonl"
MANIFEST_PATH = GRAPH_DIR / "manifest.json"

MODEL = "gpt-4o-mini-2024-07-18"
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60

CONCURRENCY = 16
MAX_BODY_CHARS = 6000  # truncate huge articles before extraction

ENTITY_TYPES = ["person", "place", "building", "org", "event", "work", "concept", "other"]
RELATION_TYPES = [
    "located_in", "part_of", "member_of", "founded_by", "founded",
    "designed_by", "designed", "directed_by", "directed", "written_by",
    "produced_by", "starred_in", "born_in", "died_in", "married_to",
    "parent_of", "child_of", "succeeded_by", "preceded_by", "other",
]

EXTRACTION_PROMPT = f"""Extract named entities and the relationships between them from the
article below. Be conservative: only emit relationships explicitly supported
by the text. Skip generic concepts (e.g. "city", "movie") — only specific
named entities.

Entity types: {', '.join(ENTITY_TYPES)}
Relation types: {', '.join(RELATION_TYPES)}

Use canonical surface forms (e.g. "Robert Moses", not "Moses, Robert" or
"Mr. Moses"). Use snake_case relation values from the list above; use "other"
only as a last resort. Always direct relationships from subject to object
(e.g. for "X designed Y", emit src=X rel=designed dst=Y)."""


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                },
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "rel": {"type": "string", "enum": RELATION_TYPES},
                    "dst": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["src", "rel", "dst", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "edges"],
    "additionalProperties": False,
}


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s)
    s = s.strip("-")
    return s or "unknown"


def cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok * PRICE_IN_PER_M + out_tok * PRICE_OUT_PER_M) / 1_000_000


def load_done_files() -> set[str]:
    if not BY_FILE_PATH.exists():
        return set()
    done = set()
    with BY_FILE_PATH.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                done.add(rec["file"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def article_body(text: str) -> str:
    """Strip frontmatter-ish header, keep title + summary + article body."""
    # Each file looks like:
    #   # Title
    #   *Summary:* ...
    #   ## Article
    #   body
    # We send the whole thing (truncated) — gives the LLM full context.
    return text[:MAX_BODY_CHARS]


async def extract_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    file_path: Path,
    cost_tracker: dict,
    cost_cap: float,
) -> dict | None:
    async with sem:
        if cost_tracker["spent"] >= cost_cap:
            return None
        text = file_path.read_text(encoding="utf-8")
        body = article_body(text)
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"<source name=\"{file_path.name}\">\n{body}\n</source>"},
                ],
                temperature=0,
                seed=42,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "graph_extraction",
                        "schema": EXTRACTION_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except Exception as e:  # network / schema validation / etc
            return {
                "file": file_path.name,
                "error": str(e),
                "entities": [],
                "edges": [],
            }
        usage = resp.usage
        cost_tracker["in"] += usage.prompt_tokens
        cost_tracker["out"] += usage.completion_tokens
        cost_tracker["spent"] = cost_usd(cost_tracker["in"], cost_tracker["out"])
        try:
            parsed = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            parsed = {"entities": [], "edges": []}
        return {
            "file": file_path.name,
            "entities": parsed.get("entities", []),
            "edges": parsed.get("edges", []),
        }


def append_record(rec: dict) -> None:
    with BY_FILE_PATH.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_graph() -> tuple[int, int]:
    """Read by_file.jsonl, canonicalise + fuzzy-merge, write entities/edges."""
    # First pass: collect raw entities and edges with per-file provenance.
    raw_entities: list[tuple[str, str, str, str]] = []  # (file, name, type, slug)
    raw_edges: list[dict] = []

    with BY_FILE_PATH.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            file_name = rec.get("file", "")
            for e in rec.get("entities", []):
                name = (e.get("name") or "").strip()
                typ = e.get("type") or "other"
                if not name:
                    continue
                raw_entities.append((file_name, name, typ, slugify(name)))
            for edge in rec.get("edges", []):
                src = (edge.get("src") or "").strip()
                dst = (edge.get("dst") or "").strip()
                rel = edge.get("rel") or "other"
                if not src or not dst:
                    continue
                raw_edges.append({
                    "src_name": src,
                    "src_slug": slugify(src),
                    "rel": rel,
                    "dst_name": dst,
                    "dst_slug": slugify(dst),
                    "source": file_name,
                    "evidence": (edge.get("evidence") or "")[:240],
                })

    # Second pass: build canonical entities.
    #
    # Merge rule: two entities collapse only if they have the SAME type AND
    # the same sorted-token-set of word characters (ignoring case + tokens
    # ≤1 char). This catches the targeted case ("Robert Moses" ↔ "Moses,
    # Robert" ↔ "robert moses") without false-merging similarly-named-but-
    # distinct entities ("15 Hudson Yards" vs "10 Hudson Yards" — different
    # token sets because "10" ≠ "15").
    def norm_tokens(name: str) -> tuple:
        toks = re.findall(r"\w+", name.lower())
        toks = [t for t in toks if len(t) > 1]
        return tuple(sorted(toks))

    by_norm: dict[tuple[str, tuple], dict] = {}
    canonical_id_map: dict[tuple[str, str], str] = {}  # (raw_slug, type) -> canonical_id

    # Sort raw entities so the most-supported surface name wins as canonical.
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for file_name, name, typ, slug in raw_entities:
        counts[(slug, typ, name)] += 1

    # Order: by descending count, then alphabetical for determinism.
    grouped: dict[tuple[str, str], dict] = {}
    for (slug, typ, name), n in counts.items():
        key = (slug, typ)
        if key not in grouped or n > grouped[key]["count"]:
            grouped[key] = {"name": name, "count": n}

    # Now walk per-entity, attach all sources/aliases.
    per_slug: dict[tuple[str, str], dict] = {}
    for file_name, name, typ, slug in raw_entities:
        key = (slug, typ)
        if key not in per_slug:
            per_slug[key] = {"id": slug, "name": grouped[key]["name"], "type": typ,
                              "aliases": set(), "sources": set()}
        ent = per_slug[key]
        if name != ent["name"]:
            ent["aliases"].add(name)
        ent["sources"].add(file_name)

    # Sort entities by descending support so the canonical id is the most-cited one.
    sorted_ents = sorted(per_slug.values(), key=lambda e: (-len(e["sources"]), e["id"]))

    final_entities: dict[str, dict] = {}
    for ent in sorted_ents:
        norm_key = (ent["type"], norm_tokens(ent["name"]))
        if not norm_key[1]:
            # No usable tokens after normalisation — keep as standalone.
            norm_key = (ent["type"], (ent["id"],))
        existing_id = by_norm.get(norm_key, {}).get("id")
        if existing_id is None:
            final_entities[ent["id"]] = {
                "id": ent["id"],
                "name": ent["name"],
                "type": ent["type"],
                "aliases": set(ent["aliases"]),
                "sources": set(ent["sources"]),
            }
            by_norm[norm_key] = {"id": ent["id"]}
            canonical_id_map[(ent["id"], ent["type"])] = ent["id"]
        else:
            tgt = final_entities[existing_id]
            if ent["name"] != tgt["name"]:
                tgt["aliases"].add(ent["name"])
            tgt["aliases"].update(ent["aliases"])
            tgt["sources"].update(ent["sources"])
            canonical_id_map[(ent["id"], ent["type"])] = existing_id

    # Resolve edges to canonical ids. We don't know the type at edge-time,
    # so try each known type bucket; if both endpoints resolve, keep the edge.
    def resolve(slug: str) -> str | None:
        for typ in ENTITY_TYPES:
            cid = canonical_id_map.get((slug, typ))
            if cid is not None:
                return cid
        return None

    final_edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for e in raw_edges:
        src_id = resolve(e["src_slug"])
        dst_id = resolve(e["dst_slug"])
        if not src_id or not dst_id or src_id == dst_id:
            continue
        key = (src_id, e["rel"], dst_id)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        final_edges.append({
            "src": src_id,
            "rel": e["rel"],
            "dst": dst_id,
            "source": e["source"],
            "evidence": e["evidence"],
        })

    # Write outputs.
    with ENTITIES_PATH.open("w") as fh:
        for ent in sorted(final_entities.values(), key=lambda e: e["id"]):
            ent_out = {
                "id": ent["id"],
                "name": ent["name"],
                "type": ent["type"],
                "aliases": sorted(ent["aliases"]),
                "sources": sorted(ent["sources"]),
            }
            fh.write(json.dumps(ent_out, ensure_ascii=False) + "\n")

    with EDGES_PATH.open("w") as fh:
        for edge in final_edges:
            fh.write(json.dumps(edge, ensure_ascii=False) + "\n")

    return len(final_entities), len(final_edges)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only extract the first N corpus files (for dry-run).")
    ap.add_argument("--max-cost-usd", type=float, default=3.0,
                    help="Abort extraction once this spend is reached.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip extraction; re-merge from existing by_file.jsonl.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (copy .env.example to .env)")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    cost_tracker = {"in": 0, "out": 0, "spent": 0.0}

    if not args.merge_only:
        all_files = sorted(CORPUS_DIR.glob("*.md"))
        if args.limit:
            all_files = all_files[: args.limit]

        done = load_done_files()
        todo = [f for f in all_files if f.name not in done]
        print(f"Corpus: {len(all_files)} files. Already extracted: {len(done)}. To do: {len(todo)}.")
        print(f"Model: {MODEL}. Cost cap: ${args.max_cost_usd:.2f}. Concurrency: {CONCURRENCY}.")

        client = AsyncOpenAI(api_key=api_key)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def run_and_save(fp: Path) -> dict | None:
            rec = await extract_one(client, sem, fp, cost_tracker, args.max_cost_usd)
            if rec is not None:
                append_record(rec)
            return rec

        if todo:
            tasks = [run_and_save(fp) for fp in todo]
            for _ in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="extracting"):
                await _
                if cost_tracker["spent"] >= args.max_cost_usd:
                    print(f"\n[cost cap hit] ${cost_tracker['spent']:.4f} >= ${args.max_cost_usd}. Stopping new extractions; in-flight will finish.")
                    break

        await client.close()
        print(f"Spent: ${cost_tracker['spent']:.4f} ({cost_tracker['in']:,} in / {cost_tracker['out']:,} out tokens)")

    print("Merging into canonical graph...")
    t_merge = time.time()
    n_ent, n_edge = merge_graph()
    print(f"  {n_ent:,} entities, {n_edge:,} edges (merge {int((time.time() - t_merge) * 1000)} ms)")

    manifest = {
        "model": MODEL,
        "prompt_hash": hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest()[:16],
        "n_entities": n_ent,
        "n_edges": n_edge,
        "cost_usd": round(cost_tracker["spent"], 6),
        "in_tokens": cost_tracker["in"],
        "out_tokens": cost_tracker["out"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
