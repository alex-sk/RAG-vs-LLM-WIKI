"""Build a markdown wiki + curated demo questions from HotpotQA.

Pulls the dev (distractor) split, samples N questions, writes each unique
context paragraph as corpus/<slug>.md with title, summary, body, and
[[wiki-links]] to other entities in the corpus.

Run: uv run scripts/build_corpus.py [--num-questions 300]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"


def slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "untitled"


def find_wiki_links(body: str, self_slug: str, titles_by_slug: dict[str, str]) -> str:
    """Wrap occurrences of other entity titles in [[wiki-link]] syntax.

    Greedy longest-first match so 'United States Army' beats 'United States'.
    Skips the article's own title (no self-links). Only links the first
    occurrence per article to avoid noise.
    """
    candidates = [(s, t) for s, t in titles_by_slug.items() if s != self_slug]
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    used: set[str] = set()
    out = body
    for slug, title in candidates:
        if slug in used:
            continue
        pattern = re.compile(r"(?<!\[)\b" + re.escape(title) + r"\b(?!\])", re.IGNORECASE)
        m = pattern.search(out)
        if not m:
            continue
        replacement = f"[[{slug}|{m.group(0)}]]"
        out = out[: m.start()] + replacement + out[m.end():]
        used.add(slug)
    return out


def write_article(slug: str, title: str, body: str, links_to: list[str]) -> None:
    summary = body.split(". ")[0].strip()
    if not summary.endswith("."):
        summary += "."

    see_also = ""
    if links_to:
        bullets = "\n".join(f"- [[{s}]]" for s in sorted(set(links_to)))
        see_also = f"\n\n## See also\n\n{bullets}\n"

    content = f"# {title}\n\n*Summary:* {summary}\n\n## Article\n\n{body}\n{see_also}"
    (CORPUS_DIR / f"{slug}.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    CORPUS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    for old in CORPUS_DIR.glob("*.md"):
        old.unlink()

    print(f"Loading HotpotQA dev split (distractor)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=args.seed).select(range(args.num_questions))

    titles_by_slug: dict[str, str] = {}
    raw_bodies: dict[str, str] = {}

    for ex in ds:
        ctx = ex["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            slug = slugify(title)
            if slug in raw_bodies:
                continue
            body = "".join(sentences).strip()
            if len(body) < 40:
                continue
            titles_by_slug[slug] = title
            raw_bodies[slug] = body

    print(f"Collected {len(raw_bodies)} unique entities from {args.num_questions} questions.")

    for slug, body in raw_bodies.items():
        linked = find_wiki_links(body, slug, titles_by_slug)
        links_to = [m.group(1) for m in re.finditer(r"\[\[([a-z0-9-]+)(?:\|[^\]]*)?\]\]", linked)]
        write_article(slug, titles_by_slug[slug], linked, links_to)

    # Save the question set for curation/eval
    questions = []
    for ex in ds:
        questions.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "level": ex["level"],
            "supporting_titles": list(ex["supporting_facts"]["title"]),
        })
    (DATA_DIR / "all_questions.json").write_text(json.dumps(questions, indent=2))
    print(f"Saved {len(questions)} questions to data/all_questions.json")
    print(f"Wrote {len(raw_bodies)} markdown files to corpus/")


if __name__ == "__main__":
    main()
