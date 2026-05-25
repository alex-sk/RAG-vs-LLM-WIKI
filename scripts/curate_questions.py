"""Curate demo questions from data/all_questions.json.

Two modes:

1. Default (no flags) — original cherry-picked set: 5 bridge/hard + 2 comparison
   curated to showcase the agent's multi-hop strength. Overwrites
   data/demo_questions.json.

2. --add-strata — reads existing data/demo_questions.json and APPENDS a
   stratified-random sample of 8 additional questions (4 bridge/hard +
   4 comparison/hard) drawn from the remaining valid questions with a
   fixed seed (default 42). Deduplicates by question id so re-running
   is idempotent.

   Note: HotpotQA's validation split is uniformly level='hard' — easy
   and medium only exist in the train split. The corpus is built from
   validation, so we stratify by type only. Adding true level diversity
   would require sourcing extra paragraphs from the train split and
   re-indexing.

The two-mode split is intentional: the cherry-picks are explicitly
illustrative; the strata additions broaden the dropdown so an interviewer
sees more topic variety than the cherry-pick alone.

Run:
  uv run scripts/curate_questions.py                 # 7 cherry-picks (resets)
  uv run scripts/curate_questions.py --add-strata    # appends 8 stratified
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "demo_questions.json"

STRATA_TO_ADD = [
    ("bridge", "hard", 4),
    ("comparison", "hard", 4),
]


def slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "untitled"


def load_valid_questions() -> list[dict]:
    questions = json.loads((DATA_DIR / "all_questions.json").read_text())
    corpus_slugs = {f.stem for f in CORPUS_DIR.glob("*.md")}
    valid = []
    for q in questions:
        support_slugs = [slugify(t) for t in q["supporting_titles"]]
        if all(s in corpus_slugs for s in support_slugs):
            q["supporting_slugs"] = support_slugs
            valid.append(q)
    return valid


def slim(q: dict) -> dict:
    return {
        "id": q["id"],
        "question": q["question"],
        "answer": q["answer"],
        "type": q["type"],
        "level": q["level"],
        "supporting_slugs": q["supporting_slugs"],
    }


def cherry_pick(valid: list[dict]) -> list[dict]:
    bridges_hard = [q for q in valid if q["type"] == "bridge" and q["level"] == "hard"]
    bridges_med = [q for q in valid if q["type"] == "bridge" and q["level"] == "medium"]
    comparisons = [q for q in valid if q["type"] == "comparison"]
    picks = (bridges_hard[:5] + bridges_med[:3] + comparisons[:2])[:10]
    return [slim(q) for q in picks]


def add_strata(existing: list[dict], valid: list[dict], seed: int) -> list[dict]:
    """Append stratified-random samples, skipping ids already in existing."""
    used_ids = {q["id"] for q in existing}
    rng = random.Random(seed)
    additions: list[dict] = []
    for qtype, level, count in STRATA_TO_ADD:
        pool = [q for q in valid if q["type"] == qtype and q["level"] == level and q["id"] not in used_ids]
        rng.shuffle(pool)
        chosen = pool[:count]
        if len(chosen) < count:
            print(f"  warning: stratum {qtype}/{level} only had {len(chosen)} candidates (wanted {count})")
        additions.extend(slim(q) for q in chosen)
        used_ids.update(q["id"] for q in chosen)
    return existing + additions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-strata", action="store_true",
                        help="Append a stratified-random sample to existing demo_questions.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    valid = load_valid_questions()
    print(f"{len(valid)} questions have all supporting paragraphs in corpus.")

    if args.add_strata:
        if not OUT_PATH.exists():
            raise SystemExit("data/demo_questions.json missing — run without --add-strata first to seed.")
        existing = json.loads(OUT_PATH.read_text())
        combined = add_strata(existing, valid, args.seed)
        OUT_PATH.write_text(json.dumps(combined, indent=2))
        print(f"\nAppended {len(combined) - len(existing)} stratified questions "
              f"({len(existing)} → {len(combined)} total).")
        print("Final dropdown:")
        for q in combined:
            print(f"  [{q['type']}/{q['level']}] {q['question']}")
    else:
        demo = cherry_pick(valid)
        OUT_PATH.write_text(json.dumps(demo, indent=2))
        print(f"Wrote {len(demo)} cherry-picked questions to {OUT_PATH}")
        for q in demo:
            print(f"  [{q['type']}/{q['level']}] {q['question']}")


if __name__ == "__main__":
    main()
