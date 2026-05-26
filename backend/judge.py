"""LLM-as-judge: grade a pipeline answer against the HotpotQA gold answer."""
from __future__ import annotations

import json

from openai import AsyncOpenAI

JUDGE_MODEL = "gpt-4o-mini-2024-07-18"

_openai_client: AsyncOpenAI | None = None


def init(openai_client: AsyncOpenAI) -> None:
    global _openai_client
    _openai_client = openai_client


SYSTEM_PROMPT = """You grade short-answer QA against a HotpotQA gold answer. The gold is authoritative — it specifies the exact fact required, at the exact level of specificity required.

Mark the candidate CORRECT when it explicitly states the gold fact, allowing for:
- Paraphrasing ("is a poet" for gold "poet")
- Formatting, quote, and punctuation differences ('Hotel (1965)' for '"Hotel" (1965)')
- Reordering of multi-item lists
- Extra surrounding context

Mark the candidate INCORRECT when:
- It uses a broader category instead of the gold's specific term ("German" for "Prussian"; "a country" for "France"; "a musician" for "guitarist")
- It uses a related-but-different term ("Bavarian" for "Prussian"; "Sony" for "Sony Music")
- It is missing any item from a multi-item gold answer
- It hedges, refuses, or says the answer is unknown

Be strict on specificity. Do not infer that a broader answer "implies" the gold — if the gold term (or a true synonym) is not present in the candidate, the candidate is incorrect.

Reply with JSON: {"correct": bool, "reason": "<one short sentence quoting the relevant part of the candidate>"}."""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "correct": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["correct", "reason"],
            "additionalProperties": False,
        },
    },
}


async def judge(question: str, gold: str, answer: str) -> dict:
    assert _openai_client is not None, "judge.init() not called"

    user_msg = (
        f"Question: {question}\n\n"
        f"Gold answer: {gold}\n\n"
        f"Candidate answer: {answer}"
    )

    resp = await _openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0,
        seed=42,
    )

    return json.loads(resp.choices[0].message.content or "{}")
