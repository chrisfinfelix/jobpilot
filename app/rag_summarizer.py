"""
rag_summarizer.py
-------------------
This is the "R" + "AG" of RAG in practice:
  - Retrieval already happened in web_research.py (real search results)
  - Here, we Augment a prompt with those real snippets, and Generate
    a summary — but we FORCE the model to only use what's in front of
    it, and to cite which numbered source backs each claim.

This is what keeps the app from making things up: the model never
answers "what are Google's interview rounds" from its own training
memory — it answers from the specific snippets we hand it, and shows
its work.
"""

import json
import os
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a research summarizer for a job-application assistant.

You will be given a numbered list of SOURCES (search result snippets about a
company and role) and must summarize ONLY what those sources actually say.

Respond with ONLY a raw JSON object — no markdown fences, no commentary —
matching exactly this shape:

{
  "company_overview": "<2-3 sentence overview, citing sources like [1] [2]>",
  "role_expectations": [<strings: what the role likely requires, each ending with a citation like [3]>],
  "interview_process": [<strings: what the interview process looks like, each with a citation>],
  "culture_notes": [<strings: culture/work-environment notes, each with a citation>],
  "confidence": "<'high' | 'medium' | 'low' — how much the sources actually covered>"
}

CRITICAL RULES:
- Every bullet point MUST end with a citation to a source number, e.g. "...on-site interviews. [2]"
- NEVER state a fact that isn't supported by the sources. If the sources don't
  cover something (e.g. interview process), say so honestly in that field
  (e.g. ["Sources did not cover the interview process for this role."]) rather
  than inventing one.
- If sources conflict, mention both perspectives.
- Do not use outside knowledge beyond what's in the sources, even if you
  recognize the company. Sources may be incomplete or outdated — treat them
  as the only truth available to you right now.
- Set "confidence" to "low" if the sources are thin, off-topic, or mostly
  about a different role/company than asked.
"""


class SummarizerError(Exception):
    pass


def _format_sources(sources: list[dict]) -> str:
    """Turns the raw source list into a numbered block the model can cite."""
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s['title']}\nURL: {s['url']}\n{s['snippet']}\n")
    return "\n".join(lines)


def summarize_research(company: str, role: str, sources: list[dict]) -> dict:
    """
    Takes the raw source list from web_research.gather_research() and
    returns a structured, cited summary dict.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SummarizerError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    client = Groq(api_key=api_key)

    sources_block = _format_sources(sources)
    user_prompt = f"""COMPANY: {company}
ROLE: {role}

SOURCES:
{sources_block}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise SummarizerError(f"Groq API request failed: {e}") from e

    raw_content = response.choices[0].message.content

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise SummarizerError(f"Groq returned non-JSON output: {e}") from e

    required_keys = {"company_overview", "role_expectations", "interview_process", "culture_notes", "confidence"}
    if not required_keys.issubset(result.keys()):
        raise SummarizerError(f"Groq response missing expected fields: {result}")

    # Attach the actual source list (title + url) so the frontend can show
    # real clickable citations, not just "[1]" with nothing behind it.
    result["sources"] = [
        {"index": i + 1, "title": s["title"], "url": s["url"]}
        for i, s in enumerate(sources)
    ]

    return result
