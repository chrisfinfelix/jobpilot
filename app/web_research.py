"""
web_research.py
-----------------
Collects raw, real information about a company + role from the open web.
This module ONLY gathers sourced snippets — it does not summarize or
interpret anything. That separation is the whole point of RAG: gather
real facts first, then (in rag_summarizer.py) ask the AI to summarize
ONLY from what was actually found.

Uses DuckDuckGo search (via the `ddgs` package) because it's free and
needs no API key — good fit for a student project. Swap this module out
later for a paid search API (SerpAPI, Google CSE, etc.) without touching
any other file, since it only needs to return the same source-list shape.
"""

from ddgs import DDGS

# Each query targets a different angle of "researching a company/role" —
# mirrors what a person would actually Google before applying.
QUERY_TEMPLATES = [
    "{company} careers {role}",
    "{company} {role} interview process",
    "{company} interview experience review",
    "{company} company culture engineering",
]

RESULTS_PER_QUERY = 4


class WebResearchError(Exception):
    pass


def _run_query(query: str, max_results: int) -> list[dict]:
    """Run one DuckDuckGo search and return normalized result dicts."""
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        # A single failed query shouldn't kill the whole research pass —
        # we just return nothing for this angle and move on.
        return []

    normalized = []
    for r in raw_results:
        normalized.append({
            "title": r.get("title", "").strip(),
            "url": r.get("href", r.get("link", "")).strip(),
            "snippet": r.get("body", r.get("snippet", "")).strip(),
            "query": query,
        })
    return normalized


def gather_research(company: str, role: str) -> list[dict]:
    """
    Runs several targeted searches and returns a deduplicated list of
    source snippets: [{title, url, snippet, query}, ...]

    Raises WebResearchError only if EVERY query failed (e.g. no internet).
    """
    all_results = []
    seen_urls = set()

    for template in QUERY_TEMPLATES:
        query = template.format(company=company, role=role)
        results = _run_query(query, RESULTS_PER_QUERY)
        for r in results:
            if r["url"] and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)

    if not all_results:
        raise WebResearchError(
            f"No research results found for '{company}' / '{role}'. "
            "This could mean no internet connection, or the search "
            "returned nothing usable — try a more specific company name."
        )

    return all_results
