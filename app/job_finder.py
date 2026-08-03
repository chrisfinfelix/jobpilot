"""
job_finder.py
--------------
Auto-fetches a real job description for a given company + role, instead
of requiring the user to paste one manually.

This is the same RAG discipline as Step 2, applied to a different problem:
1. Retrieval — search the web for an actual job posting page, fetch its
   raw HTML, strip it down to plain text (web_research.py's search +
   requests/BeautifulSoup for the fetch).
2. Augmented Generation — hand that messy raw page text to Groq and ask
   it to EXTRACT (not invent) the actual job description content: title,
   responsibilities, requirements. The model is explicitly told to pull
   only what's really on the page, and to say so honestly if the page
   didn't actually contain a real job posting.

Results are cached (Step 2's cache, "jobdesc" namespace) for 7 days so
repeat lookups of the same company+role don't re-fetch or re-spend an AI call.

Like Step 3's relationship to Step 2, this always lets the user see and
edit the result before it's used anywhere else — nothing here bypasses
human review.
"""

import json
import os
import re
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SEARCH_QUERY_TEMPLATE = "{company} {role} job description apply"
MAX_SEARCH_RESULTS = 5
FETCH_TIMEOUT_SECONDS = 8
MAX_PAGE_CHARS = 8000  # keep the raw page text within a reasonable prompt size

REQUEST_HEADERS = {
    # Some career pages block requests with no user agent at all.
    "User-Agent": "Mozilla/5.0 (compatible; JobPilotBot/1.0; student project)"
}

SYSTEM_PROMPT = """You are a job-description extractor.

You will be given the raw, messy text scraped from a web page that MIGHT
be a real job posting. Your job is to EXTRACT the actual job description
content from it — do not invent, guess, or fill in anything that isn't
really on the page.

Respond with ONLY a raw JSON object — no markdown fences, no commentary —
matching exactly this shape:

{
  "found_real_posting": <true or false>,
  "job_description": "<the extracted job title, responsibilities, and requirements, formatted as clean readable plain text with blank lines between sections. Empty string if found_real_posting is false>",
  "note": "<1 sentence: if found_real_posting is true, briefly say what was extracted; if false, honestly explain why (e.g. 'page appears to be a search results list, not a posting' or 'page was blocked/empty')>"
}

Rules:
- If the page text doesn't actually look like a real job posting (e.g. it's
  a generic careers landing page, a login wall, a cookie notice, a search
  results list, or clearly unrelated content), set found_real_posting to
  false and leave job_description empty. Do NOT invent a plausible-sounding
  job description to fill the gap.
- Strip out navigation menus, cookie banners, unrelated links, and site
  boilerplate — keep only the actual role title, responsibilities, and
  requirements/qualifications.
- Preserve real specifics from the page (technologies, years of experience,
  degree requirements) exactly as written. Do not add requirements that
  aren't on the page.
"""


class JobFinderError(Exception):
    pass


def _search_job_urls(company: str, role: str) -> list[dict]:
    """Search the web for likely job posting pages. Returns [{title, url}, ...]."""
    query = SEARCH_QUERY_TEMPLATE.format(company=company, role=role)
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    except Exception as e:
        raise JobFinderError(f"Web search failed: {e}") from e

    results = []
    for r in raw_results:
        url = r.get("href", r.get("link", "")).strip()
        title = r.get("title", "").strip()
        if url:
            results.append({"title": title, "url": url})
    return results


def _fetch_page_text(url: str) -> str | None:
    """
    Fetches a URL and strips it down to plain visible text.
    Returns None (rather than raising) on any failure — a single bad URL
    shouldn't kill the whole search, we just try the next one.
    """
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except Exception:
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "svg", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return text[:MAX_PAGE_CHARS] if text else None
    except Exception:
        return None


def _extract_job_description(company: str, role: str, page_text: str, source_url: str) -> dict:
    """Asks Groq to extract the real job description from raw page text."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise JobFinderError("GROQ_API_KEY is not set. Add it to your .env file.")
    client = Groq(api_key=api_key)

    user_prompt = f"""COMPANY: {company}
ROLE: {role}
SOURCE URL: {source_url}

RAW PAGE TEXT:
{page_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # this is extraction, not writing — stay literal
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise JobFinderError(f"Groq API request failed: {e}") from e

    raw_content = response.choices[0].message.content
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise JobFinderError(f"Groq returned non-JSON output: {e}") from e

    required_keys = {"found_real_posting", "job_description", "note"}
    if not required_keys.issubset(result.keys()):
        raise JobFinderError(f"Groq response missing expected fields: {result}")

    return result


def find_job_description(company: str, role: str) -> dict:
    """
    Main entry point. Searches for real postings, tries each URL until one
    yields an extractable job description, and returns:
      { job_description, source_url, source_title, note }

    Raises JobFinderError if search fails entirely, or if no candidate page
    yielded a real, extractable job posting.
    """
    candidates = _search_job_urls(company, role)
    if not candidates:
        raise JobFinderError(
            f"No search results found for '{company}' / '{role}'. Try a more specific role title."
        )

    last_note = "No candidate pages could be checked."
    for candidate in candidates:
        page_text = _fetch_page_text(candidate["url"])
        if not page_text or len(page_text) < 200:
            continue  # page didn't load or had almost no content — try the next one

        extracted = _extract_job_description(company, role, page_text, candidate["url"])
        if extracted["found_real_posting"] and extracted["job_description"].strip():
            return {
                "job_description": extracted["job_description"],
                "source_url": candidate["url"],
                "source_title": candidate["title"],
                "note": extracted["note"],
            }
        last_note = extracted.get("note", last_note)

    raise JobFinderError(
        f"Searched {len(candidates)} pages but couldn't find an extractable job posting. "
        f"Last attempt: {last_note} Try pasting the job description manually instead."
    )
