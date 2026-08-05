"""
contact_finder.py
------------------
Finds an email address to send the application to, for a given company.

This deliberately mirrors job_finder.py's structure — same search → fetch
→ extract discipline — but with one addition, because the failure mode
here is worse than a bad job-description prefill:

  - job_finder.py: if extraction fails, the user just sees an empty text
    box and pastes manually. No harm done.
  - contact_finder.py: if we got this wrong silently, an application could
    go to a nonexistent inbox, or worse, someone else's. So this module
    never hands back a single "best guess" as if it were verified — every
    result carries a `tier` field the caller (and the UI) must respect:

    tier="verified"  -> a real email address was actually found stated on
                         a real page (e.g. a mailto: link or plain text on
                         a careers/contact page). Comes with a source URL.
    tier="guess"      -> nothing real was found. These are unverified
                         pattern guesses (careers@domain, jobs@domain,
                         hr@domain) off a best-guess company domain. May
                         not exist or may not be monitored. Many companies
                         only accept applications through an ATS portal,
                         not email at all — that's expected, not a bug.
    tier="none"        -> couldn't even guess a domain. User must type the
                         recipient manually.

  Nothing in this module or its caller should ever fill the "To" field
  without the user seeing which tier it came from — see the review-gate
  modal in static/index.html, which labels tier="guess" results as
  unverified suggestions, not a filled-in answer.
"""

import json
import os
import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from ddgs import DDGS
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

CONTACT_SEARCH_QUERY_TEMPLATE = "{company} careers contact email jobs HR"
DOMAIN_SEARCH_QUERY_TEMPLATE = "{company} official website"
MAX_SEARCH_RESULTS = 5
FETCH_TIMEOUT_SECONDS = 8
MAX_PAGE_CHARS = 8000

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JobPilotBot/1.0; student project)"
}

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

SYSTEM_PROMPT = """You are a contact-email extractor.

You will be given the raw, messy text scraped from a web page that MIGHT
belong to a company's careers/contact/HR page. Your job is to find a REAL
email address that is actually written on the page for job applications,
recruiting, or general HR contact — do not invent, guess, or construct one.

Respond with ONLY a raw JSON object — no markdown fences, no commentary —
matching exactly this shape:

{
  "found_email": <true or false>,
  "email": "<the exact email address as written on the page, or empty string if none found>",
  "note": "<1 sentence: if found_email is true, say where on the page it appeared (e.g. 'listed as the recruiting contact on the careers page'); if false, explain briefly why (e.g. 'page only has a contact form, no email listed' or 'page is unrelated to this company')>"
}

Rules:
- Only report an email that is literally present in the page text (including
  from mailto: links if visible as text). Never construct or guess one from
  the company name or domain, even if it seems obvious.
- Prefer an address that's actually meant for applications/recruiting/HR
  over a generic support or sales address, if multiple are present.
- If the page has no email at all (e.g. it's just a contact FORM with no
  visible address, or is unrelated content), set found_email to false.
"""


class ContactFinderError(Exception):
    pass


def _search_contact_urls(company: str) -> list[dict]:
    query = CONTACT_SEARCH_QUERY_TEMPLATE.format(company=company)
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    except Exception as e:
        raise ContactFinderError(f"Web search failed: {e}") from e

    results = []
    for r in raw_results:
        url = r.get("href", r.get("link", "")).strip()
        title = r.get("title", "").strip()
        if url:
            results.append({"title": title, "url": url})
    return results


def _fetch_page_text(url: str) -> str | None:
    """Same approach as job_finder._fetch_page_text — failures return None
    rather than raising, so one bad URL doesn't kill the whole search."""
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


def _extract_contact_email(company: str, page_text: str, source_url: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ContactFinderError("GROQ_API_KEY is not set. Add it to your .env file.")
    client = Groq(api_key=api_key)

    user_prompt = f"""COMPANY: {company}
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
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise ContactFinderError(f"Groq API request failed: {e}") from e

    raw_content = response.choices[0].message.content
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ContactFinderError(f"Groq returned non-JSON output: {e}") from e

    required_keys = {"found_email", "email", "note"}
    if not required_keys.issubset(result.keys()):
        raise ContactFinderError(f"Groq response missing expected fields: {result}")

    return result


def _guess_official_domain(company: str) -> str | None:
    """
    Best-effort: search for the company's official site and take the
    domain of the first result. Falls back to a naive slug if search
    fails. This is only ever used to build labeled GUESSES (tier="guess"),
    never presented as a verified address.
    """
    query = DOMAIN_SEARCH_QUERY_TEMPLATE.format(company=company)
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=3))
        for r in raw_results:
            url = r.get("href", r.get("link", "")).strip()
            if not url:
                continue
            netloc = urlparse(url).netloc.lower()
            netloc = netloc[4:] if netloc.startswith("www.") else netloc
            # Skip obvious non-company domains that show up in generic searches
            if netloc and not any(bad in netloc for bad in ["wikipedia", "linkedin", "glassdoor", "indeed", "crunchbase"]):
                return netloc
    except Exception:
        pass

    # Fallback: naive slug (e.g. "Acme Corp" -> "acmecorp.com"). Crude, but
    # clearly labeled as a guess downstream — better than nothing for the
    # "guess" tier, never used for the "verified" tier.
    slug = re.sub(r"[^a-z0-9]", "", company.lower())
    return f"{slug}.com" if slug else None


def find_contact_email(company: str) -> dict:
    """
    Main entry point. Tries to find a real, sourced contact email first.
    Falls back to labeled pattern guesses if nothing real turns up.
    Never raises for "nothing found" — that's a valid, expected outcome
    (most companies use an ATS portal, not email) — only raises
    ContactFinderError if the search itself fails outright.

    Returns one of:
      {"tier": "verified", "email": ..., "source_url": ..., "source_title": ..., "note": ...}
      {"tier": "guess", "guesses": [...], "domain_guess": ..., "note": ...}
      {"tier": "none", "note": ...}
    """
    if not company or not company.strip():
        raise ContactFinderError("Company name is required.")

    candidates = _search_contact_urls(company)

    last_note = "No candidate pages could be checked."
    for candidate in candidates:
        page_text = _fetch_page_text(candidate["url"])
        if not page_text or len(page_text) < 100:
            continue

        extracted = _extract_contact_email(company, page_text, candidate["url"])
        email = extracted.get("email", "").strip()
        if extracted.get("found_email") and EMAIL_REGEX.match(email):
            return {
                "tier": "verified",
                "email": email,
                "source_url": candidate["url"],
                "source_title": candidate["title"],
                "note": extracted.get("note", ""),
            }
        last_note = extracted.get("note", last_note)

    domain = _guess_official_domain(company)
    if not domain:
        return {
            "tier": "none",
            "note": f"Couldn't find a real contact email or even guess a domain for '{company}'. Enter the recipient manually.",
        }

    guesses = [f"careers@{domain}", f"jobs@{domain}", f"hr@{domain}"]
    return {
        "tier": "guess",
        "guesses": guesses,
        "domain_guess": domain,
        "note": (
            f"No published contact email found ({last_note}) These are unverified common-pattern "
            f"guesses at {domain} — they may not exist or may not be monitored. Many companies only "
            f"accept applications through their careers portal, not email at all."
        ),
    }
