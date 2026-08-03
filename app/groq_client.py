"""
groq_client.py
---------------
Wraps calls to the Groq API. For Step 1, we use it for one job: compare
a resume against a job description and return a structured match score,
missing skills, keyword-alignment ATS warnings, and a summary.

Step 4 note: this prompt's ats_warnings is scoped to KEYWORD ALIGNMENT
only (does the resume use the terms this specific JD uses) — the fuzzy
judgment call that actually needs a model. Structural ATS issues (missing
section headers, table layouts, no contact info, scanned PDFs, etc.) are
deterministic and checked separately in resume_parser.py, since those are
objectively true/false and don't need an LLM to guess at. main.py merges
both lists before returning the response.

We ask the model to return ONLY JSON (no preamble) so we can parse it
reliably, and we validate the shape before trusting it.
"""

import json
import os
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a strict, honest resume-to-job matching engine.
You will be given a RESUME and a JOB DESCRIPTION.

Compare them carefully and respond with ONLY a raw JSON object — no markdown
fences, no commentary, no preamble. The JSON must exactly match this shape:

{
  "match_score": <integer 0-100>,
  "matched_skills": [<strings: skills/requirements the resume DOES cover>],
  "missing_skills": [<strings: skills/requirements from the JD the resume does NOT show>],
  "ats_warnings": [<strings: KEYWORD ALIGNMENT issues only — see rules below>],
  "summary": "<one or two sentence honest summary of the fit>"
}

Rules:
- Be honest and specific. Do not inflate the score to be encouraging.
- match_score should reflect real overlap between resume content and JD requirements.
- missing_skills should only list things genuinely absent or unclear in the resume.
- ats_warnings is SCOPED TO KEYWORD ALIGNMENT ONLY. Do not comment on formatting,
  layout, section headers, tables, images, or file type — those are checked
  separately by a different system. Only flag things like: the JD's exact job
  title or key terms not appearing anywhere in the resume, an important
  JD-specific keyword/tool/technology missing from the resume text, or the
  resume using different terminology than the JD for the same thing (e.g. JD
  says "React.js", resume only says "front-end frameworks"). Keep it to 0-4 items.
- If there are no real keyword-alignment issues, return an empty list — do not
  invent minor items just to fill the list.
- If the resume is clearly unrelated to the job, say so honestly in the summary
  and give a low score.
"""


class GroqClientError(Exception):
    pass


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqClientError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def score_resume_against_job(resume_text: str, job_description: str) -> dict:
    """
    Sends resume + job description to Groq and returns a parsed dict:
    { match_score, matched_skills, missing_skills, ats_warnings, summary }

    Raises GroqClientError on any failure (missing key, API error, bad JSON).
    """
    client = _get_client()

    user_prompt = f"""RESUME:
{resume_text}

---

JOB DESCRIPTION:
{job_description}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # low temperature: we want consistent, factual scoring
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise GroqClientError(f"Groq API request failed: {e}") from e

    raw_content = response.choices[0].message.content

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise GroqClientError(f"Groq returned non-JSON output: {e}") from e

    # Basic shape validation so bad AI output doesn't silently break the app
    required_keys = {"match_score", "matched_skills", "missing_skills", "ats_warnings", "summary"}
    if not required_keys.issubset(result.keys()):
        raise GroqClientError(f"Groq response missing expected fields: {result}")

    # Clamp score defensively in case the model ignores the 0-100 instruction
    result["match_score"] = max(0, min(100, int(result["match_score"])))

    return result