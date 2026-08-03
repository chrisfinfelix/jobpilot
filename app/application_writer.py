"""
application_writer.py
-----------------------
Step 3: writes a tailored cover letter and resume bullet suggestions.

This is where Step 1 and Step 2 pay off — instead of asking the AI to
write a generic cover letter, we feed it:
  - the resume text itself
  - the job description
  - the match gaps from Step 1 (missing_skills) so it can honestly
    address them instead of overselling
  - the cited company research from Step 2 (if available) so the
    letter can reference real, specific things about the company
    instead of generic flattery ("I've always admired your innovative
    culture...")

Nothing here is sent to the user without them reviewing it first —
that review step happens in the frontend / Send Service (Step 5),
not here. This module only drafts.
"""

import json
import os
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a career-writing assistant that drafts a tailored
cover letter and resume bullet suggestions for a specific job application.

You will be given: a RESUME, a JOB DESCRIPTION, a list of MATCHED skills,
a list of MISSING skills, and (optionally) COMPANY RESEARCH notes.

Respond with ONLY a raw JSON object — no markdown fences, no commentary —
matching exactly this shape:

{
  "cover_letter": "<a complete, ready-to-edit cover letter, 3-4 paragraphs, plain text with \\n\\n between paragraphs>",
  "resume_bullets": [<strings: 3-5 NEW or IMPROVED resume bullet points tailored to this job, each starting with a strong action verb and, where the resume supports it, a quantified result>],
  "gaps_addressed_honestly": "<1-2 sentences on how the letter handles the candidate's missing skills, e.g. by highlighting transferable experience, WITHOUT lying about skills they don't have>",
  "talking_points": [<strings: 2-3 short bullet points the candidate could bring up in an interview, grounded in the resume and, if provided, the company research>]
}

CRITICAL RULES:
- NEVER claim the candidate has a skill or experience that isn't actually
  supported by their resume. If a required skill is missing, either omit
  it, or honestly frame nearby/transferable experience — never fabricate.
- The cover letter must be specific to THIS resume and THIS job description,
  not generic boilerplate. Reference actual resume details (projects,
  companies, quantified achievements) and actual job requirements.
- If COMPANY RESEARCH notes are provided, you may reference specific,
  real details from them (e.g. a specific product, value, or fact) to show
  genuine interest — but do NOT invent company facts that weren't given to
  you, and do not fabricate a personal connection to the company.
- If no COMPANY RESEARCH was provided, write a strong letter based on the
  resume and job description alone — do not invent company details to
  fill the gap.
- Keep the tone professional but human — avoid cliches like "I am writing
  to express my interest" or "team player" without substance behind it.
- resume_bullets should be usable as literal resume lines: concise,
  action-verb-first, no first-person pronouns ("I", "my").
"""


class ApplicationWriterError(Exception):
    pass


def _build_user_prompt(
    resume_text: str,
    job_description: str,
    matched_skills: list[str],
    missing_skills: list[str],
    research_context: dict | None,
) -> str:
    parts = [
        f"RESUME:\n{resume_text}",
        f"\nJOB DESCRIPTION:\n{job_description}",
        f"\nMATCHED SKILLS (candidate already has these):\n{', '.join(matched_skills) or 'None identified'}",
        f"\nMISSING SKILLS (candidate should not falsely claim these):\n{', '.join(missing_skills) or 'None identified'}",
    ]

    if research_context:
        overview = research_context.get("company_overview", "")
        culture = research_context.get("culture_notes", [])
        parts.append(
            "\nCOMPANY RESEARCH (real, sourced notes — you may reference these):\n"
            f"Overview: {overview}\n"
            f"Culture notes: {'; '.join(culture) if culture else 'none available'}"
        )
    else:
        parts.append("\nCOMPANY RESEARCH: none provided — do not invent company-specific details.")

    return "\n".join(parts)


def generate_application_materials(
    resume_text: str,
    job_description: str,
    matched_skills: list[str],
    missing_skills: list[str],
    research_context: dict | None = None,
) -> dict:
    """
    Returns a dict: { cover_letter, resume_bullets, gaps_addressed_honestly, talking_points }
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ApplicationWriterError("GROQ_API_KEY is not set. Add it to your .env file.")
    client = Groq(api_key=api_key)

    user_prompt = _build_user_prompt(
        resume_text, job_description, matched_skills, missing_skills, research_context
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,  # a bit higher than Step 1/2 — this is creative writing, not fact-checking
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise ApplicationWriterError(f"Groq API request failed: {e}") from e

    raw_content = response.choices[0].message.content

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ApplicationWriterError(f"Groq returned non-JSON output: {e}") from e

    required_keys = {"cover_letter", "resume_bullets", "gaps_addressed_honestly", "talking_points"}
    if not required_keys.issubset(result.keys()):
        raise ApplicationWriterError(f"Groq response missing expected fields: {result}")

    return result
