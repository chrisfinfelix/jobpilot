"""
main.py
-------
JobPilot Steps 1-4: resume parsing + match score + ATS compatibility checks,
company/role research (RAG), and tailored cover letter + resume bullet
generation.

Step 4 (ATS compatibility) is NOT a separate endpoint — it's merged directly
into /api/match's ats_warnings field, combining deterministic structural
checks (resume_parser.py) with AI keyword-alignment checks (groq_client.py).
See the /api/match handler below for details.

Endpoints:
  POST /api/match               -> upload a resume + job description, get a match score
                                    + merged ATS warnings (structural + keyword alignment).
  GET  /api/research             -> research a company/role with cited AI summary.
  GET  /api/find-job-description -> auto-fetch a real job posting for a company/role,
                                     to pre-fill the job description field.
  POST /api/generate             -> generate a tailored cover letter + resume bullets,
                                     using the match gaps and (if cached) company research.
  GET  /api/companies            -> bundled list of company names for the picker.
  GET  /api/roles                -> bundled list of role titles for the picker.
  GET  /health                    -> simple health check.

Step 5 (Gmail send flow):
  GET  /api/gmail/auth-url       -> builds the Google OAuth consent URL.
  GET  /api/gmail/callback       -> Google redirects here after consent; exchanges
                                     the code for tokens and stores them locally.
  GET  /api/gmail/status         -> whether Gmail is currently connected.
  POST /api/gmail/disconnect     -> deletes the locally stored token.
  POST /api/gmail/send           -> sends an email via Gmail API. Only ever called
                                     from the frontend's review-gate modal, after the
                                     user has seen and approved the final To/Subject/
                                     Body/Attachment. See app/gmail_client.py.

Run locally with:
  python -m uvicorn app.main:app --reload
"""

from dotenv import load_dotenv
load_dotenv()  # must run before we read GROQ_API_KEY anywhere

import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.resume_parser import parse_resume, UnsupportedFileType, EmptyResumeError, check_ats_structural_issues
from app.groq_client import score_resume_against_job, GroqClientError
from app.models import MatchResult, ResearchResult, GenerationResult, JobDescriptionResult
from app.research_cache import init_db, get_cached_research, set_cached_research
from app.web_research import gather_research, WebResearchError
from app.rag_summarizer import summarize_research, SummarizerError
from app.application_writer import generate_application_materials, ApplicationWriterError
from app.job_finder import find_job_description, JobFinderError
from app import gmail_client

app = FastAPI(title="JobPilot API", version="0.3.0")
init_db()  # make sure the research_cache.db table exists before we take traffic

# Load the bundled company/role lists once at startup — small static files,
# no need to re-read from disk on every request.
DATA_DIR = Path(__file__).parent / "data"
COMPANIES = json.loads((DATA_DIR / "companies.json").read_text())
ROLES = json.loads((DATA_DIR / "roles.json").read_text())

# Allow the local test frontend (and any dev frontend) to call this API.
# Tighten this to specific origins before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/companies")
def list_companies():
    """Returns the bundled list of company names for the search dropdown."""
    return {"companies": COMPANIES}


@app.get("/api/roles")
def list_roles():
    """Returns the bundled list of role titles for the search dropdown."""
    return {"roles": ROLES}


@app.post("/api/match", response_model=MatchResult)
async def match_resume(
    job_description: str = Form(...),
    pasted_resume_text: str | None = Form(default=None),
    resume_file: UploadFile | None = File(default=None),
):
    """
    Accepts a job description plus EITHER an uploaded resume file
    (PDF/DOCX/TXT) OR pasted resume text, and returns an AI-generated
    match score plus ATS compatibility warnings.

    ats_warnings in the response is a MERGE of two different kinds of
    checks (Step 4):
      - deterministic structural checks (resume_parser.check_ats_structural_issues) —
        missing section headers, table layouts, scanned PDFs, no contact info, etc.
        These are rule-based, instant, and don't depend on the AI's opinion.
      - AI keyword-alignment checks (groq_client.score_resume_against_job) —
        whether the resume's wording actually matches THIS job description's
        terms. This needs judgment, so it stays an AI call.
    """
    if not job_description or len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is too short or missing.")

    file_bytes = None
    filename = None
    if resume_file is not None:
        filename = resume_file.filename
        file_bytes = await resume_file.read()

    # --- Step A: parse the resume into plain text ---
    try:
        resume_text = parse_resume(filename, file_bytes, pasted_resume_text)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmptyResumeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Step B: deterministic ATS structural checks (Step 4, no AI, instant) ---
    structural_warnings = check_ats_structural_issues(filename, file_bytes, resume_text)

    # --- Step C: send resume + JD to Groq for scoring + keyword-alignment warnings ---
    try:
        result = score_resume_against_job(resume_text, job_description)
    except GroqClientError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # --- Step D: merge both warning sources, structural first, deduped ---
    combined_warnings = structural_warnings + [
        w for w in result.get("ats_warnings", []) if w not in structural_warnings
    ]
    result["ats_warnings"] = combined_warnings

    return MatchResult(**result)


@app.get("/api/research", response_model=ResearchResult)
def research_company_role(company: str, role: str, force_refresh: bool = False):
    """
    Researches a company + role using real web sources, then asks the AI
    to summarize ONLY from those sources (RAG) — with citations.

    Results are cached for 7 days per company+role so we don't re-research
    (and re-spend AI calls) every time someone searches the same thing.

    Query params:
      company        e.g. "Google"
      role            e.g. "Backend Engineer Intern"
      force_refresh   bypass the cache and re-research (default false)
    """
    if not company or not company.strip():
        raise HTTPException(status_code=400, detail="Company name is required.")
    if not role or not role.strip():
        raise HTTPException(status_code=400, detail="Role is required.")

    # --- Step A: check the cache first ---
    if not force_refresh:
        cached = get_cached_research(company, role)
        if cached is not None:
            return ResearchResult(
                **cached,
                from_cache=True,
                cached_age_hours=cached.get("_cached_age_hours"),
            )

    # --- Step B: gather real sources from the web ---
    try:
        sources = gather_research(company, role)
    except WebResearchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # --- Step C: RAG summarize — AI explains ONLY what the sources say ---
    try:
        result = summarize_research(company, role, sources)
    except SummarizerError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # --- Step D: cache it for next time ---
    set_cached_research(company, role, result)

    return ResearchResult(**result, from_cache=False)


@app.get("/api/find-job-description", response_model=JobDescriptionResult)
def find_job_description_endpoint(company: str, role: str, force_refresh: bool = False):
    """
    Auto-fetches a real job description for a company + role, instead of
    requiring the user to paste one manually.

    Searches the web for an actual posting, fetches the page, and asks the
    AI to EXTRACT (not invent) the real job description text from it — the
    same RAG discipline as Step 2, applied to job postings instead of
    company research. Cached for 7 days under a separate "jobdesc"
    namespace so it never collides with Step 2's company research cache.

    IMPORTANT: this returns text meant to pre-fill the job description
    field for the user to review and edit — never auto-submitted anywhere
    on its own. If no real posting could be found or extracted, this
    returns a clean error so the user can fall back to pasting manually.

    Query params:
      company        e.g. "Google"
      role            e.g. "Backend Engineer Intern"
      force_refresh   bypass the cache and re-search (default false)
    """
    if not company or not company.strip():
        raise HTTPException(status_code=400, detail="Company name is required.")
    if not role or not role.strip():
        raise HTTPException(status_code=400, detail="Role is required.")

    if not force_refresh:
        cached = get_cached_research(company, role, namespace="jobdesc")
        if cached is not None:
            return JobDescriptionResult(
                **cached,
                from_cache=True,
                cached_age_hours=cached.get("_cached_age_hours"),
            )

    try:
        result = find_job_description(company, role)
    except JobFinderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    set_cached_research(company, role, result, namespace="jobdesc")

    return JobDescriptionResult(**result, from_cache=False)


@app.post("/api/generate", response_model=GenerationResult)
async def generate_application(
    job_description: str = Form(...),
    pasted_resume_text: str | None = Form(default=None),
    resume_file: UploadFile | None = File(default=None),
    company: str | None = Form(default=None),
    role: str | None = Form(default=None),
):
    """
    Step 3: generates a tailored cover letter + resume bullets.

    Reuses Step 1 (parses the resume, scores it against the job) and,
    if company + role are given AND a cached Step 2 research result
    already exists, folds that in as real context for the letter.

    We deliberately do NOT trigger a fresh web search here even if
    company/role are given — that's a slower, separate action the user
    already took via /api/research. This endpoint only uses what's
    already cached, so generating a cover letter stays fast.
    """
    if not job_description or len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is too short or missing.")

    file_bytes = None
    filename = None
    if resume_file is not None:
        filename = resume_file.filename
        file_bytes = await resume_file.read()

    # --- Step A: parse the resume (same as Step 1) ---
    try:
        resume_text = parse_resume(filename, file_bytes, pasted_resume_text)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmptyResumeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Step B: score it against the job (same as Step 1) ---
    try:
        match_result = score_resume_against_job(resume_text, job_description)
    except GroqClientError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # --- Step C: pull cached research if we have it (from Step 2) ---
    research_context = None
    if company and role:
        research_context = get_cached_research(company, role)

    # --- Step D: generate the cover letter + bullets ---
    try:
        generated = generate_application_materials(
            resume_text=resume_text,
            job_description=job_description,
            matched_skills=match_result["matched_skills"],
            missing_skills=match_result["missing_skills"],
            research_context=research_context,
        )
    except ApplicationWriterError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return GenerationResult(
        match_score=match_result["match_score"],
        matched_skills=match_result["matched_skills"],
        missing_skills=match_result["missing_skills"],
        used_company_research=research_context is not None,
        **generated,
    )


@app.get("/api/gmail/auth-url")
def gmail_auth_url():
    """Frontend calls this, then does window.location.href = data.auth_url
    to send the browser to Google's consent screen."""
    try:
        url = gmail_client.build_auth_url()
        return {"auth_url": url}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gmail/callback")
def gmail_callback(code: str, state: str):
    """Google redirects here after the user approves/denies consent.
    On success, redirects back to the main page with a query flag the
    frontend checks on load to refresh its 'connected' status."""
    try:
        gmail_client.handle_callback(code, state)
    except ValueError as e:
        # bad/replayed state — send them back with an error flag instead of 500ing
        return RedirectResponse(url="/?gmail_error=" + str(e).replace(" ", "+"))
    except Exception as e:
        return RedirectResponse(url="/?gmail_error=" + str(e).replace(" ", "+"))

    return RedirectResponse(url="/?gmail_connected=true")


@app.get("/api/gmail/status")
def gmail_status():
    return {"connected": gmail_client.is_connected()}


@app.post("/api/gmail/disconnect")
def gmail_disconnect():
    gmail_client.disconnect()
    return {"connected": False}


@app.post("/api/gmail/send")
async def gmail_send(
    to: str = Form(...),
    subject: str = Form(...),
    body_text: str = Form(...),
    attachment: UploadFile | None = File(default=None),
):
    """
    Called ONLY from the review-gate modal's 'Confirm & Send' button —
    every field here has already been shown to and approved by the user.
    This endpoint does not re-generate or alter content; it just sends
    exactly what's in the form.
    """
    attachments = []
    if attachment is not None:
        file_bytes = await attachment.read()
        attachments.append((attachment.filename, file_bytes, attachment.content_type or "application/octet-stream"))

    try:
        result = gmail_client.send_email(to=to, subject=subject, body_text=body_text, attachments=attachments)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"sent": True, "message_id": result["id"], "thread_id": result["threadId"]}


# Serve the simple test UI at http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")