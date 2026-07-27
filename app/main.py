

from dotenv import load_dotenv
load_dotenv()  # must run before we read GROQ_API_KEY anywhere

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.resume_parser import parse_resume, UnsupportedFileType, EmptyResumeError
from app.groq_client import score_resume_against_job, GroqClientError
from app.models import MatchResult

app = FastAPI(title="JobPilot API", version="0.1.0")

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


@app.post("/api/match", response_model=MatchResult)
async def match_resume(
    job_description: str = Form(...),
    pasted_resume_text: str | None = Form(default=None),
    resume_file: UploadFile | None = File(default=None),
):
    """
    Accepts a job description plus EITHER an uploaded resume file
    (PDF/DOCX/TXT) OR pasted resume text, and returns an AI-generated
    match score.
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

    # --- Step B: send resume + JD to Groq for scoring ---
    try:
        result = score_resume_against_job(resume_text, job_description)
    except GroqClientError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return MatchResult(**result)


# Serve the simple test UI at http://localhost:8000/
app.mount("/", StaticFiles(directory="static", html=True), name="static")
