# JobPilot — Step 1: Resume Upload + Parsing + AI Match Score

This is the first build milestone from the JobPilot plan: upload a resume
(PDF, DOCX, or pasted text), give it a job description, and get back an
AI-generated match score, missing skills, and ATS warnings — powered by
Groq's free-tier API.

## What's in here

```
jobpilot/
├── app/
│   ├── main.py          # FastAPI app + the /api/match endpoint
│   ├── resume_parser.py # Extracts text from PDF / DOCX / pasted text
│   ├── groq_client.py   # Calls Groq's API, validates the JSON it returns
│   └── models.py        # Pydantic response models
├── static/
│   └── index.html       # A minimal test UI (upload + see the score)
├── requirements.txt
└── .env.example
```

## Setup

1. **Create a free Groq API key**
   Go to https://console.groq.com/keys, sign up (no credit card), and
   generate a key.

2. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your key**
   ```bash
   cp .env.example .env
   # then edit .env and paste your real key in:
   # GROQ_API_KEY=gsk_your_real_key_here
   ```

4. **Run it**
   ```bash
   uvicorn app.main:app --reload
   ```
   Open http://localhost:8000 — you'll see the test page. Upload a resume,
   paste a job description, and hit "Get Match Score."

   You can also test the API directly at http://localhost:8000/docs
   (FastAPI's auto-generated Swagger UI).

## How it works

1. `resume_parser.py` turns whatever you gave it (PDF bytes, DOCX bytes,
   or pasted text) into a single clean text string. This step does **no**
   AI — it's pure text extraction, so it's instant and free.
2. `groq_client.py` sends that resume text + the job description to
   Llama 3.3 70B on Groq, with a system prompt that forces it to respond
   in strict JSON (`match_score`, `matched_skills`, `missing_skills`,
   `ats_warnings`, `summary`). We validate the shape of what comes back
   before trusting it — if the model ever returns something malformed,
   the API returns a clean 502 instead of crashing.
3. `main.py` wires it together into one endpoint: `POST /api/match`.

## What was tested

- PDF, DOCX, and pasted-text parsing (unit tested, all pass)
- Unsupported file type → clean 400 error
- Empty/too-short resume → clean 422 error
- Missing/short job description → clean 400 error
- Missing `GROQ_API_KEY` → clean error message instead of a crash
- **Note:** I hit and fixed a real version bug during testing — `groq==0.11.0`
  crashes with newer `httpx` versions (`unexpected keyword argument 'proxies'`).
  `requirements.txt` is pinned to `groq==1.6.0`, which works correctly.

What I could **not** test in this environment: an actual live call to Groq's
API (this sandbox can't reach external AI APIs). Once you drop in your real
key and run it locally, that's the one thing to double check — everything
around it is verified.

## Known limitations (intentional, for Step 1 scope)

- No database yet — nothing is saved between requests. That's Step 6
  (tracker dashboard) in the build order.
- No auth — this is a single-user local test app for now.
- Scanned/image-only PDFs won't extract text (no OCR yet) — the app will
  tell you to paste the text instead if that happens.

## Next step

Per the build order: **Step 2 — Company/role research + AI summary (RAG)**.
That's the "wow" feature — plugging in real company data before asking the
AI to summarize it, instead of letting it guess.
