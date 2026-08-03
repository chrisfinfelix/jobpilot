from pydantic import BaseModel


class MatchResult(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    ats_warnings: list[str]
    summary: str


class ErrorResponse(BaseModel):
    error: str


class SourceRef(BaseModel):
    index: int
    title: str
    url: str


class ResearchResult(BaseModel):
    company_overview: str
    role_expectations: list[str]
    interview_process: list[str]
    culture_notes: list[str]
    confidence: str
    sources: list[SourceRef]
    from_cache: bool = False
    cached_age_hours: float | None = None


class GenerationResult(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    cover_letter: str
    resume_bullets: list[str]
    gaps_addressed_honestly: str
    talking_points: list[str]
    used_company_research: bool


class JobDescriptionResult(BaseModel):
    job_description: str
    source_url: str
    source_title: str
    note: str
    from_cache: bool = False
    cached_age_hours: float | None = None