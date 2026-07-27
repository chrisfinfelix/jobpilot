from pydantic import BaseModel


class MatchResult(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    ats_warnings: list[str]
    summary: str


class ErrorResponse(BaseModel):
    error: str
