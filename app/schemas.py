from typing import Any, Literal
from pydantic import BaseModel, Field


SourceType = Literal["resume", "jobs", "projects", "notes", "unknown"]


class DocumentRecord(BaseModel):
    path: str
    source_type: SourceType
    size_bytes: int


class ChunkMetadata(BaseModel):
    chunk_id: str
    source_type: SourceType
    file_name: str
    relative_path: str
    chunk_index: int


class ChunkRecord(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    vector: list[float] | None = None


class Citation(BaseModel):
    label: str
    score: float
    source_type: str
    file_name: str
    relative_path: str
    text_preview: str


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    source_type: SourceType | None = None


class HybridSearchResponse(BaseModel):
    query: str
    results: list[Citation]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    source_type: SourceType | None = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace_id: str


class JobMatchRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    human_confirmed: bool = False
    top_k: int = Field(6, ge=1, le=20)


class JobProfile(BaseModel):
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    responsibilities: list[str] = []
    keywords: list[str] = []


class ResumeProfile(BaseModel):
    skills: list[str] = []
    projects: list[str] = []
    tools: list[str] = []
    evidence: list[str] = []


class MatchReport(BaseModel):
    score: int = Field(0, ge=0, le=100)
    matched_items: list[str] = []
    missing_items: list[str] = []
    evidence_sources: list[str] = []
    recommendations: list[str] = []
    narrative: str = ""


class AgentMatchResponse(BaseModel):
    job_profile: JobProfile
    resume_profile: ResumeProfile
    report: MatchReport
    citations: list[Citation]
    trace_id: str


class ReindexResponse(BaseModel):
    status: str
    document_count: int
    chunk_count: int
    collection: str


class EvalResult(BaseModel):
    question: str
    latency_ms: float
    retrieved_count: int
    keyword_recall: float
    source_type_hit: float
    citation_coverage: float


class EvalResponse(BaseModel):
    results: list[EvalResult]
    averages: dict[str, float]